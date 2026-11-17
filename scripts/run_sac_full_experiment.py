#!/usr/bin/env python3
"""Wait for qualified preparation, then execute paired BC/SAC and baseline runs.

Never manufactures gate approval or edits source. Any failed dependency/run
stops with preserved evidence. The initial SAC budget is10000 transitions/seed.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "pretraining", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--transitions", type=int, default=10000)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.transitions < 10000 or args.transitions % 5000:
        p.error("online budget must be >=10000 and a multiple of5000")
    args.manifest = args.manifest.resolve()
    args.pretraining = args.pretraining.resolve()
    args.output = args.output.resolve()
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / ".job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "src", "scripts", "configs/contracts"],
        cwd=ROOT,
    ).returncode:
        raise ValueError("use a clean fixed revision")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    state = dict(
        status="WAITING_FOR_PRETRAINING",
        pid=os.getpid(),
        revision=revision,
        manifest=str(args.manifest),
        pretraining=str(args.pretraining),
        budget_per_seed=args.transitions,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        phases={},
    )
    if args.resume:
        prior = json.loads((args.output / "job.json").read_text())
        for key in (
            "revision",
            "manifest",
            "manifest_sha256",
            "pretraining",
            "budget_per_seed",
        ):
            if prior[key] != state[key]:
                raise ValueError(f"experiment resume provenance mismatch: {key}")
    children = []

    def save():
        state["updated_unix"] = time.time()
        tmp = args.output / "job.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.output / "job.json")

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def run(name, script, arguments, backend="cpu"):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        (args.output / "logs").mkdir(exist_ok=True)
        env = dict(
            os.environ,
            JAX_PLATFORMS=backend,
            JAX_ENABLE_X64="true",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            TF_CPP_MIN_LOG_LEVEL="3",
            PYTHONUNBUFFERED="1",
        )
        with (args.output / "logs" / f"{name}.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            children.append(child)
            state["phases"][name] = dict(
                pid=child.pid, status="RUNNING", command=command
            )
            save()
            while child.poll() is None:
                producer_now = json.loads((args.pretraining / "job.json").read_text())
                if producer_now["status"] not in (
                    "RUNNING",
                    "GATES_PASSED_ONLINE_SAC_NOT_STARTED",
                ):
                    raise RuntimeError(
                        f"pretraining stopped: {producer_now.get('error', producer_now['status'])}"
                    )
                save()
                time.sleep(10)
            state["phases"][name].update(
                exit_code=child.returncode,
                status="COMPLETE" if child.returncode == 0 else "FAILED",
            )
            save()
            if child.returncode:
                raise RuntimeError(f"{name} failed; inspect logs/{name}.log")

    def evaluated(name, policy, seed, checkpoint=None):
        destination = args.output / "evaluation" / f"{name}.json"
        if destination.exists():
            old = json.loads(destination.read_text())
            if old.get("status") == "COMPLETE" and old.get("qualified") is True:
                if (
                    checkpoint is not None
                    and old.get("checkpoint_sha256")
                    != hashlib.sha256(
                        (checkpoint / "manifest.json").read_bytes()
                    ).hexdigest()
                ):
                    raise ValueError("cached evaluation checkpoint changed")
                return destination
            # Preserve partial/error evidence on explicit resume; evaluate afresh.
            destination.rename(
                destination.with_name(
                    destination.stem + f"-interrupted-{time.time_ns()}.json"
                )
            )
        params = [
            "evaluate",
            "--manifest",
            args.manifest,
            "--output",
            destination,
            "--policy",
            policy,
            "--seed",
            seed,
        ]
        if checkpoint is not None:
            params.extend(["--checkpoint", checkpoint])
        run(name, "evaluate_sac_models.py", params, "cuda,cpu" if checkpoint else "cpu")
        return destination

    save()
    try:
        producer = json.loads((args.pretraining / "job.json").read_text())
        if Path(producer["manifest"]).resolve() != args.manifest:
            raise ValueError("pretraining manifest mismatch")
        state["status"] = "RUNNING_BASELINES"
        reports = []
        # Independent baseline evaluations can proceed while data is collected.
        # Every proposal still passes the same production execution boundary.
        for policy in (
            "teacher",
            "stationary_proposal",
            "braking_proposal",
            "always_rejected",
            "collision_seeking",
        ):
            reports.append(evaluated(f"baseline-{policy}", policy, 0))
        state["status"] = "WAITING_FOR_PRETRAINING"
        while True:
            producer = json.loads((args.pretraining / "job.json").read_text())
            if Path(producer["manifest"]).resolve() != args.manifest:
                raise ValueError("pretraining manifest mismatch")
            if producer["status"] == "GATES_PASSED_ONLINE_SAC_NOT_STARTED":
                break
            if producer["status"] != "RUNNING":
                raise RuntimeError(
                    f"pretraining stopped: {producer.get('error',producer['status'])}"
                )
            if time.time() - producer["updated_unix"] > 120:
                raise RuntimeError(
                    "pretraining heartbeat stale; review recovery before continuing"
                )
            save()
            time.sleep(30)
        state["status"] = "RUNNING"
        save()
        run(
            "collection-audit",
            "audit_sac_collection.py",
            [
                "--trace",
                args.pretraining / "collection" / "ticks.jsonl",
                "--output",
                args.output / "collection-audit.json",
            ],
        )
        for seed in (0, 1, 2):
            bc = args.pretraining / f"bc-seed{seed}"
            gate = Path(producer["phases"][f"gate-seed{seed}"]["report"])
            reports.append(evaluated(f"bc-seed{seed}", "BC", seed, bc))
            online = args.output / f"online-seed{seed}"
            completed = False
            if (online / "report.json").exists():
                previous = json.loads((online / "report.json").read_text())
                completed = (
                    previous["status"] == "COMPLETE"
                    and previous.get("qualified") is True
                )
            if not completed:
                params = [
                    "--mode",
                    "bc-sac",
                    "--manifest",
                    args.manifest,
                    "--bc",
                    bc,
                    "--gate",
                    gate,
                    "--synthetic",
                    args.pretraining / "synthetic.json",
                    "--output",
                    online,
                    "--seed",
                    seed,
                    "--transitions",
                    args.transitions,
                ]
                if online.exists():
                    params.append("--resume")
                run(f"online-seed{seed}", "run_sac_online.py", params, "cuda,cpu")
            trained = json.loads((online / "report.json").read_text())
            if (
                trained["budget"] != args.transitions
                or trained.get("qualified") is not True
            ):
                raise ValueError("online run incomplete/budget mismatch")
            checkpoint = online / trained["latest_checkpoint"]
            reports.append(evaluated(f"sac-seed{seed}", "SAC", seed, checkpoint))
        run(
            "comparison",
            "evaluate_sac_models.py",
            [
                "compare",
                "--reports",
                *reports,
                "--output",
                args.output / "comparison.json",
            ],
        )
        state.update(
            status="COMPLETE",
            comparison=str(args.output / "comparison.json"),
            interpretation="Experiment completed; improvement must be read from paired metrics, not inferred from this status.",
        )
    except BaseException as exc:
        state.update(status="STOPPED_FOR_REVIEW", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        save()


if __name__ == "__main__":
    main()

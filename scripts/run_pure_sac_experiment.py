#!/usr/bin/env python3
"""Run Rule-based vs pure SAC, with separate evidence and no BC dependency."""
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
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "synthetic", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--transitions", type=int, default=15000)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--baseline-report",
        type=Path,
        help="Reuse a complete identical-runtime Frenet evaluation",
    )
    args = p.parse_args()
    if args.transitions < 15000 or args.transitions % 5000:
        p.error("total budget must be >=15000 and a multiple of5000")
    for name in ("manifest", "synthetic", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
        cwd=ROOT,
        text=True,
    ).strip():
        raise ValueError("use a clean fixed source revision")
    from scripts.pure_sac_protocol import protocol_metadata

    state = dict(
        status="RUNNING",
        pid=os.getpid(),
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest=str(args.manifest),
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        synthetic_sha256=hashlib.sha256(args.synthetic.read_bytes()).hexdigest(),
        protocol=protocol_metadata(),
        budget_per_seed=args.transitions,
        phases={},
        baseline_report=(
            str(args.baseline_report.resolve()) if args.baseline_report else None
        ),
        baseline_sha256=(
            hashlib.sha256(args.baseline_report.read_bytes()).hexdigest()
            if args.baseline_report
            else None
        ),
    )
    args.output.mkdir(parents=True, exist_ok=args.resume)
    lock = (args.output / ".job.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.resume:
        prior = json.loads((args.output / "job.json").read_text())
        for key in (
            "revision",
            "runner_sha256",
            "manifest",
            "manifest_sha256",
            "synthetic_sha256",
            "protocol",
            "budget_per_seed",
            "baseline_report",
            "baseline_sha256",
        ):
            if prior[key] != state[key]:
                raise ValueError(f"pure experiment resume mismatch: {key}")
        state["phases"] = prior["phases"]
    child = None

    def save():
        state["updated_unix"] = time.time()
        tmp = args.output / "job.json.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(args.output / "job.json")

    def interrupted(signum, frame):
        raise InterruptedError(f"signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    def run(name, script, params, backend="cpu"):
        nonlocal child
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, params)]
        env = dict(
            os.environ,
            JAX_PLATFORMS=backend,
            JAX_ENABLE_X64="true",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            TF_CPP_MIN_LOG_LEVEL="3",
            PYTHONUNBUFFERED="1",
        )
        (args.output / "logs").mkdir(exist_ok=True)
        with (args.output / "logs" / f"{name}.log").open("a") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            state["phases"][name] = dict(
                status="RUNNING", pid=child.pid, command=command
            )
            save()
            while child.poll() is None:
                save()
                time.sleep(10)
            state["phases"][name].update(
                status="COMPLETE" if child.returncode == 0 else "FAILED",
                exit_code=child.returncode,
            )
            save()
            if child.returncode:
                raise RuntimeError(f"{name} failed; inspect logs/{name}.log")

    def evaluated(name, policy, seed, checkpoint=None):
        destination = args.output / "evaluation" / f"{name}.json"
        if destination.exists():
            old = json.loads(destination.read_text())
            if (
                old.get("status") == "COMPLETE"
                and old.get("qualified") is True
                and old.get("policy") == policy
                and old.get("seed") == seed
                and old["versions"]["manifest_sha256"] == state["manifest_sha256"]
            ):
                if (
                    checkpoint
                    and old.get("checkpoint_sha256")
                    != hashlib.sha256(
                        (checkpoint / "manifest.json").read_bytes()
                    ).hexdigest()
                ):
                    raise ValueError("cached evaluation checkpoint changed")
                return destination
            destination.rename(
                destination.with_name(
                    f"{destination.stem}-interrupted-{time.time_ns()}.json"
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
        if checkpoint:
            params.extend(["--checkpoint", checkpoint])
        run(name, "evaluate_sac_models.py", params, "cuda,cpu" if checkpoint else "cpu")
        return destination

    save()
    try:
        # Validates/resumes the full existing synthetic test without teacher data.
        run(
            "synthetic",
            "validate_sac_exploration.py",
            ["--manifest", args.manifest, "--output", args.synthetic],
        )
        if args.baseline_report:
            from scripts.pure_sac_protocol import validate_rule_baseline

            manifest = json.loads(args.manifest.read_text())
            synthetic = json.loads(args.synthetic.read_text())
            validate_rule_baseline(
                json.loads(args.baseline_report.read_text()),
                synthetic["versions"],
                [
                    (r["scene_id"], r["provider"])
                    for r in manifest["rows"]
                    if r["split"] == "validation"
                ],
            )
            reports = [args.baseline_report.resolve()]
            state["phases"]["rule-based"] = dict(
                status="REUSED_VERIFIED",
                report=state["baseline_report"],
                sha256=state["baseline_sha256"],
            )
            save()
        else:
            reports = [evaluated("rule-based", "teacher", 0)]
        for seed in (0, 1, 2):
            online = args.output / f"online-seed{seed}"
            prior = (
                json.loads((online / "report.json").read_text())
                if (online / "report.json").exists()
                else {}
            )
            if prior.get("status") != "COMPLETE" or prior.get("qualified") is not True:
                params = [
                    "--mode",
                    "pure-sac",
                    "--manifest",
                    args.manifest,
                    "--synthetic",
                    args.synthetic,
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
                trained.get("training_mode") != "pure-sac"
                or trained.get("protocol") != state["protocol"]
                or trained.get("budget") != args.transitions
                or trained.get("qualified") is not True
            ):
                raise ValueError("pure SAC training provenance/budget mismatch")
            reports.append(
                evaluated(
                    f"pure-sac-seed{seed}",
                    "PURE_SAC",
                    seed,
                    online / trained["latest_checkpoint"],
                )
            )
        run(
            "comparison",
            "evaluate_sac_models.py",
            [
                "compare",
                "--experiment",
                "pure-sac",
                "--reports",
                *reports,
                "--output",
                args.output / "comparison.json",
            ],
        )
        state.update(
            status="COMPLETE",
            comparison=str(args.output / "comparison.json"),
            interpretation="Completion is not evidence of improvement; inspect all paired metrics.",
        )
    except BaseException as exc:
        state.update(status="STOPPED_FOR_REVIEW", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        save()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run 20k collection, 4096/case exploration, BC5000 and validation gates.

This job does not launch online SAC. It stops for diagnosis at the first failed
gate, retaining all evidence; loss reduction never overrides acceptance gates.
Run from a fixed revision/worktree. Child logs and atomic status survive the UI.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    manifest = args.manifest.resolve()
    if output.exists() and not args.resume:
        parser.error("fresh output required, or explicitly --resume")
    output.mkdir(parents=True, exist_ok=args.resume)
    lock = (output / ".job.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("another job already owns this output directory")
    (output / "logs").mkdir(exist_ok=True)
    children = []
    logs = []
    status = dict(
        status="RUNNING",
        pid=os.getpid(),
        started_unix=time.time(),
        worktree=str(ROOT),
        manifest=str(manifest),
        phases={},
        seeds=[0, 1, 2],
        scope="PRETRAINING_AND_GATES_ONLY",
    )
    status["revision"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", "src", "scripts", "configs/contracts"],
        cwd=ROOT,
    ).returncode:
        parser.error("use a fixed revision with clean source/scripts/contracts")

    def save():
        status["updated_unix"] = time.time()
        temporary = output / "job.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(output / "job.json")

    def start(name, script, arguments, backend="cpu"):
        command = [sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)]
        log = (output / "logs" / f"{name}.log").open("a")
        logs.append(log)
        env = dict(
            os.environ,
            JAX_PLATFORMS=backend,
            JAX_ENABLE_X64="true",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            TF_CPP_MIN_LOG_LEVEL="3",
            PYTHONUNBUFFERED="1",
        )
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append(child)
        status["phases"][name] = dict(pid=child.pid, status="RUNNING", command=command)
        save()
        return child

    def wait(name, child):
        while child.poll() is None:
            for phase_name, phase in status["phases"].items():
                process = next(p for p in children if p.pid == phase["pid"])
                code = process.poll()
                if code is not None:
                    phase.update(
                        exit_code=code, status="COMPLETE" if code == 0 else "FAILED"
                    )
                    if code:
                        raise RuntimeError(f"{phase_name} failed with exit {code}")
            save()
            time.sleep(10)
        code = child.returncode
        status["phases"][name].update(
            exit_code=code, status="COMPLETE" if code == 0 else "FAILED"
        )
        save()
        if code:
            raise RuntimeError(
                f"{name} failed with exit {code}; inspect logs/{name}.log"
            )

    def interrupted(signum, frame):
        raise InterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        synthetic = start(
            "synthetic",
            "validate_sac_exploration.py",
            ["--manifest", manifest, "--output", output / "synthetic.json"],
        )
        collection_args = [
            "collect",
            "--manifest",
            manifest,
            "--output",
            output / "collection",
        ]
        if (output / "collection").exists():
            collection_args.append("--resume")
        collection = start("collection", "run_sac_pipeline.py", collection_args)
        wait("collection", collection)
        wait("synthetic", synthetic)
        for seed in (0, 1, 2):
            checkpoint = output / f"bc-seed{seed}"
            if not checkpoint.exists():
                name = f"bc-seed{seed}"
                wait(
                    name,
                    start(
                        name,
                        "run_sac_pipeline.py",
                        [
                            "bc",
                            "--manifest",
                            manifest,
                            "--data",
                            output / "collection",
                            "--output",
                            checkpoint,
                            "--seed",
                            seed,
                        ],
                        "cuda",
                    ),
                )
            # A fresh attempt keeps incomplete gate evidence for diagnosis. The
            # validator verifies all checkpoint/data/source hashes on every run.
            gate = output / f"gate-seed{seed}-{time.time_ns()}.json"
            name = f"gate-seed{seed}"
            wait(
                name,
                start(
                    name,
                    "validate_sac_bc.py",
                    [
                        "--manifest",
                        manifest,
                        "--checkpoint",
                        checkpoint,
                        "--synthetic",
                        output / "synthetic.json",
                        "--output",
                        gate,
                    ],
                    "cuda,cpu",
                ),
            )
            if not json.loads(gate.read_text()).get("passed"):
                raise RuntimeError(f"gate did not pass: {gate.name}")
            status["phases"][name]["report"] = str(gate)
            save()
        status["status"] = "GATES_PASSED_ONLINE_SAC_NOT_STARTED"
    except BaseException as exc:
        status.update(status="STOPPED_FOR_REVIEW", error=f"{type(exc).__name__}: {exc}")
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
        for log in logs:
            log.close()
        save()


if __name__ == "__main__":
    main()

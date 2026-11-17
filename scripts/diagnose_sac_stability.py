#!/usr/bin/env python3
"""Paired critic ablation on identical frozen self-collected warmup batches.

Diagnostic only: no simulator, no new experience, no qualified training output.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=2000)
    args = p.parse_args()
    import jax, numpy as np
    from src.contracts.observation import CoreObservation
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import Replay
    from src.training.sac.networks import Critic
    from src.training.sac import learner
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.runtime import write_report

    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    meta = json.loads((args.checkpoint / "manifest.json").read_text())
    if file_hash(args.checkpoint / "replay.npz") != meta["hashes"]["replay.npz"]:
        raise ValueError("source replay checksum mismatch")
    replay = Replay.load(args.checkpoint / "replay.npz", meta["versions"])
    if replay.size < 5000 or np.any(replay.arrays["teacher_accepted"][:5000]):
        raise ValueError("requires 5000 self-collected warmup transitions")
    # Only the original 5000 pre-update transitions, not diverged-policy data.
    replay.size = replay.cursor = 5000
    result = dict(
        scope="FIXED_SELF_WARMUP_CRITIC_ABLATION",
        qualified=False,
        source_checkpoint_sha256=file_hash(args.checkpoint / "manifest.json"),
        source_replay_sha256=meta["hashes"]["replay.npz"],
        requested_updates=args.updates,
        note="Both arms use float64 gradient-norm accumulation and identical replay indices. This is numerical stability, not online policy performance.",
        arms={},
    )
    original = learner.Critic
    for name, layer_norm in (("plain_critic", False), ("layer_norm_critic", True)):
        learner.Critic = lambda: Critic(layer_norm=layer_norm)
        state = learner.initialize(tensor_observation(CoreObservation.empty()), 0)
        replay.rng = np.random.default_rng(0)
        arm = dict(status="RUNNING", updates=0, metrics=[], elapsed_s=0.0)
        result["arms"][name] = arm
        start = time.monotonic()
        try:
            for i in range(args.updates):
                state, metrics = learner.update(state, replay.sample(256), replay.size)
                values = {k: float(v) for k, v in metrics.items()}
                arm["updates"] = i + 1
                if i < 10 or (i + 1) % 25 == 0:
                    arm["metrics"].append(dict(update=i + 1, **values))
                if i == 0 or (i + 1) % 100 == 0:
                    arm["elapsed_s"] = time.monotonic() - start
                    write_report(args.output / "report.json", result)
                    print(
                        json.dumps(dict(arm=name, update=i + 1, **values)), flush=True
                    )
                if (
                    max(
                        values["actor_q1_abs_max"],
                        values["actor_q2_abs_max"],
                        values["target_abs_max"],
                    )
                    > 1e9
                ):
                    arm.update(status="DIVERGED", last_metrics=values)
                    break
            else:
                arm.update(status="FINITE", last_metrics=values)
        except FloatingPointError as exc:
            arm.update(status="NONFINITE", error=str(exc))
        finally:
            arm["elapsed_s"] = time.monotonic() - start
            write_report(args.output / "report.json", result)
            del state
            jax.clear_caches()
    learner.Critic = original


if __name__ == "__main__":
    main()

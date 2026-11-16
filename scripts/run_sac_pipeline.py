#!/usr/bin/env python3
"""Run versioned SAC preparation. Smoke output is never a training gate pass."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "bc"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    from src.training.sac.runtime import collect_teacher, train_bc

    if args.stage == "collect":
        collect_teacher(
            args.manifest,
            args.output,
            args.seed,
            target=20 if args.smoke else 20000,
            smoke=args.smoke,
        )
    else:
        if args.data is None:
            parser.error("bc requires --data collection directory")
        train_bc(
            args.manifest,
            args.data,
            args.output,
            args.seed,
            updates=2 if args.smoke else 5000,
            smoke=args.smoke,
        )


if __name__ == "__main__":
    main()

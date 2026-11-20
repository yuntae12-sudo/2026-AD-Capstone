#!/usr/bin/env python3
"""Run an unchanged fresh diagnostic driver while recording its initial learner.

Example: --driver refresh --expect-initial CONTROL/initial-learner.msgpack --
         --checkpoint ... --configuration ... --output ... [driver arguments]
The wrapper intercepts the first update for serialization only; it neither
restores a model nor modifies the loss, batch, RNG, parameters or returned state.
"""
import argparse
import importlib
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", choices=("retention", "refresh"), required=True)
    parser.add_argument("--expect-initial", type=Path)
    parser.add_argument("driver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.driver_args
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if any(a.split("=", 1)[0] == "--continue-from" for a in argv):
        parser.error("initial capture is only for fresh runs")
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(argv)
    if output.output.exists():
        raise FileExistsError(output.output)
    if args.expect_initial is not None and not args.expect_initial.is_file():
        raise FileNotFoundError(args.expect_initial)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from flax import serialization
    from src.training.sac import learner
    from src.training.sac.checkpoint import file_hash
    from scripts.sac_initial_state_capture import capture_first_update
    driver = importlib.import_module("scripts.diagnose_sac_terminal_" + args.driver)
    original_update, original_argv = learner.update, sys.argv
    learner.update = capture_first_update(
        original_update, serialization.to_bytes, output.output,
        expected=args.expect_initial,
        provenance=dict(wrapper_sha256=file_hash(__file__),
                        capture_sha256=file_hash(Path(__file__).with_name("sac_initial_state_capture.py")),
                        driver_sha256=file_hash(driver.__file__), driver_arguments=argv),
    )
    sys.argv = [driver.__file__, *argv]
    try:
        driver.main()
        if not (output.output / "initial-state-audit.json").is_file():
            raise RuntimeError("driver completed without recording its initial learner")
    finally:
        learner.update, sys.argv = original_update, original_argv


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Revalidate every candidate with the same production reset, preserving exclusions."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.parent.resolve() != args.input.parent.resolve():
        raise ValueError(
            "keep the same inventory directory for candidate paths/checksums"
        )
    from src.training.sac.runtime import selected_scenes, versions, write_report
    from src.training.sac.checkpoint import file_hash
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.environment.womd_control.geometry import ResetRejected
    from src.scenarios.womd20 import parse_scenario

    original = json.loads(args.input.read_text())
    rows = []
    exclusions = []
    for split in ("train", "validation"):
        for row, path in selected_scenes(args.input, split):
            try:
                SACPlanningEnvironment().reset(
                    parse_scenario(path.read_bytes()), provider_spec=row["provider"]
                )
            except ResetRejected as exc:
                exclusions.append(
                    dict(
                        scene_id=row["scene_id"],
                        provider=row["provider"],
                        split=split,
                        reason=str(exc),
                    )
                )
            else:
                rows.append(
                    dict(
                        row,
                        revalidated_runtime_sha256=versions(args.input)["code_sha256"],
                    )
                )
    result = dict(
        original,
        rows=rows,
        reset_exclusions=exclusions,
        parent_manifest_sha256=file_hash(args.input),
        revalidation_runtime=versions(args.input),
    )
    write_report(args.output, result)
    print(json.dumps(dict(kept=len(rows), exclusions=exclusions)), flush=True)


if __name__ == "__main__":
    main()

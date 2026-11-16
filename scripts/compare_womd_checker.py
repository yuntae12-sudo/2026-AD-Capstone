#!/usr/bin/env python3
"""Differential check against an explicit historical implementation.

Loads trusted local Git source, never downloads or changes the working tree.
Both implementations receive the same real current state and raw proposals.
"""
import argparse
from dataclasses import asdict, fields
import os
from pathlib import Path
import subprocess
import sys
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.environment.womd_control import control
from src.environment.womd_control.environment import (
    SACPlanningEnvironment,
    route_reference_proposal,
    scripted_proposal,
)
from src.scenarios.womd20 import parse_scenario
from scripts.screen_womd_training import development_split, write_json


def historical(revision):
    package = "_womd_checker_oracle"
    parent = types.ModuleType(package)
    parent.__path__ = []
    sys.modules[package] = parent
    loaded = {}
    for name in ("geometry", "control"):
        source = subprocess.check_output(
            ["git", "show", f"{revision}:src/environment/womd_control/{name}.py"],
            cwd=ROOT,
            text=True,
        )
        module = types.ModuleType(f"{package}.{name}")
        module.__package__ = package
        sys.modules[module.__name__] = module
        exec(compile(source, f"git:{revision}/{name}.py", "exec"), module.__dict__)
        loaded[name] = module
    return loaded


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", default="568d6d9")
    p.add_argument(
        "--inventory",
        type=Path,
        default=Path("outputs/sac_readiness_20261116/inventory/candidates"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sac_environment_gate/checker_parity.json"),
    )
    args = p.parse_args()
    old = historical(args.revision)
    report = dict(reference_revision=args.revision, cases=[], all_passed=False)
    for filename, provider in (
        ("00005-00044.pb", "fixed_keep"),
        ("00006-00029.pb", "fixed_keep"),
        ("00060-00013.pb", "fixed_follow"),
    ):
        scenario = parse_scenario((args.inventory / filename).read_bytes())
        if development_split(scenario.scenario_id)[0] == "test":
            raise ValueError("held-out test scene")
        e = SACPlanningEnvironment()
        boundary = e.reset(scenario, provider_spec=provider)
        route = old["geometry"].Route(
            **{f.name: getattr(e.route, f.name) for f in fields(old["geometry"].Route)}
        )
        for tick in range(3):
            for planner in (scripted_proposal, route_reference_proposal):
                proposal = planner(e, boundary.token)
                objects = e._check_objects()
                expected = old["control"].check_trajectory(
                    old["control"].Trajectory(proposal),
                    e.ego,
                    route,
                    objects,
                    e.memory,
                    e.tick,
                )
                actual = control.check_trajectory(
                    control.Trajectory(proposal),
                    e.ego,
                    e.route,
                    objects,
                    e.memory,
                    e.tick,
                )
                a, b = asdict(actual), asdict(expected)
                passed = all(
                    (
                        np.isclose(a[k], b[k], atol=1e-8, rtol=0)
                        if isinstance(a[k], float)
                        else a[k] == b[k]
                    )
                    for k in a
                )
                row = dict(
                    scene=scenario.scenario_id,
                    tick=e.tick,
                    planner=planner.__name__,
                    passed=bool(passed),
                    actual=a,
                    expected=b,
                )
                report["cases"].append(row)
                print(
                    {k: v for k, v in row.items() if k not in ("actual", "expected")},
                    flush=True,
                )
                write_json(args.output, report)
            result, boundary = e.step_waypoints(
                route_reference_proposal(e, boundary.token), boundary.token
            )
            if result.terminated or result.truncated:
                break
    report["all_passed"] = all(r["passed"] for r in report["cases"])
    write_json(args.output, report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

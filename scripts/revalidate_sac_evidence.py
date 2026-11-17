#!/usr/bin/env python3
"""Reuse measured sampler/baseline evidence only when its dependencies are identical.

This records a new provenance envelope, not a new rollout or synthetic run.
Learner weights and replay are never imported by this tool.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
FOLDERS = (
    "src/contracts",
    "src/environment/womd_control",
    "src/training/sac",
    "src/planning",
)


def excluding_definitions(source, allowed):
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
        or node.name not in allowed
    ]
    return ast.dump(tree, include_attributes=False)


def require_unchanged_dependencies(before, after):
    if set(before) != set(after):
        raise ValueError("runtime source file set changed")
    allowed = {
        "src/training/sac/learner.py": None,
        "src/training/sac/networks.py": {"Critic"},
        "src/training/sac/runtime.py": {"versions"},
    }
    changed = []
    for name in sorted(before):
        if before[name] == after[name]:
            continue
        changed.append(name)
        if name not in allowed:
            raise ValueError(f"evidence dependency changed: {name}")
        definitions = allowed[name]
        if definitions is not None and excluding_definitions(
            before[name], definitions
        ) != excluding_definitions(after[name], definitions):
            raise ValueError(
                f"sampler/actor/environment evidence dependency changed: {name}"
            )
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-revision", required=True)
    for name in ("manifest", "synthetic", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from src.training.sac.runtime import versions, selected_scenes, write_report
    from src.training.sac.checkpoint import file_hash
    from scripts.pure_sac_protocol import (
        require_synthetic_report,
        validate_rule_baseline,
    )

    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", args.source_revision, "--", *FOLDERS],
        cwd=ROOT,
        text=True,
    ).splitlines()
    names = [
        name
        for name in names
        if Path(name).suffix == ".py" and str(Path(name).parent) in FOLDERS
    ]
    old = {
        name: subprocess.check_output(
            ["git", "show", f"{args.source_revision}:{name}"], cwd=ROOT
        )
        for name in names
    }
    current = {
        str(path.relative_to(ROOT)): path.read_bytes()
        for folder in FOLDERS
        for path in (ROOT / folder).glob("*.py")
    }
    digest = hashlib.sha256()
    for name in sorted(old):
        digest.update(name.encode() + b"\0" + old[name])
    changed = require_unchanged_dependencies(old, current)
    synthetic = json.loads(args.synthetic.read_text())
    baseline = json.loads(args.baseline.read_text())
    source = synthetic["versions"]
    target = versions(args.manifest)
    if source != baseline["versions"] or source["code_sha256"] != digest.hexdigest():
        raise ValueError("measured evidence/source revision mismatch")
    for key, value in source.items():
        if key != "code_sha256" and target.get(key) != value:
            raise ValueError(f"manifest/config/dependency changed: {key}")
    # The synthetic runner and the actual baseline evaluation path also match.
    for name in ("scripts/validate_sac_exploration.py",):
        prior = subprocess.check_output(
            ["git", "show", f"{args.source_revision}:{name}"], cwd=ROOT
        )
        if prior != (ROOT / name).read_bytes():
            raise ValueError("synthetic runner changed")
    for name, definitions in (
        (
            "scripts/sac_experiment_runtime.py",
            ("scripted_proposer", "evaluate", "summarize_episodes"),
        ),
    ):
        prior = ast.parse(
            subprocess.check_output(
                ["git", "show", f"{args.source_revision}:{name}"], cwd=ROOT
            )
        )
        now = ast.parse((ROOT / name).read_bytes())
        for definition in definitions:
            a = next(
                n
                for n in prior.body
                if isinstance(n, ast.FunctionDef) and n.name == definition
            )
            b = next(
                n
                for n in now.body
                if isinstance(n, ast.FunctionDef) and n.name == definition
            )
            # evaluate acquired an optional metadata field; caller/current baseline
            # execution must otherwise remain the same. Require byte-identical AST
            # by choosing the source revision that already has that metadata field.
            if ast.dump(a, include_attributes=False) != ast.dump(
                b, include_attributes=False
            ):
                raise ValueError(f"baseline execution changed: {definition}")
    require_synthetic_report(
        synthetic, source, file_hash(ROOT / "scripts/validate_sac_exploration.py")
    )
    pairs = [
        (r["scene_id"], r["provider"])
        for r, _ in selected_scenes(args.manifest, "validation")
    ]
    validate_rule_baseline(baseline, source, pairs)
    args.output.mkdir(parents=True)
    for name, report, path in (
        ("synthetic", synthetic, args.synthetic),
        ("baseline", baseline, args.baseline),
    ):
        report["evidence_reuse"] = dict(
            method="verified_unchanged_execution_dependencies",
            source_sha256=file_hash(path),
            source_versions=source,
            compatibility_revision=args.source_revision,
            changed_sources=changed,
            note="No new measurements. Only critic/learner/metadata changed; sampler, environment, PD and Frenet execution remain identical.",
        )
        report["versions"] = target
        write_report(args.output / (name + ".json"), report)
    print(json.dumps(dict(changed_sources=changed, new_version=target)))


if __name__ == "__main__":
    main()

"""Require the correct actual collection audit before frozen-policy diagnostics."""
import json
from pathlib import Path


def audit(directory, manifest):
    directory, manifest = Path(directory), Path(manifest)
    report = json.loads((directory / "report.json").read_text())
    if report.get("scope") == "OWN_SUCCESSOR_REFRESH_SAC_COMPARISON":
        if ("successor_refresh" not in report
                or any(k in report for k in ("stratified_refresh", "rejection_refresh", "terminal_refresh"))):
            raise ValueError("successor refresh requires an unambiguous collection contract")
        from scripts.audit_sac_successor_replay import audit as validate
    else:
        if "successor_refresh" in report:
            raise ValueError("successor refresh cannot be relabeled as historical collection")
        from scripts.sac_stratified_diagnostic_audit import audit as validate
    checked = validate(directory, manifest)
    if checked.get("status") != "COMPLETE":
        raise ValueError("actual collection validation did not complete")
    from src.training.sac.checkpoint import file_hash
    return dict(checked, report_sha256=file_hash(directory / "report.json"),
                replay_sha256=file_hash(directory / "replay.npz"),
                successor_audit_adapter_sha256=file_hash(Path(__file__)))

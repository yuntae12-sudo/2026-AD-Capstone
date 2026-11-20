"""Validate supported actual collection histories before read-only policy probes."""
import json
from pathlib import Path


def audit(directory, manifest):
    directory, manifest = Path(directory), Path(manifest)
    report = json.loads((directory / "report.json").read_text())
    scope = report.get("scope")
    if scope == "OWN_TERMINAL_REFRESH_SAC_COMPARISON":
        from scripts.audit_sac_refresh_replay import audit as validate
        if "terminal_refresh" not in report:
            raise ValueError("refresh history is missing its collection contract")
    elif scope in ("SHORT_ONLINE_SAC_DIAGNOSTIC",
                   "OWN_GOAL_FAILURE_TERMINAL_RETENTION_SAC_COMPARISON",
                   "BOUNDED_ONLINE_SAC_CONTINUATION"):
        from scripts.audit_sac_terminal_branch_replay import audit as validate
        if "terminal_refresh" in report:
            raise ValueError("refresh collection cannot be relabeled as legacy history")
    else:
        raise ValueError("unsupported collection scope for this diagnostic")
    checked = validate(directory, manifest)
    if checked.get("status") != "COMPLETE":
        raise ValueError("actual collection validation did not complete")
    from src.training.sac.checkpoint import file_hash
    return dict(checked, report_sha256=file_hash(directory / "report.json"),
                replay_sha256=file_hash(directory / "replay.npz"),
                collection_audit_adapter_sha256=file_hash(Path(__file__)))

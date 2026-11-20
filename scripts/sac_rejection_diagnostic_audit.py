"""Validate rejection-refresh histories for read-only policy diagnostics.

The older adapter stays frozen for an active comparison. Historical scopes
delegate to it only when they do not claim a rejection-refresh contract.
"""

import json
from pathlib import Path


def audit(directory, manifest):
    directory, manifest = Path(directory), Path(manifest)
    report = json.loads((directory / "report.json").read_text())
    if report.get("scope") == "OWN_REJECTION_REFRESH_SAC_COMPARISON":
        if "rejection_refresh" not in report or "terminal_refresh" in report:
            raise ValueError("rejection refresh requires an unambiguous collection contract")
        from scripts.audit_sac_rejection_replay import audit as validate
    else:
        if "rejection_refresh" in report:
            raise ValueError("rejection refresh cannot be relabeled as historical collection")
        from scripts.sac_collection_diagnostic_audit import audit as validate
    checked = validate(directory, manifest)
    if checked.get("status") != "COMPLETE":
        raise ValueError("actual collection validation did not complete")
    from src.training.sac.checkpoint import file_hash
    return dict(
        checked,
        report_sha256=file_hash(directory / "report.json"),
        replay_sha256=file_hash(directory / "replay.npz"),
        rejection_audit_adapter_sha256=file_hash(Path(__file__)),
    )

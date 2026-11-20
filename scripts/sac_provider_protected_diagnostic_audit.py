"""Dispatch exact collection audits without relabeling protected source histories."""
import json
from pathlib import Path


def audit(directory, manifest):
    directory, manifest = Path(directory), Path(manifest)
    report = json.loads((directory / "report.json").read_text())
    if report.get("scope") == "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON":
        if ("provider_protected_refresh" not in report or any(k in report for k in
                ("successor_refresh", "stratified_refresh", "rejection_refresh", "terminal_refresh"))):
            raise ValueError("provider protection requires an unambiguous collection contract")
        from scripts.audit_sac_provider_protected_replay import audit as validate
    else:
        if "provider_protected_refresh" in report:
            raise ValueError("provider protection cannot be relabeled as historical collection")
        from scripts.sac_successor_diagnostic_audit import audit as validate
    result = validate(directory, manifest)
    if result.get("status") != "COMPLETE":
        raise ValueError("actual collection validation did not complete")
    from src.training.sac.checkpoint import file_hash
    return dict(result, report_sha256=file_hash(directory / "report.json"),
                replay_sha256=file_hash(directory / "replay.npz"),
                provider_protected_audit_adapter_sha256=file_hash(Path(__file__)))

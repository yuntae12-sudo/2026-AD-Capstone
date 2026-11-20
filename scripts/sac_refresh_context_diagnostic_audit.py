"""Dispatch read-only diagnostics without relabeling continued collection."""
import json
from pathlib import Path


def audit(directory, manifest):
    directory, manifest = Path(directory), Path(manifest)
    report = json.loads((directory / "report.json").read_text())
    if report.get("scope") == "OWN_REFRESH_CONTEXT_CONTINUATION":
        contracts = {key for key in ("early_moving_refresh", "provider_protected_refresh")
                     if key in report}
        forbidden = {"successor_refresh", "stratified_refresh", "rejection_refresh", "terminal_refresh"}
        if len(contracts) != 1 or forbidden.intersection(report):
            raise ValueError("context continuation requires one unambiguous refresh contract")
        from scripts.audit_sac_refresh_context_replay import audit as validate
    else:
        from scripts.sac_early_moving_diagnostic_audit import audit as validate
    result = validate(directory, manifest)
    if result.get("status") != "COMPLETE":
        raise ValueError("actual collection validation did not complete")
    from src.training.sac.checkpoint import file_hash
    return dict(result, report_sha256=file_hash(directory / "report.json"),
                replay_sha256=file_hash(directory / "replay.npz"),
                context_audit_adapter_sha256=file_hash(Path(__file__)))

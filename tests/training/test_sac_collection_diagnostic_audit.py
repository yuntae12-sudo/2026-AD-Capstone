import json

import pytest

from scripts.sac_collection_diagnostic_audit import audit


def report(tmp_path, scope, refresh=False):
    value = {"scope": scope}
    if refresh:
        value["terminal_refresh"] = {"mode": "actual_terminal_refresh_v1"}
    (tmp_path / "report.json").write_text(json.dumps(value))
    (tmp_path / "replay.npz").write_bytes(b"audit fixture, never loaded as a replay")


@pytest.mark.parametrize("scope,refresh,module", [
    ("OWN_TERMINAL_REFRESH_SAC_COMPARISON", True, "scripts.audit_sac_refresh_replay"),
    ("OWN_GOAL_FAILURE_TERMINAL_RETENTION_SAC_COMPARISON", False, "scripts.audit_sac_terminal_branch_replay"),
    ("SHORT_ONLINE_SAC_DIAGNOSTIC", False, "scripts.audit_sac_terminal_branch_replay"),
    ("BOUNDED_ONLINE_SAC_CONTINUATION", False, "scripts.audit_sac_terminal_branch_replay"),
])
def test_each_collection_requires_its_actual_lineage_validator(tmp_path, monkeypatch, scope, refresh, module):
    report(tmp_path, scope, refresh)
    calls = []
    def validate(directory, manifest):
        calls.append((directory, manifest))
        return {"status": "COMPLETE", "checked_by": module}
    monkeypatch.setattr(module+".audit", validate)
    result = audit(tmp_path, tmp_path / "manifest.json")
    assert result["checked_by"] == module and len(calls) == 1
    assert len(result["replay_sha256"]) == 64


@pytest.mark.parametrize("scope,refresh", [
    ("UNKNOWN", False),
    ("OWN_TERMINAL_REFRESH_SAC_COMPARISON", False),
    ("OWN_GOAL_FAILURE_TERMINAL_RETENTION_SAC_COMPARISON", True),
])
def test_missing_or_mislabeled_refresh_cannot_bypass_validation(tmp_path, scope, refresh):
    report(tmp_path, scope, refresh)
    with pytest.raises(ValueError):
        audit(tmp_path, tmp_path / "manifest.json")


def test_failed_validator_is_not_reported_as_complete(tmp_path, monkeypatch):
    report(tmp_path, "OWN_TERMINAL_REFRESH_SAC_COMPARISON", True)
    monkeypatch.setattr("scripts.audit_sac_refresh_replay.audit", lambda *args: {"status": "FAILED"})
    with pytest.raises(ValueError, match="did not complete"):
        audit(tmp_path, tmp_path / "manifest.json")

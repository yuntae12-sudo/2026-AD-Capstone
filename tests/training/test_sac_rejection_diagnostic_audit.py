import json

import pytest

from scripts.sac_rejection_diagnostic_audit import audit


def write_report(tmp_path, scope, **contracts):
    (tmp_path / "report.json").write_text(json.dumps(dict(scope=scope, **contracts)))
    (tmp_path / "replay.npz").write_bytes(b"fixture; validator mocked")


def test_new_history_requires_rejection_lineage_audit(tmp_path, monkeypatch):
    write_report(tmp_path, "OWN_REJECTION_REFRESH_SAC_COMPARISON", rejection_refresh={})
    calls = []

    def validate(directory, manifest):
        calls.append((directory, manifest))
        return {"status": "COMPLETE", "refresh_steps": 20}

    monkeypatch.setattr("scripts.audit_sac_rejection_replay.audit", validate)
    result = audit(tmp_path, tmp_path / "manifest.json")
    assert calls == [(tmp_path, tmp_path / "manifest.json")]
    assert result["refresh_steps"] == 20
    assert len(result["rejection_audit_adapter_sha256"]) == 64


def test_historical_validator_is_still_required(tmp_path, monkeypatch):
    write_report(tmp_path, "OWN_TERMINAL_REFRESH_SAC_COMPARISON", terminal_refresh={})

    def reject(*args):
        raise ValueError("corrupt historical lineage")

    monkeypatch.setattr("scripts.sac_collection_diagnostic_audit.audit", reject)
    with pytest.raises(ValueError, match="corrupt historical lineage"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("scope,contracts", [
    ("OWN_REJECTION_REFRESH_SAC_COMPARISON", {}),
    ("OWN_REJECTION_REFRESH_SAC_COMPARISON", {"rejection_refresh": {}, "terminal_refresh": {}}),
    ("OWN_TERMINAL_REFRESH_SAC_COMPARISON", {"rejection_refresh": {}}),
    ("SHORT_ONLINE_SAC_DIAGNOSTIC", {"rejection_refresh": {}}),
    ("UNKNOWN", {}),
])
def test_missing_mixed_or_mislabeled_contracts_fail(tmp_path, scope, contracts):
    write_report(tmp_path, scope, **contracts)
    with pytest.raises(ValueError):
        audit(tmp_path, tmp_path / "manifest.json")


def test_noncomplete_audit_cannot_pass(tmp_path, monkeypatch):
    write_report(tmp_path, "OWN_REJECTION_REFRESH_SAC_COMPARISON", rejection_refresh={})
    monkeypatch.setattr("scripts.audit_sac_rejection_replay.audit", lambda *args: {"status": "FAILED"})
    with pytest.raises(ValueError, match="did not complete"):
        audit(tmp_path, tmp_path / "manifest.json")

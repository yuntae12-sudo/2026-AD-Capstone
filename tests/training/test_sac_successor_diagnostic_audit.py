import json

import pytest

from scripts.sac_successor_diagnostic_audit import audit


def write_report(path, scope, **contracts):
    (path / "report.json").write_text(json.dumps(dict(scope=scope, **contracts)))
    (path / "replay.npz").write_bytes(b"fixture; actual lineage validator mocked")


def test_new_history_cannot_bypass_successor_lineage_validation(tmp_path, monkeypatch):
    write_report(tmp_path, "OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", successor_refresh={})
    calls = []

    def validate(directory, manifest):
        calls.append((directory, manifest))
        return dict(status="COMPLETE", refresh_steps=20)

    monkeypatch.setattr("scripts.audit_sac_successor_replay.audit", validate)
    result = audit(tmp_path, tmp_path / "manifest.json")
    assert calls == [(tmp_path, tmp_path / "manifest.json")]
    assert result["refresh_steps"] == 20 and len(result["successor_audit_adapter_sha256"]) == 64


@pytest.mark.parametrize("scope,contracts", [
    ("OWN_STRATIFIED_REFRESH_SAC_COMPARISON", {"stratified_refresh": {}}),
    ("OWN_REJECTION_REFRESH_SAC_COMPARISON", {"rejection_refresh": {}}),
    ("OWN_TERMINAL_REFRESH_SAC_COMPARISON", {"terminal_refresh": {}}),
])
def test_historical_collection_audits_still_run(tmp_path, monkeypatch, scope, contracts):
    write_report(tmp_path, scope, **contracts)

    def reject(*args):
        raise ValueError("corrupt historical lineage")

    monkeypatch.setattr("scripts.sac_stratified_diagnostic_audit.audit", reject)
    with pytest.raises(ValueError, match="corrupt historical lineage"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("scope,contracts", [
    ("OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", {}),
    ("OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", {"successor_refresh": {}, "stratified_refresh": {}}),
    ("OWN_STRATIFIED_REFRESH_SAC_COMPARISON", {"successor_refresh": {}}),
    ("OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", {"successor_refresh": {}, "rejection_refresh": {}}),
    ("OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", {"successor_refresh": {}, "terminal_refresh": {}}),
    ("OWN_REJECTION_REFRESH_SAC_COMPARISON", {"successor_refresh": {}}),
    ("OWN_TERMINAL_REFRESH_SAC_COMPARISON", {"successor_refresh": {}}),
    ("SHORT_ONLINE_SAC_DIAGNOSTIC", {"successor_refresh": {}}),
    ("UNKNOWN", {}),
])
def test_missing_mixed_or_mislabeled_contracts_fail(tmp_path, scope, contracts):
    write_report(tmp_path, scope, **contracts)
    with pytest.raises(ValueError):
        audit(tmp_path, tmp_path / "manifest.json")


def test_noncomplete_validation_cannot_pass(tmp_path, monkeypatch):
    write_report(tmp_path, "OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", successor_refresh={})
    monkeypatch.setattr("scripts.audit_sac_successor_replay.audit", lambda *args: dict(status="FAILED"))
    with pytest.raises(ValueError, match="did not complete"):
        audit(tmp_path, tmp_path / "manifest.json")

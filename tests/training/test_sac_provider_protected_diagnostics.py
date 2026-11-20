import ast
import json
from pathlib import Path

import pytest

from scripts.sac_provider_protected_diagnostic_audit import audit


def fixture(path, scope, **contracts):
    (path / "report.json").write_text(json.dumps(dict(scope=scope, **contracts)))
    (path / "replay.npz").write_bytes(b"fixture: lineage validator mocked")


def test_actual_protected_audit_is_required(tmp_path, monkeypatch):
    fixture(tmp_path, "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", provider_protected_refresh={})
    calls = []
    def validate(*args):
        calls.append(args)
        return dict(status="COMPLETE", refresh_steps=200)
    monkeypatch.setattr("scripts.audit_sac_provider_protected_replay.audit", validate)
    result = audit(tmp_path, tmp_path / "manifest.json")
    assert calls == [(tmp_path, tmp_path / "manifest.json")]
    assert result["refresh_steps"] == 200


def test_historical_successor_still_uses_historical_validator(tmp_path, monkeypatch):
    fixture(tmp_path, "OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", successor_refresh={})
    def reject(*args): raise ValueError("corrupt historical lineage")
    monkeypatch.setattr("scripts.sac_successor_diagnostic_audit.audit", reject)
    with pytest.raises(ValueError, match="corrupt historical"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("scope,contracts", [
    ("OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", {}),
    ("OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", {"provider_protected_refresh": {}, "successor_refresh": {}}),
    ("OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", {"provider_protected_refresh": {}, "terminal_refresh": {}}),
    ("OWN_SUCCESSOR_REFRESH_SAC_COMPARISON", {"provider_protected_refresh": {}}),
    ("UNKNOWN", {"provider_protected_refresh": {}}),
])
def test_missing_mixed_and_mislabeled_contracts_rejected(tmp_path, scope, contracts):
    fixture(tmp_path, scope, **contracts)
    with pytest.raises(ValueError): audit(tmp_path, tmp_path / "manifest.json")


def test_failed_audit_is_not_promoted_to_complete(tmp_path, monkeypatch):
    fixture(tmp_path, "OWN_PROVIDER_PROTECTED_REFRESH_SAC_COMPARISON", provider_protected_refresh={})
    monkeypatch.setattr("scripts.audit_sac_provider_protected_replay.audit", lambda *args: dict(status="FAILED"))
    with pytest.raises(ValueError, match="did not complete"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("kind", ["branch_q", "soft_returns"])
def test_policy_diagnostics_only_change_collection_audit(kind):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    old = (scripts / f"diagnose_sac_successor_{kind}.py").read_text()
    new = (scripts / f"diagnose_sac_provider_protected_{kind}.py").read_text()
    assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(new.replace(
        "sac_provider_protected_diagnostic_audit", "sac_successor_diagnostic_audit")))


def test_coverage_math_and_actual_replay_verification_remain_unchanged():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    for old_name,new_name,replacements in [
        ("audit_sac_successor_coverage.py", "audit_sac_provider_protected_coverage.py", [
            ("scripts.audit_sac_provider_protected_refresh", "scripts.audit_sac_successor_refresh"),
            ('provider_protected_refresh=source["provider_protected_refresh"]', 'successor_refresh=source["successor_refresh"]')]),
        ("verify_sac_successor_refresh_execution.py", "verify_sac_provider_protected_execution.py", [
            ("scripts.audit_sac_provider_protected_refresh", "scripts.audit_sac_successor_refresh")])]:
        old=(scripts/old_name).read_text();new=(scripts/new_name).read_text()
        for a,b in replacements:new=new.replace(a,b)
        assert ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))

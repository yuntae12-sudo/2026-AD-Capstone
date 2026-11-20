"""Require context provenance and preserve the established diagnostic math."""
import ast
import json
from pathlib import Path

import pytest

from scripts.sac_refresh_context_diagnostic_audit import audit


def fixture(path, **contracts):
    (path / "report.json").write_text(json.dumps(dict(
        scope="OWN_REFRESH_CONTEXT_CONTINUATION", **contracts)))
    (path / "replay.npz").write_bytes(b"dispatch fixture only; no actual replay")


@pytest.mark.parametrize("contract", ["early_moving_refresh", "provider_protected_refresh"])
def test_continuation_cannot_skip_its_provenance_audit(tmp_path, monkeypatch, contract):
    fixture(tmp_path, **{contract: {}})
    def reject(*args):
        raise ValueError("parent preservation failed")
    monkeypatch.setattr("scripts.audit_sac_refresh_context_replay.audit", reject)
    with pytest.raises(ValueError, match="parent preservation failed"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("contracts", [
    {},
    {"early_moving_refresh": {}, "provider_protected_refresh": {}},
    {"early_moving_refresh": {}, "terminal_refresh": {}},
    {"provider_protected_refresh": {}, "successor_refresh": {}},
])
def test_ambiguous_contract_rejected_before_diagnostics(tmp_path, contracts):
    fixture(tmp_path, **contracts)
    with pytest.raises(ValueError, match="unambiguous"):
        audit(tmp_path, tmp_path / "manifest.json")


def test_failed_audit_cannot_be_reported_as_complete(tmp_path, monkeypatch):
    fixture(tmp_path, early_moving_refresh={})
    monkeypatch.setattr("scripts.audit_sac_refresh_context_replay.audit",
                        lambda *args: {"status": "FAILED"})
    with pytest.raises(ValueError, match="did not complete"):
        audit(tmp_path, tmp_path / "manifest.json")


@pytest.mark.parametrize("kind", ["branch_q", "soft_returns"])
def test_q_and_rollout_math_unchanged(kind):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    old = (scripts / f"diagnose_sac_early_moving_{kind}.py").read_text()
    new = (scripts / f"diagnose_sac_refresh_context_{kind}.py").read_text()
    new = new.replace("sac_refresh_context_diagnostic_audit", "sac_early_moving_diagnostic_audit")
    assert ast.dump(ast.parse(new)) == ast.dump(ast.parse(old))


def test_coverage_calculation_and_sampling_checks_unchanged():
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    old = (scripts / "audit_sac_early_moving_coverage.py").read_text()
    new = (scripts / "audit_sac_refresh_context_coverage.py").read_text()
    new = new.replace("scripts.audit_sac_refresh_context_continuation", "scripts.audit_sac_early_moving_refresh")
    new = new.replace('        continuation_history=source["continuation_history"],\n'
                      '        refresh_contract={key: source[key] for key in\n'
                      '                          ("early_moving_refresh", "provider_protected_refresh") if key in source},',
                      '        early_moving_refresh=source["early_moving_refresh"],')
    assert ast.dump(ast.parse(new)) == ast.dump(ast.parse(old))

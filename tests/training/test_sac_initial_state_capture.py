import json
from types import SimpleNamespace

import pytest

from scripts.sac_initial_state_capture import capture_first_update


def test_capture_preserves_update_inputs_outputs_and_does_not_repeat(tmp_path):
    state = SimpleNamespace(updates=0, bc_updates=0)
    batch, result, calls = object(), object(), []
    def update(s, b, size):
        calls.append((s, b, size))
        return result
    reference = tmp_path / "reference"
    reference.write_bytes(b"initial")
    wrapped = capture_first_update(update, lambda s: b"initial", tmp_path, expected=reference)
    assert wrapped(state, batch, size=5001) is result
    assert calls == [(state, batch, 5001)]
    state.updates = 1
    assert wrapped(state, batch, size=5002) is result
    assert len(calls) == 2
    assert (tmp_path / "initial-learner.msgpack").read_bytes() == b"initial"
    assert json.loads((tmp_path / "initial-state-audit.json").read_text())["reference_bytes_equal"] is True


def test_mismatch_preserves_evidence_and_never_updates(tmp_path):
    reference = tmp_path / "reference"
    reference.write_bytes(b"different")
    def forbidden(*args):
        pytest.fail("update must not run after an initial-state mismatch")
    wrapped = capture_first_update(forbidden, lambda s: b"actual", tmp_path, expected=reference)
    with pytest.raises(ValueError, match="differ"):
        wrapped(SimpleNamespace(updates=0, bc_updates=0))
    assert (tmp_path / "initial-learner.msgpack").read_bytes() == b"actual"
    assert json.loads((tmp_path / "initial-state-audit.json").read_text())["status"] == "FAILED"


@pytest.mark.parametrize("updates,bc", [(1, 0), (0, 1)])
def test_nonfresh_or_bc_state_is_rejected_before_serializing(tmp_path, updates, bc):
    def forbidden(*args):
        pytest.fail("must reject before serializing or updating")
    wrapped = capture_first_update(forbidden, forbidden, tmp_path)
    with pytest.raises(ValueError, match="fresh pure"):
        wrapped(SimpleNamespace(updates=updates, bc_updates=bc))
    assert not (tmp_path / "initial-learner.msgpack").exists()

import copy
import numpy as np
import pytest
from PIL import Image

from src.visualization.sac_rollout import capture, frame_at, require_paired
from src.visualization.sac_gif import render, selected_ticks, transform
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.training.sac.collector import collect_step
from src.training.sac.teacher import FrenetTeacher
from tests.environment.test_womd_control import scene


def test_capture_preserves_actual_controller_and_checker():
    teacher = FrenetTeacher()
    propose = lambda e, token: teacher.propose(e, token).proposal
    captured = capture(scene(), "fixed_keep", propose, "test", max_ticks=2)
    e = SACPlanningEnvironment()
    b = e.reset(scene(), provider_spec="fixed_keep")
    for recorded in captured["frames"][1:]:
        step = collect_step(e, b, propose)
        b = step.boundary
        np.testing.assert_array_equal(recorded["ego"]["xy"], e.ego.position)
        assert recorded["source"] == step.result.source.value
        assert recorded["reason"] == step.result.check_result.reason
        assert recorded["reward"] == step.reward.training
        assert len(recorded["executed_xy"]) == 30
    assert captured["outcome"] == "PREVIEW_LIMIT"


def test_emergency_does_not_display_the_rejected_path_as_executed():
    from src.environment.womd_control.environment import scripted_proposal

    def invalid(e, token):
        xy = scripted_proposal(e, token).xy_world + np.array([0.0, 100.0])
        return e.proposal_from_world(xy, token)

    episode = capture(scene(), "fixed_keep", invalid, "invalid", max_ticks=1)
    frame = episode["frames"][1]
    assert frame["source"] == "EMERGENCY" and frame["accepted"] is False
    assert len(frame["raw_xy"]) == 30 and frame["executed_xy"] == []


def evidence():
    v = dict(
        id=None,
        xy=[0.0, 0.0],
        yaw=0.0,
        length=4.0,
        width=2.0,
        speed=1.0,
        velocity=[1.0, 0.0],
    )
    frame = dict(
        tick=0,
        time_s=0.0,
        ego=v,
        objects=[],
        target_id=None,
        raw_xy=[],
        executed_xy=[],
        accepted=None,
        source="RESET",
        reason="RESET",
        violations=[],
        events=[],
        outcome=None,
        colliding_ids=[],
    )
    second = copy.deepcopy(frame)
    second.update(
        tick=1,
        time_s=0.1,
        source="EMERGENCY",
        reason="KINEMATIC_LIMIT",
        accepted=False,
        outcome="COLLISION",
    )
    second["ego"]["xy"] = [0.1, 0.0]
    ep = dict(
        scene_id="test-scene",
        provider="fixed_keep",
        seed=0,
        label="Test",
        map=dict(
            lines=[],
            route=[[0.0, 0.0], [30.0, 0.0]],
            corridor=[[-5.0, -3.0], [30.0, -3.0], [30.0, 3.0], [-5.0, 3.0]],
        ),
        frames=[frame, second],
        outcome="COLLISION",
    )
    return dict(episodes=[ep, copy.deepcopy(ep)])


def test_pairing_rejects_different_initial_conditions_and_holds_terminal_state():
    data = evidence()
    require_paired(data["episodes"])
    frame, held = frame_at(data["episodes"][0], 3)
    assert frame["tick"] == 1 and held
    data["episodes"][1]["frames"][0]["ego"]["xy"][0] = 1
    with pytest.raises(ValueError, match="initial"):
        require_paired(data["episodes"])


def test_coordinate_transform_and_stride_preserve_last_tick():
    np.testing.assert_allclose(
        transform([[10, 21]], [10, 20], np.pi / 2), [[1, 0]], atol=1e-12
    )
    assert selected_ticks(8, 3) == [0, 3, 6, 7]
    with pytest.raises(ValueError):
        selected_ticks(8, 0)


def test_export_gif_is_animated_and_has_physical_frame_timing(tmp_path):
    output = tmp_path / "comparison.gif"
    summary = render(evidence(), output)
    with Image.open(output) as gif:
        assert gif.n_frames == 2 and gif.size == (1440, 900)
        assert gif.info["duration"] == 100
        gif.seek(1)
        assert gif.info["duration"] == 1500
    assert output.with_suffix(".png").exists()
    assert summary["simulation_duration_s"] == 0.1

import pytest
from tests.environment.test_womd_control import scene
from src.environment.womd_control.environment import SACPlanningEnvironment
from src.environment.womd_control.geometry import ResetRejected


@pytest.mark.parametrize(
    "velocity,accel_y,reason",
    [
        (-0.02, 0.0, "INITIAL_REVERSE_VELOCITY"),
        (0.1, 0.01, "INITIAL_CURVATURE_LIMIT"),
        (10.0, 5.0, "INITIAL_LATERAL_ACCELERATION_LIMIT"),
    ],
)
def test_reset_rejects_measured_motion_that_no_anchored_path_can_repair(
    velocity, accel_y, reason
):
    proto = scene()
    start = max(2, proto.current_time_index)
    for state in proto.tracks[0].states:
        state.velocity_x = velocity
        state.velocity_y = 0.0
    proto.tracks[0].states[start].velocity_y = accel_y * 0.1
    with pytest.raises(ResetRejected, match=reason):
        SACPlanningEnvironment().reset(proto)


def test_tiny_drift_below_motion_threshold_preserves_actual_velocity():
    proto = scene()
    for state in proto.tracks[0].states:
        state.velocity_x = -0.001
        state.velocity_y = 0.001
    e = SACPlanningEnvironment()
    e.reset(proto)
    assert e.ego.velocity[0] < 0 and e.ego.velocity[1] > 0

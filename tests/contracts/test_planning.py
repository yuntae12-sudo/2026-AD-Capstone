import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.contracts import planning as p
from src.contracts.observation import (
    CORE_SHAPES, ITEM_SHAPES, FEATURE_COUNT, CoreObservation, PPOObservation,
    masked_probabilities,
)

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = json.loads((Path(__file__).parent / "fixtures/planning_v3.json").read_text())


def test_frozen_schema_parity():
    spec = json.loads((ROOT / "configs/contracts/planning_v3.json").read_text())
    for name, enum in (("maneuver", p.Maneuver), ("direction", p.Direction),
                       ("stage", p.Stage), ("source", p.Source),
                       ("execution_request", p.ExecutionRequest), ("task_kind", p.TaskKind),
                       ("rejection_reason", p.RejectionReason), ("terminal_reason", p.TerminalReason)):
        assert [item.value for item in enum] == spec["enums"][name]
    assert list(p.Maneuver).index(p.Maneuver.STOP) == 4
    assert {k: list(v) for k, v in CORE_SHAPES.items()} == spec["core_observation"]["shapes"]
    assert {k: list(v) for k, v in ITEM_SHAPES.items()} == spec["core_observation"]["item_masks"]
    assert FEATURE_COUNT == spec["core_observation"]["float_feature_count"] == 5975
    assert p.SCHEMA_VERSION == spec["schema_version"]
    assert p.DISTRIBUTION_VERSION == spec["action_distribution_version"]


@pytest.mark.parametrize("key,cls", [
    ("initial_observation_token", p.ObservationToken), ("task_spec", p.TaskSpec),
    ("command_request", p.CommandRequest), ("receipt", p.Receipt),
    ("accepted_command", p.BehaviorCommand), ("waypoint_proposal", p.WaypointProposal),
])
def test_original_normative_wire_examples_roundtrip(key, cls):
    result = p.from_wire(cls, EXAMPLE[key])
    assert p.to_wire(result) == EXAMPLE[key]


def test_clocks_schema_and_expiry():
    token = p.from_wire(p.ObservationToken, EXAMPLE["initial_observation_token"])
    for changed in (replace(token, revision=1), replace(token, sim_tick=13),
                    replace(token, episode_uid="next-episode"), replace(token, scene_id="other")):
        with pytest.raises(p.ContractError, match="STALE"):
            changed.require_current(token)
    with pytest.raises(p.ContractError, match="SCHEMA"):
        replace(token, schema_version=2)
    command = p.from_wire(p.BehaviorCommand, EXAMPLE["accepted_command"])
    assert not command.is_valid_at(11)
    assert command.is_valid_at(21)
    assert not command.is_valid_at(22)


@pytest.mark.parametrize("value", [True, 1.2, -1, 2**63])
def test_tick_is_strict_int64(value):
    with pytest.raises(p.ContractError):
        p.ObservationToken("episode", "scene", value, 0)


def test_codec_uses_world_velocity_and_acceleration_independent_of_heading():
    rng = np.random.default_rng(12)
    u = rng.normal(size=(30, 2))
    position = np.array([123456., -234567.])
    velocity = np.array([3., -2.])
    acceleration = np.array([.1, .3])
    yaw = 1.2
    xy = p.decode_waypoints(u, position, yaw, velocity, acceleration)
    np.testing.assert_allclose(p.encode_waypoints(xy, position, yaw, velocity, acceleration), u, atol=1e-8)
    zero = p.decode_waypoints(np.zeros((30, 2)), position, yaw, velocity, acceleration)
    tau = p.future_times(0)[:, None]
    np.testing.assert_allclose(zero, position + velocity*tau + .5*acceleration*tau**2)


@pytest.mark.parametrize("field,mutator", [
    ("times", lambda x: np.asarray(x) + .1),
    ("xy_world", lambda x: np.asarray(x) + 1e-5),
    ("raw_u", lambda x: np.full((30, 2), np.nan)),
    ("xy_world", lambda x: np.asarray(x, np.float32)),
    ("anchor_acceleration_valid", lambda x: False),
    ("reference_point", lambda x: "REAR_AXLE"),
])
def test_proposal_rejects_corrupt_payload(field, mutator):
    proposal = p.from_wire(p.WaypointProposal, EXAMPLE["waypoint_proposal"])
    with pytest.raises(p.ContractError):
        replace(proposal, **{field: mutator(getattr(proposal, field))})


def test_wire_invalid_optional_and_integer_enum_fail():
    payload = copy.deepcopy(EXAMPLE["command_request"])
    payload["command_draft"]["target_agent_id"] = {"value": 42, "valid": False}
    with pytest.raises(p.ContractError):
        p.from_wire(p.CommandRequest, payload)
    payload = copy.deepcopy(EXAMPLE["command_request"])
    payload["command_draft"]["maneuver"] = 3
    with pytest.raises(p.ContractError):
        p.from_wire(p.CommandRequest, payload)


def test_payloads_are_independent_immutable_snapshots():
    proposal = p.from_wire(p.WaypointProposal, EXAMPLE["waypoint_proposal"])
    assert proposal.raw_u.ravel(order="C").shape == (60,)
    with pytest.raises(ValueError):
        proposal.raw_u.setflags(write=True)
    core = CoreObservation.empty()
    assert core.flatten().shape == (5975,)
    assert core.flatten().dtype == np.float32
    with pytest.raises(TypeError):
        core.features["ego"] = np.zeros(14)
    features = {k: v.copy() for k, v in core.features.items()}
    features["ego"][0] = 1
    with pytest.raises(p.ContractError, match="must be zero"):
        CoreObservation(features, core.field_masks, core.item_masks)


def make_ppo():
    commands = np.zeros((5, 19), np.float32)
    commands[0, 0] = 1
    masks = np.zeros((5, 19), bool)
    masks[0, :5] = True
    return PPOObservation(p.ObservationToken("ep", "s", 0, 0), CoreObservation.empty(),
                          commands, masks,
                          np.full((5, 3), -1, np.int32), np.full(5, -1, np.int32),
                          np.array([True, False, False, False, False]),
                          ("keep-token", "", "", "", ""))


def test_masked_policy_and_binding_validation():
    obs = make_ppo()
    obs.validate_selection(0, "keep-token")
    for index, token in ((4, ""), (0, "stale"), (5, "bad")):
        with pytest.raises(p.ContractError):
            obs.validate_selection(index, token)
    probs, entropy = masked_probabilities(np.array([1., 1e6, 0., 0., 0.]), obs.action_mask)
    np.testing.assert_array_equal(probs, [1, 0, 0, 0, 0])
    assert entropy == 0
    with pytest.raises(p.ContractError):
        masked_probabilities(np.zeros(5), np.zeros(5, bool))
    with pytest.raises(p.ContractError, match="invalid slot"):
        replace(obs, candidate_lane_slots=np.zeros(5, np.int32))

"""Immutable policy tensors; raw simulator state and future logs are excluded."""
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .planning import ContractError, ObservationToken, array, integer

CORE_SHAPES = {
    "task": (3,), "ego": (14,), "agents": (3, 32, 17),
    "lanes": (16, 20, 7), "boundaries": (16, 20, 5), "route": (40, 4),
    "signals": (8, 12), "command": (19,), "execution": (25,),
    "previous_plan": (30, 6), "controller_memory": (6,),
}
ITEM_SHAPES = {k: v[:-1] for k, v in CORE_SHAPES.items()
               if k in {"agents", "lanes", "boundaries", "route", "signals", "previous_plan"}}
FEATURE_COUNT = sum(int(np.prod(shape)) for shape in CORE_SHAPES.values())


@dataclass(frozen=True)
class CoreObservation:
    features: Mapping[str, np.ndarray]
    field_masks: Mapping[str, np.ndarray]
    item_masks: Mapping[str, np.ndarray]

    def __post_init__(self):
        for name, expected, dtype in (("features", CORE_SHAPES, np.float32),
                                      ("field_masks", CORE_SHAPES, bool),
                                      ("item_masks", ITEM_SHAPES, bool)):
            source = getattr(self, name)
            if set(source) != set(expected):
                raise ContractError(f"incorrect {name} keys")
            values = {k: array(source[k], shape, k, dtype) for k, shape in expected.items()}
            object.__setattr__(self, name, MappingProxyType(values))
        for name, data in self.features.items():
            mask = self.field_masks[name]
            if np.any(data[~mask] != 0):
                raise ContractError(f"invalid {name} features must be zero")
            if name in ITEM_SHAPES:
                if np.any(mask[~self.item_masks[name]]):
                    raise ContractError(f"invalid {name} items cannot contain valid fields")
                if np.any(self.item_masks[name] & ~mask[..., :2].all(axis=-1)):
                    raise ContractError(f"valid {name} items require XY geometry")

    def flatten(self):
        """Fixed order C layout; masks must also be passed to the future encoder."""
        return np.concatenate([self.features[k].ravel() for k in CORE_SHAPES])

    @classmethod
    def empty(cls):
        return cls({k: np.zeros(v, np.float32) for k, v in CORE_SHAPES.items()},
                   {k: np.zeros(v, bool) for k, v in CORE_SHAPES.items()},
                   {k: np.zeros(v, bool) for k, v in ITEM_SHAPES.items()})


@dataclass(frozen=True)
class PPOObservation:
    token: ObservationToken
    core: CoreObservation
    candidate_commands: np.ndarray
    candidate_command_field_mask: np.ndarray
    candidate_agent_slots: np.ndarray
    candidate_lane_slots: np.ndarray
    action_mask: np.ndarray
    candidate_tokens: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.token, ObservationToken) or not isinstance(self.core, CoreObservation):
            raise ContractError("token and CoreObservation required")
        for name, shape, dtype in (("candidate_commands", (5, 19), np.float32),
                                  ("candidate_command_field_mask", (5, 19), bool),
                                  ("candidate_agent_slots", (5, 3), np.int32),
                                  ("candidate_lane_slots", (5,), np.int32),
                                  ("action_mask", (5,), bool)):
            object.__setattr__(self, name, array(getattr(self, name), shape, name, dtype))
        tokens = tuple(self.candidate_tokens)
        if len(tokens) != 5 or any(not isinstance(t, str) for t in tokens):
            raise ContractError("five candidate token strings required")
        enabled_tokens = [tokens[i] for i in np.flatnonzero(self.action_mask)]
        if any(not t for t in enabled_tokens) or len(set(enabled_tokens)) != len(enabled_tokens):
            raise ContractError("enabled candidate tokens must be nonempty and unique")
        object.__setattr__(self, "candidate_tokens", tokens)
        if np.any(self.candidate_commands[~self.candidate_command_field_mask] != 0):
            raise ContractError("invalid command fields must be zero")
        for index in np.flatnonzero(self.action_mask):
            if not self.candidate_command_field_mask[index, :5].all() or not np.array_equal(
                self.candidate_commands[index, :5], np.eye(5, dtype=np.float32)[index]
            ):
                raise ContractError("candidate maneuver must match its fixed action index")
        for slots, upper, present in ((self.candidate_agent_slots, 32, self.core.item_masks["agents"][-1]),
                                      (self.candidate_lane_slots, 16, self.core.item_masks["lanes"].any(axis=1))):
            if np.any((slots < -1) | (slots >= upper)):
                raise ContractError("slot outside schema range; source IDs are not slot indices")
            selected = slots[slots >= 0]
            if not present[selected].all():
                raise ContractError("candidate refers to an invalid slot")

    def validate_selection(self, action_index, candidate_token):
        integer(action_index, "action_index")
        if action_index >= 5 or not self.action_mask[action_index]:
            raise ContractError("POLICY_ERROR: masked or out-of-range action")
        if candidate_token != self.candidate_tokens[action_index]:
            raise ContractError("POLICY_ERROR: candidate token mismatch")


def masked_probabilities(logits, mask):
    """Stable categorical probabilities; no-action boundaries must not call PPO."""
    logits = array(logits, (5,), "logits")
    mask = array(mask, (5,), "mask", bool)
    if not mask.any():
        raise ContractError("no available actions; do not invoke PPO")
    probabilities = np.zeros(5, np.float64)
    weights = np.exp(logits[mask] - logits[mask].max())
    probabilities[mask] = weights / weights.sum()
    positive = probabilities > 0
    entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
    return probabilities, float(entropy)

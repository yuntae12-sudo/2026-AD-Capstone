"""Explicit critic input-gain experiment; production Critic is unchanged."""

import math
import hashlib
from pathlib import Path
from src.training.sac.networks import Critic


class InputGainCritic(Critic):
    action_gain: float = 1.0

    def __call__(self, observation, raw_u):
        if not math.isfinite(self.action_gain) or self.action_gain <= 0:
            raise ValueError("critic action gain must be finite and positive")
        # Scaling is invertible. Only the critic's input copy changes: the
        # sampler, log density, decoder, checker and replay still use raw u.
        return super().__call__(observation, raw_u * self.action_gain)


def critic_factory_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def diagnostic_critic(arm, *, restoring=False):
    gain = arm.get("critic_action_gain", 1.0)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("critic action gain must be finite and positive")
    if restoring and gain != 1.0 and arm.get("critic_factory_sha256") != critic_factory_hash():
        raise ValueError("critic input-gain implementation hash mismatch")
    options = dict(zero_output=arm["zero_q"],
                   action_coordinates=arm.get("critic_action_coordinates", "raw"))
    # Preserve the exact historical module for old artifacts and gain=1.
    return Critic(**options) if gain == 1.0 else InputGainCritic(action_gain=gain, **options)

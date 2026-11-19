"""Explicit critic feature experiments; production Critic is unchanged."""

import math
import hashlib
from pathlib import Path
from src.training.sac.networks import Critic

# The legacy gain implementation below is unchanged. Allow its archived models
# to restore after adding a new opt-in coordinate mode; unknown hashes fail.
LEGACY_GAIN_FACTORY_HASHES = frozenset({
    "9986af114dd291e6664e007782287c07d0e75bcfee9c2d05526b08b384a8ed12",
})


class InputGainCritic(Critic):
    action_gain: float = 1.0

    def __call__(self, observation, raw_u):
        if not math.isfinite(self.action_gain) or self.action_gain <= 0:
            raise ValueError("critic action gain must be finite and positive")
        # Scaling is invertible. Only the critic's input copy changes: the
        # sampler, log density, decoder, checker and replay still use raw u.
        return super().__call__(observation, raw_u * self.action_gain)


def critic_factory_hash():
    digest = hashlib.sha256()
    for name in ("sac_diagnostic_critic.py", "sac_physical_critic.py"):
        digest.update(name.encode() + b"\0" + Path(__file__).with_name(name).read_bytes())
    return digest.hexdigest()


def diagnostic_critic(arm, *, restoring=False):
    gain = arm.get("critic_action_gain", 1.0)
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("critic action gain must be finite and positive")
    coordinates = arm.get("critic_action_coordinates", "raw")
    if coordinates == "physical_whitened":
        if gain != 1.0:
            raise ValueError("physical coordinate trial cannot combine another input gain")
        if restoring and arm.get("critic_factory_sha256") != critic_factory_hash():
            raise ValueError("physical critic implementation hash mismatch")
        from scripts.sac_physical_critic import PhysicalActionCritic
        return PhysicalActionCritic(zero_output=arm["zero_q"], action_coordinates="whitened")
    if coordinates not in ("raw", "whitened"):
        raise ValueError("unknown diagnostic critic coordinates")
    if (restoring and gain != 1.0 and arm.get("critic_factory_sha256") not in
            LEGACY_GAIN_FACTORY_HASHES | {critic_factory_hash()}):
        raise ValueError("critic input-gain implementation hash mismatch")
    options = dict(zero_output=arm["zero_q"],
                   action_coordinates=coordinates)
    # Preserve the exact historical module for old artifacts and gain=1.
    return Critic(**options) if gain == 1.0 else InputGainCritic(action_gain=gain, **options)

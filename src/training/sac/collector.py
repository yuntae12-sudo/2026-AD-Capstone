"""Actual action/next-observation collection, with no synthetic HOLD actions."""

from dataclasses import dataclass
import numpy as np
from src.training.sac.replay import Transition
from src.training.sac.reward import reward_after_step
from src.environment.womd_control.geometry import project


class PendingTransition:
    """Accumulate real forced ticks until the next policy boundary or final obs."""

    def __init__(self, observation, proposal, teacher_accepted=False):
        self.observation = observation
        self.proposal = proposal
        self.teacher_accepted = teacher_accepted
        self.reward = 0.0
        self.ticks = 0
        self.closed = False

    def append(self, reward, result, boundary):
        if self.closed:
            raise ValueError("pending transition already finalized")
        token = self.proposal.observation_token
        if (
            result.pre_token.episode_uid != token.episode_uid
            or result.pre_token.sim_tick != token.sim_tick + self.ticks
        ):
            raise ValueError("noncontiguous actual ticks")
        if result.post_tick != result.pre_token.sim_tick + 1:
            raise ValueError("environment tick must advance exactly once")
        self.reward += 0.99**self.ticks * reward.training
        self.ticks += 1
        if not reward.sample_valid or not result.sample_valid:
            self.closed = True
            return None
        if not (boundary.needs_sac_action or result.terminated or result.truncated):
            return None
        self.closed = True
        return Transition(
            self.observation,
            self.proposal.raw_u,
            self.reward,
            boundary.sac_observation,
            self.ticks,
            not (result.terminated or result.truncated),
            True,
            self.teacher_accepted,
            token.episode_uid,
            token.scene_id,
            token.sim_tick,
            result.terminated,
            result.truncated,
        )


@dataclass(frozen=True)
class CollectedStep:
    result: object
    boundary: object
    reward: object
    transition: object
    teacher_diagnostics: dict | None


def collect_step(environment, boundary, proposer, replay=None, teacher=False):
    if not boundary.needs_sac_action:
        raise ValueError("do not call policy/teacher at a forced boundary")
    if environment.token != boundary.token:
        raise ValueError("collector has stale observation")
    before = environment.ego
    previous = environment.ego_at(environment.frame - 1)
    progress = (
        project(environment.route.center, before.position)[0]
        - environment.route.start_s
    )
    generated = proposer(environment, boundary.token)
    proposal = generated.proposal if teacher else generated
    if not np.isfinite(proposal.raw_u).all():
        raise ValueError("nonfinite raw action: stop collection and learner")
    result, next_boundary = environment.step_waypoints(proposal, boundary.token)
    reward = reward_after_step(environment, result, before, previous, progress)
    pending = PendingTransition(
        boundary.sac_observation, proposal, teacher and result.check_result.valid
    )
    transition = pending.append(reward, result, next_boundary)
    if not pending.closed:
        raise NotImplementedError(
            "forced lifecycle requires persistent PendingTransition; initial KEEP/FOLLOW must use one tick"
        )
    if transition is not None and replay is not None:
        replay.add(transition)
    return CollectedStep(
        result,
        next_boundary,
        reward,
        transition,
        generated.diagnostics if teacher else None,
    )

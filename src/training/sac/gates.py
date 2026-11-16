"""Per-behavior acceptance gates; finite loss never qualifies a policy."""

from collections import Counter
import jax
import numpy as np
from .networks import observation_speed, sample_action
from .policy import proposal_from_action, tensor_observation
from src.environment.womd_control.control import Trajectory


def probe_boundary(environment, boundary, actor, key, samples=256):
    """Read-only actual-state probe. Does not advance or reset the environment."""
    if samples != 256:
        raise ValueError("contract stochastic probe requires 256 samples")
    e = environment
    e._require(boundary.token)
    observation = tensor_observation(boundary.sac_observation)
    mu, rho = actor.apply_fn({"params": actor.params}, observation)
    objects = e._check_objects()
    mean_proposal = proposal_from_action(
        e, boundary.token, np.asarray(mu[0], np.float64), sequence="gate-mean"
    )
    mean_check = e._check(Trajectory(mean_proposal), objects)
    row = dict(
        scene_id=boundary.token.scene_id,
        tick=boundary.token.sim_tick,
        behavior=e.command.draft.maneuver.value,
        speed_mps=e.ego.speed,
        mean_pass=mean_check.valid,
        mean_reason=mean_check.reason,
        samples=0,
        accepted=0,
        reasons={},
    )
    if not mean_check.valid:
        return row
    keys = jax.random.split(key, samples)
    us = jax.vmap(
        lambda k: sample_action(mu[0], rho[0], observation_speed(observation)[0], k)[0]
    )(keys)
    reasons = Counter()
    for index, raw in enumerate(np.asarray(us)):
        proposal = proposal_from_action(
            e, boundary.token, raw, sequence=f"gate-{index}"
        )
        check = e._check(Trajectory(proposal), objects)
        reasons[check.reason] += 1
        row["accepted"] += int(check.valid)
    row.update(samples=samples, reasons=dict(reasons))
    return row


def summarize_acceptance(rows):
    """Mean-failed states stay in the mean denominator and remain visible."""
    result = {}
    for behavior in ("KEEP", "FOLLOW"):
        states = [row for row in rows if row["behavior"] == behavior]
        passed = [row for row in states if row["mean_pass"]]
        complete = bool(states) and all(
            row["samples"] == 256 and 0 <= row["accepted"] <= 256 for row in passed
        )
        if any(row["samples"] != 0 for row in states if not row["mean_pass"]):
            raise ValueError("mean-failed states must not enter stochastic denominator")
        mean_rate = len(passed) / len(states) if states else 0.0
        count = sum(row["samples"] for row in passed)
        probability = sum(row["accepted"] for row in passed) / count if count else 0.0
        per_state = [
            row["accepted"] / row["samples"] for row in passed if row["samples"]
        ]
        result[behavior] = dict(
            states=len(states),
            mean_pass_rate=mean_rate,
            mean_failure_fraction=1.0 - mean_rate,
            stochastic_pass_rate=probability,
            stochastic_samples=count,
            per_state_pass_rates=per_state,
            complete=complete,
            passed=complete and mean_rate >= 0.9 and probability >= 0.9,
        )
    return dict(
        by_behavior=result,
        passed=all(r["passed"] for r in result.values()),
        scope="actual-state acceptance only; synthetic 4096/case and policy performance are separate gates",
    )


def require_training_gate(report, versions, checkpoint_sha256):
    if (
        report.get("versions") != versions
        or report.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("gate does not match runtime/checkpoint")
    if report.get("scope") != "QUALIFIED_BC_GATE":
        raise ValueError("smoke/provisional gate cannot authorize SAC")
    if not summarize_acceptance(report.get("states", []))["passed"]:
        raise ValueError("KEEP and FOLLOW must each pass mean and stochastic gates")
    synthetic = report.get("synthetic_cases", [])
    case_ids = {r["case"] for r in synthetic}
    pairs = {(r["case"], r["sigma"]) for r in synthetic}
    if (
        len(synthetic) != 18
        or len(case_ids) != 9
        or pairs != {(case, sigma) for case in case_ids for sigma in (0.02, 0.03)}
    ):
        raise ValueError("missing initial/max-sigma synthetic cases")
    if any(
        r["samples"] != 4096
        or not r.get("mean_valid", False)
        or not 0 <= r["accepted"] <= 4096
        or r["accepted"] / r["samples"] < 0.95
        for r in synthetic
    ):
        raise ValueError("synthetic exploration gate failed")

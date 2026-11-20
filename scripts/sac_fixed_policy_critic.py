"""Value-function diagnosis, never an alternate production SAC update.

Run the canonical update on a copied learner, but retain the original actor and
temperature (including optimizer states). Q/target/critic optimizer/RNG follow
the exact canonical computation. This keeps the policy-evaluation objective
fixed; the result is not a coupled SAC training checkpoint.
"""

import numpy as np

from src.training.sac.learner import update


def critic_evaluation_step(state, batch, replay_size):
    if int(state.bc_updates) or np.asarray(batch["teacher_accepted"]).any():
        raise ValueError("fixed-policy self-experience diagnostic requires BC/teacher0")
    evaluated, metrics = update(state, batch, replay_size)
    evaluated = evaluated.replace(
        actor=state.actor,
        log_alpha=state.log_alpha,
        alpha_optimizer=state.alpha_optimizer,
    )
    # Canonical metrics include a hypothetical actor/temperature update which
    # was discarded. Keep that distinction visible rather than relabeling it.
    return evaluated, metrics

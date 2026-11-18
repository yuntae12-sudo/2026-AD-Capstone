"""Opt-in diagnostic initialization, never a rewrite of learned weights."""


def zero_critic_action_initialization(state):
    import jax

    if (
        int(state.updates)
        or int(state.bc_updates)
        or int(state.q1.step)
        or int(state.q2.step)
    ):
        raise ValueError("action initialization requires a fresh learner")

    def parameters(params):
        result = jax.tree_util.tree_map(lambda x: x, params)
        kernel = result["Dense_0"]["kernel"]
        if kernel.ndim != 2 or kernel.shape[0] <= 60 or kernel.shape[1] != 256:
            raise ValueError("unexpected critic first-layer layout")
        result["Dense_0"]["kernel"] = kernel.at[-60:, :].set(0)
        return result

    q1 = state.q1.replace(params=parameters(state.q1.params))
    q2 = state.q2.replace(params=parameters(state.q2.params))
    return state.replace(q1=q1, q2=q2, target1=q1.params, target2=q2.params)

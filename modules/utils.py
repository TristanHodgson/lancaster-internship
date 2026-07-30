def policy_sum(mdp, state, action, V):
    # Computes \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    assert not mdp.is_terminal(
        state), f"{state} is terminal PS"
    return sum(prob * (reward + mdp.gamma * V[next_state]) for prob, next_state, reward in mdp.outcomes(state, action))


def argmax_policy_sum(mdp, state, V):
    # Computes  \argmax_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    assert not mdp.is_terminal(state), f"{state} is terminal APS"
    return max(mdp.actions(state), key=lambda action: policy_sum(mdp, state, action, V))


def max_policy_sum(mdp, state, V):
    # Computes  $\max_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s'))$ for all actions a in state s
    assert not mdp.is_terminal(state), f"{state} is terminal MPS"
    return max([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])


def policy_sum_gamma_1(mdp, state, action, u):
    return sum(
        probability * (reward + u[next_state])
        for probability, next_state, reward
        in mdp.outcomes(state, action)
    )


def argmax_policy_sum_gamma_1(mdp, state, u, old_action=None, tol=1e-8):
    # Computes \argmax_a \sum_{s', r} p(s', r | s, a) * (r + u(s'))
    values = {action: policy_sum_gamma_1(
        mdp, state, action, u) for action in mdp.actions(state)}
    best = max(values.values())
    # Don't change the policy unless we have to
    if old_action is not None and values[old_action] >= best - tol * max(1.0, abs(best)):
        return old_action
    return max(values, key=lambda action: values[action])


def span_norm(x):
    # Computes the span norm of list x
    return max(x) - min(x)


def action_from_state(state, policy):
    # Computes \pi(state)
    # Requires a deterministic policy, please do not put in actions with probability 0
    if state not in policy:
        raise KeyError(f"{state} is missing from the policy")
    assert len(
        policy[state]) == 1, f"Stochastic policy {state}, {policy[state]}"
    return next(iter(policy[state]))
    # next(iter(policy[state])) is used to get the first action in the policy for the state


def get_max_action(policy):
    max_action = -float("inf")
    max_action_state = None
    for state in policy.keys():
        assert len(policy[state]) == 1
        action = next(iter(policy[state]))
        if action > max_action:
            max_action = action
            max_action_state = state
    return max_action_state, max_action


def greedy_policy(mdp):
    # Returns a policy for the MDP that is greedy wrt the maximum possible reward (regardless of probability)
    return {
        state: {max(mdp.actions(state), key=lambda action: max(
            r for _, _, r in mdp.outcomes(state, action))): 1.0}
        for state in mdp.states()
        if mdp.actions(state)
    }


def repair_all_policy(mdp):
    # Repairs every failed component immediately.
    return {state: {max(mdp.actions(state)): 1.0} for state in mdp.states()}

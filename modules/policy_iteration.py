from modules.utils import action_from_state, policy_sum, argmax_policy_sum, argmax_policy_sum_gamma_1, greedy_policy, policy_sum_gamma_1, span_norm, argmax_policy_sum_gamma_1

#################
### Gamma < 1 ###
#################


def policy_evaluation(mdp, policy, original_value, epsilon):
    # We do the inplace method since it has faster convergence
    delta = float("inf")
    V = original_value.copy()
    while delta > epsilon:
        delta = 0
        for state in mdp.states():
            if mdp.is_terminal(state):
                continue
            v = V[state]
            V[state] = policy_sum(
                mdp, state, action_from_state(state, policy), V)
            delta = max(delta, abs(v - V[state]))
    return V


def policy_improvement(mdp, V, policy):
    new_policy = {}
    policy_stable = True
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        old_action = action_from_state(state, policy)
        # old_action = \pi(state)
        pi_s = argmax_policy_sum(mdp, state, V)
        new_policy[state] = {pi_s: 1}
        if old_action != pi_s:
            policy_stable = False
    return new_policy, policy_stable


def discounted_policy_iteration(mdp, policy, epsilon):
    policy_stable = False
    V = {state: 0 for state in mdp.states()}
    while not policy_stable:
        V = policy_evaluation(mdp, policy, V, epsilon)
        policy, policy_stable = policy_improvement(mdp, V, policy)
    return policy, V


#################
### Gamma = 1 ###
#################

def policy_improvement_gamma_1(mdp, u, policy):
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        old_action = action_from_state(state, policy)
        new_action = argmax_policy_sum_gamma_1(mdp, state, u, old_action)
        policy[state] = {new_action: 1.0}
    return policy


def policy_evaluation_sweep_gamma_1(mdp, policy, u):
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        action = action_from_state(state, policy)
        u[state] = policy_sum_gamma_1(mdp, state, action, u)
    return u


def policy_evaluation_gamma_1(mdp, policy, v, epsilon):
    delta = float("inf")
    while delta > epsilon:
        u = policy_evaluation_sweep_gamma_1(mdp, policy, v.copy())
        delta = span_norm([u[state] - v[state]
                          for state in mdp.states() if not mdp.is_terminal(state)])
        ref_val = next(iter(u.values()))
        v = {state: val - ref_val for state, val in u.items()}
    return v


def policy_iteration_gamma_1(mdp, policy, epsilon):
    policy_stable = False
    v = {state: 0 for state in mdp.states()}
    while not policy_stable:
        v = policy_evaluation_gamma_1(mdp, policy, v, epsilon)
        old_policy = policy.copy()
        policy = policy_improvement_gamma_1(mdp, v, policy)
        policy_stable = (policy == old_policy)
    return policy, v

#########################
### Combining the two ###
#########################


def policy_iteration(mdp, initial_policy, epsilon):
    if mdp.gamma < 1:
        policy = greedy_policy(mdp)
        return discounted_policy_iteration(mdp, policy, epsilon)
    else:
        return policy_iteration_gamma_1(mdp, initial_policy, epsilon)

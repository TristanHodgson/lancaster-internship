from modules.utils import max_policy_sum, argmax_policy_sum, policy_sum_gamma_1, argmax_policy_sum_gamma_1, span_norm

def value_iteration(mdp, epsilon):
    V = {state: 0 for state in mdp.states()}
    delta = float("inf")
    while delta > epsilon:
        delta = 0
        for state in mdp.states():
            if mdp.is_terminal(state):
                continue
            v = V[state]
            V[state] = max_policy_sum(mdp, state, V)
            delta = max(delta, abs(v - V[state]))
    policy = {}
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        pi_s = argmax_policy_sum(mdp, state, V)
        policy[state] = {pi_s: 1}
    return policy, V


def value_iteration_gamma_1(mdp, epsilon):
    s_star = next(iter(mdp.states()))
    u = w = {state: 0.0 for state in mdp.states()}
    delta = float("inf")
    while delta >= epsilon:
        new_u = {state: max(policy_sum_gamma_1(mdp, state, action, w) for action in mdp.actions(state)) for state in mdp.states()}
        w = {state: new_u[state] - new_u[s_star] for state in mdp.states()}
        delta = span_norm([new_u[state] - u[state] for state in mdp.states()])
        u = new_u
    return {state: {argmax_policy_sum_gamma_1(mdp, state, u): 1.0} for state in mdp.states()}, w


#########################
### Combining the two ###
#########################


def value_iteration(mdp, epsilon):
    if mdp.gamma == 1:
        return value_iteration_gamma_1(mdp, epsilon)
    else:
        return value_iteration(mdp, epsilon)

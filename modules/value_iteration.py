from modules.helper import max_policy_sum, argmax_policy_sum


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

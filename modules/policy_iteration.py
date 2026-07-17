from modules.helper import policy_sum, argmax_policy_sum
from modules.helper import action_from_state

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
                mdp, state, action_from_state(mdp, state, policy), V)
            delta = max(delta, abs(v - V[state]))
    return V


def policy_improvement(mdp, V, policy):
    new_policy = {}
    policy_stable = True
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        old_action = action_from_state(mdp, state, policy)
        # old_action = \pi(state)
        pi_s = argmax_policy_sum(mdp, state, V)
        new_policy[state] = {pi_s: 1}
        if old_action != pi_s:
            policy_stable = False
    return new_policy, policy_stable


def policy_iteration(mdp, policy, epsilon):
    policy_stable = False
    V = {state: 0 for state in mdp.states()}
    # Note: we use an initial V that is all zero, is there something better we should do instead?
    while not policy_stable:
        V = policy_evaluation(mdp, policy, V, epsilon)
        policy, policy_stable = policy_improvement(mdp, V, policy)
    return policy, V
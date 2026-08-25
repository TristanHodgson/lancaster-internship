import numpy as np
from modules.utils import action_from_state, argmax_policy_sum, greedy_policy, policy_sum_gamma_1, policy_matrices
from mpmath import mp, mpf


#################
### Gamma < 1 ###
#################


def policy_evaluation(mdp, policy):
    # The value of a fixed policy d solves v = r_d + \gamma P_d v, that is (I - \gamma P_d) v = r_d (Puterman 6.1.1)
    state_to_idx, P, R = policy_matrices(mdp, policy)
    A = np.eye(len(state_to_idx)) - float(mdp.gamma) * np.array(P.tolist(), dtype=float)
    V = np.linalg.solve(A, np.array(R.tolist(), dtype=float).ravel())
    return {state: V[state_to_idx[state]] for state in mdp.states()}


def policy_improvement(mdp, V, policy):
    # d_{n+1}(s) \in \argmax_a r(s, a) + \gamma \sum_j p(j | s, a) V(j), keeping d_n(s) when it ties
    return {
        state: policy[state] if mdp.is_terminal(state)
        else {argmax_policy_sum(mdp, state, V, action_from_state(state, policy)): 1.0}
        for state in mdp.states()
    }


def discounted_policy_iteration(mdp, policy):
    while True:
        V = policy_evaluation(mdp, policy)
        new_policy = policy_improvement(mdp, V, policy)
        if new_policy == policy:
            return policy, V
        policy = new_policy


#################
### Gamma = 1 ###
#################


def average_evaluation(mdp, policy):
    # Step 2 of Puterman 8.6.1
    state_to_idx, P, R = policy_matrices(mdp, policy)
    A = np.eye(len(state_to_idx)) - np.array(P.tolist(), dtype=float)
    A[:, 0] = 1.0
    x = np.linalg.solve(A, np.array(R.tolist(), dtype=float).ravel())
    return x[0], {state: 0.0 if state_to_idx[state] == 0 else x[state_to_idx[state]] for state in mdp.states()}


def average_improvement(mdp, h, policy):
    # Step 3 of Puterman 8.6.1
    new_policy = {}
    for state in mdp.states():
        scores = {action: policy_sum_gamma_1(mdp, state, action, h) for action in mdp.actions(state)}
        old_action = action_from_state(state, policy)
        best_action = old_action if scores[old_action] == max(scores.values()) else max(scores, key=scores.get)
        new_policy[state] = {best_action: 1.0}
    return new_policy


def policy_iteration_gamma_1(mdp, policy):
    # Puterman's unichain policy iteration (8.6.1, pg 378)
    while True:
        _, h = average_evaluation(mdp, policy)  # Step 2
        new_policy = average_improvement(mdp, h, policy)  # Step 3
        if new_policy == policy:
            return policy, h
        policy = new_policy


#########################
### Combining the two ###
#########################


def policy_iteration(mdp, initial_policy):
    if mdp.gamma < 1:
        policy = greedy_policy(mdp)
        return discounted_policy_iteration(mdp, policy)
    else:
        return policy_iteration_gamma_1(mdp, initial_policy)
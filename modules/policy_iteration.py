import numpy as np
from modules.utils import action_from_state, policy_sum, argmax_policy_sum, multichain_policy_gain, greedy_policy, policy_sum_gamma_1, expected_gain_sum, reward_scale, better_action, gain_maximising_actions, policy_matrices

#################
### Gamma < 1 ###
#################


def policy_evaluation(mdp, policy):
    # The value of a fixed policy d solves v = r_d + \gamma P_d v, that is (I - \gamma P_d) v = r_d (Puterman 6.1.1)
    state_to_idx, P, R = policy_matrices(mdp, policy)
    V = np.linalg.solve(np.eye(len(state_to_idx)) - mdp.gamma * P, R)
    return {state: V[state_to_idx[state]] for state in mdp.states()}


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


def discounted_policy_iteration(mdp, policy):
    policy_stable = False
    while not policy_stable:
        V = policy_evaluation(mdp, policy)
        policy, policy_stable = policy_improvement(mdp, V, policy)
    return policy, V


#################
### Gamma = 1 ###
#################




def gain_improvement(mdp, g, policy, threshold):
    # Step 3(a) of Puterman 9.2.1
    # d_{n+1}(s) \in \argmax_{a} \sum_j p(j | s, a) g(j)
    return {
        state: {better_action(
            state, policy,
            lambda s, a: expected_gain_sum(mdp, s, a, g),
            mdp.actions(state), threshold
        ): 1.0}
        for state in mdp.states()
    }


def bias_improvement(mdp, g, h, policy, threshold):
    # Step 3(b) of Puterman 9.2.1
    # d_{n+1}(s) \in \argmax_{a \in B_s} r(s, a) + \sum_j p(j | s, a) h(j)
    return {
        state: {better_action(
            state, policy,
            lambda s, a: policy_sum_gamma_1(mdp, s, a, h),
            gain_maximising_actions(mdp, state, g, threshold), threshold
        ): 1.0}
        for state in mdp.states()
    }


def policy_iteration_gamma_1(mdp, policy, tol=1e-8):
    # Puterman's multichain policy iteration (9.2.1, pg 452)
    threshold = tol * reward_scale(mdp)
    while True:
        g, h = multichain_policy_gain(mdp, policy) # Step 2
        new_policy = gain_improvement(mdp, g, policy, threshold)  # Step 3(a)
        if new_policy != policy:
            policy = new_policy
            continue
        new_policy = bias_improvement(mdp, g, h, policy, threshold)  # Step 3(b)
        if new_policy == policy:  # Step 4
            return policy, h
        policy = new_policy


#########################
### Combining the two ###
#########################


def policy_iteration(mdp, initial_policy, epsilon):
    if mdp.gamma < 1:
        policy = greedy_policy(mdp)
        return discounted_policy_iteration(mdp, policy)
    else:
        return policy_iteration_gamma_1(mdp, initial_policy)
from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import action_from_state, greedy_policy, all_close, graph_policy


########################
###  Hyperparameters ###
########################

EPSILON = 1e-16  # Error for policy evaluation
THETA = 1e-16  # Error for value iteration
TOL = 1e-8 # Tolerance for testing equality of policy iteration and value iteration

PARAMS = {
    "N": 10,
    "alpha": 0.01,
    "tau": 1,
    "p": 120,
    "r": 1,
    "delta": 0.001,
}
GAMMA = 0.9

########################
###     Calculate    ###
########################

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=GAMMA)
initial_policy = greedy_policy(mdp)
PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
VI_policy, VI_V = value_iteration(mdp, THETA)

assert all_close(PI_V, VI_V, tol=TOL), "Policy Iteration and Value Iteration did not converge to the same value function"

graph_policy(mdp, PI_policy, PARAMS["N"])
from time import perf_counter

from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import action_from_state, greedy_policy, all_close, graph_policy


########################
###  Hyperparameters ###
########################

EPSILON = 1e-10  # Error for policy evaluation
THETA = 1e-10  # Error for value iteration
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


########################
###    Speed Test    ###
########################

"""
Ns = [10*i for i in range(1, 11)]
Ps = [10*i for i in range(5, 16)]

start = perf_counter()
for N in Ns:
    for P in Ps:
        PARAMS["N"] = N
        PARAMS["p"] = P
        actions = generate_mdp(**PARAMS)
        mdp = MDP(actions=actions, gamma=GAMMA)
        initial_policy = greedy_policy(mdp)
        PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
        # graph_policy(mdp, PI_policy, N, title=f"Policy Heatmap for N={N}, P={P}")
end = perf_counter()
print(f"Time taken for policy iteration: {end - start:.4f} seconds")

start = perf_counter()
for N in Ns:
    for P in Ps:
        PARAMS["N"] = N
        PARAMS["p"] = P
        actions = generate_mdp(**PARAMS)
        mdp = MDP(actions=actions, gamma=GAMMA)
        initial_policy = greedy_policy(mdp)
        VI_policy, VI_V = value_iteration(mdp, THETA)
        # graph_policy(mdp, VI_policy, N, title=f"Policy Heatmap for N={N}, P={P}")
end = perf_counter()
print(f"Time taken for value iteration: {end - start:.4f} seconds")
"""

# Time taken for policy iteration: 135.5179 seconds
# Time taken for value iteration: 2013.8827 seconds

########################
###    Experiment    ###
########################


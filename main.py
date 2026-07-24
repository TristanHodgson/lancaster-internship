from time import perf_counter
from tqdm import tqdm
from pprint import pprint
import tabulate

from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import action_from_state, greedy_policy, all_close, graph_policy, get_max_action


########################
###  Hyperparameters ###
########################

PARAMS = {
    "N": 10, # Number of components
    "alpha": 1, # rate of failure
    "tau": 1000, # Rate of repair
    "p": 120, # Penalty for system going down
    "r": 1, # Repair cost
    "gamma": 1 # Discount factor
}


EPSILON = (1-PARAMS["gamma"])/(PARAMS["gamma"]) * 1e-8  # Error for policy evaluation
THETA = (1-PARAMS["gamma"])/(2*PARAMS["gamma"]) * 1e-8  # Error for value iteration (see 6.3.3 of Puterman)
TOL = 1e-6 # Tolerance for testing equality of policy iteration and value iteration

PARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])

########################
###     Calculate    ###
########################

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
initial_policy = greedy_policy(mdp)
PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
VI_policy, VI_V = value_iteration(mdp, THETA)

pprint(get_max_action(PI_policy))

assert all_close(PI_V, VI_V, tol=TOL), "Policy Iteration and Value Iteration did not converge to the same value function"

# graph_policy(mdp, PI_policy, PARAMS["N"])


########################
###    Speed Test    ###
########################

# Commented out to speed up run time. Results below.

"""
Ns = [10*i for i in range(1, 11)]
Ps = [10*i for i in range(5, 16)]

start = perf_counter()
for N in Ns:
    for P in Ps:
        PARAMS["N"] = N
        PARAMS["p"] = P
        actions = generate_mdp(**PARAMS)
        mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
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
        mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
        initial_policy = greedy_policy(mdp)
        VI_policy, VI_V = value_iteration(mdp, THETA)
        # graph_policy(mdp, VI_policy, N, title=f"Policy Heatmap for N={N}, P={P}")
end = perf_counter()
print(f"Time taken for value iteration: {end - start:.4f} seconds")
"""

# Time taken for policy iteration: 135.5179 seconds
# Time taken for value iteration: 2013.8827 secondsPARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])


########################
###    Experiment    ###
########################


Ns = [20,30,50]
Ps = [i for i in range(1, 10)] + [10**i for i in range(4,16, 4)]

table_data = []
for N in Ns:
    for P in tqdm(Ps):
        PARAMS["N"] = N
        PARAMS["p"] = P * N**2
        actions = generate_mdp(**PARAMS)
        mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
        initial_policy = greedy_policy(mdp)
        PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
        graph_policy(mdp, PI_policy, N, title=f"Policy Heatmap for N={N}, P={P * N**2}, tau={PARAMS['tau']}", SAVE=True, filename=f"N{N}_P{P * N**2}_t{PARAMS['tau']}")
        table_data.append([N, P * N**2, get_max_action(PI_policy)[0], get_max_action(PI_policy)[1]])

print(tabulate.tabulate(table_data, headers=["N", "P", "Max Action State", "Max Action"], tablefmt="github"))
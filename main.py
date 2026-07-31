from time import perf_counter
from tqdm import tqdm
from pprint import pprint
import tabulate

from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import graph_policy, policy_gain_gamma_1, policy_gain
from modules.utils import repair_all_policy, greedy_policy 
from modules.lp import lp

########################
###  Gurobi License  ###
########################

import os
os.environ["GRB_LICENSE_FILE"] = "secret/gurobi.lic"


########################
###  Hyperparameters ###
########################

PARAMS = {
    "N": 8, # Number of components
    "alpha": 1, # rate of failure, do not change
    "tau": 100, # Rate of repair
    "p": 1000, # Penalty for system going down
    "r": 1, # Repair cost, do not change
    "gamma": 1 # Discount factor
}


EPSILON = 1e-8  # Error for policy evaluation
THETA = 1e-8  # Error for value iteration

if PARAMS["gamma"] != 1:
    EPSILON = max((1-PARAMS["gamma"])/(PARAMS["gamma"]) * EPSILON, 1e-14)
    THETA = max((1-PARAMS["gamma"])/(PARAMS["gamma"]) * THETA, 1e-14)


PARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])

########################
###     Calculate    ###
########################

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
initial_policy = repair_all_policy(mdp)
PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
# VI_policy, VI_V = value_iteration(mdp, THETA)
LP_policy, LP_transient = lp(mdp)


graph_policy(mdp, LP_policy, PARAMS["N"])
graph_policy(mdp, PI_policy, PARAMS["N"])
# graph_policy(mdp, VI_policy, PARAMS["N"])

if PARAMS["gamma"] == 1:
    gain_lp, bias_lp = policy_gain_gamma_1(mdp, LP_policy)
    gain_pi, bias_pi = policy_gain_gamma_1(mdp, PI_policy)
else:
    gain_lp, bias_lp = policy_gain(mdp, LP_policy)
    gain_pi, bias_pi = policy_gain(mdp, PI_policy)


print(f"Gain from LP: {gain_lp:.12f} \t\t\tBias from LP:\n")
pprint(bias_lp)
print("\n\n\n\n")
print(f"Gain from PI: {gain_pi:.12f} \t\t\tBias from PI:\n")
pprint(bias_pi)
print(f"Disagreements: {[s for s in mdp.states() if LP_policy[s] != PI_policy[s]]}")


########################
### Testing equality ###
########################

failures = []
for N in range(4,12,2):
    for P in [10**i for i in range(2,7,1)]:
        PARAMS["N"] = N
        PARAMS["p"] = P
        PARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])
        actions = generate_mdp(**PARAMS)
        mdp = MDP(actions=actions, gamma=1)
        initial_policy = repair_all_policy(mdp)
        PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
        LP_policy, LP_transient = lp(mdp)
        if PI_policy != LP_policy:
            disagreement_states = {s for s in mdp.states() if LP_policy[s] != PI_policy[s]}
            very_bad_states = {s for s in disagreement_states if s not in LP_transient}
            print(f"Disagreement for N={N}, P={P}: {disagreement_states}")
            if very_bad_states:
                print(f"\n\n\nDisagreement for N={N}, P={P} that are not transient: {very_bad_states}\n\n\n")
            # graph_policy(mdp, LP_policy, N, title=f"LP Policy Heatmap for N={N}, P={P}")
            # graph_policy(mdp, PI_policy, N, title=f"PI Policy Heatmap for N={N}, P={P}")
            failures.append((N, P, disagreement_states))


print(tabulate.tabulate(failures, headers=["N", "P", "Disagreement States"], tablefmt="github"))

########################
###    Speed Test    ###
########################

# Commented out to speed up run time. Results below.
# VI only implemented for gamma < 1 so don't try a speed test with gamma = 1

"""
Ns = [10*i for i in range(1, 11)]
Ps = [10*i for i in range(5, 16)]
# Remember to set PARAMS["delta"] <= 1 / (PARAMS["N"] * PARAMS["tau"]) for each iteration

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

# Ns = [50]
# Ps = [i for i in range(1, 10)] + [10**i for i in range(4,12, 4)]

# table_data = []
# for N in Ns:
#     for P in tqdm(Ps):
#         PARAMS["N"] = N
#         PARAMS["p"] = P * N
#         PARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])
#         actions = generate_mdp(**PARAMS)
#         mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
#         initial_policy = greedy_policy(mdp)
#         PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
#         graph_policy(mdp, PI_policy, N, title=f"Policy Heatmap for N={N}, P={P * N }, tau={PARAMS['tau']}, gamma={PARAMS['gamma']}", SAVE=True, filename=f"gamma{PARAMS['gamma']}_N{N}_P{P * N}_t{PARAMS['tau']}")
#         table_data.append([N, P * N, get_max_action(PI_policy)[0], get_max_action(PI_policy)[1]])

# print(tabulate.tabulate(table_data, headers=["N", "P", "Max Action State", "Max Action"], tablefmt="github"))
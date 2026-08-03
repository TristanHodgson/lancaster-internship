from time import perf_counter
from tqdm import tqdm
from pprint import pprint
import tabulate

from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import graph_policy, policy_equal, uptime
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
    "gamma": 0.99, # Discount factor
    "k": 1 # k out of N system
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
VI_policy, VI_V = value_iteration(mdp, THETA)
LP_policy, LP_transient = lp(mdp)

print(f"Uptime for PI policy: {uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k'])}")
print(f"Uptime for VI policy: {uptime(mdp, VI_policy, PARAMS['N'], PARAMS['k'])}")
# print(f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")

graph_policy(mdp, LP_policy, PARAMS["N"])
graph_policy(mdp, PI_policy, PARAMS["N"])
graph_policy(mdp, VI_policy, PARAMS["N"])

########################
###    Speed Test    ###
########################

# Commented out to speed up run time. Results below.


# table_data = []
# for gamma in [1-10**(-i) for i in range(2,7)] + [1]:
#     row = []
#     for i in ["PI", "VI", "LP"]:
#         start = perf_counter()
#         for _ in range(500):
#             if i == "PI":
#                 PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
#             elif i == "VI":
#                 VI_policy, VI_V = value_iteration(mdp, THETA)
#             elif i == "LP":
#                 LP_policy, LP_transient = lp(mdp)
#         end = perf_counter()
#         row.append(end - start)
#     table_data.append([gamma] + row)
# print(tabulate.tabulate(table_data, headers=["Gamma", "PI time (s)", "VI time (s)", "LP time (s)"], tablefmt="github", floatfmt=".4f"))


# |     Gamma |   PI time (s) |   VI time (s) |   LP time (s) |
# |-----------|---------------|---------------|---------------|
# |  0.990000 |        3.6834 |      219.6479 |        2.4225 |
# |  0.999000 |        3.7741 |      221.0431 |        2.4227 |
# |  0.999900 |        3.7732 |      223.5712 |        2.4269 |
# |  0.999990 |        3.7714 |      224.3156 |        2.3885 |
# |  0.999999 |        3.7537 |      224.9884 |        2.3967 |
# |  1.000000 |        3.7694 |      220.9227 |        2.4012 |

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
from time import perf_counter
from tqdm import tqdm
from pprint import pprint
import tabulate
import matplotlib.pyplot as plt
import numpy as np

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
    "gamma": 1, # Discount factor
    "k": 1 # Number of components needed to be healthy
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

# graph_policy( LP_policy, PARAMS["N"])

# Fill in missing states in LP_policy with the maximum possible action
for state in mdp.states():
    if state not in LP_policy:
        LP_policy[state] = {max(mdp.actions(state)): 1}

print(f"Uptime for PI policy: {uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k'])}")
# print(f"Uptime for VI policy: {uptime(mdp, VI_policy, PARAMS['N'], PARAMS['k'])}")
print(f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")

graph_policy(LP_policy, PARAMS["N"], transient_states=LP_transient)
# graph_policy(PI_policy, PARAMS["N"])
# graph_policy(VI_policy, PARAMS["N"])

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
#         graph_policy(PI_policy, N, title=f"Policy Heatmap for N={N}, P={P * N }, tau={PARAMS['tau']}, gamma={PARAMS['gamma']}", SAVE=True, filename=f"gamma{PARAMS['gamma']}_N{N}_P{P * N}_t{PARAMS['tau']}")
#         table_data.append([N, P * N, get_max_action(PI_policy)[0], get_max_action(PI_policy)[1]])

# print(tabulate.tabulate(table_data, headers=["N", "P", "Max Action State", "Max Action"], tablefmt="github"))


###########################
### Binary Search for P ###
###########################

def uptime_binary_search(target_uptime, a, b,  N, tau, gamma, k=1, alpha=1, r=1):
    if b-a < 0.01:
        return int((a + b) / 2)
    # Finding uptime for p=(a+b)/2
    actions = generate_mdp(N=N, alpha=alpha, tau=tau, p=(a+b)/2, r=r, gamma=gamma, delta=1/(N*tau), k=k)
    mdp = MDP(actions=actions, gamma=gamma)
    initial_policy = repair_all_policy(mdp)
    PI_policy, _ = policy_iteration(mdp, initial_policy, EPSILON)
    actual_uptime = uptime(mdp, PI_policy, N, k)
    # Next iteration
    # print(f"Target uptime: {target_uptime}, actual uptime: {actual_uptime}, a: {a}, b: {b}")
    if actual_uptime < target_uptime:
        return uptime_binary_search(target_uptime, (a+b)/2, b, N, tau, gamma, k, alpha, r)
    else:
        return uptime_binary_search(target_uptime, a, (a+b)/2, N, tau, gamma, k, alpha, r)


# uptime_p = uptime_binary_search(
#     target_uptime=0.999999999,
#     a=0,
#     b=100000,
#     N=10,
#     tau=500,
#     gamma=1,
#     k=1
# )
# print(f"Binary search for P to achieve uptime of 0.999999999: {uptime_p}")


########################
###   Monotonicity   ###
########################

# table_data = []

# for p in range(1,100000, 100):
#     actions = generate_mdp(N=10, alpha=1, tau=1000, p=p, r=1, gamma=0.99, delta=1/(10*1000), k=1)
#     mdp = MDP(actions=actions, gamma=0.99)
#     initial_policy = repair_all_policy(mdp)
#     PI_policy, _ = policy_iteration(mdp, initial_policy, EPSILON)
#     actual_uptime = uptime(mdp, PI_policy, 10, 1)
#     table_data.append([p, actual_uptime])

# plt.plot([row[0] for row in table_data], [-np.log10(1-row[1]) for row in table_data], color="#426A5A")
# plt.title("Uptime vs P for N=10, tau=1000, gamma=0.99, k=1")
# plt.xlabel("P")
# plt.ylabel("9s of Uptime, -log(Downtime)")
# plt.savefig("plots/monotonicity_10_1000_0.99_1.svg", format="svg")
# plt.show()

# print(tabulate.tabulate(table_data, headers=["P", "Uptime"], tablefmt="github", floatfmt=".16f"))



########################
###   Target Uptime  ###
########################

PARAMS = {
    "N": 8, # Number of components
    "alpha": 1, # rate of failure, do not change
    "tau": 100, # Rate of repair
    "p": 1000, # Penalty for system going down
    "r": 1, # Repair cost, do not change
    "gamma": 1, # Discount factor
    "k": 1 # Number of components needed to be healthy
}
PARAMS["delta"] = 1 / (PARAMS["N"] * PARAMS["tau"])

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
LP_policy, LP_transient = lp(mdp, target_uptime=0.999999, N_k=PARAMS["N"] - PARAMS["k"])
pprint(LP_policy)
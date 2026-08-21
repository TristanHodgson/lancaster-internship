from time import perf_counter
from tqdm import tqdm
from pprint import pprint
import tabulate

import matplotlib.pyplot as plt
import numpy as np

from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import graph_policy, policy_equal, uptime, graph_policy_grid
from modules.utils import repair_all_policy, greedy_policy, policy_sum_gamma_1, action_from_state
from modules.lp import lp
from modules.conjectures import optimal_policy, test_monotonicity, test_action_monotonicity, test_recurrent_shape

########################
###  Gurobi License  ###
########################

import os
os.environ["GRB_LICENSE_FILE"] = "secret/gurobi.lic"


########################
###  Hyperparameters ###
########################

PARAMS = {
    "N": [4],  # Number of components of each type
    "alpha": [1],  # rate of failure, do not change
    "tau": [10],  # Rate of repair
    "p": 20,  # Penalty for system going down
    "r": [1],  # Repair cost, do not change
    "k": [1]  # Number of components needed to be healthy
}

GAMMA = 1  # Discount factor
CANCEL = True  # Whether to allow cancellation of repairs

THETA = 1e-8  # Error for value iteration

if GAMMA != 1:
    THETA = max((1-GAMMA)/(GAMMA) * THETA, 1e-14)


PARAMS["delta"] = 1 / sum(n * t for n, t in zip(PARAMS["N"], PARAMS["tau"]))
PARAMS["cancellation"] = CANCEL

########################
###     Calculate    ###
########################

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=GAMMA)

initial_policy = repair_all_policy(mdp)
PI_policy, PI_V = policy_iteration(mdp, initial_policy)
# VI_policy, VI_V = value_iteration(mdp, THETA)
LP_policy, LP_transient = lp(mdp)

# graph_policy(LP_policy, PARAMS["N"], LP_transient)

print(f"Uptime for PI policy: {uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k'])}")
# print(f"Uptime for VI policy: {uptime(mdp, VI_policy, PARAMS['N'], PARAMS['k'])}")
print(
    f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")

# graph_policy(LP_policy, PARAMS["N"], component=0)
# graph_policy(PI_policy, PARAMS["N"], component=1)

FILENAME = f"Cancel={CANCEL}, N={PARAMS['N']}, alpha={PARAMS['alpha']}, tau={PARAMS['tau']}, r={PARAMS['r']}, k={PARAMS['k']}, p={PARAMS['p']}, gamma={GAMMA}"
TITLE = "Example policy with cancellation" if CANCEL else "Example policy"
TITLE += f" (uptime={uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k']):.2%})"
graph_policy(PI_policy, PARAMS["N"], LP_transient, title=TITLE, filename=FILENAME, SAVE=True, cancellation=CANCEL)


########################
###   Target Uptime  ###
########################


# actions = generate_mdp(**PARAMS)
# mdp = MDP(actions=actions, gamma=GAMMA)
# target_uptime = 0.999
# LP_policy, LP_transient = lp(mdp, target_uptime=target_uptime, N=PARAMS["N"], k=PARAMS["k"])
# pprint(LP_policy)
# print(f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")


###########################
### Binary Search for P ###
###########################

def uptime_binary_search(target_uptime, a, b,  N, tau, gamma, k=1, alpha=1, r=1):
    if b-a < 0.01:
        return int((a + b) / 2)

    actions = generate_mdp(N=[N], alpha=[alpha], tau=[tau], p=(a + b) / 2, r=[r], delta=1/(N*tau), k=[k])
    mdp = MDP(actions=actions, gamma=gamma)
    initial_policy = repair_all_policy(mdp)
    PI_policy, _ = policy_iteration(mdp, initial_policy)
    actual_uptime = uptime(mdp, PI_policy, [N], [k])

    print(f"Target uptime: {target_uptime}, actual uptime: {actual_uptime}, a: {a}, b: {b}, p_mid: {(a + b) / 2}")
    if actual_uptime < target_uptime:
        return uptime_binary_search(target_uptime, (a + b) / 2, b, N, tau, gamma, k, alpha, r)
    else:
        return uptime_binary_search(target_uptime, a, (a + b) / 2, N, tau, gamma, k, alpha, r)


uptime_p = uptime_binary_search(
    target_uptime=0.999999,
    a=0,
    b=1000,
    N=10,
    tau=500,
    gamma=1,
    k=1
)
print(f"Binary search for P to achieve uptime of 0.999999: {uptime_p}")


########################
###    Speed Test    ###
########################

# # Commented out to speed up run time. Results below.
# table_data = []
# for gamma in [0.8,0.9,0.99,0.999,0.9999,1]:
#     row = [gamma]
#     for i in ["PI", "VI", "LP"]:
#         count = 0
#         start = perf_counter()
#         for N in range(2, 6):
#             for tau in [1, 10, 50, 100, 500, 1000]:
#                 for k in range(1, N//2):
#                     for p in [0, 5, 10, 50, 100, 500, 1000, 5000, 10000]:
#                         for _ in range(5):
#                             count += 1
#                             print(f"Testing algo={i}, gamma={gamma}, N={N}, tau={tau}, k={k}, p={p}")
#                             actions = generate_mdp(N=[N], alpha=[1], tau=[tau], p=p, r=[1], delta=1/(N*(tau+1)), k=[k])
#                             mdp = MDP(actions=actions, gamma=gamma)
#                             initial_policy = repair_all_policy(mdp)
#                             if i == "PI":
#                                 PI_policy, PI_V = policy_iteration(mdp, initial_policy, 1e-6)
#                             elif i == "VI":
#                                 VI_policy, VI_V = value_iteration(mdp, 1e-6)
#                             elif i == "LP":
#                                 LP_policy, LP_transient = lp(mdp)
#         end = perf_counter()
#         t = (end - start)/count
#         row.append(t)
#     print(row)
#     table_data.append(row)
    

# print(tabulate.tabulate(table_data, headers=["Gamma", "PI time (s)", "VI time (s)", "LP time (s)"], tablefmt="github", floatfmt=".4f"))
# print(f"Averaged over: {count}")

# |   Gamma |   PI time (s) |   VI time (s) |   LP time (s) |
# |---------|---------------|---------------|---------------|
# |  0.8000 |        0.0004 |        0.0023 |        0.0017 |
# |  0.9000 |        0.0004 |        0.0043 |        0.0017 |
# |  0.9900 |        0.0005 |        0.0381 |        0.0017 |
# |  0.9990 |        0.0005 |        0.3659 |        0.0017 |
# |  0.9999 |        0.0005 |        3.6259 |        0.0018 |
# |  1.0000 |        0.0012 |        0.1268 |        0.0029 |
# Averaged over: 540, took 37m37s


########################
###   Monotonicity   ###
########################

# table_data = []

# for p in range(1,50000, 100):
#     actions = generate_mdp(N=[10], alpha=[1], tau=[100], p=p, r=[1], delta=1/(10*1000), k=[1])
#     mdp = MDP(actions=actions, gamma=1)
#     LP_policy, _ = lp(mdp)
#     actual_uptime = uptime(mdp, LP_policy, [10], [1])
#     table_data.append([p, actual_uptime])

# plt.figure(figsize=(2.5*3.5/2, 1.5*3.5/2))
# plt.plot([row[0] for row in table_data], [-np.log10(1-row[1]) for row in table_data], color="#002147")
# plt.title("Uptime vs P")
# plt.xlabel("Downtime penalty (p)")
# plt.ylabel("9s of Uptime, -log(Downtime)")
# plt.savefig("img/mono/uptime.svg", format="svg", transparent=True, bbox_inches="tight", pad_inches=0.1)
# plt.show()

# print(tabulate.tabulate(table_data, headers=["P", "Uptime"], tablefmt="github", floatfmt=".16f"))


########################
###    Experiment    ###
########################
"""
PARAMS = {
    "N": [6], # Number of components of each type
    "alpha": [1], # rate of failure, do not change
    "tau": [100], # Rate of repair
    "r": [1], # Repair cost, do not change
    "k": [1] # Number of components needed to be healthy
}
PARAMS["delta"] = 1 / sum(n * t for n, t in zip(PARAMS["N"], PARAMS["tau"]))
GAMMA = 1 # Discount factor

# table_data = []
# for p in tqdm([i for i in range(0,100000,100)]+[i for i in range(100000,10000000, 1000)]):
#     PARAMS["p"] = p
    
#     actions = generate_mdp(**PARAMS)
#     mdp = MDP(actions=actions, gamma=GAMMA)
#     lp_policy, lp_transient = lp(mdp)
#     uptime_value = uptime(mdp, lp_policy, PARAMS["N"], PARAMS["k"])
#     table_data.append([p, uptime_value, lp_transient])
#     graph_policy(lp_policy, PARAMS["N"], lp_transient, title=f"p={p}", filename=f"experiment1/experiment_p_{p:010d}", SAVE=True)


PARAMS = {
    "N": [3,3], # Number of components of each type
    "alpha": [1,1], # rate of failure, do not change
    "tau": [100,100], # Rate of repair
    "r": [2,1], # Repair cost, do not change
    "k": [1,1] # Number of components needed to be healthy
}
PARAMS["delta"] = 1 / sum(n * t for n, t in zip(PARAMS["N"], PARAMS["tau"]))
GAMMA = 1 # Discount factor

table_data = []
for p in tqdm([i for i in range(0,100000,100)]+[i for i in range(100000,10000000, 10000)]):
    PARAMS["p"] = p
    
    actions = generate_mdp(**PARAMS)
    mdp = MDP(actions=actions, gamma=GAMMA)
    lp_policy, lp_transient = lp(mdp)
    uptime_value = uptime(mdp, lp_policy, PARAMS["N"], PARAMS["k"])
    table_data.append([p, uptime_value, lp_transient])
    graph_policy_grid(lp_policy, PARAMS["N"], lp_transient, title=f"p={p}", filename=f"experiment2/experiment_p_{p:010d}", SAVE=True)
"""


########################
###   Experiment 2   ###
########################


# conjecture1 = []
# conjecture2 = []
# conjecture3 = []
# tested = 0
# compared = 0

# for gamma in tqdm([0.5, 0.7, 0.8, 0.9, 0.99, 0.999, 1]):
#     for tau in [1, 10, 100, 200, 500, 700, 850, 1000]:
#         for N in range(1, 8):
#             for k in [1]:
#                 previous_p, previous_policy = None, None
#                 for p in sorted([0, 10, 100, 200, 500, 700, 1000, 10000, 100000]):
#                     print(f"Testing gamma={gamma}, N={N}, tau={tau}, k={k}, p={p}")
#                     mdp, policy, h = optimal_policy(N, tau, k, p, gamma)
#                     tested += 1
#                     conjecture1 += [(gamma, tau, N, k, p) + violation for violation in test_monotonicity(mdp, policy, h)]
#                     conjecture3 += [(gamma, tau, N, k, p) + violation for violation in test_recurrent_shape(mdp, policy)]
#                     if previous_policy is not None:
#                         compared += 1
#                         conjecture2 += [(gamma, tau, N, k, previous_p, p) + violation for violation in test_action_monotonicity(mdp, policy, previous_policy)]
#                     previous_p, previous_policy = p, policy

# print("\n"*5)
# print(f"Tested {tested} instances")
# ### Conjecture 1: x > a \implies \pi(s_1 + x, s_2 - x) = 0 ###
# print("\n"*5)
# print(f"Conjecture 1: {len(conjecture1)} violations, {sum(1 for violation in conjecture1 if violation[-1])} of them with both states in the same recurrent class")
# for gamma, tau, N, k, p, state, x1, x2, f1, f2, same_class in conjecture1:
#     print(f"Violation for gamma={gamma}, tau={tau}, N={N}, k={k}, p={p}, state={state}, same class={same_class}: f({x1})={f1} < f({x2})={f2}")


# ### Conjecture 2: increasing p never decreases the optimal action ###
# print("\n"*5)
# print(f"Conjecture 2: {len(conjecture2)} violations over {compared} comparisons, {sum(1 for violation in conjecture2 if violation[-1])} of them at a recurrent state")
# for gamma, tau, N, k, p1, p2, state, a1, a2, recurrent in conjecture2:
#     print(f"Violation for gamma={gamma}, tau={tau}, N={N}, k={k}, state={state}, recurrent={recurrent}: pi_{p1}({state})={a1} > pi_{p2}({state})={a2}")


# ### Conjecture 3: each column of the recurrent class is downward closed in s_1 ###
# print("\n"*5)
# print(f"Conjecture 3: {len(conjecture3)} violations")
# for gamma, tau, N, k, p, state, missing in conjecture3:
#     print(f"Violation for gamma={gamma}, tau={tau}, N={N}, k={k}, p={p}: {state} is recurrent but {missing} is not in its class")
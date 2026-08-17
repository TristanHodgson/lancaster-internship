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

########################
###  Gurobi License  ###
########################

import os
os.environ["GRB_LICENSE_FILE"] = "secret/gurobi.lic"


########################
###  Hyperparameters ###
########################

PARAMS = {
    "N": [4, 4], # Number of components of each type
    "alpha": [1.5,1 ], # rate of failure, do not change
    "tau": [100,100], # Rate of repair
    "p": 100, # Penalty for system going down
    "r": [2, 1], # Repair cost, do not change
    "k": [1,1] # Number of components needed to be healthy
}

GAMMA = 1 # Discount factor

EPSILON = 1e-8  # Error for policy evaluation
THETA = 1e-8  # Error for value iteration

if GAMMA != 1:
    EPSILON = max((1-GAMMA)/(GAMMA) * EPSILON, 1e-14)
    THETA = max((1-GAMMA)/(GAMMA) * THETA, 1e-14)


PARAMS["delta"] = 1 / sum(n * t for n, t in zip(PARAMS["N"], PARAMS["tau"]))

########################
###     Calculate    ###
########################

actions = generate_mdp(**PARAMS)
mdp = MDP(actions=actions, gamma=GAMMA)

# initial_policy = repair_all_policy(mdp)
# PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
# VI_policy, VI_V = value_iteration(mdp, THETA)
LP_policy, LP_transient = lp(mdp)

graph_policy(LP_policy, PARAMS["N"], LP_transient)

# print(f"Uptime for PI policy: {uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k'])}")
# print(f"Uptime for VI policy: {uptime(mdp, VI_policy, PARAMS['N'], PARAMS['k'])}")
print(f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")

# graph_policy(LP_policy, PARAMS["N"], component=0)
# graph_policy(PI_policy, PARAMS["N"], component=1)

TITLE = f"N={PARAMS['N']}, alpha={PARAMS['alpha']}, tau={PARAMS['tau']}, r={PARAMS['r']}, k={PARAMS['k']}, p={PARAMS['p']}, gamma={GAMMA}"
graph_policy_grid(LP_policy, PARAMS["N"], LP_transient, title=TITLE, filename=TITLE, SAVE=True)


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

# Note currently only works for gamma=1


def test_monotonicity(N, tau, k, p):
    violations = 0
    violations_desc = []
    params = {"N": [N], "alpha": [1], "tau": [tau], "p": p, "r": [1], "k": [k]}
    params["delta"] = 1 / sum(n * t for n, t in zip(params["N"], params["tau"]))
    mdp = MDP(actions=generate_mdp(**params), gamma=1)
    print(f"Testing monotonicity for N={N}, tau={tau}, k={k}, p={p}")
    lp_policy, _ = lp(mdp)
    policy = {state: {max(lp_policy[state], key=lp_policy[state].get) if lp_policy.get(state) else (0,): 1.0} for state in mdp.states()}
    policy, h = policy_iteration(mdp, policy, EPSILON)

    f = {state: policy_sum_gamma_1(mdp, state, (0,), h) for state in mdp.states()} # evaluating f assumes 0 action
    tol = 1e-8 * max(abs(value) for value in f.values())

    for state in mdp.states():
        (s1, s2), = state
        a, = action_from_state(state, policy)
        F = [f[((s1 + x, s2 - x),)] for x in range(a, s2 + 1)]
        for i in range(len(F) - 1):
            if F[i + 1] > F[i] + tol:
                violations += 1
                violations_desc.append((tau, N, k, p, state, a, a + i, F[i], F[i + 1]))
    return violations, violations_desc


v = 0
desc = []
tested = 0
for tau in [1,10, 100, 1000, 500, 200,700, 850]:
    for N in range(1, 8):
        for k in range(1, N):
            for p in [0, 10, 100, 1000, 500, 200, 700, 10000, 100000, 1000000]:
                tested += 1
                mono = test_monotonicity(N, tau, k, p)
                v += mono[0]
                desc.extend(mono[1])

print(f"Total violations: {v} out of {tested} tests")

for violation in desc:
    tau, N, k, p, state, a1, a2, f1, f2 = violation
    print(f"Violation for tau={tau}, N={N}, k={k}, p={p}, state={state}: f({a1})={f1} < f({a2})={f2}")
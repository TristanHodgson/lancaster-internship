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
    "N": [4, 4], # Number of components of each type
    "alpha": [1, 1], # rate of failure, do not change
    "tau": [100, 50], # Rate of repair
    "p": 1000, # Penalty for system going down
    "r": [1, 2], # Repair cost, do not change
    "k": [1, 1] # Number of components needed to be healthy
}

GAMMA = 0.99 # Discount factor

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
initial_policy = repair_all_policy(mdp)

PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
# VI_policy, VI_V = value_iteration(mdp, THETA)
LP_policy, LP_transient = lp(mdp)

graph_policy(LP_policy, PARAMS["N"], LP_transient)

# Fill in missing states in LP_policy with the maximum possible action
for state in mdp.states():
    if state not in LP_policy:
        LP_policy[state] = {max(mdp.actions(state)): 1}

print(f"Uptime for PI policy: {uptime(mdp, PI_policy, PARAMS['N'], PARAMS['k'])}")
# print(f"Uptime for VI policy: {uptime(mdp, VI_policy, PARAMS['N'], PARAMS['k'])}")
print(f"Uptime for LP policy: {uptime(mdp, LP_policy, PARAMS['N'], PARAMS['k'])}")

# graph_policy(LP_policy, PARAMS["N"], component=0)
# graph_policy(PI_policy, PARAMS["N"], component=1)


########################
###   Target Uptime  ###
########################

# PARAMS["gamma"] = 1
# actions = generate_mdp(**PARAMS)
# mdp = MDP(actions=actions, gamma=PARAMS["gamma"])
# LP_policy, LP_transient = lp(mdp, target_uptime=0.999999, N=PARAMS["N"], k=PARAMS["k"])
# pprint(LP_policy)

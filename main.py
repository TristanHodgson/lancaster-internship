from tabulate import tabulate
from modules.mdp import *
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import json_import, action_from_state, greedy_policy, all_close
from pprint import pprint

########################
###  Hyperparameters ###
########################

EPSILON = 1e-16  # Error for policy evaluation
THETA = 1e-16  # Error for value iteration
TOL = 1e-8 # Tolerance for convergence in policy iteration and value iteration

########################
###     Load Data    ###
########################

actions = generate_mdp(
    N=10,
    alpha=0.1,
    tau=1,
    p=10000,
    r=1,
    delta=0.001
)


########################
###     Calculate    ###
########################


mdp = MDP(actions=actions, gamma=0.9)
initial_policy = greedy_policy(mdp)
PI_policy, PI_V = policy_iteration(mdp, initial_policy, EPSILON)
VI_policy, VI_V = value_iteration(mdp, THETA)

assert all_close(PI_V, VI_V, tol=TOL), "Policy Iteration and Value Iteration did not converge to the same value function"

########################
###      Display     ###
########################

table_data = []
terminal_rows = []
for state in sorted(mdp.states()):
    if mdp.is_terminal(state):
        terminal_rows.append([state, "NA", "NA", "NA"])
    else:
        table_data.append([
            state,
            # action_from_state(mdp, state, initial_policy),
            action_from_state(mdp, state, PI_policy),
            PI_V[state]
        ])

headers = ["State", "Initial Action", "New Action", "New Value"]
print(tabulate(table_data + terminal_rows,
               headers=headers, tablefmt="github", floatfmt=".4f"))

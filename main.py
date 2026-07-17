from tabulate import tabulate
from modules.mdp import MDP
from modules.policy_iteration import policy_iteration
from modules.value_iteration import value_iteration
from modules.helper import json_import, action_from_state

########################
###  Hyperparameters ###
########################

EPSILON = 1e-16 # Error for policy evaluation
THETA = 1e-16 # Error for value iteration
# TODO: How do we pick a good epsilon

########################
###     Load Data    ###
########################

# Dictionary of states:{state:[(probability, next_state, reward)]}
# Must be stochastic, i.e. the sum of the probabilities for each state, action pair must be 1
actions = json_import("data/mdp.json")

# Dictionary of states:{action: probability}
# This allows us to have both stochastic and deterministic policies
# However our implementation of policy iteration is explicitly for deterministic policies e.g.
# policy = {"s0": {"left": 0.5, "right": 0.5}, "s1": {"finish": 1.0}}
initial_policy = {
    "s0": {"down": 1.0},
    "s1": {"left": 1.0},
    "s2": {"down": 1.0},
    "s3": {"left": 1.0},
    "s4": {"left": 1.0},
    "s5": {"left": 1.0}
}

########################
###     Calculate    ###
########################


mdp = MDP(actions=actions, gamma=0.9)
policy, V = policy_iteration(mdp, initial_policy, EPSILON)
assert policy_iteration(mdp, policy, EPSILON) == value_iteration(mdp, THETA)


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
            action_from_state(mdp, state, initial_policy),
            action_from_state(mdp, state, policy),
            V[state]
        ])

headers = ["State", "Initial Action", "New Action", "New Value"]
print(tabulate(table_data + terminal_rows, headers=headers, tablefmt="github", floatfmt=".4f"))

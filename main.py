import json


########################
###  Hyperparameters ###
########################

EPSILON = 1e-6 # Convergence threshold for policy evaluation

########################
###     MDP class    ###
########################

class MDP:
    def __init__(self, actions, gamma):
        self.transitions = actions
        self.gamma = gamma # The discount factor

    def states(self):
        # Return the set of all states
        return self.transitions.keys()

    def actions(self, state):
        # For a given state, return a list of possible actions
        return self.transitions[state].keys()

    def outcomes(self, state, action):
        # For a given state and action, return a list of possible outcomes
        return self.transitions[state][action]

    def is_terminal(self, state):
        # For a given state, return True if the state is terminal
        return len(self.transitions[state]) == 0

    def terminal_states(self):
        # Return the set of all terminal states
        return {state for state in self.states() if self.is_terminal(state)}


########################
### Helper functions ###
########################

def policy_sum(mdp, state, action, V):
    # Computes \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    return sum(prob * (reward + mdp.gamma * V[next_state])
        for prob, next_state, reward in mdp.outcomes(state, action))

def argmax_policy_sum(mdp, state, V, policy):
    # Computes  \argmax_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    return max(mdp.actions(state), key=lambda action: policy_sum(mdp, state, action, V))

def max_policy_sum(mdp, state, V, policy):
    # Computes  \max_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    return max([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])

def action_from_state(mdp, state, policy):
    # Computes \pi(state)
    # Requires a deterministic policy, please do not put in actions with probability 0
    if state not in policy:
        raise KeyError(f"State '{state}' is missing from the policy. Is it a terminal state?")
    assert len(policy[state]) == 1, f"Policy for {state} is not deterministic: {policy[state]}"
    return next(iter(policy[state]))
    # next(iter(policy[state])) is used to get the first action in the policy for the state

def json_import(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

########################
### Policy iteration ###
########################

def policy_evaluation(mdp, policy, original_value, epsilon=EPSILON):
    # We do the inplace method since it has faster convergence
    delta = float("inf")
    V = original_value.copy()
    while delta > epsilon:
        delta = 0
        for state in mdp.states():
            if mdp.is_terminal(state):
                continue
            v = V[state]
            V[state] = policy_sum(mdp, state, action_from_state(mdp, state, policy), V)
            delta = max(delta, abs(v - V[state]))
    return V

def policy_improvement(mdp, V, policy):
    new_policy = {}
    policy_stable = True
    for state in mdp.states():
        if mdp.is_terminal(state):
            continue
        old_action = action_from_state(mdp, state, policy)
        # old_action = \pi(state)
        pi_s = argmax_policy_sum(mdp, state, V, policy)
        new_policy[state] = {pi_s: 1}
        if old_action != pi_s: policy_stable = False
    return new_policy, policy_stable
    
        
def policy_iteration(mdp, policy):
    policy_stable = False
    V = {state: 0 for state in mdp.states()}
    while not policy_stable:
        V = policy_evaluation(mdp, policy, V)
        policy, policy_stable = policy_improvement(mdp, V, policy)
    return policy, V


########################
### Creating a model ###
########################

# Dictionary of states:{state:[(probability, next_state, reward)]}
# Must be stochastic, i.e. the sum of the probabilities for each state, action pair must be 1
actions = json_import("data/mdp.json")

# Dictionary of states:{action: probability}
# This allows us to have both stochastic and deterministic policies
# However our implementation of policy iteration is explicitly for deterministic policies e.g.
# policy = {"s0": {"left": 0.5, "right": 0.5}, "s1": {"finish": 1.0}}

policy = {
    "s0": {"right": 1.0},
    "s1": {"finish": 1.0},
}

mdp = MDP(actions=actions, gamma=0.9)


print("Initial policy:", policy)
print("New policy and value function after policy iteration:")
policy, V = policy_iteration(mdp, policy)
print("Final policy:", policy)
print("Final value function:", V)
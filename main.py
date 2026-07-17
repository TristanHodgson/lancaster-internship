import numpy as np

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
    return np.sum(prob * (reward + mdp.gamma * V[next_state])
        for prob, next_state, reward in mdp.outcomes(state, action))

def argmax_policy_sum(mdp, state, V, policy):
    return np.argmax([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])

def max_policy_sum(mdp, state, V, policy):
    return np.max([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])


########################
### Policy iteration ###
########################

def policy_evaluation(mdp, policy, original_value, epsilon=EPSILON):
    # Requires a deterministic policy, do not put in actions with probability 0
    # We do the inplace method since it has faster convergence
    delta = float("inf")
    V = original_value.copy()
    while delta > epsilon:
        delta = 0
        for state in mdp.states():
            if mdp.is_terminal(state):
                continue
            v = V[state]
            assert len(policy[state]) == 1, f"Policy for {state} is not deterministic: {policy[state]}"
            V[state] = policy_sum(mdp, state, next(iter(policy[state])), V)
            # next(iter(policy[state])) is used to get the first action in the policy for the state
            delta = max(delta, abs(v - V[state]))
    return V


########################
### Creating a model ###
########################

# Dictionary of states:{state:[(probability, next_state, reward)]}
# Must be stochastic, i.e. the sum of the probabilities for each state, action pair must be 1
actions = {
    "s0": {
        "left": [(0.8, "s1", 2), (0.2, "s3", -1)],
        "right": [(0.3, "s3", 0), (0.7, "s3", 1)]
    },
    "s1": {
        "finish": [(1.0, "s3", 5)]
    },
    "s2": {}
}

# Dictionary of states:{action: probability}
# This allows us to have both stochastic and deterministic policies
policy = {
    "s0": {"left": 0.5, "right": 0.5},
    "s1": {"finish": 1.0},
}


mdp = MDP(actions=actions, gamma=0.9)


V = {state: 0 for state in mdp.states()}

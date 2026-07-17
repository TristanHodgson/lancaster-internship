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


def policy_sum(mdp, state, action, V):
    # Computes \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    return sum(prob * (reward + mdp.gamma * V[next_state])
        for prob, next_state, reward in mdp.outcomes(state, action))


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
    "s3": {}
}

# Dictionary of states:{action: probability}
# This allows us to have both stochastic and deterministic policies
policy = {
    "s0": {"left": 0.5, "right": 0.5},
    "s1": {"finish": 1.0},
}


mdp = MDP(actions=actions, gamma=0.9)


V = {state: 0 for state in mdp.states()}

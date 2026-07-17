class MDP:
    def __init__(self, actions, gamma):
        self.actions = actions
        self.gamma = gamma # The discount factor

    def states(self):
        # Return the set of all states
        return self.actions.keys()

    def actions(self, state):
        # For a given state, return a list of possible actions
        return self.actions[state].keys()

    def outcomes(self, state, action):
        # For a given state and action, return a list of possible outcomes
        return self.actions[state][action]

    def is_terminal(self, state):
        # For a given state, return True if the state is terminal
        return len(self.actions[state]) == 0

    def terminal_states(self):
        # Return the set of all terminal states
        return {state for state in self.states() if self.is_terminal(state)}
    

# Dictionary of states:[action: [(probability, next_state, reward)]]
actions = {
    "s0": {
        "left": [(0.8, "s1", 2), (0.2, "s3", -1)],
        "right": [(1.0, "s3", 0)]
    },
    "s1": {
        "finish": [(1.0, "s3", 5)]
    },
    "s3": {}
}

mdp = MDP(actions=actions, gamma=0.9)
print(mdp.terminal_states())
print(mdp.actions["s0"])
print(mdp.outcomes("s0", "left"))
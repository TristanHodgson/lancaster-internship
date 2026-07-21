class MDP:
    def __init__(self, actions, gamma):
        self.transitions = actions
        self.gamma = gamma  # The discount factor

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


def reward_function(state, action, r,  p, N, gamma):
    cost = r * (state[0] + action)
    if state[0] + state[1] == N:
        cost += p
    return -cost * gamma


def generate_mdp(N, alpha, tau, p, r, delta, gamma):
    model = {}
    for s1 in range(N + 1):
        for s2 in range(N + 1 - s1):
            state = (s1, s2)
            actions = {}
            for action in range(0, s2 + 1):
                degradation = (
                    (N - s1 - s2)*alpha * delta,
                    (s1 + action, s2 - action + 1),
                    delta * reward_function(state, action, r, p, N, gamma)
                )
                repair = (
                    (s1 + action)*tau * delta,
                    (s1 + action - 1, s2 - action),
                    delta * reward_function(state, action, r, p, N, gamma)
                )
                nothing = (
                    1 - degradation[0] - repair[0],
                    (s1 + action, s2 - action),
                    delta * reward_function(state, action, r,  p, N,gamma)
                )
                outcomes = [degradation, repair, nothing]
                actions[action] = [outcome for outcome in outcomes if outcome[0] > 0]
            model[state] = actions
    return model

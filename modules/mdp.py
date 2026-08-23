from mpmath import mp, mpf

mp.dps = 50

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


def is_healthy(state, N, k):
    # The system is up iff every type has at least k_j working components, i.e. s1_j + s2_j \le N_j - k_j
    return all(N[j]- s1 - s2 >= k[j] for j, (s1, s2) in enumerate(state))


def reward_function(state, action, r, p, N, k):
    cost = sum(r[j] * (s1 + action[j]) for j, (s1, _) in enumerate(state))
    if not is_healthy(state, N, k):
        cost += p
    return -cost


def generate_mdp(N, alpha, tau, p, r, delta, k, cancellation=False):
    # N, alpha, tau, r and k are lists indexed by component type, p is the system-wide downtime penalty
    alpha = [mpf(a) for a in alpha]
    tau = [mpf(t) for t in tau]
    r = [mpf(x) for x in r]
    p = mpf(p)
    delta = mpf(delta)

    model = {}

    state_space = [()]
    for n in N:            
        state_space = [state + ((s1, s2),) # Concatenate two tuples
                        for state in state_space
                        for s1 in range(n + 1) for s2 in range(n + 1 - s1)]

    for state in state_space:
        actions = {}
        action_space = [()]
        for s1, s2 in state:
            action_space = [action + (a,)
                            for action in action_space for a in range(-s1 if cancellation else 0, s2 + 1)]

        for action in action_space:
            # Every transition is taken from the post-action state, and they all earn the same reward
            new_state = tuple((s1 + action[j], s2 - action[j])
                                for j, (s1, s2) in enumerate(state))
            reward = reward_function(state, action, r, p, N, k)
            outcomes = []
            for j, (s1, s2) in enumerate(new_state):
                degradation = (
                    (N[j] - s1 - s2) * alpha[j] * delta,
                    new_state[:j] + ((s1, s2 + 1),) + new_state[j + 1:],
                    reward
                )
                repair = (
                    s1 * tau[j] * delta,
                    new_state[:j] + ((s1 - 1, s2),) + new_state[j + 1:],
                    reward
                )
                outcomes += [degradation, repair]
            self_transition_prob = 1 - sum(probability for probability, _, _ in outcomes)
            assert self_transition_prob >= 0, "Delta too large"
            nothing = (self_transition_prob, new_state, reward)
            outcomes.append(nothing)
            actions[action] = [outcome for outcome in outcomes if outcome[0] > 0] # Need to remove prob=0 events
        model[state] = actions
    return model

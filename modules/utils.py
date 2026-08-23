from mpmath import mp


def policy_sum(mdp, state, action, V):
    # Computes \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    assert not mdp.is_terminal(
        state), f"{state} is terminal PS"
    return sum(prob * (reward + mdp.gamma * V[next_state]) for prob, next_state, reward in mdp.outcomes(state, action))


def argmax_policy_sum(mdp, state, V):
    # Computes  \argmax_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    assert not mdp.is_terminal(state), f"{state} is terminal APS"
    return max(mdp.actions(state), key=lambda action: policy_sum(mdp, state, action, V))


def max_policy_sum(mdp, state, V):
    # Computes  $\max_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s'))$ for all actions a in state s
    assert not mdp.is_terminal(state), f"{state} is terminal MPS"
    return max([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])


def policy_sum_gamma_1(mdp, state, action, u):
    return sum(
        probability * (reward + u[next_state])
        for probability, next_state, reward
        in mdp.outcomes(state, action)
    )


def argmax_policy_sum_gamma_1(mdp, state, u, old_action=None, tol=1e-8):
    # Computes \argmax_a \sum_{s', r} p(s', r | s, a) * (r + u(s'))
    values = {action: policy_sum_gamma_1(
        mdp, state, action, u) for action in mdp.actions(state)}
    best = max(values.values())
    # Don't change the policy unless we have to
    if old_action is not None and values[old_action] >= best - tol * max(1.0, abs(best)):
        return old_action
    return max(values, key=lambda action: values[action])


def span_norm(x):
    # Computes the span norm of list x
    return max(x) - min(x)


def action_from_state(state, policy):
    # Computes \pi(state)
    # Requires a deterministic policy, please do not put in actions with probability 0
    if state not in policy:
        raise KeyError(f"{state} is missing from the policy")
    assert len(
        policy[state]) == 1, f"Stochastic policy {state}, {policy[state]}"
    return next(iter(policy[state]))
    # next(iter(policy[state])) is used to get the first action in the policy for the state


def get_max_action(policy):
    max_action = None
    max_action_state = None
    for state in policy.keys():
        assert len(policy[state]) == 1
        action = next(iter(policy[state]))
        if max_action is None or sum(action) > sum(max_action):
            max_action = action
            max_action_state = state
    return max_action_state, max_action


def greedy_policy(mdp):
    # Returns a policy for the MDP that is greedy wrt the maximum possible reward (regardless of probability)
    return {
        state: {max(mdp.actions(state), key=lambda action: max(
            r for _, _, r in mdp.outcomes(state, action))): 1.0}
        for state in mdp.states()
        if mdp.actions(state)
    }


def repair_all_policy(mdp):
    # Repairs every failed component immediately
    # The action set is a product of ranges, so the lexicographic maximum is (s2_1, ..., s2_J)
    return {state: {max(mdp.actions(state)): 1.0} for state in mdp.states()}


def expected_gain_sum(mdp, state, action, g):
    # Computes \sum_{j} p(j | s, a) g(j)
    return sum(probability * g[next_state] for probability, next_state, _ in mdp.outcomes(state, action))


def reward_scale(mdp):
    # The largest absolute reward in the model
    return max(abs(reward) for state in mdp.states() for action in mdp.actions(state) for _, _, reward in mdp.outcomes(state, action))


def reachable_states(mdp, policy, start):
    # Runs the search algorithm to find all states reachable from start under policy
    seen = {start}
    stack = [start]
    while stack:
        state = stack.pop()
        for _, next_state, _ in mdp.outcomes(state, action_from_state(state, policy)):
            if next_state not in seen:
                seen.add(next_state)
                stack.append(next_state)
    return seen


def recurrent_classes(mdp, policy):
    # The recurrent classes R_1, ..., R_k of the Markov chain induced by policy
    # A class is recurrent iff a closed communicating class
    reachable = {state: reachable_states(mdp, policy, state) for state in mdp.states()}
    classes = []
    for state in mdp.states():
        if any(state not in reachable[other] for other in reachable[state]):
            continue  # state is transient
        for recurrent_class in classes:
            if state in reachable[recurrent_class[0]]:
                recurrent_class.append(state)
                break
        else:
            classes.append([state])
    return classes


def policy_matrices(mdp, policy):
    # Builds the transition matrix P and expected reward vector R induced by policy,
    # together with the state ordering used to index them
    n = len(mdp.states())
    state_to_idx = {state: i for i, state in enumerate(mdp.states())}
    P = mp.zeros(n, n)  # Transition matrix
    R = mp.zeros(n, 1)  # Reward vector
    for s in mdp.states():
        for a, action_prob in policy[s].items():
            for prob, next_s, reward in mdp.outcomes(s, a):
                P[state_to_idx[s], state_to_idx[next_s]] += action_prob * prob
                R[state_to_idx[s]] += action_prob * prob * reward
    return state_to_idx, P, R


def multichain_policy_gain(mdp, policy):
    # Solves Puterman's multichain evaluation equations (9.2.1) and (9.2.2)
    # (I - P) g = 0
    # g + (I - P) h = R
    # h is only unique up to an additive constant on each recurrent class, so we impose Puterman's Condition 9.2.3: h = 0 at the first state of every recurrent class
    state_to_idx, P, R = policy_matrices(mdp, policy)
    n = len(mdp.states())
    I = mp.eye(n)

    # Form the block matrix \begin{pmatrix} I - P & 0 \\ I & I - P \end{pmatrix}
    A = mp.zeros(2 * n, 2 * n)
    A[0:n, 0:n] = I - P          # (I - P) g = 0
    A[n:2 * n, 0:n] = I          # g + (I - P) h = R
    A[n:2 * n, n:2 * n] = I - P
    # Form the block vector \begin{pmatrix} 0 \\ R \end{pmatrix}
    b = mp.zeros(2 * n, 1)
    for i in range(n):
        b[n + i] = R[i]

    # Impose Puterman's Condition 9.2.3: h = 0 at the first state of every recurrent class
    for recurrent_class in recurrent_classes(mdp, policy):
        i = state_to_idx[recurrent_class[0]]
        for j in range(2 * n):
            A[i, j] = 0
        A[i, n + i] = 1
        b[i] = 0

    x = mp.lu_solve(A, b)
    g = {state: x[state_to_idx[state]] for state in mdp.states()}
    h = {state: x[n + state_to_idx[state]] for state in mdp.states()}
    return g, h


def better_action(state, policy, value, allowed, threshold):
    # Finds the action that maximises value(state,action), keeping the current action if it is within threshold of the best
    scores = {action: value(state, action) for action in allowed}
    best = max(scores.values())
    old_action = action_from_state(state, policy)
    if old_action in scores and scores[old_action] >= best - threshold:
        return old_action
    return max(scores, key=lambda action: scores[action])


def gain_maximising_actions(mdp, state, g, threshold):
    # Returns the set of actions that maximise \sum_j p(j | s, a) g(j) within threshold
    scores = {action: expected_gain_sum(mdp, state, action, g) for action in mdp.actions(state)}
    best = max(scores.values())
    return [action for action, score in scores.items() if score >= best - threshold]
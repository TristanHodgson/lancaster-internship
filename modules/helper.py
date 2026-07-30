import numpy as np
import matplotlib.pyplot as plt


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


def span_norm(x):
    # Computes the span norm of list x
    return max(x) - min(x)


def action_from_state(mdp, state, policy):
    # Computes \pi(state)
    # Requires a deterministic policy, please do not put in actions with probability 0
    if state not in policy:
        raise KeyError(f"{state} is missing from the policy")
    assert len(
        policy[state]) == 1, f"Stochastic policy {state}, {policy[state]}"
    return next(iter(policy[state]))
    # next(iter(policy[state])) is used to get the first action in the policy for the state


def greedy_policy(mdp):
    # Returns a policy for the MDP that is greedy wrt the maximum possible reward (regardless of probability)
    return {
        state: {max(mdp.actions(state), key=lambda action: max(
            r for _, _, r in mdp.outcomes(state, action))): 1.0}
        for state in mdp.states()
        if mdp.actions(state)
    }


def all_close(V1, V2, tol=1e-8):
    # Check if two value functions are close enough
    return all(abs(V1[state] - V2[state]) < tol for state in V1)


def graph_policy(mdp, policy, N, title="Policy Heatmap", SAVE=False, filename="policy_heatmap"):
    policy_matrix = np.full((N + 1, N + 1), np.nan)

    for s1 in range(N + 1):
        for s2 in range(N + 1 - s1):
            state = (s1, s2)
            if state in policy:
                action = action_from_state(mdp, state, policy)
                policy_matrix[s1, s2] = action

    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(policy_matrix, cmap="viridis",
                   origin="lower", vmin=0, vmax=N)

    for s1 in range(N + 1):
        for s2 in range(N + 1):
            value = policy_matrix[s1, s2]
            if not np.isnan(value):
                ax.text(
                    s2, s1, f"{int(value)}",
                    ha="center", va="center",
                    color="white", fontsize=8
                )

    ax.set_xlabel("s2")
    ax.set_ylabel("s1")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    if SAVE:
        fig.savefig(f"img/{filename}.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def get_max_action(policy):
    max_action = -float("inf")
    max_action_state = None
    for state in policy.keys():
        assert len(policy[state]) == 1
        action = next(iter(policy[state]))
        if action > max_action:
            max_action = action
            max_action_state = state
    return max_action_state, max_action



def policy_sum_gamma_1(mdp, state, action, u):
    return sum(
        probability * (reward + u[next_state])
        for probability, next_state, reward
        in mdp.outcomes(state, action)
    )

def argmax_policy_sum_gamma_1(mdp, state, u, old_action=None, tol=1e-8):
    # Computes \argmax_a \sum_{s', r} p(s', r | s, a) * (r + u(s'))
    values = {action: policy_sum_gamma_1(mdp, state, action, u) for action in mdp.actions(state)}
    best = max(values.values())
    # Don't change the policy unless we have to
    if old_action is not None and values[old_action] >= best - tol * max(1.0, abs(best)):
        return old_action
    return max(values, key=lambda action: values[action])

def repair_all_policy(mdp):
    # Repairs every failed component immediately.
    return {state: {max(mdp.actions(state)): 1.0} for state in mdp.states()}


def policy_gain(mdp, policy):
    n = len(mdp.states())
    state_to_idx = {state: i for i, state in enumerate(mdp.states())}
    P = np.zeros((n, n)) # Transition matrix
    R = np.zeros(n) # Reward vector    
    for s in mdp.states():
        if mdp.is_terminal(s):
            continue
        i = state_to_idx[s]        
        for a, action_prob in policy.get(s, {}).items():
            for prob, next_s, reward in mdp.outcomes(s, a):
                j = state_to_idx[next_s]                
                P[i, j] += action_prob * prob
                R[i] += action_prob * prob * reward
    I = np.eye(n)
    V = np.linalg.solve(I - (mdp.gamma * P), R) # Solve the linear system (I - gamma * P) * V = R
    # Re-map the calculated values back to their corresponding states
    return {state: V[i] for i, state in enumerate(mdp.states())}

def policy_gain_gamma_1(mdp, policy):
    # Solves the evaluation equations g + (I - P) h = R exactly for the gain g and the bias h
    # h is only defined up to a constant, so we pin h = 0 at the last state by replacing that
    # column of (I - P) with ones, which puts g in its place
    n = len(mdp.states())
    state_to_idx = {state: i for i, state in enumerate(mdp.states())}
    P = np.zeros((n, n)) # Transition matrix
    R = np.zeros(n) # Reward vector
    for s in mdp.states():
        i = state_to_idx[s]
        for a, action_prob in policy[s].items():
            for prob, next_s, reward in mdp.outcomes(s, a):
                P[i, state_to_idx[next_s]] += action_prob * prob
                R[i] += action_prob * prob * reward

    A = np.eye(n) - P
    A[:, -1] = 1.0
    x = np.linalg.solve(A, R) # Not lstsq: A is singular exactly when the policy is multichain
    g, h = x[-1], np.append(x[:-1], 0.0)
    # A scalar gain only solves the equations if the policy is unichain, so we always check
    residual = np.max(np.abs(g + h - P @ h - R))
    assert residual <= 1e-8 * max(1.0, np.max(np.abs(R))), f"Residual {residual:.3e}: policy is multichain, so has no scalar gain"
    return g, {state: h[state_to_idx[state]] for state in mdp.states()}


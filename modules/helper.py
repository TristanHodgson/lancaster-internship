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
    # Computes  \max_a \sum_{s',r} p(s',r|s,a) (r + \gamma V(s')) for all actions a in state s
    assert not mdp.is_terminal(state), f"{state} is terminal MPS"
    return max([policy_sum(mdp, state, action, V) for action in mdp.actions(state)])


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

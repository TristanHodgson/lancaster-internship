import numpy as np
import matplotlib.pyplot as plt
from modules.utils import action_from_state


def all_close(V1, V2, tol=1e-8):
    # Check if two value functions are close enough
    return all(abs(V1[state] - V2[state]) < tol for state in V1)


def graph_policy(mdp, policy, N, title="Policy Heatmap", SAVE=False, filename="policy_heatmap"):
    policy_matrix = np.full((N + 1, N + 1), np.nan)

    for s1 in range(N + 1):
        for s2 in range(N + 1 - s1):
            state = (s1, s2)
            if state in policy:
                action = action_from_state(state, policy)
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


def policy_gain(mdp, policy):
    n = len(mdp.states())
    state_to_idx = {state: i for i, state in enumerate(mdp.states())}
    P = np.zeros((n, n))  # Transition matrix
    R = np.zeros(n)  # Reward vector
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
    # Solve the linear system (I - gamma * P) * V = R
    V = np.linalg.solve(I - (mdp.gamma * P), R)
    # Re-map the calculated values back to their corresponding states
    return {state: V[i] for i, state in enumerate(mdp.states())}


def policy_gain_gamma_1(mdp, policy):
    # Solves the evaluation equations g + (I - P) h = R exactly for the gain g and the bias h
    # h is only defined up to a constant, so we pin h = 0 at the last state by replacing that
    # column of (I - P) with ones, which puts g in its place
    n = len(mdp.states())
    state_to_idx = {state: i for i, state in enumerate(mdp.states())}
    P = np.zeros((n, n))  # Transition matrix
    R = np.zeros(n)  # Reward vector
    for s in mdp.states():
        i = state_to_idx[s]
        for a, action_prob in policy[s].items():
            for prob, next_s, reward in mdp.outcomes(s, a):
                P[i, state_to_idx[next_s]] += action_prob * prob
                R[i] += action_prob * prob * reward

    A = np.eye(n) - P
    A[:, -1] = 1.0
    # Not lstsq: A is singular exactly when the policy is multichain
    x = np.linalg.solve(A, R)
    g, h = x[-1], np.append(x[:-1], 0.0)
    # A scalar gain only solves the equations if the policy is unichain, so we always check
    residual = np.max(np.abs(g + h - P @ h - R))
    assert residual <= 1e-8 * \
        max(1.0, np.max(np.abs(
            R))), f"Residual {residual:.3e}: policy is multichain, so has no scalar gain"
    return g, {state: h[state_to_idx[state]] for state in mdp.states()}


def policy_equal(lp_policy, pi_policy):
    return all(lp_policy[state] == pi_policy[state] for state in lp_policy)
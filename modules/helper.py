import matplotlib.pyplot as plt
import numpy as np
from modules.mdp import MDP
from modules.utils import action_from_state, policy_matrices


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


def policy_equal(lp_policy, pi_policy):
    return all(lp_policy[state] == pi_policy[state] for state in lp_policy)


def policy_gain(mdp, policy):
    # Solves (I - gamma P) V = R for the expected total discounted reward
    _, P, R = policy_matrices(mdp, policy)
    return np.linalg.solve(np.eye(len(R)) - mdp.gamma * P, R).mean()


def policy_gain_gamma_1(mdp, policy):
    # Solves the evaluation equations g + (I - P) h = R for the scalar gain g
    # h is only defined up to a constant, so we pin h = 0 at the last state by replacing that column of (I - P) with ones, which puts g in its place
    _, P, R = policy_matrices(mdp, policy)
    A = np.eye(len(R)) - P
    A[:, -1] = 1.0
    x = np.linalg.solve(A, R)
    g, h = x[-1], np.append(x[:-1], 0.0)
    return g


def policy_mrp(mdp, policy, reward):
    # The Markov reward process induced by policy with the reward function replaced
    actions = {state: action_from_state(state, policy)
                for state in mdp.states()}
    return MDP({state: {actions[state]: [(prob, next_state, reward(state, actions[state]))
                                            for prob, next_state, _ in mdp.outcomes(state, actions[state])]}
                                            for state in mdp.states()}, mdp.gamma)


def evaluate_policy(mdp, policy, reward):
    # Gives the gain of the Markov reward process induced by the policy under the reward function
    mrp = policy_mrp(mdp, policy, reward)
    return policy_gain_gamma_1(mrp, policy)


def uptime(mdp, policy, N, k=1):
    # Gain of the indicator of the system being healthy
    return evaluate_policy(mdp, policy, lambda state, action: state[0] + state[1] <= N - k)

    
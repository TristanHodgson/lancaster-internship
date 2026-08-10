import matplotlib.pyplot as plt
import numpy as np
from modules.mdp import MDP, is_healthy
from modules.utils import action_from_state, policy_matrices


def all_close(V1, V2, tol=1e-8):
    # Check if two value functions are close enough
    return all(abs(V1[state] - V2[state]) < tol for state in V1)

def graph_policy(policy, N, transient_states=None, fixed=None, title="Policy Heatmap", SAVE=False, filename="policy_heatmap", ax=None):
    # Plots the action taken on the component type marked True in fixed, with every other type held at the state given in fixed
    if fixed is None:
        fixed = [True] + [(0, 0)] * (len(N) - 1)
    component = fixed.index(True)
    policy_matrix = np.full((N[component] + 1, N[component] + 1), np.nan)

    for s1 in range(N[component] + 1):
        for s2 in range(N[component] + 1 - s1):
            state = tuple(fixed[:component]) + ((s1, s2),) + tuple(fixed[component + 1:])
            if state in policy:
                action = action_from_state(state, policy)
                policy_matrix[s1, s2] = action[component]

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(policy_matrix, cmap="viridis", origin="lower", vmin=0, vmax=N[component])

    for s1 in range(N[component] + 1):
        for s2 in range(N[component] + 1 - s1):
            state = tuple(fixed[:component]) + ((s1, s2),) + tuple(fixed[component + 1:])
            value = policy_matrix[s1, s2]
            if not np.isnan(value):
                ax.text(
                    s2, s1, f"{int(value)}",
                    ha="center", va="center",
                    color="red" if state in transient_states else "white", fontsize=8
                )

    ax.set_xlabel("s2")
    ax.set_ylabel("s1")
    ax.set_title(title)
    if own_figure:
        fig.colorbar(im, ax=ax)
        if SAVE:
            fig.savefig(f"img/{filename}.svg", format="svg", bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
    return im


def graph_policy_grid(policy, N, transient_states=None, title="Policy Heatmap", SAVE=False, filename="policy_heatmap_grid"):
    # One subplot per state (i, j) of the first component, each plotting the action on the second component
    assert len(N) == 2, "graph_policy_grid requires exactly two component types"
    fig, axes = plt.subplots(N[0] + 1, N[0] + 1, figsize=(2.4 * (N[0] + 1), 2.4 * (N[0] + 1)), squeeze=False, layout="constrained")

    im = None
    for i in range(N[0] + 1):
        for j in range(N[0] + 1):
            ax = axes[N[0] - i][j]
            if i + j > N[0]:
                fig.delaxes(ax)
                continue
            im = graph_policy(policy, N, transient_states, fixed=[(i, j), True], ax=ax)
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel(f"$s_1^1 = {i}$" if j == 0 else "")
            ax.tick_params(labelbottom=(i == 0), labelleft=(j == 0))
            if i + j == N[0]:
                ax.set_title(f"$s_2^1 = {j}$")

    fig.colorbar(im, ax=list(fig.axes), shrink=0.9, aspect=40, label="components of type 2 sent for repair")

    fig.suptitle(title)
    fig.supxlabel(r"grid column: $s_2^1$   $\cdot$   within panel: $s_2^2$")
    fig.supylabel(r"grid row: $s_1^1$   $\cdot$   within panel: $s_1^2$")
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
    return policy_gain_gamma_1(mrp, policy) if mdp.gamma == 1.0 else policy_gain(mrp, policy)


def uptime(mdp, policy, N, k):
    # Gain of the indicator of the system being healthy
    return evaluate_policy(mdp, policy, lambda state, action: is_healthy(state, N, k))

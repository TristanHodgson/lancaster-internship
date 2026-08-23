import matplotlib.pyplot as plt
import numpy as np
from modules.mdp import MDP, is_healthy
from modules.utils import action_from_state, policy_matrices
from mpmath import mp

def all_close(V1, V2, tol=1e-8):
    # Check if two value functions are close enough
    return all(abs(V1[state] - V2[state]) < tol for state in V1)

def randomised_note(policy, states):
    # Lists the states where q_{d(s)} puts mass on more than one action, heaviest first
    randomised = {state: policy[state] for state in states if len(policy[state]) > 1}
    if not randomised:
        return ""
    return "\n".join(["Randomised states:"] + [
        f"{state}: " + ", ".join(f"{action} {q:.3g}" for action, q in sorted(distribution.items(), key=lambda item: -item[1]))
        for state, distribution in randomised.items()])


def graph_policy(policy, N, transient_states=None, fixed=None, title="Policy Heatmap", SAVE=False, filename="policy_heatmap", ax=None, cancellation=False):
    # Plots the action taken on the component type marked True in fixed, with every other type held at the state given in fixed
    # The policy may be randomised, in which case the value plotted is the expected action \sum_a q_{d(s)}(a) a_component
    if fixed is None:
        fixed = [True] + [(0, 0)] * (len(N) - 1)
    component = fixed.index(True)
    policy_matrix = np.full((N[component] + 1, N[component] + 1), np.nan)
    plotted = []

    for s1 in range(N[component] + 1):
        for s2 in range(N[component] + 1 - s1):
            state = tuple(fixed[:component]) + ((s1, s2),) + tuple(fixed[component + 1:])
            if state in policy:
                plotted.append(state)
                policy_matrix[s1, s2] = sum(q * action[component] for action, q in policy[state].items())

    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(policy_matrix, cmap="coolwarm" if cancellation else "viridis", origin="lower", vmin=-N[component] if cancellation else 0, vmax=N[component])
    for s1 in range(N[component] + 1):
        for s2 in range(N[component] + 1 - s1):
            state = tuple(fixed[:component]) + ((s1, s2),) + tuple(fixed[component + 1:])
            value = policy_matrix[s1, s2]
            if not np.isnan(value):
                if state in (transient_states or set()):
                    ax.add_patch(plt.Rectangle((s2 - 0.5, s1 - 0.5), 1, 1, color="black")) # Transient states are blacked out
                ax.text(
                    s2, s1, f"{value:g}",
                    ha="center", va="center",
                    color="black" if len(policy[state]) > 1 else "white", fontsize=8
                )

    ax.set_xlabel(r"Not being repaired $(s_2)$")
    ax.set_ylabel(r"Under repair $(s_1)$")
    ax.set_title(title)
    if own_figure:
        note = randomised_note(policy, plotted)
        if note:
            ax.set_xlabel("s2\n\n" + note)
        fig.colorbar(im, ax=ax)
        if SAVE:
            fig.savefig(f"img/{filename}.svg", format="svg", bbox_inches="tight", transparent=True)
        else:
            plt.show()
        plt.close(fig)
    return im


def graph_policy_grid(policy, N, transient_states=None, title="Policy Heatmap", SAVE=False, filename="policy_heatmap_grid", cancellation=False):
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
            im = graph_policy(policy, N, transient_states, fixed=[(i, j), True], ax=ax, cancellation=cancellation)
            ax.set_title("")
            ax.set_xlabel("")
            ax.set_ylabel(f"$s_1^1 = {i}$" if j == 0 else "")
            ax.tick_params(labelbottom=(i == 0), labelleft=(j == 0))
            if i + j == N[0]:
                ax.set_title(f"$s_2^1 = {j}$")

    fig.colorbar(im, ax=list(fig.axes), shrink=0.9, aspect=40, label="components of type 2 sent for repair")

    fig.suptitle(title)
    note = randomised_note(policy, policy.keys())
    fig.supxlabel(r"grid column: $s_2^1$   $\cdot$   within panel: $s_2^2$" + ("\n\n" + note if note else ""), fontsize=8)
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
    n = P.rows
    V = mp.lu_solve(mp.eye(n) - mdp.gamma * P, R)
    return sum(V[i] for i in range(n)) / n


def policy_gain_gamma_1(mdp, policy):
    # Solves the evaluation equations g + (I - P) h = R for the scalar gain g
    # h is only defined up to a constant, so we pin h = 0 at the last state by replacing that column of (I - P) with ones, which puts g in its place
    _, P, R = policy_matrices(mdp, policy)
    n = P.rows
    A = mp.eye(n) - P
    for i in range(n):
        A[i, n - 1] = 1
    x = mp.lu_solve(A, R)
    return x[n - 1]


def policy_mrp(mdp, policy, reward):
    # The Markov reward process induced by policy with the reward function replaced
    transitions = {}
    for state in mdp.states():
        outcomes = {}
        for action, q in policy[state].items():
            for prob, next_state, _ in mdp.outcomes(state, action):
                outcomes[next_state] = outcomes.get(next_state, 0.0) + q * prob
        mixed_reward = sum(q * reward(state, action) for action, q in policy[state].items())
        transitions[state] = {None: [(prob, next_state, mixed_reward) for next_state, prob in outcomes.items()]}
    return MDP(transitions, mdp.gamma)


def evaluate_policy(mdp, policy, reward):
    # Gives the gain of the Markov reward process induced by the policy under the reward function
    mrp = policy_mrp(mdp, policy, reward)
    mrp_policy = {state: {None: 1.0} for state in mrp.states()}
    return policy_gain_gamma_1(mrp, mrp_policy) if mdp.gamma == 1.0 else policy_gain(mrp, mrp_policy)


def uptime(mdp, policy, N, k):
    # Gain of the indicator of the system being healthy
    return evaluate_policy(mdp, policy, lambda state, action: is_healthy(state, N, k))

import numpy as np

from modules.helper import policy_gain_gamma_1
from modules.mdp import MDP


def max_damage(mdp):
    return max(s1 + s2 for ((s1, s2),) in mdp.states())


def has_cancellation(mdp):
    # Each action set is a range, so a negative action anywhere means the model allows cancellation
    return any(a < 0 for state in mdp.states() for (a,) in mdp.actions(state))


def bottom_rows(mdp, slack=None):
    return_rows = [[0]]
    for damage in range(1, max_damage(mdp) + 1):
        return_rows = [row + [target]
                       for row in return_rows
                       for target in range(0 if slack is None else max(0, row[-1] - slack), damage + 1)]
    return return_rows


def after_action(row, state, cancellation):
    # The state the policy moves to before anything else happens
    ((s1, s2),) = state
    target = row[s1 + s2] if cancellation else max(s1, row[s1 + s2])
    return ((target, s1 + s2 - target),)


def generate_policy(mdp, row, cancellation=None):
    cancellation = has_cancellation(
        mdp) if cancellation is None else cancellation
    return {state: {(after_action(row, state, cancellation)[0][0] - state[0][0],): 1.0}
            for state in mdp.states()}


def lumped(mdp, row, cancellation):
    index, order, weight = {}, [], []
    for state in mdp.states():
        b = after_action(row, state, cancellation)
        if b not in index:
            index[b] = len(order)
            order.append(b)
            weight.append(0)
        weight[index[b]] += 1
    return index, order, weight


def gain(mdp, row, cancellation):
    index, order, weight = lumped(mdp, row, cancellation)
    n = len(order)
    A, c = np.eye(n), np.zeros(n)
    for i, b in enumerate(order):
        for prob, next_state, reward in mdp.outcomes(b, (0,)):
            A[i, index[after_action(row, next_state, cancellation)]] -= mdp.gamma * prob
            c[i] += prob * reward
    f = np.linalg.solve(A, c)
    return sum(w * f[i] for i, w in enumerate(weight)) / sum(weight)


def gain_gamma_1(mdp, row, cancellation):
    transitions, stack = {}, [after_action(row, ((0, 0),), cancellation)]
    while stack:
        state = stack.pop()
        if state not in transitions:
            transitions[state] = {None: [(prob, after_action(row, next_state, cancellation), reward)
                                         for prob, next_state, reward in mdp.outcomes(state, (0,))]}
            stack += [next_state for _, next_state,
                      _ in transitions[state][None]]
    mrp = MDP(transitions, 1.0)
    return policy_gain_gamma_1(mrp, {state: {None: 1.0} for state in mrp.states()})


def brute_force(mdp, slack=None, tol=1e-9):
    cancellation = has_cancellation(mdp)
    rows = bottom_rows(mdp, slack)
    key = gain_gamma_1 if mdp.gamma == 1 else gain
    scores = [key(mdp, row, cancellation) for row in rows]
    best = max(scores)
    winner = next(row for row, score in zip(rows, scores) if score >= best - tol * max(1.0, abs(best)))
    return generate_policy(mdp, winner, cancellation)
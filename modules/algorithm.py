import numpy as np
from mpmath import mp

from modules.helper import policy_gain_gamma_1
from modules.mdp import MDP

# The reward and the transition rates depend only on the post-action state, so the optimality
# equations collapse to v(s) = max_{b in B(s)} f(b) with B(s) = {(s1 + x, s2 - x) : 0 <= x <= s2}.
#
#   Proved (discounted and undiscounted): if 0 <= x <= pi(s) and both states are recurrent then
#   pi(s1 + x, s2 - x) = pi(s) - x, so the states at or below the target on a diagonal share a
#   single post-action state.
#
#   Conjectured: if x > pi(s) then pi(s1 + x, s2 - x) = 0, which needs f to be unimodal along
#   the diagonal. This is what lets the target extend from the recurrent class to every state.
#
# Together these give pi(s1, s2) = max(0, t(s1 + s2) - s1) for a bottom row t, which is the
# O(N) description of an O(N^2) policy that the enumeration below searches over.
#
#   Proved with cancellation (discounted and undiscounted): B(s) depends only on s1 + s2, so
#   B(s) = B(s') for every state on a diagonal and the floor at s1 disappears. The post-action
#   state is exactly (t(m), m - t(m)), no conjecture required.


def max_damage(mdp):
    return max(s1 + s2 for ((s1, s2),) in mdp.states())


def has_cancellation(mdp):
    # Each action set is a range, so a negative action anywhere means the model allows cancellation
    return any(a < 0 for state in mdp.states() for (a,) in mdp.actions(state))


def bottom_rows(mdp, slack=None):
    # Rows with row[0] = 0 and 0 <= row[m] <= m
    # row[m] >= row[m - 1] is exactly horizontal monotonicity of the action, conjectured for
    # gamma = 1 but with a known counterexample when discounting, so it is only imposed there.
    # slack relaxes it to row[m] >= row[m - 1] - slack; None leaves the row unrestricted
    if mdp.gamma == 1:
        slack = 0
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
    cancellation = has_cancellation(mdp) if cancellation is None else cancellation
    return {state: {(after_action(row, state, cancellation)[0][0] - state[0][0],): 1.0}
            for state in mdp.states()}


def lumped(mdp, row, cancellation):
    # v(s) = f(b(s)), so the value on the whole state space is carried by f on the post-action
    # states the row actually uses, with weight counting the states that share each one.
    # Cancellation collapses every diagonal to a single post-action state, giving exactly N + 1
    index, order, weight = {}, [], []
    for state in mdp.states():
        b = after_action(row, state, cancellation)
        if b not in index:
            index[b] = len(order)
            order.append(b)
            weight.append(0)
        weight[index[b]] += 1
    return index, order, weight


def gain(mdp, row, cancellation, exact=False):
    # Solves f = R + gamma P f on the post-action states and averages over the original ones,
    # matching policy_gain's uniform initial distribution. Ties are re-ranked in mpmath, so
    # float64 only ever decides which rows are worth looking at again
    index, order, weight = lumped(mdp, row, cancellation)
    n = len(order)
    A, c = (mp.eye(n), mp.zeros(n, 1)) if exact else (np.eye(n), np.zeros(n))
    for i, b in enumerate(order):
        for prob, next_state, reward in mdp.outcomes(b, (0,)):
            j = index[after_action(row, next_state, cancellation)]
            A[i, j] -= mdp.gamma * (prob if exact else float(prob))
            c[i] += (prob * reward) if exact else float(prob) * float(reward)
    f = mp.lu_solve(A, c) if exact else np.linalg.solve(A, c)
    return sum(w * f[i] for i, w in enumerate(weight)) / sum(weight)


def gain_gamma_1(mdp, row, cancellation):
    # The gain ignores transient states, so only the post-action states reachable from (0, 0)
    # are needed and the sub-MDP has O(N) states rather than O(N^2)
    transitions, stack = {}, [after_action(row, ((0, 0),), cancellation)]
    while stack:
        state = stack.pop()
        if state not in transitions:
            transitions[state] = {None: [(prob, after_action(row, next_state, cancellation), reward)
                                         for prob, next_state, reward in mdp.outcomes(state, (0,))]}
            stack += [next_state for _, next_state, _ in transitions[state][None]]
    mrp = MDP(transitions, 1.0)
    return policy_gain_gamma_1(mrp, {state: {None: 1.0} for state in mrp.states()})


def brute_force(mdp, slack=None, tol=1e-9):
    cancellation = has_cancellation(mdp)
    rows = bottom_rows(mdp, slack)

    if mdp.gamma == 1:
        return generate_policy(mdp, max(rows, key=lambda row: gain_gamma_1(mdp, row, cancellation)), cancellation)

    scores = [gain(mdp, row, cancellation) for row in rows]
    best = max(scores)
    tied = [row for row, score in zip(rows, scores) if score >= best - tol * max(1.0, abs(best))]
    winner = tied[0] if len(tied) == 1 else max(tied, key=lambda row: gain(mdp, row, cancellation, exact=True))
    return generate_policy(mdp, winner, cancellation)
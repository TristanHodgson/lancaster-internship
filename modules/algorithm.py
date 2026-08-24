from modules.helper import policy_gain, policy_gain_gamma_1
from modules.mdp import MDP


def bottom_rows(mdp):
    # Rows with row[0] = 0 and row[damage] <= damage
    # Non-decreasing is an average reward conjecture, so we drop it when discounting
    rows = [[0]]
    for damage in range(1, max(s1 + s2 for ((s1, s2),) in mdp.states()) + 1):
        rows = [row + [target] for row in rows
                for target in range(row[-1] if mdp.gamma == 1 else 0, damage + 1)]
    return rows


def generate_policy(mdp, row):
    return {((s1, s2),): {(max(0, row[s1 + s2] - s1),): 1.0} for ((s1, s2),) in mdp.states()}


def after_action(row, state):
    # The state the policy moves to before anything else happens
    ((s1, s2),) = state
    target = max(s1, row[s1 + s2])
    return ((target, s1 + s2 - target),)


def gain_gamma_1(mdp, row):
    # States sharing an after-action state have the same future, so the chain the policy
    # walks is a lumped sub-MDP of the original on O(N) states rather than O(N^2)
    transitions, stack = {}, [((0, 0),)]
    while stack:
        state = stack.pop()
        if state not in transitions:
            transitions[state] = {None: [(prob, after_action(row, next_state), reward)
                                         for prob, next_state, reward in mdp.outcomes(state, (0,))]}
            stack += [next_state for _, next_state, _ in transitions[state][None]]
    mrp = MDP(transitions, 1.0)
    return policy_gain_gamma_1(mrp, {state: {None: 1.0} for state in mrp.states()})


def gain(mdp, row):
    # Lumping keeps the gain, but policy_gain averages the value over the states it is
    # given, so discounted policies are evaluated on the original state space instead
    return policy_gain(mdp, generate_policy(mdp, row)) if mdp.gamma < 1 else gain_gamma_1(mdp, row)


def brute_force(mdp):
    return generate_policy(mdp, max(bottom_rows(mdp), key=lambda row: gain(mdp, row)))
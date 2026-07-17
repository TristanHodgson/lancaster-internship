import json


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


def json_import(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
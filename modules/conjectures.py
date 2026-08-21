from modules.mdp import MDP, generate_mdp
from modules.lp import lp
from modules.policy_iteration import policy_iteration
from modules.utils import action_from_state, policy_sum, recurrent_classes, span_norm



def optimal_policy(N, tau, k, p, gamma=1, alpha=1, r=1):
    # Solves the one type model, returning the MDP, a deterministic optimal policy and h, which is the bias when gamma=1 and the discounted value function otherwise
    params = {"N": [N], "alpha": [alpha], "tau": [tau], "p": p, "r": [r], "k": [k], "cancellation": False}
    params["delta"] = 1 / sum(n * t for n, t in zip(params["N"], params["tau"]))
    mdp = MDP(actions=generate_mdp(**params), gamma=gamma)

    lp_policy, _ = lp(mdp)    
    policy = {state: {max(lp_policy[state], key=lp_policy[state].get) if lp_policy.get(state) else (0,): 1.0} for state in mdp.states()}
    policy, h = policy_iteration(mdp, policy)
    return mdp, policy, h


def class_of_state(mdp, policy):
    # Maps each recurrent state to the index of its recurrent class, transient states are absent
    return {state: i for i, recurrent_class in enumerate(recurrent_classes(mdp, policy)) for state in recurrent_class}


########################
###   Conjecture 1   ###
########################


def test_monotonicity(mdp, policy, h):
    # x > a \implies \pi(s_1 + x, s_2 - x) = 0
    # It is enough that f is non-increasing along the diagonal from the target onwards, since then no extra repair is ever worth starting
    violations = []
    R = class_of_state(mdp, policy)

    f = {state: policy_sum(mdp, state, (0,), h) for state in mdp.states()}  # evaluating f assumes 0 action
    tol = 1e-8

    for state in mdp.states():
        (s1, s2), = state
        a, = action_from_state(state, policy)
        F = [f[((s1 + x, s2 - x),)] for x in range(a, s2 + 1)]
        for i in range(len(F) - 1):
            if F[i + 1] > F[i] + tol:
                other = ((s1 + a + i + 1, s2 - a - i - 1),)
                violations.append((state, a + i, a + i + 1, F[i], F[i + 1], state in R and R[state] == R.get(other)))
    return violations


########################
###   Conjecture 2   ###
########################


def test_action_monotonicity(mdp, policy, previous_policy):
    # p \le p' \implies \pi_p(s) \le \pi_{p'}(s) for every state s, where previous_policy is optimal for the smaller p
    violations = []
    R = class_of_state(mdp, policy)

    for state in mdp.states():
        a, = action_from_state(state, previous_policy)
        b, = action_from_state(state, policy)
        if b < a:
            violations.append((state, a, b, state in R))
    return violations


########################
###   Conjecture 3   ###
########################


def test_recurrent_shape(mdp, policy):
    # (s_1, s_2) \in R \implies (s_1', s_2) \in R for every 0 \le s_1' \le s_1
    violations = []

    for recurrent_class in recurrent_classes(mdp, policy):
        R = set(recurrent_class)
        for state in R:
            (s1, s2), = state
            for x in range(s1):
                if ((x, s2),) not in R:
                    violations.append((state, ((x, s2),)))
    return violations


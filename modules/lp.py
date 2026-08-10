import pulp
import gurobipy as gp

from modules.mdp import is_healthy


def gurobi_solver(tol=1e-9):
    return pulp.GUROBI(msg=False, Method=1, Presolve=0, NumericFocus=3, FeasibilityTol=tol, OptimalityTol=tol)


def solve_lp(mdp, target_uptime=None, N=None, k=None):
    # maximises \sum_{s \in S} \sum_{a \in A(s)} r(s, a) x_{s, a}
    # Subject to:
    #   \sum_{a \in A(j)} x_{j, a} - \gamma \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = \alpha_j \quad \forall j \in S 
    #   x_{s, a} \ge 0 \quad \forall s \in S, \forall a \in A(s)
    # Note N and k are the per-type lists, only needed if target_uptime is set
    prob = pulp.LpProblem("MDP_LP_Discounted", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    alpha = 1.0 / len(list(mdp.states())) # vector that has to be positive, stochastic; we simplify to a scalar

    objective = []
    condition_term2 = {s: [] for s in mdp.states()}
    for s in mdp.states():
        for a in mdp.actions(s):
            expected_reward = sum(p * r for p, _, r in mdp.outcomes(s, a)) # r(s,a)
            for p, next_s, _ in mdp.outcomes(s, a):
                condition_term2[next_s].append(p * x[(s, a)]) # P(j | s, a) x_{s, a}, we only consider the states j with non-zero probability of being reached, hence use the set of outcomes
            objective.append(expected_reward * x[(s, a)]) # r(s,a) * x_{s,a}
    prob += pulp.lpSum(objective) # Set the objective as the sum of the elements of the list

    for j in mdp.states():
        condition_term1 = pulp.lpSum(x[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a}
        prob += (condition_term1 - mdp.gamma * pulp.lpSum(condition_term2[j]) == alpha) # Adding the whole of the first condition

    if target_uptime is not None:
        # \sum_{s, a} (1{s_1 + s_2 \le N - k} - U) x_{s, a} \ge 0, which is uptime >= U
        prob += (pulp.lpSum((is_healthy(s, N, k) - target_uptime) * x[(s, a)] for s in mdp.states() for a in mdp.actions(s)) >= 0) # Adding the uptime condition

    prob.solve(gurobi_solver()) # Solve using Gurobi
    assert pulp.LpStatus[prob.status] == "Optimal", f"LP status was {pulp.LpStatus[prob.status]}"
    # Puterman (6.9.4): q_{d(s)}(a) = \frac{x_{s, a}}{\sum_{a'} x_{s, a'}}
    values = {s: {a: max(0, pulp.value(x[(s, a)]) or 0) for a in mdp.actions(s)} for s in mdp.states()} # Gurobi can return small negatives
    return {s: {a: v / sum(vs.values()) for a, v in vs.items() if v > 0} for s, vs in values.items()}


# Multichain LP from 9.3 of Puterman
def solve_lp_gamma_1(mdp, target_uptime=None, N=None, k=None):
    # Maximises \sum_{s, a} r(s, a) x_{s, a}
    # Subject to:
    #   \forall j \in S \qquad \sum_{a \in A(j)} x_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = 0
    #   \forall j \in S \qquad \sum_{a \in A(j)} x_{j, a} + \sum_{a \in A(j)} y_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) y_{s, a} = \alpha_j
    #   x_{s, a} \ge 0, y_{s, a} \ge 0
    prob = pulp.LpProblem("MDP_LP_Gamma_1", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    y = {(s, a): pulp.LpVariable(f"y_{s}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    alpha = 1.0 / len(list(mdp.states()))

    objective = []
    condition1_term2 = {s: [] for s in mdp.states()}
    condition2_term3 = {s: [] for s in mdp.states()}
    for s in mdp.states():
        for a in mdp.actions(s):
            expected_reward = sum(p * r for p, _, r in mdp.outcomes(s, a)) # r(s,a)
            objective.append(expected_reward * x[(s, a)]) # r(s,a) * x_{s,a}
            for p, next_s, _ in mdp.outcomes(s, a):
                condition1_term2[next_s].append(p * x[(s, a)]) # P(j | s, a) x_{s, a}
                condition2_term3[next_s].append(p * y[(s, a)]) # P(j | s, a) y_{s, a}
    prob += pulp.lpSum(objective) # Set the objective as the sum of the elements of the list

    for j in mdp.states():
        condition1_term1 = condition2_term1 = pulp.lpSum(x[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a}
        condition2_term2 = pulp.lpSum(y[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} y_{j, a}
        prob += (condition1_term1 - pulp.lpSum(condition1_term2[j]) == 0) # Adding the whole of the first condition
        prob += (condition2_term1 + condition2_term2 - pulp.lpSum(condition2_term3[j]) == alpha) # Adding the whole of the second condition

    if target_uptime is not None:
        # \sum_{s, a} (1{s_1 + s_2 \le N - k} - U) x_{s, a} \ge 0, which is uptime \ge U once
        prob += (pulp.lpSum((is_healthy(s, N, k) - target_uptime) * x[(s, a)] for s in mdp.states() for a in mdp.actions(s)) >= 0) # Adding the uptime condition

    prob.solve(gurobi_solver()) # Solve using Gurobi
    assert pulp.LpStatus[prob.status] == "Optimal", f"LP status was {pulp.LpStatus[prob.status]}"

    # Puterman (9.3.7): q_{d(s)}(a) = x_{s, a} / \sum_{a'} x_{s, a'} on S_x = {s : \sum_a x_{s, a} > 0},
    # and y_{s, a} / \sum_{a'} y_{s, a'} elsewhere; the policy is defined at every state.
    policy = {}
    transient_states = set()
    for s in mdp.states():
        values = {a: pulp.value(x[(s, a)]) for a in mdp.actions(s) if pulp.value(x[(s, a)]) > 0}
        if not values:
            transient_states.add(s) # States are transient if the \sum_a x_{s, a} = 0
            values = {a: pulp.value(y[(s, a)]) for a in mdp.actions(s) if pulp.value(y[(s, a)]) > 0}
        total = sum(values.values())
        policy[s] = {a: v / total for a, v in values.items()}
        # assert target_uptime is not None or len(policy[s]) == 1, f"Randomised policy at {s}: {policy[s]}"
    return policy, transient_states


def lp(mdp, target_uptime=None, N=None, k=None):
    if mdp.gamma == 1: return solve_lp_gamma_1(mdp, target_uptime, N, k)
    else: return solve_lp(mdp, target_uptime, N, k), set()

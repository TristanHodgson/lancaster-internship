import pulp


def gurobi_solver(tol=1e-9):
    # TODO: check settings
    return pulp.GUROBI(msg=False, Presolve=0, NumericFocus=3, FeasibilityTol=tol, OptimalityTol=tol)


def lp_action(mdp, s, x, tol):
    # Given a state s and the LP variables x, returns the action maximising x_{s, a},
    # or None if x_{s, a} = 0 for every a
    values = {a: max(0, pulp.value(x[(s, a)]) or 0) for a in mdp.actions(s)} # Gurobi can return small negatives
    if sum(values.values()) <= tol:
        return None
    return max(values, key=lambda a: values[a])

def solve_lp(mdp):
    # maximises \sum_{s \in S} \sum_{a \in A(s)} r(s, a) x_{s, a}
    # Subject to:
    #   \sum_{a \in A(j)} x_{j, a} - \gamma \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = \alpha_j \quad \forall j \in S 
    #   x_{s, a} \ge 0 \quad \forall s \in S, \forall a \in A(s)
    prob = pulp.LpProblem("MDP_LP_Discounted", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
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

    prob.solve(gurobi_solver()) # Solve using Gurobi
    assert pulp.LpStatus[prob.status] == "Optimal", f"LP status was {pulp.LpStatus[prob.status]}"
    # Here \sum_a x_{j,a} = \alpha_j + \gamma \sum P(j|s,a) x_{s,a} >= \alpha > 0 at every state,
    # so the argmax is always well determined and needs no fallback
    return {s: {max(mdp.actions(s), key=lambda a: pulp.value(x[(s, a)]) or 0.0): 1.0} for s in mdp.states()}

# Unichain LP from 8.8 of Puterman
def solve_lp_gamma_1(mdp, tol=1e-9):
    # Maximises \sum_{s, a} r(s, a) x_{s, a}
    # Subject to:
    #   \forall j \in S \qquad \sum_{a \in A(j)} x_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = 0
    #   \sum_{s \in S} \sum_{a \in A(s)} x_{s, a} = 1
    #   x_{s, a} \ge 0
    prob = pulp.LpProblem("MDP_LP_Gamma_1", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}

    objective = []
    condition_term2 = {s: [] for s in mdp.states()}
    for s in mdp.states():
        for a in mdp.actions(s):
            expected_reward = sum(p * r for p, _, r in mdp.outcomes(s, a)) # r(s,a)
            objective.append(expected_reward * x[(s, a)]) # r(s,a) * x_{s,a}
            for p, next_s, _ in mdp.outcomes(s, a):
                condition_term2[next_s].append(p * x[(s, a)]) # P(j | s, a) x_{s, a}
    prob += pulp.lpSum(objective) # Set the objective as the sum of the elements of the list

    for j in mdp.states():
        condition_term1 = pulp.lpSum(x[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a}
        prob += (condition_term1 - pulp.lpSum(condition_term2[j]) == 0) # Adding the whole of the first condition
    prob += (pulp.lpSum(x.values()) == 1) # Adding the whole of the second condition

    prob.solve(gurobi_solver()) # Solve using Gurobi
    assert pulp.LpStatus[prob.status] == "Optimal", f"LP status was {pulp.LpStatus[prob.status]}"

    policy = {}
    unassigned_states = set()
    for s in mdp.states():
        action = lp_action(mdp, s, x, tol)
        if action is None:
            unassigned_states.add(s)
        else:
            policy[s] = {action: 1.0}
    return policy, unassigned_states

def lp(mdp):
    if mdp.gamma == 1: return solve_lp_gamma_1(mdp)
    else: return solve_lp(mdp), set()
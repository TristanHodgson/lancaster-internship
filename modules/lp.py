import pulp
from modules.policy_iteration import policy_iteration_gamma_1


def gurobi_solver(tol=1e-9):
    # TODO: check settings
    return pulp.GUROBI(msg=False, Presolve=0, NumericFocus=3, FeasibilityTol=tol, OptimalityTol=tol)


def lp_action(mdp, s, x, y, tol):
    # Given a state s, and the LP variables x and y, returns the action that maximises them
    # State is recurrent (i.e. sum(x) != 0) under the optimal policy => use max x variable
    # State is transient under the optimal policy => use max y variable
    values = {a: max(0.0, pulp.value(x[(s, a)]) or 0.0) for a in mdp.actions(s)} # Gurobi can return small negatives
    if sum(values.values()) <= tol:
        values = {a: max(0.0, pulp.value(y[(s, a)]) or 0.0) for a in mdp.actions(s)}
        print(f"State {s} is transient,")
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


# LP from 9.3 of Puternam, with a second stage that also optimises the bias
def solve_lp_gamma_1(mdp, tol=1e-9):
    # Stage 1 maximises \sum_{s, a} r(s, a) x_{s, a}
    # Stage 2 holds the gain at g* and maximises \sum_{s, a} (r(s, a) - g*) y_{s, a}
    # In both stages subject to:
    #   \forall j\in S \qquad \sum_a x_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = 0
    #   \forall j \in S \qquad \sum_{a \in A(j)} x_{j, a} + \sum_{a\in A(j)} y_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) y_{s, a} = \alpha_j
    #   x_{s, a} \ge 0, y_{s, a} \ge 0
    prob = pulp.LpProblem("MDP_LP_Gamma_1", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    y = {(s, a): pulp.LpVariable(f"y_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    alpha = 1.0 / len(list(mdp.states())) # vector that just has to be positive, stochastic; we simplify to a scalar

    gain_terms = []
    bias_terms = []
    condition1_term2 = {s: [] for s in mdp.states()}
    condition2_term2 = {s: [] for s in mdp.states()}
    for s in mdp.states():
        for a in mdp.actions(s):
            expected_reward = sum(p * r for p, _, r in mdp.outcomes(s, a)) # r(s,a)
            gain_terms.append(expected_reward * x[(s, a)]) # r(s,a) * x_{s,a}
            bias_terms.append(expected_reward * y[(s, a)]) # r(s,a) * y_{s,a}
            for p, next_s, _ in mdp.outcomes(s, a):
                condition1_term2[next_s].append(p * x[(s, a)]) # P(j | s, a) x_{s, a}
                condition2_term2[next_s].append(p * y[(s, a)]) # P(j | s, a) y_{s, a}

    for j in mdp.states():
        condition1_term1 = pulp.lpSum(x[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a}
        condition2_term1 = pulp.lpSum(x[(j, a)] + y[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a} + \sum_{a\in A(j)} y_{j, a}
        prob += (condition1_term1 - pulp.lpSum(condition1_term2[j]) == 0) # Adding the whole of the first condition
        prob += (condition2_term1 - pulp.lpSum(condition2_term2[j]) == alpha) # Adding the whole of the second condition

    # Stage 1: maximise the gain
    gain = pulp.lpSum(gain_terms)
    prob += gain # Stage 1 objective
    prob.solve(gurobi_solver())
    g = pulp.value(prob.objective)

    # Stage 2: maximise the bias, holding the gain at g*
    prob += (gain >= g - tol)
    prob += pulp.lpSum(bias_terms) - g * pulp.lpSum(y.values()) # Stage 2 objective
    prob.solve(gurobi_solver())

    return {s: {lp_action(mdp, s, x, y, tol): 1.0} for s in mdp.states()}

def lp(mdp):
    if mdp.gamma == 1: return solve_lp_gamma_1(mdp)
    else: return solve_lp(mdp)
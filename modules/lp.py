import pulp
from modules import mdp
from modules.mdp import MDP


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

    prob.solve(pulp.GUROBI(msg=False)) # Solve using Gurobi
    return {s: {max(mdp.actions(s), key=lambda a: pulp.value(x[(s, a)]) or 0): 1.0} for s in mdp.states()} # Produce a deterministic policy by taking the action with the highest x(s,a) value for each state s, or the first if all 0


def solve_lp_gamma_1(mdp):
    # maximises \sum_{s \in S} \sum_{a \in A(s)} r(s, a) x_{s, a}
    # Subject to:
    #   \forall j\in S \qquad \sum_a x_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) x_{s, a} = 0
    #   \forall j \in S \qquad \sum_{a \in A(j)} x_{j, a} + \sum_{a\in A(j)} y_{j, a} - \sum_{s \in S} \sum_{a \in A(s)} P(j | s, a) y_{s, a} = \alpha_j
    #   x_{s, a} \ge 0 \quad \forall s \in S, \forall a \in A(s)
    #   y_{s, a} \ge 0 \quad \forall s \in S, \forall a \in A(s)
    prob = pulp.LpProblem("MDP_LP_Gamma_1", pulp.LpMaximize)
    x = {(s, a): pulp.LpVariable(f"x_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    y = {(s, a): pulp.LpVariable(f"y_{s[0]}_{s[1]}_{a}", lowBound=0) for s in mdp.states() for a in mdp.actions(s)}
    alpha = 1.0 / len(list(mdp.states())) # vector that just has to be positive, stochastic; we simplify to a scalar

    objective = []
    condition1_term2 = {s: [] for s in mdp.states()}
    condition2_term2 = {s: [] for s in mdp.states()}
    for s in mdp.states():
        for a in mdp.actions(s):
            expected_reward = sum(p * r for p, _, r in mdp.outcomes(s, a)) # r(s,a)
            objective.append(expected_reward * x[(s, a)]) # r(s,a) * x_{s,a}
            for p, next_s, _ in mdp.outcomes(s, a):
                condition1_term2[next_s].append(p * x[(s, a)]) # working out P(j | s, a) x_{s, a}, we only consider the states j with non-zero probability of being reached
                condition2_term2[next_s].append(p * y[(s, a)]) # working out P(j | s, a) y_{s, a}, we only consider the states j with non-zero probability of being reached
    prob += pulp.lpSum(objective) # Set the objective as the sum of the elements of the list

    for j in mdp.states():
        condition1_term1 = pulp.lpSum(x[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a}
        condition2_term1 = pulp.lpSum(x[(j, a)] + y[(j, a)] for a in mdp.actions(j)) # \sum_{a \in A(j)} x_{j, a} + \sum_{a\in A(j)} y_{j, a}
        prob += (condition1_term1 - pulp.lpSum(condition1_term2[j]) == 0) # Adding the whole of the first condition
        prob += (condition2_term1 - pulp.lpSum(condition2_term2[j]) == alpha) # Adding the whole of the second condition

    prob.solve(pulp.GUROBI(msg=False)) # Solve using Gurobi

    policy = {}
    for s in mdp.states():
        # Sum the x values for the current state to check if it's recurrent
        x_sum = sum(pulp.value(x[(s, a)]) or 0 for a in mdp.actions(s))
        # We use a small tolerance (e.g., 1e-6) to account for floating-point inaccuracies
        if x_sum > 1e-6:
            # State is recurrent under the optimal policy -> use x variables
            best_action = max(mdp.actions(s), key=lambda a: pulp.value(x[(s, a)]) or 0)
        else:
            # State is transient under the optimal policy -> use y variables
            best_action = max(mdp.actions(s), key=lambda a: pulp.value(y[(s, a)]) or 0)
        policy[s] = {best_action: 1.0}
        
    return policy



def lp(mdp):
    if mdp.gamma == 1:
        return solve_lp_gamma_1(mdp)
    else:
        return solve_lp(mdp)
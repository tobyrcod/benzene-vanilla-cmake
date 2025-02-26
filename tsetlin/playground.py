from pathlib import Path
from utils import *

boardsize = 6

clauses, weights = UtilsTM.Model.load_trained_tmu_model_clauses(Path("models/tmu/6x6-baseline_exact_8limit/weighted_clauses.json"), boardsize)

i = 252

UtilsPlot.plot_clause(clauses[i], boardsize, Path("democlause.png"))

print(weights[i], clauses[i])

# -------------------------------------------------------------
print('------------------------------------------------------')
# -------------------------------------------------------------

def check():
    ones = 0
    for j, clause in enumerate(clauses):
        _, codes = UtilsTM.Model._clean_to_sat_clause(clause)
        ones += Counter(codes)[1]
        if 1 in Counter(codes):
            print('1', j)
    print(ones)

check()

# -------------------------------------------------------------
print('------------------------------------------------------')
# -------------------------------------------------------------

print('before', len(clauses), len(weights))
clauses, weights = UtilsTM.Model.make_model_clauses_satisfiable(clauses, weights)
print('after', len(clauses), len(weights))

branch_factors = list(map(UtilsTM.Model.calculate_clause_branch_factor, clauses))

print('upper bound total bf', sum(branch_factors))

# -------------------------------------------------------------
print('------------------------------------------------------')
# -------------------------------------------------------------

def fix():
    non_negated_clauses, non_negated_weights = [], []
    for j in range(len(clauses)):
        clause = clauses[j]
        weight = weights[j]

        expanded_clauses = UtilsTM.Model.expand_negated_literals_in_clause(clause)
        for expanded_clause in expanded_clauses:
            non_negated_clauses.append(expanded_clause)
            non_negated_weights.append(weight)

    return non_negated_clauses, non_negated_weights

# TODO: move to utils like the rest
print('before', len(clauses), len(weights))
clauses, weights = fix()
print('after', len(clauses), len(weights))

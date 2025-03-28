import json
import sys
from pathlib import Path
from typing import DefaultDict
import numpy as np

from utils import *

UtilsHex.SearchPattern.initialise()

# [Lost, Empty, Inconclusive, Won]
BEST_TYPE_WEIGHTS = [0.1, 0.1, 0.2, 1.0]

directory = Path("models/tmu/onevsone/6x6-baseline_exact_8limit")
clause_path = directory / "weighted_clauses.json"

# Calculate matches
# print('Finding matches...')
# UtilsHex.SearchPattern.calculate_matches_in_clauses(clause_path, 6)
# sys.exit()

# Reload the clauses & matches
clauses, clauses_weights, clauses_matches = UtilsHex.SearchPattern.load_matches_in_clauses(clause_path, 6)
print(len(clauses), len(clauses_weights), len(clauses_matches))

# Visualise a clause
# UtilsPlot.plot_literals(clauses[132], 6, directory / "plot.png")
# print(clauses_matches[132])
# sys.exit()

# ------------------------------------------------------------------------------------------------------------------


sys.exit()

# Perform the analysis
for i in range(len(clauses)):
    # If this clause has no templates, then we ignore it
    if i not in clauses_matches:
        continue

    # This clause has a template, so we want to know more about it
    clause = clauses[i]
    weights = clauses_weights[i]
    matches = clauses_matches[i]

    # We look one by one at each match in the clause
    for match in matches:
        # Each match type has its own best found weights for how important to a win it is
        match_type = match['MatchType']
        match_type_weight = BEST_TYPE_WEIGHTS[match_type.value-1]

        # This match is either found for black or white
        match_winner = match['MatchPlayer']




import json
import sys
from pathlib import Path
from typing import DefaultDict
import numpy as np

from utils import *

UtilsHex.SearchPattern.initialise()

# [Lost, Empty, Inconclusive, Won]
BEST_TYPE_WEIGHTS = [0.1, 0.1, 0.2, 1.0]

directory = Path("models/tmu/6x6-equalunder_8limit")
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

# Work out the score for the found clauses templates
# These matrices define scores for the 4 possible discrete cases:
# [0, 0]: Black Won the Clause and Black Won the Match
# [0, 1]: Black Won the Clause and White Won the Match
# [1, 0]: White Won the Clause and Black Won the Match
# [1, 1]: White Won the Clause and White Won the Match
discrete_clause_discrete_match_matrix = [[0, 0], [0, 0]]
discrete_clause_weighted_match_matrix = [[0, 0], [0, 0]]
# These lists define scores for the 2 possible weighted cases:
# [0, 0]: Black's clause weight total when Black Won the Match
# [0, 1]: Black's clause weight total when White Won the Match
# [1, 0]: White's clause weight total when Black Won the Match
# [1, 1]: White's clause weight total when White Won the Match
weighted_clause_discrete_match_matrix = [[0, 0], [0, 0]]
weighted_clause_weighted_match_matrix = [[0, 0], [0, 0]]
# We also just want to track the total number of matches for black and white
total_player_matches = [defaultdict(int), defaultdict(int)]
# And the total clause weight used by each player
total_player_weights = [0, 0]
total_player_absolute_weights = [0, 0]

# Perform the analysis
for i in range(len(clauses)):
    # If this clause has no templates, then we ignore it
    if i not in clauses_matches:
        continue

    # This clause has a template, so we want to know more about it
    clause = clauses[i]
    weights = clauses_weights[i]
    matches = clauses_matches[i]

    # Save information about the total weights we have seen
    total_player_weights[0] += weights[0]
    total_player_weights[1] += weights[1]
    total_player_absolute_weights[0] += abs(weights[0])
    total_player_absolute_weights[1] += abs(weights[1])

    # We look one by one at each match in the clause
    for match in matches:
        # Each match type has its own best found weights for how important to a win it is
        match_type = match['MatchType']
        match_type_weight = BEST_TYPE_WEIGHTS[match_type.value-1]
        match_type_discrete = -1 if match_type == UtilsHex.SearchPattern.Match.MatchType.LOST else 1

        # This match is either found for black or white
        match_winner = match['MatchPlayer']
        total_player_matches[match_winner][match_type.name] += 1

        # Discrete Clause Analysis
        discrete_clause_winner = np.argmax(weights)
        discrete_clause_discrete_match_matrix[discrete_clause_winner][match_winner] += 1 * match_type_discrete
        discrete_clause_weighted_match_matrix[discrete_clause_winner][match_winner] += 1 * match_type_weight

        # Weighted Clause Analysis
        weighted_clause_discrete_match_matrix[0][match_winner] += weights[0] * match_type_discrete
        weighted_clause_discrete_match_matrix[1][match_winner] += weights[1] * match_type_discrete
        weighted_clause_weighted_match_matrix[0][match_winner] += weights[0] * match_type_weight
        weighted_clause_weighted_match_matrix[1][match_winner] += weights[1] * match_type_weight

def print_discrete_clause_result_matrix(matrix):
    print('[0, 0]: Black Won the Clause and Black Won the Match')
    print(matrix[0][0])
    print('[0, 1]: Black Won the Clause and White Won the Match')
    print(matrix[0][1])
    print('[1, 0]: White Won the Clause and Black Won the Match')
    print(matrix[1][0])
    print('[1, 1]: White Won the Clause and White Won the Match')
    print(matrix[1][1])

def print_weighted_clause_result_matrix(matrix):
    print("[0, 0]: Black's clause weight total when Black Won the Match")
    print(matrix[0][0])
    print("[0, 1]: Black's clause weight total when White Won the Match")
    print(matrix[0][1])
    print("[1, 0]: White's clause weight total when Black Won the Match")
    print(matrix[1][0])
    print("[1, 1]: White's clause weight total when White Won the Match")
    print(matrix[1][1])

print('------------------------------')
print('black total matches')
print(len(total_player_matches[0]), total_player_matches[0])
print('white total matches')
print(len(total_player_matches[1]), total_player_matches[1])
print('------------------------------')
print('winning weights %')
print('black', len([1 for b, w in clauses_weights if b > w]) / len(clauses_weights))
print('white', len([1 for b, w in clauses_weights if w > b]) / len(clauses_weights))
print('------------------------------')
print('negative weights')
print('black', len([1 for b, w in clauses_weights if b < 0]))
print('white', len([1 for b, w in clauses_weights if w < 0]))
print('------------------------------')
print('black total weight')
print('sum: ', total_player_weights[0], 'abs: ', total_player_absolute_weights[0], 'avg: ', total_player_weights[0]/len(clauses_weights))
print('white total weight')
print('sum: ', total_player_weights[1], 'abs: ', total_player_absolute_weights[1], 'avg: ', total_player_weights[1]/len(clauses_weights))
print('------------------------------')
print('discrete clause, discrete match')
print_discrete_clause_result_matrix(discrete_clause_discrete_match_matrix)
print('------------------------------')
print('discrete clause, weighted match')
print_discrete_clause_result_matrix(discrete_clause_weighted_match_matrix)
print('------------------------------')
print('weighted clause, discrete match')
print_weighted_clause_result_matrix(weighted_clause_discrete_match_matrix)
print('------------------------------')
print('weighted clause, weighted match')
print_weighted_clause_result_matrix(weighted_clause_weighted_match_matrix)
print('------------------------------')
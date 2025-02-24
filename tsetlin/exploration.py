import csv
import itertools
import os.path
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path
from tqdm import tqdm

from tsetlin.utils import UtilsPlot, UtilsHex, UtilsTM, UtilsDataset, Helpers

def find_random_match_in_random_board():
    # Randomly select a boardsize
    boardsizes = list(range(6, 14))
    boardsize = random.choice(boardsizes)

    # Randomly select a pattern
    template_names = ['wheel'] # UtilsHex.SearchPattern.get_pattern_names()
    template_name = random.choice(template_names)

    # Randomly select a variation of this pattern
    variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
    search_pattern = variations[0] # random.choice(variations)

    # Randomly select a match type
    match_type = random.choice(list(UtilsHex.SearchPattern.Match.MatchType))

    i = 0
    matches = None
    while not matches:
        literals = UtilsTM.Literals.make_random_board(boardsize)
        matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
        matches = [match for match in matches if match.match_type == match_type]
        i += 1

    # For now, just plot the first match and ignore the rest
    match = matches[0]
    UtilsPlot.plot_search_pattern_match(match)
    print(match)


# TODO: think about fact that random boards have no continuity but dataset states do (actually from played games)
#  so they have some extra dependence on each other. May need to actually generate random games instead of random states
# TODO: combine shared logic in random and dataset versions?

def calculate_template_matches_in_random(ds_dist: UtilsDataset.Dataset):
    boardsize = ds_dist.boardsize
    start_board = 0
    num_boards = 1_000_000

    possible_num_pieces = list(ds_dist.state_num_pieces_counts.keys())
    num_pieces_distribution = [count / ds_dist.num_rows for count in ds_dist.state_num_pieces_counts.values()]
    assert sum(num_pieces_distribution) == 1

    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_dist.name}_dist_random_template_matches.csv"
    csv_headers = ['Board#', 'NumPieces', 'MatchPlayer', 'MatchType', 'MatchBaseName', 'MatchVarName', 'MatchX', 'MatchY']

    # If this match dataset already exist, we need to correctly add to it
    if os.path.exists(filepath):
        with open(filepath, mode='r', newline='') as ds_match:
            csv_reader = csv.reader(ds_match)

            curr_dataset_name = next(csv_reader)[1]
            assert curr_dataset_name == ds_dist.name
            curr_boardsize = int(next(csv_reader)[1])
            assert curr_boardsize == boardsize
            curr_num_boards = int(next(csv_reader)[1])
            assert curr_num_boards == num_boards
            headers = next(csv_reader)

            last_row = None
            for match in csv_reader:
                last_row = match
            if last_row == ['# Finished']:
                print('This search is already finished!')
                return
            last_board = int(last_row[headers.index('Board#')])
            start_board = last_board + 1
            print(f'Resuming existing search from board {start_board}...')

    # If this match dataset doesn't already exist, we need to make it
    else:
        with open(filepath, mode='w', newline='') as ds_match:
            csv_writer = csv.writer(ds_match)
            csv_writer.writerow(['dataset_distribution', ds_dist.name])
            csv_writer.writerow(['boardsize', boardsize])
            csv_writer.writerow(['num_boards', num_boards])
            csv_writer.writerow(csv_headers)

    # Continue generating boards to match until we are complete
    with open(filepath, mode='a', newline='') as ds_match:
        csv_writer = csv.writer(ds_match)
        for board in tqdm(range(start_board, num_boards)):
            # print(f"Board: {board}, Progress: {100 * board / num_boards:.3f}%")

            num_pieces = np.random.choice(possible_num_pieces, p=num_pieces_distribution)
            literals = UtilsTM.Literals.make_random_board(boardsize, num_pieces)
            for template_name in UtilsHex.SearchPattern.get_pattern_names() :
                variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
                for search_pattern in variations:
                    matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
                    for match in matches:
                        base_name = match.search_pattern.base_name
                        var_name = match.search_pattern.variation_name
                        csv_writer.writerow([board, num_pieces, match.player, match.match_type, base_name, var_name, match.coord[0], match.coord[1]])
        csv_writer.writerow(['# Finished'])

def load_template_matches_in_random(ds_dist: UtilsDataset.Dataset) -> Tuple[int, List]:
    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_dist.name}_dist_random_template_matches.csv"

    return _load_template_matches(filepath)

# ANALYSIS

def analyse_occurrence(dataset: UtilsDataset.Dataset):
    num_dataset, matches_dataset = load_template_matches_in_dataset(dataset)
    print(len(matches_dataset) / num_dataset)

    occurrence_counts = Counter(match['MatchBaseName'] for match in matches_dataset)
    normalized_counts = {key: value / num_dataset for key, value in occurrence_counts.items()}
    print(normalized_counts)

    matches_dataset = [match for match in matches_dataset if
                       match['MatchType'] == UtilsHex.SearchPattern.Match.MatchType.WON]
    occurrence_counts = Counter(match['MatchBaseName'] for match in matches_dataset)
    normalized_counts = {key: value / num_dataset for key, value in occurrence_counts.items()}
    print(normalized_counts)

def analyse_dataset_matches(dataset: UtilsDataset.Dataset):
    # TODO: check distribution of moves is the same between both
    # TODO: plot distribution of templates at all in both
    # TODO: plot found templates in percentage of games (at all, and then per win/lose)
    # TODO: use pandas dataframes instead

    num_dataset, matches_dataset = load_template_matches_in_dataset(dataset)

    # random_occurrences = Counter(match['MatchBaseName'] for match in matches_random)
    # random_frac = {key: value / num_random for key, value in random_occurrences.items()}
    # print(random_frac, num_random)

    # For searching regular games, we want the templates we won.
    # matches_dataset = [match for match in matches_dataset if
    #                    match['MatchType'] == UtilsHex.SearchPattern.Match.MatchType.WON]
    # For searching clauses, we want all the templates that we just haven't lost (maybe).
    matches_dataset = [match for match in matches_dataset if
                       match['MatchType'] != UtilsHex.SearchPattern.Match.MatchType.LOST]

    # Split dataset by winner/looser and matches by black/white and see changes

    # Get the matches grouped by who they matched for
    matches_black = [match for match in matches_dataset if match['MatchPlayer'] == 0]
    matches_white = [match for match in matches_dataset if match['MatchPlayer'] == 1]
    print("matches_black", "matches_white", "frac_black")
    print(len(matches_black), len(matches_white), len(matches_black) / len(matches_dataset))
    assert len(matches_black) + len(matches_white) == len(matches_dataset)

    # Get the matches grouped by which player wins
    matches_black_win = [match for match in matches_dataset if dataset.Y[match['Board#']] == 0]
    matches_white_win = [match for match in matches_dataset if dataset.Y[match['Board#']] == 1]
    print("matches_black_win", "matches_white_win", "frac_black_win")
    print(len(matches_black_win), len(matches_white_win), len(matches_black_win) / len(matches_dataset))
    assert len(matches_black_win) + len(matches_white_win) == len(matches_dataset)

    # Exploring Black Wins
    print('Black Wins')
    matches_black_and_black_win = [match for match in matches_black_win if match['MatchPlayer'] == 0]
    matches_white_and_black_win = [match for match in matches_black_win if match['MatchPlayer'] == 1]
    print(len(matches_black_and_black_win), len(matches_white_and_black_win), len(matches_black_and_black_win) / len(matches_black_win))
    assert len(matches_black_and_black_win) + len(matches_white_and_black_win) == len(matches_black_win)

    # Exploring White Wins
    print('White Wins')
    matches_black_and_white_win = [match for match in matches_white_win if match['MatchPlayer'] == 0]
    matches_white_and_white_win = [match for match in matches_white_win if match['MatchPlayer'] == 1]
    print(len(matches_black_and_white_win), len(matches_white_and_white_win), len(matches_white_and_white_win) / len(matches_white_win))
    assert len(matches_black_and_white_win) + len(matches_white_and_white_win) == len(matches_white_win)

    match_black_and_black_win_occurrences = Counter(match['MatchBaseName'] for match in matches_black_and_black_win)
    match_white_and_black_win_occurrences = Counter(match['MatchBaseName'] for match in matches_white_and_black_win)

    match_black_and_white_win_occurrences = Counter(match['MatchBaseName'] for match in matches_black_and_white_win)
    match_white_and_white_win_occurrences = Counter(match['MatchBaseName'] for match in matches_white_and_white_win)

    print('Black Wins')
    print(match_black_and_black_win_occurrences)
    print(match_white_and_black_win_occurrences)

    print('White Wins')
    print(match_white_and_white_win_occurrences)
    print(match_black_and_white_win_occurrences)


if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()

    # nbs, nws, pbs, pws = UtilsTM.Model.load_trained_model_clauses(Path("models/6x6-baseline_exact_model.pkl"), boardsize=6)
    # UtilsPlot.plot_clause(nbs[747], 6, Path("models/6x6-baseline_exact_clauses/test_clause_plot21.png"))

    # nbs_dataset = UtilsDataset.clauses_to_dataset("negative_black_clauses", clauses=nbs, clause_player=0, clause_winner=1, boardsize=6)
    # nws_dataset = UtilsDataset.clauses_to_dataset("negative_white_clauses", clauses=nws, clause_player=1, clause_winner=0, boardsize=6)
    # pbs_dataset = UtilsDataset.clauses_to_dataset("positive_black_clauses", clauses=pbs, clause_player=0, clause_winner=0, boardsize=6)
    # pws_dataset = UtilsDataset.clauses_to_dataset("positive_white_clauses", clauses=pws, clause_player=1, clause_winner=1, boardsize=6)
    # UtilsPlot.plot_literals(nbs_dataset.X[747], 6, Path("models/6x6-baseline_exact_clauses/test_clause_plot22.png"))

    # ps_dataset = nbs_dataset + nws_dataset
    # ps_dataset += pbs_dataset + pws_dataset
    # ps_dataset.name = "6x6-baseline_exact_clauses_option3"

    # calculate_template_matches_in_dataset(ps_dataset)
    # analyse_dataset_matches(ps_dataset)

    # calculate_template_matches_in_random(UtilsDataset.BASELINE)
    # analyse_occurrence(UtilsDataset.BASELINE)

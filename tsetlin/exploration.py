import csv
import itertools
import os.path
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import numpy as np
from pathlib import Path
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

def calculate_intertemplate_matchings() -> Dict[str, Dict[str, List[np.array]]]:
    # See which templates can be found entirely inside other templates (probably always just bridges...)

    # Use a default dict so we don't need to check for the key existing already
    matchings = defaultdict(lambda: defaultdict(list))

    def check_for_in_pair(looking_for: UtilsHex.SearchPattern, looking_in: UtilsHex.SearchPattern):
        local_origin = looking_in.induced_include_coords[0]
        matches = UtilsHex.SearchPattern.search_search_pattern(looking_for, looking_in)
        for match in matches:
            local_match_position = match.coord - local_origin
            matchings[looking_in.full_name][looking_for.full_name].append(local_match_position)

    template_base_names = UtilsHex.SearchPattern.get_pattern_names()
    for looking_for_base_name, looking_in_base_name in itertools.permutations(template_base_names, 2):
        looking_for_patterns = UtilsHex.SearchPattern.get_pattern_variations(looking_for_base_name)
        looking_in_patterns = UtilsHex.SearchPattern.get_pattern_variations(looking_in_base_name)
        for looking_for_pattern, looking_in_pattern in itertools.product(looking_for_patterns, looking_in_patterns):
            check_for_in_pair(looking_for_pattern, looking_in_pattern)

    # Convert from the easy to create defaultdict into the easy to work with regular dict
    matchings = Helpers.defaultdict_to_dict(matchings)
    return matchings

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
        for board in range(start_board, num_boards):
            print(f"Board: {board}, Progress: {100 * board / num_boards:.3f}%")

            num_pieces = np.random.choice(possible_num_pieces, p=num_pieces_distribution)
            literals = UtilsTM.Literals.make_random_board(boardsize, num_pieces)
            for template_name in UtilsHex.SearchPattern.get_pattern_names() :
                variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
                for search_pattern in variations:
                    matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
                    for match in matches:
                        base_name = match.search_pattern.base_name
                        var_name = match.search_pattern.variation_name
                        csv_writer.writerow([board, num_pieces, match.player, match.type, base_name, var_name, match.coord[0], match.coord[1]])
        csv_writer.writerow(['# Finished'])

def calculate_template_matches_in_dataset(ds_states: UtilsDataset.Dataset):
    boardsize = ds_states.boardsize
    start_board = 0
    num_boards = ds_states.num_rows

    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_states.name}_template_matches.csv"
    csv_headers = ['Board#', 'NumPieces', 'MatchPlayer', 'MatchType', 'MatchBaseName', 'MatchVarName', 'MatchX', 'MatchY']

    # If this match dataset already exist, we need to correctly add to it
    if os.path.exists(filepath):
        with open(filepath, mode='r', newline='') as ds_match:
            csv_reader = csv.reader(ds_match)

            curr_dataset_name = next(csv_reader)[1]
            assert curr_dataset_name == ds_states.name
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
            csv_writer.writerow(['dataset', ds_states.name])
            csv_writer.writerow(['boardsize', boardsize])
            csv_writer.writerow(['num_boards', num_boards])
            csv_writer.writerow(csv_headers)

    # Continue finding matches in each row of the dataset until we are done
    with open(filepath, mode='a', newline='') as ds_match:
        csv_writer = csv.writer(ds_match)
        for board in range(start_board, num_boards):
            print(f"Board: {board}, Progress: {100 * board / num_boards:.3f}%")

            literals = ds_states.X[board]
            num_pieces = sum(literals)
            for template_name in UtilsHex.SearchPattern.get_pattern_names():
                variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
                for search_pattern in variations:
                    matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
                    for match in matches:
                        base_name = match.search_pattern.base_name
                        var_name = match.search_pattern.variation_name
                        csv_writer.writerow([board, num_pieces, match.player, match.type, base_name, var_name, match.coord[0], match.coord[1]])
        csv_writer.writerow(['# Finished'])

def _load_template_matches(filepath: Path) -> Tuple[int, List]:
    if not os.path.exists(filepath):
        return -1, []

    intertemplate_matchings = calculate_intertemplate_matchings()
    num_boards = -1
    matches = []

    def process_board_matches(board_matches):
        # To remove any matches that are dependent on others:
        # First we find where we would remove any match if it existed
        potential_remove_matches = defaultdict(set)
        for match in board_matches:

            # TODO: maybe try with and without this to see what we should do
            # If this match is LOST, then we don't mind if we keep those dependent on it
            if match['MatchType'] == UtilsHex.SearchPattern.Match.MatchType.LOST:
                continue

            full_name = match['MatchBaseName']
            if match['MatchVarName']:
                full_name += f"_{match['MatchVarName']}"

            dependent_on_us = intertemplate_matchings.get(full_name, [])
            if not dependent_on_us:
                continue

            origin = np.array([match['MatchX'], match['MatchY']])
            for remove_full_name, remove_offsets in dependent_on_us.items():
                for remove_offset in remove_offsets:
                    potential_remove_matches[remove_full_name].add(tuple(origin+remove_offset))

        # We found no places we would want to remove matches, so exit
        if not potential_remove_matches:
            matches.extend(board_matches)
            return

        # Now we go through the matches again to see if we have any of the ones we want to remove
        kept_board_matches = []
        for match in board_matches:
            full_name = match['MatchBaseName']
            if match['MatchVarName']:
                full_name += f"_{match['MatchVarName']}"

            remove_coords = potential_remove_matches.get(full_name, [])
            if (match['MatchX'], match['MatchY']) in remove_coords:
                continue

            kept_board_matches.append(match)

        # Finally saving the ones we want to keep
        matches.extend(kept_board_matches)

    with open(filepath, mode='r', newline='') as ds_match:
        csv_reader = csv.reader(ds_match)

        ds_state_name = next(csv_reader)[1]
        boardsize = int(next(csv_reader)[1])
        num_boards = int(next(csv_reader)[1])
        headers = next(csv_reader)

        current_board = None
        current_board_matches = []
        for line in csv_reader:
            # Check we aren't are the end of the file
            if line == ['# Finished']:
                break
            match = dict(zip(headers, [int(x) if x.isdigit() else x for x in line]))
            board = match['Board#']
            if board != current_board:
                # We have reached the end of the current board
                # Process this board
                process_board_matches(current_board_matches)
                # Start the next board
                current_board = board
                current_board_matches = []
            current_board_matches.append(match)
        # Process the final board
        process_board_matches(current_board_matches)

    return num_boards, matches

def load_template_matches_in_random(ds_dist: UtilsDataset.Dataset) -> Tuple[int, List]:
    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_dist.name}_dist_random_template_matches.csv"

    return _load_template_matches(filepath)

def load_template_matches_in_dataset(ds_states: UtilsDataset.Dataset) -> Tuple[int, List]:
    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_states.name}_template_matches.csv"

    return _load_template_matches(filepath)

# ANALYSIS

def analyse():
    # TODO: check distribution of moves is the same between both
    # TODO: plot distribution of templates at all in both
    # TODO: plot found templates in percentage of games (at all, and then per win/lose)

    dataset = UtilsDataset.PLY_4
    num_random, matches_random = load_template_matches_in_random(dataset)
    num_dataset, matches_dataset = load_template_matches_in_dataset(dataset)

    random_occurrences = Counter(match['MatchBaseName'] for match in matches_random)
    random_frac = {key: value / num_random for key, value in random_occurrences.items()}
    # print(random_frac, num_random)

    dataset_occurrences = Counter(match['MatchBaseName'] for match in matches_dataset)
    dataset_frac = {key: value / num_dataset for key, value in dataset_occurrences.items()}
    # print(dataset_frac, num_dataset)

    # Split dataset by winner/looser and matches by black/white and see changes

    # Get the matches grouped by who they matched for
    matches_black = [match for match in matches_dataset if match['MatchPlayer'] == 0]
    matches_white = [match for match in matches_dataset if match['MatchPlayer'] == 1]
    print(len(matches_black), len(matches_white), len(matches_black) / len(matches_dataset))
    assert len(matches_black) + len(matches_white) == len(matches_dataset)

    # Get the matches grouped by which player wins
    matches_black_win = [match for match in matches_dataset if dataset.Y[match['Board#']] == 0]
    matches_white_win = [match for match in matches_dataset if dataset.Y[match['Board#']] == 1]
    print(len(matches_black_win), len(matches_white_win), len(matches_black_win) / len(matches_dataset))
    assert len(matches_black_win) + len(matches_white_win) == len(matches_dataset)

    # Exploring Black Wins
    matches_black_and_black_win = [match for match in matches_black_win if match['MatchPlayer'] == 0]
    matches_white_and_black_win = [match for match in matches_black_win if match['MatchPlayer'] == 1]
    print(len(matches_black_and_black_win), len(matches_white_and_black_win), len(matches_black_and_black_win) / len(matches_black_win))
    assert len(matches_black_and_black_win) + len(matches_white_and_black_win) == len(matches_black_win)

    # Exploring White Wins
    matches_black_and_white_win = [match for match in matches_white_win if match['MatchPlayer'] == 0]
    matches_white_and_white_win = [match for match in matches_white_win if match['MatchPlayer'] == 1]
    print(len(matches_black_and_white_win), len(matches_white_and_white_win), len(matches_black_and_white_win) / len(matches_white_win))
    assert len(matches_black_and_white_win) + len(matches_white_and_white_win) == len(matches_white_win)

    match_black_and_black_win_occurrences = Counter(match['MatchBaseName'] for match in matches_black_and_black_win)
    match_white_and_black_win_occurrences = Counter(match['MatchBaseName'] for match in matches_white_and_black_win)

    match_black_and_white_win_occurrences = Counter(match['MatchBaseName'] for match in matches_black_and_white_win)
    match_white_and_white_win_occurrences = Counter(match['MatchBaseName'] for match in matches_white_and_white_win)

    print(match_black_and_black_win_occurrences)
    print(match_white_and_black_win_occurrences)

    print(match_white_and_white_win_occurrences)
    print(match_black_and_white_win_occurrences)

find_random_match_in_random_board()
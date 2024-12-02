import csv
import os.path
import random
from collections import defaultdict

import numpy as np
from pathlib import Path
from tsetlin.utils import UtilsPlot, UtilsHex, UtilsTM, UtilsDataset

def find_random_match_in_random_board():
    # Randomly select a boardsize
    boardsizes = list(range(6, 14))
    boardsize = random.choice(boardsizes)

    # Randomly select a pattern
    template_names = UtilsHex.SearchPattern.get_pattern_names()
    template_name = random.choice(template_names)

    # Randomly select a variation of this pattern
    variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
    search_pattern = random.choice(variations)

    i = 0
    matches = None
    while not matches:
        literals = UtilsTM.Literals.make_random_board(boardsize)
        matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
        i += 1

    # For now, just plot the first match and ignore the rest
    match = matches[0]
    UtilsPlot.plot_search_pattern_match(match)
    print(search_pattern, match)

def calculate_background_template_occurrences(ds_states: UtilsDataset.Dataset):
    # TODO: find template co-dependence and use that to remove over counts (e.g. wheel is 3 bridges, but shoudln't count those 3 bridges)
    # TODO: keep track of where the matches happen
    # TODO: plot to some representation, thinking heatmap?

    boardsize = ds_states.boardsize
    start_board = 0
    num_boards = 1_000_000

    possible_num_pieces = list(ds_states.state_num_pieces_counts.keys())
    num_pieces_distribution = [count / ds_states.num_rows for count in ds_states.state_num_pieces_counts.values()]
    assert sum(num_pieces_distribution) == 1

    file_dir: Path = UtilsPlot.PLOT_TEMPLATES_DIR
    filepath: Path = file_dir / f"{ds_states.name}_dist_random_template_matches.csv"
    csv_headers = ['Board#', 'MatchBaseName', 'MatchVarName', 'MatchX', 'MatchY']

    # If this match dataset already exist, we need to correctly add to it
    if os.path.exists(filepath):
        with open(filepath, mode='r', newline='') as ds_match:
            csv_reader = csv.reader(ds_match)

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
                        csv_writer.writerow([board, base_name, var_name, match.x, match.y])
        csv_writer.writerow(['# Finished'])


dataset = UtilsDataset.BASELINE
calculate_background_template_occurrences(dataset)
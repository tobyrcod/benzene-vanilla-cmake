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

def calculate_background_template_occurrences(dataset: UtilsDataset.Dataset):
    # TODO: find template co-dependence and use that to remove over counts (e.g. wheel is 3 bridges, but shoudln't count those 3 bridges)
    # TODO: keep track of where the matches happen
    # TODO: plot to some representation, thinking heatmap?

    # Store the number of pieces in a state everytime the template is found
    template_names = UtilsHex.SearchPattern.get_pattern_names()
    template_found_num_pieces = {template_name: defaultdict(int) for template_name in template_names}

    boardsize = 6
    num_boards = 1_000

    possible_num_pieces = list(dataset.state_num_pieces_counts.keys())
    num_pieces_distribution = [count / dataset.num_rows for count in dataset.state_num_pieces_counts.values()]
    assert sum(num_pieces_distribution) == 1

    for i in range(num_boards):
        print(f"Board: {i}, Progress: {100 * i / num_boards:.3f}%")

        num_pieces = np.random.choice(possible_num_pieces, p=num_pieces_distribution)
        literals = UtilsTM.Literals.make_random_board(boardsize, num_pieces)
        for template_name in template_names:
            variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
            for search_pattern in variations:
                matches = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
                for match in matches:
                    template_found_num_pieces[match.search_pattern.base_name][num_pieces] += 1

    print(template_found_num_pieces)

dataset = UtilsDataset.BASELINE
calculate_background_template_occurrences(UtilsDataset.BASELINE)
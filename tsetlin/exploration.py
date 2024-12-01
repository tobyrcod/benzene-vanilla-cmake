import random
from pathlib import Path
from tsetlin.utils import UtilsPlot, UtilsHex, UtilsTM

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

find_random_match_in_random_board()
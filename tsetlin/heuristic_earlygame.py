from itertools import combinations
from sklearn.metrics import classification_report
from utils import *

"""
Play during the early stages of the game (and also play into unclaimed board territory) is
characterized by moves that are well-spaced from a player's own pieces, minimizing the
largest gap between them, and also well-spaced from opponent's pieces to allow room to
block if necessary.
"""

UtilsHex.SearchPattern.initialise()
UtilsDataset.load_raw_datasets(boardsize=6, blunder=0)

ds = UtilsDataset.COMBINED
assert ds.complete

# Roughly define the first 20% of a given game to be the early game for which we can apply the strategy
EARLY_GAME_FRAC = 0.33

X_game = ds.X_game
Y_game = ds.Y_game
assert len(X_game) == len(Y_game)

X = []
Y = []
for X_game_states, y in zip(X_game, Y_game):
    num_states = len(X_game_states)
    early_game_start = 4  # We need at least 4 pieces (2 each) for anything to make sense
    early_game_end = max(early_game_start, int(num_states * EARLY_GAME_FRAC))
    X_early_game_states = X_game_states[early_game_start:early_game_end]
    for X_early_game_state in X_early_game_states:
        X.append(X_early_game_state)
        Y.append(y)
assert len(X) == len(Y)
print(len(X))

def average_distance(coords):
    total_distance = 0
    count = 0

    for (x1, y1), (x2, y2) in combinations(coords, 2):
        dist = math.hypot(x2 - x1, y2 - y1)
        total_distance += dist
        count += 1

    return total_distance / count

def maximum_distance(coords):
    max_distance = 0

    for (x1, y1), (x2, y2) in combinations(coords, 2):
        dist = math.hypot(x2 - x1, y2 - y1)
        max_distance = max(dist, max_distance)

    return max_distance

y_preds = []
literals_per_player = ds.boardsize ** 2
for state in X:
    literals_black = state[:literals_per_player]
    literals_white = state[literals_per_player:]

    indices_black = list(np.where(literals_black == 1)[0])
    indices_white = list(np.where(literals_white == 1)[0])

    coords_black = [UtilsHex.Coordinates.index_to_coord(i, ds.boardsize) for i in indices_black]
    coords_white = [UtilsHex.Coordinates.index_to_coord(i, ds.boardsize) for i in indices_white]

    # print(literals_black)
    # print(indices_black)
    # print(coords_black)

    avg_black = average_distance(coords_black)
    avg_white = average_distance(coords_white)

    max_black = maximum_distance(coords_black)
    max_white = maximum_distance(coords_white)

    # y_pred = 0 if avg_black >= avg_white else 1
    # y_pred = 0 if max_black <= max_white else 1
    y_pred = 0 if max_black - avg_black <= max_white - avg_white else 0

    y_preds.append(y_pred)

print(classification_report(Y, y_preds))


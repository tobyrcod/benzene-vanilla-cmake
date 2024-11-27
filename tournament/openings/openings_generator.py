import itertools
import random

BOARD_SIZE = 6
PLY = 4
RANDOM_ORDER = True

FILE_NAME = f"{BOARD_SIZE}x{BOARD_SIZE}-{'all-random' if RANDOM_ORDER else 'all'}-{PLY}ply"
NUMBERS = [i+1 for i in range(BOARD_SIZE)]
LETTERS = [chr(ord('a')+i-1) for i in NUMBERS]
TILES = [f"{l}{n}" for l in LETTERS for n in NUMBERS]

OPENINGS = list(itertools.combinations(TILES, PLY))
random.shuffle(OPENINGS)
with open(FILE_NAME, "w") as file:
    for x in OPENINGS:
        file.write(" ".join(x) + "\n")

print(f"{FILE_NAME} has been created.")

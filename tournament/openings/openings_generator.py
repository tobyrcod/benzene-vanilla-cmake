import itertools

BOARD_SIZE = 6
PLY = 3

FILE_NAME = f"{BOARD_SIZE}x{BOARD_SIZE}-all-{PLY}ply"
NUMBERS = [i+1 for i in range(BOARD_SIZE)]
LETTERS = [chr(ord('a')+i-1) for i in NUMBERS]
TILES = [f"{l}{n}" for l in LETTERS for n in NUMBERS]

with open(FILE_NAME, "w") as file:
    for x in itertools.combinations(TILES, PLY):
        file.write(" ".join(x) + "\n")

print(f"{FILE_NAME} has been created.")

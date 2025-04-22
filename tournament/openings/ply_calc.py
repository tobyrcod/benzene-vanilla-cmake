import math

BOARD_SIZE = 9

xs = []
for ply in range(1, 5):
    x = math.comb(BOARD_SIZE**2, ply)
    xs.append(str(x))
print(BOARD_SIZE**2)
print(", ".join(xs))

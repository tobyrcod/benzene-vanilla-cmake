import sys
import csv
from enum import IntFlag
from pathlib import Path
from pysgf import SGF

# TODO: document every method with """ """

class UtilsBenzene:

    @staticmethod
    def load_tournament_games(tournament_path: Path) -> tuple[list[tuple[int, list]], int]:
        result_keys = ['GAME', 'ROUND', 'OPENING', 'BLACK', 'WHITE', 'RES_B', 'RES_W', 'LENGTH', 'TIME_B', 'TIME_W', 'ERR', 'ERR_MSG']

        try:
            games = []
            game_boardsize = None

            results_path = tournament_path / "results"
            with results_path.open('r') as results_file:
                for line_r in results_file:
                    if not line_r:
                        continue

                    # Get the boardsize used for each sgf file
                    if line_r[0] == '#':
                        if not game_boardsize and 'Boardsize' in line_r:
                            game_boardsize = int(line_r.split(':')[1].strip())
                        continue

                    # Parse the result line into values
                    result = line_r.strip().split(sep='\t', maxsplit=11)
                    result = dict(zip(result_keys, result))

                    # Perform metadata results checks
                    if result['ERR'] == '1':                                                            # The game isn't complete
                        continue
                    if result['RES_B'] != result['RES_W']:                                              # Players need to agree on a winner
                        continue
                    if result['RES_B'] not in ("B+", "W+"):                                             # The winner must be a player
                        continue
                    game_winner = result['RES_B']

                    # We need to open the sgf file for the game history matching this result
                    sgf_path = tournament_path / f"{result["GAME"]}.sgf"
                    sgf = SGF.parse_file(sgf_path)

                    # Perform metadata game checks
                    if int(sgf.get_property('GM')) != 11:                                               # GM[11] means the game of Hex
                        continue
                    if sgf.get_property("US") != "twogtp.py":                                           # Need sgf files to be in the right format
                        continue
                    if sgf.get_property('RE') != game_winner:                                           # We need the right winner for this game
                        continue
                    if any(n != game_boardsize for n in sgf.board_size):                                # Matching boardsize
                        continue

                    # Convert the sgf move tree into a list
                    game_moves = []
                    move_node = sgf.children[0]
                    while children := move_node.children:
                        player = move_node.player
                        game_moves.append(move_node.get_property(player))
                        move_node = children[0]
                    game_moves = game_moves[:-1]

                    # Save the game
                    game_winner = 1 if game_winner == "B+" else 0                                       # 1 if black wins, 0 if white wins (black loses)
                    games.append((game_winner, game_moves))

            return games, game_boardsize

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)


    @staticmethod
    def games_to_winner_prediction_dataset(games:  list[tuple[int, list]], boardsize: int, dataset_path: Path):
        try:
            csv_headers = ["Winner"] + [f"x{i}" for i in range(2*boardsize**2)]
            with open(dataset_path, mode='w', newline='') as dataset:
                csv_writer = csv.writer(dataset)
                csv_writer.writerow(['boardsize', boardsize])
                csv_writer.writerow(csv_headers)
                for game in games:
                    winner, moves = game
                    states = UtilsBenzene._get_literal_states_from_moves(moves, boardsize)
                    for state in states:
                        csv_writer.writerow([winner] + state)
                    break
        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)


    @staticmethod
    def _hex_string_to_number(hex_string: str) -> int:
        # Treat the hex_string as a base 26 number, increasing powers left-to-right
        number = 0
        for c in hex_string:
            value = ord(c) - ord('a') + 1
            number *= 26
            number += value
        return number


    @staticmethod
    def _hex_position_to_index(position: str, boardsize: int) -> int:
        # move will be e.g. 'a1'
        # First job is to split this into 'a' and '1'
        split_index = -1
        for i, c in enumerate(position):
            if c.isalpha():
                continue
            if i != 0:
                split_index = i
            break
        if split_index == -1:
            return -1
        string = position[:split_index]
        number = position[split_index:]

        # Second, convert from letter, number to x, y coordinates
        x = UtilsBenzene._hex_string_to_number(string) - 1
        y = int(number) - 1
        if x < 0 or x >= boardsize or y < 0 or y >= boardsize:  # Ensure the coordinate is on the board
            return -1

        # Finally, convert from 2D coordinates to 1D index
        return y * boardsize + x


    @staticmethod
    def _get_literal_states_from_moves(moves: list[str], boardsize: int):

        # TODO: Check literal representation process from Giri et al.
        # Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
        # https://ieeexplore.ieee.org/document/9892796

        states = []
        literals_per_player = boardsize**2
        literals = UtilsTM.make_new_literals(boardsize)

        for i, move in enumerate(moves):
            is_white_move = i % 2
            index = UtilsBenzene._hex_position_to_index(move, boardsize)
            index += is_white_move * literals_per_player
            literals[index] = 1
            states.append(literals.copy())

        return states


class UtilsTM:

    class LiteralAugmentation(IntFlag):
        NONE = 0
        AUG_MOVE_COUNTER    = 1 << 0
        AUG_TURN_INDICATOR  = 1 << 1
        AUG_PADDING         = 1 << 2
        AUG_PAIR_POSITIONS  = 1 << 3
        AUG_STATE_HISTORY   = 1 << 4
        AUG_MOVE_HISTORY    = 1 << 5
        # AUG_ALL             = ~0

        @staticmethod
        def is_valid(augmentation: "UtilsTM.LiteralAugmentation") -> bool:
            if augmentation <= 0:
                return False
            if (augmentation & UtilsTM.LiteralAugmentation.AUG_STATE_HISTORY) and \
               (augmentation & UtilsTM.LiteralAugmentation.AUG_MOVE_HISTORY):
                return False  # Only one type of history can be used
            return True

        @staticmethod
        def augment_literals(literals, augmentation: "UtilsTM.LiteralAugmentation", boardsize: int):
            # Check to see this augmentation makes sense
            if not UtilsTM.LiteralAugmentation.is_valid(augmentation):
                return

            if augmentation & UtilsTM.LiteralAugmentation.AUG_PAIR_POSITIONS:
                # Instead of listing all black and then all white literals,
                # We list positions in order, alternating black then white
                # for 6x6, x1, x2,..., x37, x38,... becomes x1, x37, x2, x38,...
                literals_per_player = boardsize**2
                old_literals = literals.copy()
                for new_index in range(len(literals)):
                    colour_index = new_index // 2
                    is_white = new_index % 2
                    old_index = colour_index + is_white * literals_per_player
                    literals[new_index] = old_literals[old_index]


            if augmentation & UtilsTM.LiteralAugmentation.AUG_MOVE_COUNTER:
                # Add a binary counter for how many moves into the game we are
                # Idea: although this is implicit already, does making it explicit help?

                max_move_count = (boardsize ** 2)
                max_count_length = max_move_count.bit_length()
                move_count = sum(literals) + 1
                move_count_binary = f'{move_count:b}'.zfill(max_count_length)
                move_count_literals = list(map(int, move_count_binary))
                literals.extend(move_count_literals)

            if augmentation & UtilsTM.LiteralAugmentation.AUG_TURN_INDICATOR:
                # Add a binary bit to represent if it is black (1) or white (0) to play
                # Idea: although this is implicit already, does making it explicit help?

                move_count = sum(literals) + 1
                is_blacks_turn = move_count % 2 != 0
                literals.append(int(is_blacks_turn))

    @staticmethod
    def make_new_literals(boardsize: int) -> list[int]:
        literals_per_player = boardsize ** 2
        return [0 for _ in range(2 * literals_per_player)]

    @staticmethod
    def train_tm_from_dataset(dataset_path: Path, batch_size=10, augmentation: LiteralAugmentation=LiteralAugmentation.NONE):
        try:

            tm = None

            with open(dataset_path, mode='r', newline='') as dataset:
                reader = csv.reader(dataset)
                boardsize = int(next(reader)[1])
                headers = next(reader)
                batch = []
                for row in reader:
                    winner = int(row[0])
                    literals = [int(l) for l in row[1:]]
                    assert len(literals) == 2*boardsize**2

                    # We may want to modify the board representation to see if it helps/hinders training
                    UtilsTM.LiteralAugmentation.augment_literals(literals, augmentation, boardsize)

                    batch.append((winner, literals))
                    if len(batch) >= batch_size:
                        UtilsTM._train_tm_from_batch(tm, batch, boardsize)
                        batch.clear()
                if batch:
                    UtilsTM._train_tm_from_batch(tm, batch, boardsize)
                    batch.clear()

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)
        except Exception as e:
            print(e, file=sys.stderr)


    @staticmethod
    def _train_tm_from_batch(tm, batch, boardsize: int):
        # Placeholder function, replace with real TM
        print(boardsize, len(batch))

if __name__ == "__main__":
    test_path = Path("test_tournament")
    tournament_path = test_path / "6x6-mohex-mohex-simple-a-vs-mohex-mohex-simple-b"
    dataset_path = test_path / "dataset.csv"

    # Loader the tournament results file
    games, boardsize = UtilsBenzene.load_tournament_games(tournament_path)

    # Convert it to a winner prediction dataset
    UtilsBenzene.games_to_winner_prediction_dataset(games, boardsize, dataset_path)

    # Train a TM for hex winner prediction from the dataset
    UtilsTM.train_tm_from_dataset(dataset_path, batch_size=5, augmentation=UtilsTM.LiteralAugmentation.AUG_PAIR_POSITIONS)

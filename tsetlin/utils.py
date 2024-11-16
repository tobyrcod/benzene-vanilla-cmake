import itertools
import random
import sys
import csv
from enum import IntFlag, Enum
from pathlib import Path
from pysgf import SGF

# TODO: document every method with """ """
# TODO: add empty board with the winner when making the dataset

class Helpers:

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    @staticmethod
    def chunk_list(l, chunk_size):
        return [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]


class UtilsHex:

    @staticmethod
    def coordinates_2d_to_1d(x: int, y: int, boardsize: int) -> int:
        return y * boardsize + x


    @staticmethod
    def coordinates_1d_to_2d(i: int, boardsize: int) -> tuple[int, int]:
        return i % boardsize, i // boardsize


    @staticmethod
    def string_to_number(hex_string: str) -> int:
        # Treat the hex_string as a base 26 number, increasing powers left-to-right
        number = 0
        for c in hex_string:
            value = ord(c) - ord('a') + 1
            number *= 26
            number += value
        return number


    @staticmethod
    def position_to_index(position: str, boardsize: int) -> int:
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
        x = UtilsHex.string_to_number(string) - 1
        y = int(number) - 1
        if x < 0 or x >= boardsize or y < 0 or y >= boardsize:  # Ensure the coordinate is on the board
            return -1

        # Finally, convert from 2D coordinates to 1D index
        return UtilsHex.coordinates_2d_to_1d(x, y, boardsize)


class UtilsTournament:

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
            csv_headers = ["Game#", "Winner"] + [f"x{i}" for i in range(2*boardsize**2)]
            with open(dataset_path, mode='w', newline='') as dataset:
                csv_writer = csv.writer(dataset)
                csv_writer.writerow(['boardsize', boardsize])
                csv_writer.writerow(csv_headers)
                for i, game in enumerate(games):
                    winner, moves = game
                    states = UtilsTournament._get_literal_states_from_moves(moves, boardsize)
                    for state in states:
                        csv_writer.writerow([i, winner] + state)

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)


    @staticmethod
    def _get_literal_states_from_moves(moves: list[str], boardsize: int):

        # TODO: Check literal representation process from Giri et al.
        # Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
        # https://ieeexplore.ieee.org/document/9892796

        literals_per_player = boardsize**2
        literals = UtilsTM.Literals.make_empty_board(boardsize)
        states = [literals.copy()]  # Start with an empty board state

        for i, move in enumerate(moves):
            is_white_move = i % 2 == 1
            index = UtilsHex.position_to_index(move, boardsize)
            index += is_white_move * literals_per_player
            literals[index] = 1
            states.append(literals.copy())  # And gradually add every played state one move at a time

        return states


class UtilsTM:

    class Literals:

        class History(Enum):
            NONE = 0
            # Give Literals 'Memory'
            HISTORY_STATE       = 1
            HISTORY_MOVE        = 2

            def apply(self, literals: list[int], history: list[list[int]], boardsize: int) -> list[int]:
                if self == UtilsTM.Literals.History.HISTORY_STATE:
                    # Convert the list of game states into a big long list of all the literals
                    # e.g [[1, 2], [3, 4], [5, 6]] -> [1, 2, 3, 4, 5, 6]
                    flattened_history = list(itertools.chain.from_iterable(history))
                    return literals + flattened_history

                if self == UtilsTM.Literals.History.HISTORY_MOVE:
                    # Convert the list of game states into just the moves made at each step
                    moves = []
                    for i in range(len(history)-1):
                        state = history[i]
                        previous_state = history[i+1]
                        move = UtilsTM.Literals.get_literal_difference(state, previous_state)
                        moves.append(move)
                    flattened_moves = list(itertools.chain.from_iterable(moves))
                    return literals + flattened_moves

                return literals.copy()

        class Augmentation(IntFlag):
            AUG_NONE = 0
            # Append Information
            AUG_MOVE_COUNTER    = 1 << 0
            AUG_TURN_INDICATOR  = 1 << 1
            # Make Literals more Spatially Aware
            AUG_PADDING         = 1 << 2
            AUG_PAIR_POSITIONS  = 1 << 3
            # AUG_ALL           = ~0

            def _is_valid(self):
                if self <= 0:
                    return False
                # Add any explicit cases that aren't allowed here
                return True

            def apply(self, literals: list[int], boardsize: int) -> list[int]:
                new_boardsize = boardsize
                new_literals = literals.copy()

                # Check to see this augmentation makes sense
                if not self._is_valid():
                    return new_literals

                if self & UtilsTM.Literals.Augmentation.AUG_PADDING:
                    # Add a border of filled in tiles around the outside of the board,
                    # Matching the colours the players need to connect to win.
                    # Idea: May help to reinforce how to win / making connections between pieces being good

                    # nxn board must become (n+2)x(n+2) board
                    new_boardsize = boardsize + 2
                    new_literals = UtilsTM.Literals.make_empty_board(new_boardsize)
                    literals_per_player = boardsize ** 2
                    new_literals_per_player = new_boardsize ** 2

                    # Move the literals on the nxn board to the centre of the new (n+2)x(n+2) board
                    for i in range(literals_per_player):
                        # First, convert from nxn 1d coordinates to nxn 2d coordinates
                        x1, y1 = UtilsHex.coordinates_1d_to_2d(i, boardsize)

                        # Second, add 1 to the coordinate to get the new coordinates in the (n+2)x(n_2) grid
                        x2, y2 = x1 + 1, y1 + 1

                        # Next, convert from (n+2)x(n+2) 2d coordinates to (n+2)x(n_2) 1d coordinates
                        j = UtilsHex.coordinates_2d_to_1d(x2, y2, boardsize + 2)

                        # Finally, move the literal to its new position
                        new_literals[j] = literals[i]  # black
                        new_literals[new_literals_per_player + j] = literals[literals_per_player + i]  # white

                    # Add the border cells to the new larger board
                    # NOTE: we leave the corner positions empty
                    # Black wins top to bottom, so the first and last row should be all black
                    for y in [0, new_boardsize - 1]:
                        for x in range(1, new_boardsize - 1):
                            i = UtilsHex.coordinates_2d_to_1d(x, y, new_boardsize)
                            new_literals[i] = 1
                    # White wins left to right, so each row should begin and end with white
                    for y in range(1, new_boardsize - 1):
                        for x in [0, new_boardsize - 1]:
                            i = UtilsHex.coordinates_2d_to_1d(x, y, new_boardsize)
                            new_literals[new_literals_per_player + i] = 1

                if self & UtilsTM.Literals.Augmentation.AUG_PAIR_POSITIONS:
                    # Instead of listing all black and then all white literals,
                    # We list positions in order, alternating black then white
                    # for 6x6, x1, x2,..., x37, x38,... becomes x1, x37, x2, x38,...
                    # Idea: May help positional reasoning, especially for CTM

                    literals_per_player = new_boardsize ** 2
                    copy_new_literals = new_literals.copy()
                    for new_index in range(len(new_literals)):
                        colour_index = new_index // 2
                        is_white = new_index % 2
                        old_index = colour_index + is_white * literals_per_player
                        new_literals[new_index] = copy_new_literals[old_index]

                if self & UtilsTM.Literals.Augmentation.AUG_MOVE_COUNTER:
                    # Add a binary counter for how many moves into the game we are
                    # Idea: although this is implicit already, does making it explicit help?

                    max_move_count = boardsize ** 2
                    max_count_length = max_move_count.bit_length()
                    move_count = sum(literals) + 1
                    move_count_binary = f'{move_count:b}'.zfill(max_count_length)
                    move_count_literals = list(map(int, move_count_binary))
                    new_literals.extend(move_count_literals)

                if self & UtilsTM.Literals.Augmentation.AUG_TURN_INDICATOR:
                    # Add a binary bit to represent if it is black (1) or white (0) to play
                    # Idea: although this is implicit already, does making it explicit help?

                    move_count = sum(literals) + 1
                    is_blacks_turn = move_count % 2 != 0
                    new_literals.append(int(is_blacks_turn))

                return new_literals

        @staticmethod
        def get_literal_difference(literalsA: list[int], literalsB: list[int]) -> list[int]:
            return [a ^ b for (a, b) in zip(literalsA, literalsB)]


        @staticmethod
        def make_empty_board(boardsize: int) -> list[int]:
            literals_per_player = boardsize ** 2
            return [0 for _ in range(2 * literals_per_player)]


        @staticmethod
        def make_random_board(boardsize: int, num_pieces: int = None) -> list[int]:
            max_num_pieces = boardsize ** 2
            if num_pieces is None:
                num_pieces = random.randint(0, max_num_pieces)
            num_pieces = Helpers.clamp(num_pieces, 0, max_num_pieces)

            # For even #pieces, half should be black and half should be white
            # For odd #pieces, black should have one more than white
            # For all #pieces, black and white can never have a piece in the same place
            num_black_pieces = (num_pieces + 1) // 2
            piece_indices = random.sample(range(max_num_pieces), num_pieces)
            black_literals = [1 if i in piece_indices[:num_black_pieces] else 0 for i in range(max_num_pieces)]
            white_literals = [1 if i in piece_indices[num_black_pieces:] else 0 for i in range(max_num_pieces)]
            return black_literals + white_literals


        @staticmethod
        def make_random_move(literals: list[int], boardsize: int) -> tuple[list[int] | None, int]:
            move = UtilsTM.Literals._get_random_move(literals, boardsize)
            return UtilsTM.Literals._make_move(literals, move), move


        @staticmethod
        def _get_random_move(literals: list[int], boardsize: int) -> int:
            # Takes a literal board representation and adds a random new valid move
            move_count = sum(literals) + 1
            if move_count > boardsize ** 2:
                # print('this game is already over')
                return -1

            is_black_turn = move_count % 2 != 0

            literals_per_player = boardsize ** 2
            player_range = range(literals_per_player) if is_black_turn else range(literals_per_player,
                                                                                  literals_per_player * 2)
            # print(f"move: {move_count}", f"black?: {is_black_turn}", f"player range: {player_range}")

            possible_move_indices = [i for i, literal in enumerate(literals) if literal == 0 and i in player_range]
            move = random.choice(possible_move_indices)
            return move


        @staticmethod
        def _make_move(literals: list[int], move: int) -> list[int] | None:
            if move < 0 or move >= len(literals):
                return None  # Move isn't on board
            if literals[move] == 1:
                return None  # Move has already been made, so something is wrong

            new_literals = literals.copy()
            new_literals[move] = 1
            return new_literals


    @staticmethod
    def train_tm_from_dataset(dataset_path: Path, batch_size=10,
                              augmentation: Literals.Augmentation = Literals.Augmentation.AUG_NONE,
                              history_type: Literals.History = Literals.History.NONE,
                              history_size: int = 0):
        try:

            tm = None

            with open(dataset_path, mode='r', newline='') as dataset:
                reader = csv.reader(dataset)

                boardsize = int(next(reader)[1])
                num_literals = len(UtilsTM.Literals.make_empty_board(boardsize))

                headers = next(reader)

                game_history = dict()
                if history_type == UtilsTM.Literals.History.NONE:
                    history_size = 0  # We don't want any history
                if history_type == UtilsTM.Literals.History.HISTORY_MOVE:
                    history_size += 1  # In order to calculate n moves, we need (n+1) games of history

                batch = []
                for row in reader:
                    game_number = int(row[headers.index('Game#')])
                    if game_number not in game_history:
                        # If we are early in the game, pad the history with empty boards
                        game_history = {game_number: [UtilsTM.Literals.make_empty_board(boardsize) for _ in range(history_size)]}

                    winner = int(row[headers.index('Winner')])
                    literals = [int(l) for l in row[headers.index('x0'):]]
                    assert len(literals) == num_literals

                    # We may want to modify the board representation to see if it helps/hinders training
                    aug_literals = augmentation.apply(literals, boardsize)

                    # We may also want to modify the board representation by adding the game history
                    history = game_history[game_number]
                    final_literals = history_type.apply(aug_literals, history, boardsize)

                    # Add this games literals to the game history
                    history.insert(0, aug_literals)
                    while len(history) > history_size:
                        history.pop()

                    # Process a batch of literals once the batch is large enough
                    batch.append((winner, final_literals))
                    if len(batch) >= batch_size:
                        UtilsTM._train_tm_from_batch(tm, batch, boardsize)
                        batch.clear()

                # Process any remaining training data that doesn't fill a batch
                if batch:
                    UtilsTM._train_tm_from_batch(tm, batch, boardsize)
                    batch.clear()

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)

    @staticmethod
    def _train_tm_from_batch(tm, batch, boardsize: int):
        # Placeholder function, replace with real TM
        # print(boardsize, len(batch))
        pass


if __name__ == "__main__":
    test_path = Path("tournaments")
    tournament_path = test_path / "6x6-1ply-simple"
    dataset_path = tournament_path / "dataset.csv"

    # Loader the tournament results file
    games, boardsize = UtilsTournament.load_tournament_games(tournament_path)

    # Convert it to a winner prediction dataset
    UtilsTournament.games_to_winner_prediction_dataset(games, boardsize, dataset_path)

    # Train a TM for hex winner prediction from the dataset
    UtilsTM.train_tm_from_dataset(dataset_path,
                                  batch_size=5,
                                  augmentation=UtilsTM.Literals.Augmentation.AUG_NONE,
                                  history_type=UtilsTM.Literals.History.HISTORY_MOVE,
                                  history_size=2)

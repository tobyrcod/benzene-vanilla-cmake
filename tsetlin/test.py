import random
import time

from utils import Helpers, UtilsHex, UtilsTM

class Tests:

    # TODO: add failed case test for all

    class UtilsHex:

        @staticmethod
        def test_hex_coordinates_1d_to_2d():
            assert UtilsHex.coordinates_1d_to_2d(0, 6) == (0, 0)
            assert UtilsHex.coordinates_1d_to_2d(1, 6) == (1, 0)
            assert UtilsHex.coordinates_1d_to_2d(6, 6) == (0, 1)
            assert UtilsHex.coordinates_1d_to_2d(7, 6) == (1, 1)
            assert UtilsHex.coordinates_1d_to_2d(35, 6) == (5, 5)


        @staticmethod
        def test_hex_coordinates_2d_to_1d():
            assert UtilsHex.coordinates_2d_to_1d(0, 0, 6) == 0
            assert UtilsHex.coordinates_2d_to_1d(1, 0, 6) == 1
            assert UtilsHex.coordinates_2d_to_1d(0, 1, 6) == 6
            assert UtilsHex.coordinates_2d_to_1d(1, 1, 6) == 7
            assert UtilsHex.coordinates_2d_to_1d(5, 5, 6) == 35


        @staticmethod
        def test_hex_string_to_number():
            assert UtilsHex.string_to_number('a') == 1
            assert UtilsHex.string_to_number('b') == 2
            assert UtilsHex.string_to_number('z') == 26
            assert UtilsHex.string_to_number('aa') == 27
            assert UtilsHex.string_to_number('ab') == 28
            assert UtilsHex.string_to_number('az') == 52
            assert UtilsHex.string_to_number('aaa') == 703


        @staticmethod
        def test_hex_coord_to_index():
            assert UtilsHex.position_to_index('a1', 6) == 0
            assert UtilsHex.position_to_index('f1', 6) == 5
            assert UtilsHex.position_to_index('d2', 6) == 9
            assert UtilsHex.position_to_index('f6', 6) == 35
            assert UtilsHex.position_to_index('g1', 6) == -1

    class UtilsTM:

        class Literals:

            class History:

                @staticmethod
                def test():

                    def test_state_history():
                        history_type = UtilsTM.Literals.History.HISTORY_STATE
                        augmentation = UtilsTM.Literals.Augmentation.AUG_NONE

                        # To check history property in simple setting
                        # Do we order the history correctly, do we use queue correctly
                        def test_dummy(history_size: int, boardsize: int):
                            history = [[0] for _ in range(history_size)]

                            for i in range(1, history_size*3):
                                literals = [i]
                                wanted_literals = [max(0, i-j) for j in range(history_size+1)]

                                final_literals = history_type.apply(literals, history, boardsize)
                                assert final_literals == wanted_literals

                                history.insert(0, literals)
                                while len(history) > history_size:
                                    history.pop()

                        # To check size is right with real literals
                        def test_real(history_size: int, boardsize: int):
                            literals_per_player = boardsize ** 2
                            literals_per_game = literals_per_player * 2
                            history = [UtilsTM.Literals.make_empty_board(boardsize) for _ in range(history_size)]

                            literals = UtilsTM.Literals.make_empty_board(boardsize)
                            for move in range(1, boardsize**2):

                                aug_literals = augmentation.apply(literals, boardsize)

                                #
                                # BEGIN TESTS

                                final_literals = history_type.apply(aug_literals, history, boardsize)
                                final_games_literals = Helpers.chunk_list(final_literals, literals_per_game)

                                assert len(final_games_literals) == history_size + 1
                                assert all(len(game_literals) == literals_per_game for game_literals in final_games_literals)
                                assert final_games_literals[0] == aug_literals
                                for i, h in enumerate(history):
                                    assert final_games_literals[i+1] == h

                                #
                                # END TESTS

                                # Add these literals to the history for next iteration
                                history.insert(0, aug_literals.copy())
                                while len(history) > history_size:
                                    history.pop()

                                # Make a random move to simulate play
                                literals = UtilsTM.Literals.make_random_move(literals, boardsize)


                        for history_size in range(2, 6):
                            for boardsize in range(6, 14):
                                for _ in range(10*boardsize):
                                    test_dummy(history_size, boardsize)
                                    test_real(history_size, boardsize)

                    test_state_history()


            class Augmentation:

                @staticmethod
                def test():

                    # HELPERS

                    def print_literal_grid(literals, boardsize):
                        for row in Helpers.chunk_list(literals, boardsize):
                            print([str(n).zfill(2) for n in row])
                        print('------')

                    def pad_player_literals(player_literals, is_black, boardsize):
                        player_rows = Helpers.chunk_list(player_literals, boardsize)
                        after = []
                        after_boardsize = boardsize + 2
                        for row in range(after_boardsize):
                            if row == 0 or row == after_boardsize - 1:
                                top_bottom = int(is_black)
                                after.extend([top_bottom for _ in range(after_boardsize)])
                            else:
                                side = int(not is_black)
                                after.extend([side] + player_rows[row-1] + [side])
                        after[UtilsHex.coordinates_2d_to_1d(0, 0, after_boardsize)] = 0
                        after[UtilsHex.coordinates_2d_to_1d(0, after_boardsize-1, after_boardsize)] = 0
                        after[UtilsHex.coordinates_2d_to_1d(after_boardsize-1, 0, after_boardsize)] = 0
                        after[UtilsHex.coordinates_2d_to_1d(after_boardsize-1, after_boardsize-1, after_boardsize)] = 0
                        return after

                    # TESTS

                    def test_padding():
                        for boardsize in range(6, 14):
                            literals_per_player = boardsize ** 2

                            before_black = [i                           for i in range(literals_per_player)]
                            before_white = [literals_per_player + i     for i in range(literals_per_player)]

                            after_black = pad_player_literals(before_black, True, boardsize)
                            after_white = pad_player_literals(before_white, False, boardsize)

                            before = before_black + before_white
                            assert len(before) == 2*literals_per_player
                            after = after_black + after_white
                            after_boardsize = boardsize + 2
                            after_literals_per_player = after_boardsize**2
                            assert len(after) == 2*after_literals_per_player

                            # print(before)
                            # print(after)
                            # print_literal_grid(before_black, boardsize)
                            # print_literal_grid(after_black, after_boardsize)
                            # print_literal_grid(before_white, boardsize)
                            # print_literal_grid(after_white, after_boardsize)

                            literals = before
                            assert literals == before
                            literals = UtilsTM.Literals.Augmentation.AUG_PADDING.apply(literals, boardsize)
                            assert literals == after

                    def test_pair_position():
                        for boardsize in range(6, 14):
                            literals_per_player = boardsize**2

                            black = [i                          for i in range(literals_per_player)]
                            white = [literals_per_player + i    for i in range(literals_per_player)]
                            before = black + white
                            after = [x for pair in zip(black, white) for x in pair]

                            literals = before
                            assert literals == before
                            literals = UtilsTM.Literals.Augmentation.AUG_PAIR_POSITIONS.apply(literals, boardsize)
                            assert literals == after

                    def test_move_counter():
                        for boardsize in range(6, 14):
                            for _ in range(10*boardsize**2):
                                max_move_count = boardsize ** 2
                                move_count = random.randint(1, max_move_count)
                                before = UtilsTM.Literals.make_random_board(boardsize, move_count - 1)

                                max_count_length = max_move_count.bit_length()
                                move_count_binary = f'{move_count:b}'.zfill(max_count_length)
                                move_count_literals = list(map(int, move_count_binary))
                                after = before + move_count_literals

                                literals = before
                                assert literals == before
                                literals = UtilsTM.Literals.Augmentation.AUG_MOVE_COUNTER.apply(literals, boardsize)
                                assert literals == after

                    def test_turn_indicator():
                        for boardsize in range(6, 14):
                            for _ in range(10*boardsize**2):
                                max_turn = boardsize ** 2
                                turn = random.randint(1, max_turn)

                                before = UtilsTM.Literals.make_random_board(boardsize, turn - 1)
                                after = before + [turn % 2]

                                literals = before
                                assert literals == before
                                literals = UtilsTM.Literals.Augmentation.AUG_TURN_INDICATOR.apply(literals, boardsize)
                                assert literals == after

                    def test_padding_position_pair():
                        for boardsize in range(6, 14):
                            for _ in range(10*boardsize**2):
                                max_piece_count = boardsize ** 2
                                piece_count = random.randint(1, max_piece_count)

                                before = UtilsTM.Literals.make_random_board(boardsize, piece_count)

                                mid_step_1 = UtilsTM.Literals.Augmentation.AUG_PADDING.apply(before, boardsize)
                                final_step = UtilsTM.Literals.Augmentation.AUG_PAIR_POSITIONS.apply(mid_step_1, boardsize+2)

                                aug = UtilsTM.Literals.Augmentation.AUG_PADDING | UtilsTM.Literals.Augmentation.AUG_PAIR_POSITIONS
                                all_in_one = aug.apply(before, boardsize)

                                assert final_step == all_in_one

                    test_padding()
                    test_pair_position()
                    test_move_counter()
                    test_turn_indicator()

                    test_padding_position_pair()

            @staticmethod
            def test():

                def test_make_random_board():
                    for boardsize in range(6, 14):
                        literals_per_player = boardsize ** 2
                        max_num_pieces = boardsize ** 2
                        for _ in range(10 * boardsize ** 2):
                            num_pieces = random.randint(1, max_num_pieces)

                            literals = UtilsTM.Literals.make_random_board(boardsize, num_pieces)
                            assert len(literals) == len(UtilsTM.Literals.make_empty_board(boardsize))

                            black_literals = literals[:literals_per_player]
                            white_literals = literals[literals_per_player:]
                            assert len(black_literals) == len(white_literals) == literals_per_player

                            assert sum(black_literals) + sum(white_literals) == num_pieces
                            assert 0 <= sum(black_literals) - sum(white_literals) <= 1

                            for i in range(literals_per_player):
                                black = black_literals[i]
                                white = white_literals[i]
                                assert not (black and white)
                                assert black + white <= 1

                def test_make_random_move():
                    for boardsize in range(6, 14):
                        literals_per_player = boardsize ** 2
                        max_num_pieces = boardsize ** 2
                        for _ in range(10 * boardsize ** 2):
                            before = UtilsTM.Literals.make_random_board(boardsize)
                            after = UtilsTM.Literals.make_random_move(before, boardsize)
                            # print(before)
                            # print(after)
                            # print('----')
                            assert (after is None) == (sum(before) == max_num_pieces)
                            if after is None:
                                continue

                            assert len(after) == len(before)

                            assert sum(after) == sum(before) + 1

                            new_count = 0
                            for i in range(len(before)):
                                b = before[i]
                                a = after[i]
                                if b == 1: assert a == 1
                                if a == 0: assert b == 0
                                if a == 1 and b == 0: new_count += 1
                            assert new_count == 1

                            before_black = before[:literals_per_player]
                            before_white = before[literals_per_player:]
                            after_black = after[:literals_per_player]
                            after_white = after[literals_per_player:]
                            assert len(before_black) == len(before_white) == len(after_black) == len(after_white)

                            before_piece_count = sum(before)
                            was_blacks_turn = before_piece_count % 2 == 0
                            if was_blacks_turn:
                                assert after_white == before_white
                                assert sum(after_black) == sum(before_black) + 1
                            else:
                                # was_whites_turn
                                assert after_black == before_black
                                assert sum(after_white) == sum(before_white) + 1


                test_make_random_move()
                test_make_random_board()
                Tests.UtilsTM.Literals.History.test()
                Tests.UtilsTM.Literals.Augmentation.test()



if __name__ == "__main__":

    start_time = time.time()
    pretty_start_time = time.strftime("%H:%M:%S", time.localtime(start_time))
    print(f"STARTING TESTS: {pretty_start_time}")

    # START TESTS

    Tests.UtilsHex.test_hex_coordinates_1d_to_2d()
    Tests.UtilsHex.test_hex_coordinates_2d_to_1d()
    Tests.UtilsHex.test_hex_string_to_number()
    Tests.UtilsHex.test_hex_coord_to_index()

    Tests.UtilsTM.Literals.test()

    # END TESTS

    end_time = time.time()
    pretty_end_time = time.strftime("%H:%M:%S", time.localtime(end_time))
    print(f"TESTS COMPLETE: {pretty_end_time}")

    duration = end_time - start_time
    print(f"TESTS DURATION: {duration}")
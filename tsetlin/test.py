from utils import UtilsTournament, UtilsTM, UtilsHex

class Tests:

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
    def test_hex_string_to_index():
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


    @staticmethod
    def test_tm_pair_position_literal_augmentation():
        for boardsize in range(6, 14):
            literals_per_player = boardsize**2

            black = [i                          for i in range(literals_per_player)]
            white = [i + literals_per_player    for i in range(literals_per_player)]
            before = black + white
            after = [x for pair in zip(black, white) for x in pair]

            literals = before
            assert literals == before
            UtilsTM.LiteralAugmentation.augment_literals(literals, UtilsTM.LiteralAugmentation.AUG_PAIR_POSITIONS, boardsize)
            assert literals == after

if __name__ == "__main__":
    Tests.test_hex_coordinates_1d_to_2d()
    Tests.test_hex_coordinates_2d_to_1d()
    Tests.test_hex_string_to_index()
    Tests.test_hex_coord_to_index()

    Tests.test_tm_pair_position_literal_augmentation()
from utils import UtilsBenzene, UtilsTM

class Tests:

    @staticmethod
    def test_hex_string_to_index():
        assert UtilsBenzene._hex_string_to_number('a') == 1
        assert UtilsBenzene._hex_string_to_number('b') == 2
        assert UtilsBenzene._hex_string_to_number('z') == 26
        assert UtilsBenzene._hex_string_to_number('aa') == 27
        assert UtilsBenzene._hex_string_to_number('ab') == 28
        assert UtilsBenzene._hex_string_to_number('az') == 52
        assert UtilsBenzene._hex_string_to_number('aaa') == 703


    @staticmethod
    def test_hex_coord_to_index():
        assert UtilsBenzene._hex_position_to_index('a1', 6) == 0
        assert UtilsBenzene._hex_position_to_index('f1', 6) == 5
        assert UtilsBenzene._hex_position_to_index('d2', 6) == 9
        assert UtilsBenzene._hex_position_to_index('f6', 6) == 35
        assert UtilsBenzene._hex_position_to_index('g1', 6) == -1


    @staticmethod
    def test_pair_position_literal_augmentation():
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
    Tests.test_hex_string_to_index()
    Tests.test_hex_coord_to_index()
    Tests.test_pair_position_literal_augmentation()
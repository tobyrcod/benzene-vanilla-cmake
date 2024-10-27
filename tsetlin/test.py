from utils import UtilsBenzene

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


if __name__ == "__main__":
    Tests.test_hex_string_to_index()
    Tests.test_hex_coord_to_index()
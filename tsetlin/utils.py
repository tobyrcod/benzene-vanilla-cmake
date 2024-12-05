import itertools
import os
import random
import sys
import csv
from linecache import cache

import matplotlib.pyplot
import numpy as np
from enum import IntFlag, Enum
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import RegularPolygon, Circle
from pysgf import SGF, SGFNode
from typing import Any, Dict, List, Tuple, Callable, Set
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# TODO: document every method with """ """
# TODO: actually instantiate & use the types
# TODO: make literals a type to carry around their augmentation for easy undo
# TODO: use numpy for hex_grid (current indexing in '_search_hex_grid' is... great)
# TODO: FIX AWFUL PATHING STUFF GOING ON...
# TODO: make all the classes going around frozen?

class Helpers:

    @staticmethod
    def clamp(value, min_value, max_value):
        return max(min(value, max_value), min_value)

    @staticmethod
    def chunk_list(l, chunk_size):
        return [l[i:i + chunk_size] for i in range(0, len(l), chunk_size)]

    @staticmethod
    def defaultdict_to_dict(dd):
        return {key: dict(value) for key, value in dd.items()}


class UtilsHex:

    class Coordinates:

        # TYPE                          DESCRIPTION                                             EXAMPLE     ALLOWED-OFF-BOARD
        # ---------------------------------------------------------------------------------------------------------------------
        # index:    int                 a 1d representation used for an array                   0           False
        # coord:    tuple[int, int]     a 2d representation used for a grid                     (0, 0)      True
        # position: str                 a letter-number representation used on the hex board    a1          True

        # ---------------------------------------------------------------------------------------------------------------------
        #  VALIDITY - ARE WE ON THE BOARD?
        # ---------------------------------------------------------------------------------------------------------------------

        @staticmethod
        def is_index_on_board(index: int, boardsize: int) -> bool:
            return 0 <= index < boardsize**2

        @staticmethod
        def is_coord_on_board(x: int, y: int, boardsize: int) -> bool:
            if not 0 <= x < boardsize: return False
            if not 0 <= y < boardsize: return False
            return True

        @staticmethod
        def is_position_on_board(position: str, boardsize: int) -> bool:
            x, y = UtilsHex.Coordinates.position_to_coord(position)
            return UtilsHex.Coordinates.is_coord_on_board(x, y, boardsize)

        # ---------------------------------------------------------------------------------------------------------------------
        #  CONVERSION - MOVE BETWEEN INDICES, COORDS, & POSITIONS
        # ---------------------------------------------------------------------------------------------------------------------

        @staticmethod
        def index_to_coord(i: int, boardsize: int) -> Tuple[int, int]:
            return i % boardsize, i // boardsize


        @staticmethod
        def coord_to_index(x: int, y: int, boardsize: int) -> int:
            if not UtilsHex.Coordinates.is_coord_on_board(x, y, boardsize): return -1
            return y * boardsize + x


        @staticmethod
        def index_to_position(index: int, boardsize: int) -> str:
            # Firstly, convert from 1d index to 2D coordinates
            x, y = UtilsHex.Coordinates.index_to_coord(index, boardsize)

            # Finally, convert from 2D coordinates to position string
            return UtilsHex.Coordinates.coord_to_position(x, y)


        @staticmethod
        def position_to_index(position: str, boardsize: int) -> int:
            # First, convert the position to 2D
            (x, y) = UtilsHex.Coordinates.position_to_coord(position)

            # Finally, convert from 2D coordinates to 1D index
            return UtilsHex.Coordinates.coord_to_index(x, y, boardsize)


        @staticmethod
        def coord_to_position(x, y):
            # First, convert to hex string representation
            x = UtilsHex.Coordinates._number_to_string(x + 1)
            y = str(y + 1)

            # Finally, combine this back into a hex string position
            return f"{x}{y}"


        @staticmethod
        def position_to_coord(position: str) -> Tuple[int, int]:
            split_index = -1
            for i, c in enumerate(position):
                if c.isalpha():
                    continue
                if i != 0:
                    split_index = i
                break
            if split_index == -1:
                raise ValueError(f"{position} is not a valid position string")
            string = position[:split_index]
            number = position[split_index:]

            # Then, convert from (letter, number) to (x, y) coordinates
            x = UtilsHex.Coordinates._string_to_number(string) - 1
            y = int(number) - 1

            return x, y


        @staticmethod
        def _string_to_number(hex_string: str) -> int:
            if not hex_string.isalnum():
                raise ValueError("Hex string must be alphanumeric")

            # Treat the hex_string as a base 26 number, increasing powers left-to-right
            number = 0
            for c in hex_string:
                value = ord(c) - ord('a') + 1
                number *= 26
                number += value
            return number


        @staticmethod
        def _number_to_string(number: int) -> str:
            if number <= 0:
                raise ValueError("Hex number must be >= 1")

            string = ""
            while number > 0:
                value = (number - 1) % 26
                string = chr(ord('a') + value) + string
                number = (number - 1) // 26

            return string

    # BOARD REPRESENTATION      DESCRIPTION                                                         USAGE
    # -------------------------------------------------------------------------------------------------------------------------------------------
    # Entire Board:
    # Hex Grid                  2D: a grid of ints                                                  Simple Base for plotting and searching in
    # Literals                  1D: all black pieces followed by all white pieces                   Tsetlin Machine
    #
    # Independent of a board:
    # Templates                 all hex positions where there should and shouldn't be a piece       Defining SGF Patterns in File
    # Search Patterns           coordinate offsets to have or not have a piece                      Searching for

    class HexGrid:

        # Grid Value Meanings:
        # 0 for black,
        # 1 for white,
        # -1 for nothing,

        @staticmethod
        def from_literals(literals: List[int], boardsize: int) -> List[List[int]]:
            hex_grid = [[-1 for x in range(boardsize)] for y in range(boardsize)]

            literals_per_player = boardsize**2
            for y in range(boardsize):
                for x in range(boardsize):
                    i = UtilsHex.Coordinates.coord_to_index(x, y, boardsize)
                    player = 0 if literals[i] else 1 if literals[literals_per_player+i] else -1
                    hex_grid[y][x] = player

            return hex_grid

        @staticmethod
        def from_search_pattern(search_pattern: "UtilsHex.SearchPattern") -> List[List[int]]:
            # Search patterns are all independent of boards
            # So we need to create a new board on which this search pattern could fit

            boardsize = search_pattern.induced_boardsize
            hex_grid = [[-1 for x in range(boardsize)] for y in range(boardsize)]
            for include_coord in search_pattern.induced_include_coords:
                hex_grid[include_coord[1]][include_coord[0]] = 0

            return hex_grid

        @staticmethod
        def print_hex_grid(hex_grid: List[List[int]]):
            for i, row in enumerate(hex_grid):
                string = i * 2 * " "
                for cell in row:
                    string += str(cell).ljust(2, str(cell)[-1]) + " " * 2
                print(string)
            print()

    class Template:

        _FILE_EXTENSION = ".sgf"

        def __init__(self, name:str, template_positions: List[str], intrusion_positions: List[str]):
            self.name = name
            self.template_positions = template_positions
            self.intrusion_positions = intrusion_positions

        def __str__(self):
            return f"Template: {self.name}"

        def __repr__(self):
            return str(self)

        @staticmethod
        def _load_from_directory(directory_path: Path) -> List["UtilsHex.Template"]:
            templates = []
            template_files = directory_path.rglob(f"*{UtilsHex.Template._FILE_EXTENSION}")
            for file in template_files:
                try:
                    template = UtilsHex.Template._load_from_file(file)
                    templates.append(template)
                except Exception as e:
                    print(e)
                    continue
            return templates

        @staticmethod
        def _load_from_file(file_path: Path) -> "UtilsHex.Template":
            sgf = SGF.parse_file(file_path)

            # File Format
            if int(sgf.get_property('FF')) != 4:
                raise Exception("Template Loading is only supported for SGF file format 4 (FF[4])")

            # Game Check
            if int(sgf.get_property('GM')) != 11:
                raise Exception("Supplied SGF file is not for the game of Hex")

            # Application
            if not (application_string := sgf.get_property('AP')) or application_string[0:6] != 'HexGui':
                raise Exception("Template Loading is only supported for SGF files created by HexGUI")

            # Position or Game?
            if sgf.get_property('RE'):
                raise Exception("Supplied SGF File represents a game, not a board position")

            # Boardsize
            if not (boardsize_string := sgf.get_property('SZ')):
                raise Exception("Template file is missing the boardsize")


            # Check we have any move information
            if not sgf.children or len(sgf.children) != 1:
                raise Exception("Supplied SGF File has the wrong number of children - 1 is expected")
            move_node: SGFNode = sgf.children[0]
            if move_node.children:
                raise Exception("Unexpected Children found for the Move Node")

            # Get the black and white pieces if they exist
            if not (black_pieces := move_node.get_list_property('AB')):
                raise Exception("Supplied SGF File has no Black Pieces")
            if not (white_pieces := move_node.get_list_property('AW')):
                raise Exception("Supplied SGF File has no White Pieces")

            # Make a new template
            return UtilsHex.Template(file_path.stem, black_pieces, white_pieces)

    class SearchPattern:

        # TODO: Add the parallelogram template

        class Match:

            class MatchType(Enum):
                LOST            = 1     # Pattern matches with intrusions, but we failed to defend them correctly
                EMPTY           = 2     # Pattern matches with no intrusions
                INCONCLUSIVE    = 3     # Pattern matches with intrusions, but not enough to know anything
                WON             = 4     # Pattern matches with intrusions, but we defended them correctly

            def __init__(self, hex_grid: List[List[int]], player: int, match_type: "UtilsHex.SearchPattern.Match.MatchType", search_pattern: "UtilsHex.SearchPattern", x: int, y: int):
                """
                This match represents a search patten found in a hex grid for a player at a position
                :param hex_grid: The hex grid we found the search pattern in
                :param search_pattern: The search pattern we found
                :param player: The player that played the search pattern
                :param x: The x position on the hex grid where the match occurs
                :param y: The y position on the hex grid where the match occurs
                """

                self.hex_grid = hex_grid
                self.search_pattern = search_pattern
                self.player = player
                self.match_type = match_type
                self.coord = np.array([x, y])
                self.boardsize = len(hex_grid)

            def __str__(self):
                player_name = "black" if self.player == 0 else "white"
                position = UtilsHex.Coordinates.coord_to_position(self.coord[0], self.coord[1])
                return f"{self.boardsize}x{self.boardsize}_{self.match_type.name.lower()}_{self.search_pattern}_at_{position}_for_{player_name}"

            def __repr__(self):
                return f"Match: {str(self)}"

        def __init__(self, base_name: str,
                     include_offsets,
                     exclude_offsets,
                     variation_name: str = ""):

            self.base_name: str = base_name
            self.variation_name: str = variation_name

            full_name: str = self.base_name
            if self.variation_name:
                full_name += f"_{self.variation_name}"
            self.full_name = full_name

            # Offsets centered at (0, 0) that define this pattern
            assert all(n == 0 for n in include_offsets[0])
            self.include_offsets = include_offsets
            self.exclude_offsets = exclude_offsets

            all_offsets = np.concatenate([self.include_offsets, self.exclude_offsets], axis=0)
            min_offsets = np.min(all_offsets, axis=0)
            max_offsets = np.max(all_offsets, axis=0)
            range_offsets = max_offsets - min_offsets
            self.induced_boardsize = max(range_offsets + 1)

            # Make all the offsets positive by subtracting the smallest value in each dimension
            # This then gives us coordinates that will fit on a standard positive indexed grid
            standardise_func = lambda v: v-min_offsets
            self.induced_include_coords = np.apply_along_axis(standardise_func, axis=1, arr=include_offsets)
            self.induced_exclude_coords = np.apply_along_axis(standardise_func, axis=1, arr=exclude_offsets)

            # Create a unique id string to identify if search patterns are the same
            induced_include_indices = [str(UtilsHex.Coordinates.coord_to_index(v[0], v[1], self.induced_boardsize))
                               for v in self.induced_include_coords]
            induced_exclude_indices = [str(UtilsHex.Coordinates.coord_to_index(v[0], v[1], self.induced_boardsize))
                               for v in self.induced_exclude_coords]
            induced_include_indices.sort()
            induced_exclude_indices.sort()
            self.uid = f"{','.join(induced_include_indices)}-{','.join(induced_exclude_indices)}"

        def __eq__(self, other: "UtilsHex.SearchPattern"):
            if not isinstance(other, UtilsHex.SearchPattern):
                return False
            return self.uid == other.uid

        def __str__(self):
            return self.full_name

        def __repr__(self):
            return f"Search Pattern: {str(self)}"

        _DATASET: Dict[str, List["UtilsHex.SearchPattern"]] = dict()

        _ROT_FUNCS = {
            "60": lambda v: np.array([v[0] + v[1], -v[0]]),
            "120": lambda v: np.array([v[1], -v[0] - v[1]]),
            "180": lambda v: -v,
            "240": lambda v: np.array([-v[0] - v[1], v[0]]),
            "300": lambda v: np.array([-v[1], v[0] + v[1]])
        }

        # TODO: plot axis and check they are labelled correctly
        _FLIP_FUNCS = {
            "Q": lambda v: np.array([v[0], -v[0] - v[1]]),
            "R": lambda v: np.array([-v[0] - v[1], v[1]]),
            "S": lambda v: np.array([v[1], v[0]]),
        }

        # ----------------------------------------
        # PUBLIC INTERFACE
        # ----------------------------------------

        # Create the patterns we want

        @staticmethod
        def initialise():
            """
            Add each of the templates manually defined as search patterns.
            So Far: Bridge, Crescent, Span, Trapezoid, Wheel
            :return:
            """

            def add_from_templates_directory(directory_path: Path):
                def create_from_template(template: "UtilsHex.Template") -> "UtilsHex.SearchPattern":
                    # First, convert from hex positions to coordinates
                    template_coords = np.array(
                        list(map(UtilsHex.Coordinates.position_to_coord, template.template_positions)))
                    intrusion_coords = np.array(
                        list(map(UtilsHex.Coordinates.position_to_coord, template.intrusion_positions)))

                    # Second, recenter the template by the top-left most template position
                    # This is to make the coordinates independent of where they appear on the original sgf grid
                    center = np.copy(template_coords[0])
                    template_offsets = template_coords - center
                    intrusion_offsets = intrusion_coords - center

                    return UtilsHex.SearchPattern(template.name, template_offsets, intrusion_offsets)

                templates = UtilsHex.Template._load_from_directory(directory_path)
                search_patterns = map(create_from_template, templates)

                for search_pattern in search_patterns:
                    UtilsHex.SearchPattern._add_search_pattern(search_pattern)

            print("Loading Search Patterns from Templates...")
            add_from_templates_directory(Path("../templates"))
            print(UtilsHex.SearchPattern.get_pattern_names())

        # Getting the patterns that exist

        @staticmethod
        def get_pattern_names() -> List[str]:
            return list(UtilsHex.SearchPattern._DATASET.keys())

        @staticmethod
        def get_pattern_variations(base_name: str) -> List["UtilsHex.SearchPattern"]:
            return UtilsHex.SearchPattern._DATASET.get(base_name, [])

        # Searching for a pattern

        @staticmethod
        def search_literals(search_pattern: "UtilsHex.SearchPattern", literals: List[int], boardsize: int) -> List["UtilsHex.SearchPattern.Match"]:
            """
            Search through a set of literals to try and find a pattern
            :param search_pattern: coordinates that must contain a piece of the players color, and coordinates that must NOT contain any piece
            :param literals: unaugmented literals representing the board state to search through
            :param boardsize: the nxn size of the board represented by the literals
            :return: the player the pattern matches for and the position of the match
            """

            # Convert the literals to a hex grid
            hex_grid = UtilsHex.HexGrid.from_literals(literals, boardsize)

            # And search the hex grid
            return UtilsHex.SearchPattern._search_hex_grid(search_pattern, hex_grid)

        @staticmethod
        def search_search_pattern(looking_for: "UtilsHex.SearchPattern", looking_in: "UtilsHex.SearchPattern") -> List["UtilsHex.SearchPattern.Match"]:
            """
            Search through one search pattern to find instances of another. Used to find dependencies between them
            :param looking_for: coordinates that must contain a piece of the players color, and coordinates that must NOT contain any piece
            :param looking_in: coordinates representing the space to search through
            :return: the player the pattern matches for and the position of the match
            """
            if looking_in.induced_boardsize < looking_for.induced_boardsize:
                # We are trying to find something in a place that's too small to even fit it, so don't bother trying
                return []

            # Convert the looking in pattern to a hex grid
            hex_grid = UtilsHex.HexGrid.from_search_pattern(looking_in)

            # And search the hex grid
            return UtilsHex.SearchPattern._search_hex_grid(looking_for, hex_grid, allowed_players={0})

        @staticmethod
        def get_match_type_from_search_pattern_rules(base_name: str, player: int, exclude_players: List[int]) -> "UtilsHex.SearchPattern.Match.MatchType":
            # TODO: may be able to just test if its possible to become fully connected
            #  via only the exclude positions with some pathfinding search?
            #  and ignore all the individual rules... yes that would work... But maybe we lose some useful insights?
            #  IDEA: if we follow the perfect path of the template we still check that manually,
            #  but the rest is then automated connectivity path search
            # TODO: maybe add (LOSING, WINNING) cases to split INCONCLUSIVE
            #  for if we are following the forced path of the template
            #  or only won because we got lucky and the opponent made the wrong move
            # TODO: maybe change some of the WON to also WON_IMPERFECT
            #  as sure it is connected, but it doesnt follow the template play you should have done for this template

            # In the plot of each template we can see 0-n labelling each of the intrusion positions.
            # We can use who is in each of these, param: exclude_players, to work out the match type

            # There are some checks we can do independent of which search_pattern we are:
            # 1. Regardless of search pattern, we need the right amount of players
            base_pattern = UtilsHex.SearchPattern.get_pattern_variations(base_name)[0]
            assert len(exclude_players) == len(base_pattern.exclude_offsets)

            # 2. Regardless of the search pattern, if everything is -1 then we are empty
            if all(exclude_player == -1 for exclude_player in exclude_players):
                return UtilsHex.SearchPattern.Match.MatchType.EMPTY

            # Now we can perform the checks for each pattern
            other_player = 1 - player

            if base_name == 'bridge':
                # GOAL: to form full connectivity from one side of the bridge to the other

                # We can achieve this by looking at the pieces in the gap of the bridge
                counter = Counter(exclude_players)
                # TODO: for example, if we have both the bridge cells, this is a definitive waste of a piece
                #  as we can win with just 1, so this case should be a different kind of win (or even a loss?)
                # If you have at least 1 of the exclude positions, you win
                if counter[player] >= 1:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If both exclude positions are the opponents, you lose
                if counter[other_player] == 2:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # There must be at least one empty position
                assert counter[-1] >= 1
                # We are still able to win by playing in the empty cell, and haven't lost yet
                return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE

            if base_name == 'wheel':
                # GOAL: to fully connect all 3 of the wheel cells

                # If you have the center, you win
                if exclude_players[2] == player:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If no one has the center...
                if exclude_players[2] == -1:
                    # we are at worst inconclusive (as you could get the center later)
                    at_worst = UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                    # We now search again assuming the opponent has the center
                    # to see if we could win any other way
                    new_exclude_players = exclude_players[:]
                    new_exclude_players[2] = other_player
                    second_attempt = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        base_name, player, new_exclude_players)
                    return max(at_worst, second_attempt, key=lambda mt: mt.value)
                # The opponent has the center
                assert exclude_players[2] == other_player
                # So we need to consider the outside ring
                outer_counter = Counter([exclude_players[i] for i in [0, 1, 3]])
                # If we have 2 or more outer pieces, we win
                if outer_counter[player] >= 2:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If the opponent has 2 or more outer pieces (and the center), we lose
                if outer_counter[other_player] >= 2:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # Both players have at most 1 outer piece, so we must wait for more play
                assert outer_counter[player] <= 1 and outer_counter[other_player] <= 1
                return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE

            if base_name == 'crescent':
                # GOAL: to form full connectivity from one side of the crescent to the other

                # If we have the bottom piece, we win
                if exclude_players[3] == player:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If no one has the bottom piece...
                if exclude_players[3] == -1:
                    # we are at worst inconclusive (as you could get the bottom later)
                    at_worst = UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                    # We now search again assuming the opponent has the bottom
                    # to see if we could win any other way
                    new_exclude_players = exclude_players[:]
                    new_exclude_players[3] = other_player
                    second_attempt = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        base_name, player, new_exclude_players)
                    return max(at_worst, second_attempt, key=lambda mt: mt.value)
                # The opponent has the bottom piece
                assert exclude_players[3] == other_player
                # so we need to consider the upper triangle
                # If we have been cut off, we lose
                if exclude_players[2] == other_player:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # If we could be cut off in the future, we are inconclusive
                if exclude_players[2] == -1:
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # We have the piece we could have been cut off with
                assert exclude_players[2] == player
                # So now we can analyse the final 2 cells
                counter = Counter([exclude_players[i] for i in [0, 1]])
                # If we have either of these cells, we have formed a full connection
                if player in counter:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If either is empty, we still have the potential to form a full connection
                if -1 in counter:
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # They are both the opponents pieces
                assert counter[other_player] == 2
                return UtilsHex.SearchPattern.Match.MatchType.LOST

            if base_name == 'trapezoid':
                # GOAL: to fully connect the top and bottom sides of the trapezoid

                # trapezoid is broken into 2 pairs, (0, 1) and (2, 3),
                # s.t. if white has one in a pair, we take the other, and win!
                pair_1 = [exclude_players[i] for i in [0, 1]]
                counter_1 = Counter(pair_1)
                pair_2 = [exclude_players[i] for i in [2, 3]]
                counter_2 = Counter(pair_2)
                # If we have at least one in each pair, we win (by construction)
                if counter_1[player] >= 1 and counter_2[player] >= 1:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If we have both of pair 2, we win:
                if counter_2[player] == 2:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # We have (0, 0), (0, 1), (1, 0) or (2, 0) in (pair1, pair2)
                assert (counter_1[player], counter_2[player]) in [(0, 0), (0, 1), (1, 0), (2, 0)]
                # If opponent has any 3 cells, we are cut off and lose
                if counter_1[other_player] + counter_2[other_player] >= 3:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # The opponent has at most 2 cells
                assert counter_1[other_player] + counter_2[other_player] <= 2
                # If those 2 are in pair 2, we are cut off and lose
                if counter_2[other_player] == 2:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # If those 2 are in pair 1,
                if counter_1[other_player] == 2:
                    # We have (0, 0) or (0, 1) only, neither of which is enough
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # The opponent has at most one piece in each pair
                assert counter_1[other_player] <= 1 and counter_2[other_player] <= 1
                # Or put another way, they have (0, 0), (0, 1), (1, 0), (1, 1) in (pair1, pair2)
                assert (counter_1[other_player], counter_2[other_player]) in [(0, 0), (0, 1), (1, 0), (1, 1)]
                # In all these remaining combinations, we don't know enough
                return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE

            if base_name == 'span':
                # GOAL: to form full connectivity from one side of the span to the other

                # If we have the bottom piece, we win
                if exclude_players[5] == player:
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # If no one has the bottom piece...
                if exclude_players[5] == -1:
                    # we are at worst inconclusive (as you could get the bottom later)
                    at_worst = UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                    # We now search again assuming the opponent has the bottom piece
                    # to see if we could win any other way
                    new_exclude_players = exclude_players[:]
                    new_exclude_players[5] = other_player
                    second_attempt = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        base_name, player, new_exclude_players)
                    return max(at_worst, second_attempt, key=lambda mt: mt.value)
                # The opponent has the bottom piece
                assert exclude_players[5] == other_player
                # So we need to consider the triangle [0, 1, 2, 3, 4]
                # If the rest of the triangle is empty,
                if all(exclude_players[i] == -1 for i in [0, 1, 2, 3, 4]) == 5:
                    # we don't know what might happen there
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # If we have the double bridge potential piece (1)
                if exclude_players[1] == player:
                    # We have the double bridge cell, so let's check them!
                    # NOTE: Check the pattern plots to see where the mappings come from
                    # The left bridge is rot60
                    bridge_left_players = [exclude_players[0], exclude_players[3]]
                    result_bridge_left = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        'bridge', player, bridge_left_players)
                    # The right bridge is the base bridge
                    bridge_right_players = [exclude_players[2], exclude_players[4]]
                    result_bridge_right = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        'bridge', player, bridge_right_players)
                    # We win this template if we won both of the bridges
                    if result_bridge_left == UtilsHex.SearchPattern.Match.MatchType.WON and \
                            result_bridge_right == UtilsHex.SearchPattern.Match.MatchType.WON:
                        return UtilsHex.SearchPattern.Match.MatchType.WON
                    # We lose if we lose either bridge
                    if result_bridge_left == UtilsHex.SearchPattern.Match.MatchType.LOST or \
                            result_bridge_right == UtilsHex.SearchPattern.Match.MatchType.LOST:
                        return UtilsHex.SearchPattern.Match.MatchType.LOST
                    # At least one bridge is inconclusive or empty
                    options = {UtilsHex.SearchPattern.Match.MatchType.EMPTY,
                               UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE}
                    assert result_bridge_left in options or result_bridge_right in options
                    # which makes this span inconclusive
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # If the opponent has the double bridge piece
                if exclude_players[1] == other_player:
                    # The opponent has a bridge between the double bridge pos and the bottom
                    # Get the results of that bridge
                    # NOTE: Check the pattern plots to see where the mappings come from
                    # The opponent bridge is rot120
                    opponent_bridge_players = [exclude_players[3], exclude_players[4]]
                    opponents_bridge_type = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                        'bridge', other_player, opponent_bridge_players)
                    # If they win this bridge, we lose
                    if opponents_bridge_type == UtilsHex.SearchPattern.Match.MatchType.WON:
                        return UtilsHex.SearchPattern.Match.MatchType.LOST
                    # If they lost this bridge, we win
                    if opponents_bridge_type == UtilsHex.SearchPattern.Match.MatchType.LOST:
                        # TODO: this is a broken template win, we win but NOT THE RIGHT WAY (NOT FORCED / GOT LUCKY)
                        return UtilsHex.SearchPattern.Match.MatchType.WON
                    # If this bridge is empty or inconclusive, the span is inconclusive
                    return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE
                # The double bridge position is empty
                assert exclude_players[1] == -1
                # If we have both of the middle row, we have won
                if all(exclude_players[i] == player for i in [3, 4]):
                    return UtilsHex.SearchPattern.Match.MatchType.WON
                # We don't have the middle row,
                # so it is not possible for us to win without the double bridge position
                # We can imagine what the result would be if we DID have the double bridge position
                new_exclude_players = exclude_players[:]
                new_exclude_players[1] = player
                second_attempt = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                    'span', player, new_exclude_players)
                # If we COULDN'T win, we have definitely lost
                if second_attempt == UtilsHex.SearchPattern.Match.MatchType.LOST:
                    return UtilsHex.SearchPattern.Match.MatchType.LOST
                # We have a chance at not loosing, so we could take it
                return UtilsHex.SearchPattern.Match.MatchType.INCONCLUSIVE

            # We must have provided an invalid base_name
            return UtilsHex.SearchPattern.Match.MatchType.LOST

        # ----------------------------------------
        # PRIVATE INTERFACE
        # ----------------------------------------

        @staticmethod
        def _add_search_pattern(search_pattern: "UtilsHex.SearchPattern"):
            def create_pattern_variations() -> List["UtilsHex.SearchPattern"]:
                # Algorithms described by Red Blob Games:
                # - https://www.redblobgames.com/grids/hexagons/#rotation
                # Help simplifying converting between coordinate systems from ChatGPT

                seen_variations = [search_pattern]

                # Each rotation goes 60 degrees further than the previous
                # So we construct [60, 120, 180, 240, 300] degree rotations
                seen_so_far_variations = seen_variations.copy()
                for variant in seen_so_far_variations:
                    for rot_angle, rot_func in UtilsHex.SearchPattern._ROT_FUNCS.items():
                        rotated_include_offsets = np.apply_along_axis(rot_func, axis=1, arr=variant.include_offsets)
                        rotated_exclude_offsets = np.apply_along_axis(rot_func, axis=1, arr=variant.exclude_offsets)

                        rotated_pattern = UtilsHex.SearchPattern(
                            search_pattern.base_name,
                            rotated_include_offsets,
                            rotated_exclude_offsets,
                            variation_name=f"rot{rot_angle}"
                        )

                        # If we have seen this rotated position before, we are just repeating work
                        if rotated_pattern in seen_variations:
                            break
                        seen_variations.append(rotated_pattern)

                # We can now also flip any of the rotation variants to make even more variant options
                seen_so_far_variations = seen_variations.copy()
                for variant in seen_so_far_variations:
                    for flip_name, flip_func in UtilsHex.SearchPattern._FLIP_FUNCS.items():
                        flipped_include_offsets = np.apply_along_axis(flip_func, axis=1, arr=variant.include_offsets)
                        flipped_exclude_offsets = np.apply_along_axis(flip_func, axis=1, arr=variant.exclude_offsets)

                        flipped_pattern = UtilsHex.SearchPattern(
                            search_pattern.base_name,
                            flipped_include_offsets,
                            flipped_exclude_offsets,
                            variation_name=f"{variant.variation_name}flip{flip_name}"
                        )

                        # If we have seen this pattern before, ignore it
                        if flipped_pattern not in seen_variations:
                            seen_variations.append(flipped_pattern)

                return seen_variations

            variations = create_pattern_variations()
            UtilsHex.SearchPattern._DATASET[search_pattern.base_name] = variations

        @staticmethod
        def _search_hex_grid(search_pattern: "UtilsHex.SearchPattern", hex_grid: List[List[int]], allowed_players: Set[int]=None) -> List["UtilsHex.SearchPattern.Match"]:
            """
            Search through a hex grid to try and find a pattern
            :param search_pattern: coordinates that must contain a piece of the players color, and coordinates that must NOT contain any piece
            :param hex_grid: 2d grid board state to search through
            :param allowed_players: list of players we are looking for matches for
            :return: the full search pattern match information
            """

            # By default, we are allowed to find a match for either player
            if allowed_players is None:
                allowed_players = {0, 1}

            def match_pattern_at_start_coord(start_x, start_y) -> UtilsHex.SearchPattern.Match:
                """
                :param start_x: the x coordinate to start the pattern at
                :param start_y: the y coordinate to start the pattern at
                :return: integer representing the player for which the pattern matched
                """

                # Check we have an allowed piece at the starting position
                player = hex_grid[start_y][start_x]
                if player not in allowed_players:
                    return None

                # We have a player piece in the start position,
                # So we check for player pieces in all the positions of this pattern
                start_coord = np.array([start_x, start_y])
                for include_offset in search_pattern.include_offsets:
                    # Check that the position we want to include is on the board from this start position
                    include_coord = start_coord + include_offset
                    if not UtilsHex.Coordinates.is_coord_on_board(*include_coord, boardsize):
                        return None

                    # Check the position has a player piece
                    if hex_grid[include_coord[1]][include_coord[0]] != player:
                        return None

                # We have a piece of the right player in every include position of the pattern
                # This means we definitely have a match!
                # Now we need to classify if this is an EMPTY, INCONCLUSIVE, WON, or LOST match

                # We do this by checking which player is in each exclude_position
                exclude_players = []
                for exclude_offset in search_pattern.exclude_offsets:
                    # If the position we want to exclude is off the board from this start position
                    exclude_coord = start_coord + exclude_offset
                    if not UtilsHex.Coordinates.is_coord_on_board(*exclude_coord, boardsize):
                        # We can't apply the rules of the template correctly, so we ignore it
                        # TODO: THIS IS ACTUALLY FINE, BUT WE NEED TO INTRODUCE EDGE TEMPLATES FOR IT TO WORK
                        #  so ignore them for now, but come back later
                        return None

                    # If it is on the board, we get the player in this position
                    exclude_players.append(hex_grid[exclude_coord[1]][exclude_coord[0]])

                # And then we ask the search_pattern what this combination of players means
                match_type = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                    search_pattern.base_name, player, exclude_players)

                return UtilsHex.SearchPattern.Match(hex_grid, player, match_type, search_pattern, start_x, start_y)

            # Every board position is a potential start position for the pattern
            matches = []
            boardsize = len(hex_grid)
            for start_y in range(boardsize):
                for start_x in range(boardsize):
                    if match:=match_pattern_at_start_coord(start_x, start_y):
                        matches.append(match)

            return matches


class UtilsTournament:

    @staticmethod
    def load_tournament_games(tournament_dir: Path) -> Tuple[List[Tuple[int, list]], int]:
        result_keys = ['GAME', 'ROUND', 'OPENING', 'BLACK', 'WHITE', 'RES_B', 'RES_W', 'LENGTH', 'TIME_B', 'TIME_W', 'ERR', 'ERR_MSG']

        try:
            games = []
            game_boardsize = None

            results_path = tournament_dir / "results"
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
                    sgf_path = tournament_dir / f"{result['GAME']}.sgf"
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
                    game_winner = 0 if game_winner == "B+" else 1                                       # 1 if black wins, 0 if white wins (black loses)
                    games.append((game_winner, game_moves))

            return games, game_boardsize

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)


    @staticmethod
    def games_to_winner_prediction_dataset(games:  List[Tuple[int, list]], boardsize: int, dataset_path: Path):
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
    def _get_literal_states_from_moves(moves: List[str], boardsize: int):

        # TODO: Check literal representation process from Giri et al.
        # Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
        # https://ieeexplore.ieee.org/document/9892796

        literals_per_player = boardsize**2
        literals = UtilsTM.Literals.make_empty_board(boardsize)
        states = [literals.copy()]  # Start with an empty board state

        for i, move in enumerate(moves):
            is_white_move = i % 2 == 1
            index = UtilsHex.Coordinates.position_to_index(move, boardsize)
            index += is_white_move * literals_per_player
            literals[index] = 1
            states.append(literals.copy())  # And gradually add every played state one move at a time

        return states


class UtilsTM:

    class Literals:

        class History(Enum):
            HISTORY_NONE        = 0
            # Give Literals 'Memory'
            HISTORY_STATE       = 1
            HISTORY_MOVE        = 2

            def apply(self, literals: List[int], history: List[List[int]], boardsize: int) -> List[int]:
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

            def apply(self, literals: List[int], boardsize: int) -> List[int]:
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
                        x1, y1 = UtilsHex.Coordinates.index_to_coord(i, boardsize)

                        # Second, add 1 to the coordinate to get the new coordinates in the (n+2)x(n_2) grid
                        x2, y2 = x1 + 1, y1 + 1

                        # Next, convert from (n+2)x(n+2) 2d coordinates to (n+2)x(n_2) 1d coordinates
                        j = UtilsHex.Coordinates.coord_to_index(x2, y2, boardsize + 2)

                        # Finally, move the literal to its new position
                        new_literals[j] = literals[i]  # black
                        new_literals[new_literals_per_player + j] = literals[literals_per_player + i]  # white

                    # Add the border cells to the new larger board
                    # NOTE: we leave the corner positions empty
                    # Black wins top to bottom, so the first and last row should be all black
                    for y in [0, new_boardsize - 1]:
                        for x in range(1, new_boardsize - 1):
                            i = UtilsHex.Coordinates.coord_to_index(x, y, new_boardsize)
                            new_literals[i] = 1
                    # White wins left to right, so each row should begin and end with white
                    for y in range(1, new_boardsize - 1):
                        for x in [0, new_boardsize - 1]:
                            i = UtilsHex.Coordinates.coord_to_index(x, y, new_boardsize)
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
        def get_literal_difference(A: List[int], B: List[int]) -> List[int]:
            return [a ^ b for (a, b) in zip(A, B)]


        @staticmethod
        def make_empty_board(boardsize: int) -> List[int]:
            literals_per_player = boardsize ** 2
            return [0 for _ in range(2 * literals_per_player)]


        @staticmethod
        def make_random_board(boardsize: int, num_pieces: int = None) -> List[int]:
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
        def make_random_move(literals: List[int], boardsize: int) -> Tuple[List[int], int]:
            move = UtilsTM.Literals._get_random_move(literals, boardsize)
            return UtilsTM.Literals._make_move(literals, move), move


        @staticmethod
        def _get_random_move(literals: List[int], boardsize: int) -> int:
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
        def _make_move(literals: List[int], move: int) -> List[int]:
            if move < 0 or move >= len(literals):
                return None  # Move isn't on board
            if literals[move] == 1:
                return None  # Move has already been made, so something is wrong

            new_literals = literals.copy()
            new_literals[move] = 1
            return new_literals


class UtilsDataset:

    class Dataset:
        """
        Winner Prediction Dataset
        """

        # TODO: are the literals augmented / only apply all the searches and everything else when unaugmented (assumption right now)

        def __init__(self, X, Y, boardsize: int, name: str, complete: bool = True):
            self.X = X
            self.Y = Y

            self.boardsize = boardsize
            self.name = name
            self.complete = complete    # Are full games used in this dataset, or are we pulling states randomly?

            self.state_win_counts = Counter(self.Y)
            self.state_num_pieces_counts = Counter([sum(state) for state in self.X])

            if complete:
                # TODO: remove and impute from states instead (what we need only as technically not games)
                #  when complete we can claim its valid even if we still use state method
                X_game, Y_game = self._group_by_game()
                self.X_game = X_game
                self.Y_game = Y_game
                self.game_win_counts = Counter(self.Y_game)

        def __add__(self, other):
            if not isinstance(other, UtilsDataset.Dataset):
                raise TypeError
            if self.boardsize != other.boardsize:
                raise ValueError

            X1, X2 = self.X, other.X
            Y1, Y2 = self.Y, other.Y

            X = np.vstack((X1, X2))
            Y = np.hstack((Y1, Y2))
            return UtilsDataset.Dataset(X, Y, self.boardsize, f"({self.name} + {other.name})")

        def __str__(self):
            return f"{self.name}: {len(self.X)}"

        def __repr__(self):
            return f"Dataset: {str(self)}"

        @property
        def shape(self):
            return self.X.shape

        @property
        def num_rows(self):
            return self.shape[0]

        @property
        def num_cols(self):
            return self.shape[1]

        def reduce_player_counts(self, black: int, white: int):
            if self.state_win_counts[0] < black:
                print('Black already has less than this many entries')
                return
            if self.state_win_counts[1] < white:
                print('White already has less than this many entries')
                return

            sampling_strategy = {0: black, 1: white}
            return self._undersample_random(sampling_strategy)

        def reduce_majority_frac(self, frac: float) -> "UtilsDataset.Dataset":
            """
            Undersample the dataset by reducing the majority class until we have N total rows
            :param N: The number of total rows after under sampling
            :return: A Dataset object with the resulting under sampled distribution
            """

            majority_winner, majority_count = map(int, max(self.state_win_counts.items(), key=lambda x: x[1]))
            minority_winner, minority_count = map(int, min(self.state_win_counts.items(), key=lambda x: x[1]))
            total_count = minority_count + majority_count
            assert total_count == self.num_rows

            original_majority_frac = majority_count / total_count
            if frac < 0.5:
                print("New percentage is too low and would require us to make the majority class a minority")
                return None
            if frac > original_majority_frac:
                print("New percentage is too high, would need increase_majority_percentage instead")
                return None

            sampling_strategy = {minority_winner: minority_count, majority_winner: int(minority_count * frac / (1-frac))}
            return self._undersample_random(sampling_strategy)

        def undersample(self) -> "UtilsDataset.Dataset":
            """
            Undersample the dataset to balance the class distribution
            :param N: The minimum number of total rows to keep. Allows you to remove most of the higher class but not all the way to 50/50
            :return: A Dataset object with the resulting under sampled distribution
            """

            return self._undersample_random()

        def _undersample_tomeklinks(self) -> "UtilsDataset.Dataset":
            # NOTE: takes too long
            tomek = TomekLinks(n_jobs=16)
            X_undersampled, Y_undersampled = tomek.fit_resample(self.X, self.Y)
            return UtilsDataset.Dataset(X_undersampled, Y_undersampled, self.boardsize,f"{self.name}_under-tomek", False)

        def _undersample_nearest_neighbours(self) -> "UtilsDataset.Dataset":
            # NOTE: takes too long
            # TODO: try to run (and tomek) on collab/ncc and save it once
            enn = EditedNearestNeighbours(n_jobs=2)
            X_undersampled, Y_undersampled = enn.fit_resample(self.X, self.Y)
            return UtilsDataset.Dataset(X_undersampled, Y_undersampled, self.boardsize, f"{self.name}_under-knn", False)

        def _undersample_random(self, sampling_strategy=None) -> "UtilsDataset.Dataset":
            if not sampling_strategy:
                sampling_strategy = "auto"

            rus = RandomUnderSampler(random_state=42, sampling_strategy=sampling_strategy)
            X_undersampled, Y_undersampled = rus.fit_resample(self.X, self.Y)
            return UtilsDataset.Dataset(X_undersampled, Y_undersampled, self.boardsize,f"{self.name}_under-rand", False)

        def oversample(self) -> "UtilsDataset.Dataset":
            return self._oversample_random()

        def _oversample_random(self) -> "UtilsDataset.Dataset":
            ros = RandomOverSampler(random_state=42)
            X_oversampled, Y_oversampled = ros.fit_resample(self.X, self.Y)
            return UtilsDataset.Dataset(X_oversampled, Y_oversampled, self.boardsize, f"{self.name}_over-rand", False)

        # TODO: replace with using states and not really games but calculating what we actually need
        def _group_by_game(self):
            # We cannot reconstruct game information from incomplete datasets (yet?)
            if not self.complete:
                return None, None

            # Split the dataset into each game
            X_game = []
            Y_game = []
            for i in range(len(self.X)):
                # If we have reached a new empty board
                if not any(self.X[i]):
                    # Start a new game
                    X_game.append([])
                    Y_game.append(int(self.Y[i]))
                # Append this game state to the current game
                X_game[-1].append(self.X[i])

            return X_game, Y_game

        def _copy(self):
            return UtilsDataset.Dataset(self.X.copy(), self.Y.copy(), self.boardsize, self.name, self.complete)

    TOURNAMENTS_DIR = Path("tournaments")
    PLY_1: "UtilsDataset.Dataset" = None
    PLY_2: "UtilsDataset.Dataset" = None
    PLY_3: "UtilsDataset.Dataset" = None
    PLY_4: "UtilsDataset.Dataset" = None
    BASELINE: "UtilsDataset.Dataset" = None

    @staticmethod
    def load_raw_datasets():
        """
        Load some hardcoded datasets
        :return:
        """

        augmentation = UtilsTM.Literals.Augmentation.AUG_NONE
        history = UtilsTM.Literals.History.HISTORY_NONE
        history_size = 0

        UtilsDataset.PLY_1 = UtilsDataset._load_winner_pred_dataset("6x6-1ply-simple", augmentation, history, history_size)
        UtilsDataset.PLY_2 = UtilsDataset._load_winner_pred_dataset("6x6-2ply-simple", augmentation, history, history_size)
        UtilsDataset.PLY_3 = UtilsDataset._load_winner_pred_dataset("6x6-3ply-simple" , augmentation, history, history_size)
        UtilsDataset.PLY_4 = UtilsDataset._load_winner_pred_dataset("6x6-4ply-simple-incomplete", augmentation, history, history_size)

        # Define a new baseline dataset to match the distribution of the original paper
        baseline_black = 175968
        baseline_white = 111826
        UtilsDataset.BASELINE = UtilsDataset.PLY_1 + UtilsDataset.PLY_2 + UtilsDataset.PLY_3 + UtilsDataset.PLY_4
        """
        baseline_total = baseline_black + baseline_white
        baseline_majority_frac = baseline_black / baseline_total
        UtilsDataset.BASELINE = UtilsDataset.BASELINE.reduce_majority_frac(baseline_majority_frac)
        UtilsDataset.BASELINE.name = '6x6-baseline_dist'
        """
        UtilsDataset.BASELINE = UtilsDataset.BASELINE.reduce_player_counts(baseline_black, baseline_white)
        UtilsDataset.BASELINE.name = '6x6-baseline_exact'


    @staticmethod
    def _load_winner_pred_dataset(tournament_name: str,
                                 augmentation: UtilsTM.Literals.Augmentation,
                                 history_type: UtilsTM.Literals.History,
                                 history_size: int) -> Dataset:
        try:

            datasetX = []
            datasetY = []

            dataset_path: Path = UtilsDataset.TOURNAMENTS_DIR / tournament_name / "dataset.csv"
            with open(dataset_path, mode='r', newline='') as dataset_file:
                reader = csv.reader(dataset_file)

                boardsize = int(next(reader)[1])
                num_literals = len(UtilsTM.Literals.make_empty_board(boardsize))

                headers = next(reader)

                game_history = dict()
                if history_type == UtilsTM.Literals.History.HISTORY_NONE:
                    history_size = 0  # We don't want any history
                if history_type == UtilsTM.Literals.History.HISTORY_MOVE:
                    history_size += 1  # In order to calculate n moves, we need (n+1) games of history

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

                    # Add the final game literals and the winner to the dataset
                    datasetX.append(final_literals)
                    datasetY.append(winner)

            assert (len(datasetX) == len(datasetY))
            return UtilsDataset.Dataset(np.array(datasetX), np.array(datasetY), boardsize, tournament_name)

        except (FileNotFoundError, NotADirectoryError) as e:
            print(e, file=sys.stderr)


class UtilsPlot:

    PLOTS_DIR = Path("plots")
    PLOT_GAME_LENGTH_DIR = PLOTS_DIR / "game_lengths"
    PLOT_TEMPLATES_DIR = PLOTS_DIR / "templates"
    PLOT_WIN_RATES_DIR = PLOTS_DIR / "win_rates"

    @staticmethod
    def save_plot(plt: matplotlib.pyplot.plot, filepath: Path, bbox_inches=None):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches=bbox_inches)
        plt.close()

    ### HEX

    @staticmethod
    def _plot_hex_grid(hex_grid: List[List[int]], filepath: Path, show_axis: bool=False, coord_text_func: Callable[[int, int], str]=None):
        # Plot a Hexagonal Grid
        # - logic from: https://www.redblobgames.com/grids/hexagons/

        # By default, the coord text is its position
        if not coord_text_func:
            coord_text_func = lambda x, y: UtilsHex.Coordinates.coord_to_position(x, y)

        # Define the grid colors
        cell_color = 'lightyellow'
        piece_colors = ['black', 'white']
        grid_color = 'black'

        # Define the grid properties
        boardsize = len(hex_grid)
        hex_radius = 1
        dy = (3 / 2) * hex_radius
        dx = np.sqrt(3) * hex_radius

        def get_cell_position(x: float, y: float) -> Tuple[float, float]:
            pos_x = x * dx + (y * dx / 2)  # Add additional row offset as we go down
            pos_y = (boardsize - 1 - y) * dy  # Row 0 is at the top, so need to reverse y
            return pos_x, pos_y

        # Set up the figure
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot each hex cell at the desired position
        for y in range(boardsize):
            for x in range(boardsize):
                position = get_cell_position(x, y)

                # Plot the grid
                hexagon = RegularPolygon(position, numVertices=6, radius=hex_radius,
                                         orientation=0, edgecolor=grid_color, facecolor=cell_color)
                ax.add_patch(hexagon)

                # Plot any piece
                player = hex_grid[y][x]
                if player != -1:
                    piece = Circle(position, radius=hex_radius * 0.6,
                                   edgecolor=grid_color, facecolor=piece_colors[player])
                    ax.add_patch(piece)

                # Write the hex position

                coord_text = coord_text_func(x, y)
                if coord_text:
                    coord_text_color = grid_color if player == -1 else piece_colors[1 - player]
                    ax.text(*position, coord_text, ha='center', va='center', size=7, color=coord_text_color)

        # Plot the axis labels for hex
        if show_axis:
            text_offset = hex_radius * 1.4
            for i in range(boardsize):
                # Along the bottom
                position = get_cell_position(i, boardsize - 1)
                string = UtilsHex.Coordinates._number_to_string(i + 1)
                ax.text(position[0], position[1] - text_offset, string.upper(), ha='center', va='center', size=10)

                # Along the top
                position = get_cell_position(i, 0)
                string = UtilsHex.Coordinates._number_to_string(i + 1)
                ax.text(position[0], position[1] + text_offset, string.upper(), ha='center', va='center', size=10)

                # Along the left
                string = str(i + 1)
                ax.text(*get_cell_position(-1, i), string.upper(), ha='center', va='center', size=10)

                # Along the right
                string = str(i + 1)
                ax.text(*get_cell_position(boardsize, i), string.upper(), ha='center', va='center', size=10)

        # Calculate the width and height of the grid
        grid_width = (boardsize + 1) * dx + boardsize * (dx / 2)
        grid_height = (boardsize + 1) * dy

        # Crop the plot to only the area used for the hex board
        grid_center = get_cell_position((boardsize - 1) / 2, (boardsize - 1) / 2)
        ax.set_xlim(grid_center[0] - grid_width / 2, grid_center[0] + grid_width / 2)
        ax.set_ylim(grid_center[1] - grid_height / 2, grid_center[1] + grid_height / 2)
        ax.set_aspect('equal')
        plt.axis('off')

        # Save the plot
        UtilsPlot.save_plot(plt, filepath, 'tight')

    @staticmethod
    def plot_literals(literals: List[int], boardsize: int, filepath: Path):
        # Convert the literals to a hex grid
        hex_grid = UtilsHex.HexGrid.from_literals(literals, boardsize)

        # And plot the hex grid
        UtilsPlot._plot_hex_grid(hex_grid, filepath, True)

    @staticmethod
    def plot_search_pattern(search_pattern: UtilsHex.SearchPattern, exclude_players: List[int]=None, filepath: Path=None):
        # Convert the search pattern to a hex grid
        hex_grid = UtilsHex.HexGrid.from_search_pattern(search_pattern)

        # Add anything additional to the grid we want to
        if exclude_players:
            assert len(exclude_players) == len(search_pattern.induced_exclude_coords)
            for exclude_player, exclude_coord in zip(exclude_players, search_pattern.induced_exclude_coords):
                hex_grid[exclude_coord[1]][exclude_coord[0]] = exclude_player

        # Set a default filepath if none is given
        if not filepath:
            if exclude_players:
                match_type = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                    search_pattern.base_name, 0, exclude_players)
                plots_dir = UtilsPlot.PLOT_TEMPLATES_DIR / search_pattern.base_name / "match_types" / match_type.name.lower()
                filepath = plots_dir / f"{search_pattern}_is_{match_type.name.lower()}_for_[{','.join(map(str, exclude_players))}]"
            else:
                plots_dir = UtilsPlot.PLOT_TEMPLATES_DIR / search_pattern.base_name
                filepath = plots_dir / f"{search_pattern}.png"

        def coord_text_func(x, y):
            is_include_coord = np.all(search_pattern.induced_include_coords == [x, y], axis=1)
            if np.any(is_include_coord):
                return str(np.argmax(is_include_coord))

            is_exclude_coord = np.all(search_pattern.induced_exclude_coords == [x, y], axis=1)
            if np.any(is_exclude_coord):
                return str(np.argmax(is_exclude_coord))

            return ""

        # And plot the hex grid
        UtilsPlot._plot_hex_grid(hex_grid, filepath, False, coord_text_func)

    @staticmethod
    def plot_search_pattern_match(match: UtilsHex.SearchPattern.Match, filepath: Path=None):
        if not filepath:
            template_search_dir = UtilsPlot.PLOT_TEMPLATES_DIR / match.search_pattern.base_name / "searches"
            template_search_dir.mkdir(parents=True, exist_ok=True)
            filepath = template_search_dir / f"{str(match)}.png"
        UtilsPlot._plot_hex_grid(match.hex_grid, filepath, True)

    ### DATASET

    # TODO: more exploration, heatmaps for all/initial/final moves by win/player,

    @staticmethod
    def plot_dataset_win_rates(dataset: UtilsDataset.Dataset, by_state: bool):
        if not dataset.complete:
            by_state = True

        title = f"Player Win Rates in {dataset.name} Mohex Selfplay"
        if by_state:
            title += " States"
        else:
            title += " Games"

        title_color = 'black'
        bg_colour = 'lightblue'

        win_counts = dataset.state_win_counts if by_state else dataset.game_win_counts
        wedge_labels = [0, 1]
        wedge_values = [win_counts[i] for i in wedge_labels]
        wedge_colors = ['black', 'white']
        wedge_text_color = ['white', 'black']

        # Custom function to display both count and percentage
        # NOTE: 'pct' stands for 'percentage of total' of the pie chart
        def pct_with_counts(pct, counts):
            count = int(round(pct / 100.0 * sum(counts)))
            return f"{pct:.1f}%\n({count})"

        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(
            x=wedge_values,
            labels=None,
            autopct=lambda pct: pct_with_counts(pct, wedge_values),
            colors=wedge_colors,
            startangle=0
        )

        plt.title(title, color=title_color)
        fig.patch.set_facecolor(bg_colour)

        for i, text in enumerate(texts):
            text.set_color(wedge_text_color[i])

        for i, autotext in enumerate(autotexts):
            autotext.set_color(wedge_text_color[i])

        # Make the plot look better
        plt.axis("equal")
        plt.tight_layout()

        # Add a legend (key)
        # ax.legend(wedges, wedge_labels, title="Legend"s, loc="upper right", fontsize=12, bbox_to_anchor=(1, 1))

        # Save the plot to a file
        sub_directory = "state" if by_state else "game"
        filepath = UtilsPlot.PLOT_WIN_RATES_DIR / sub_directory / f"{dataset.name}_winrate_{sub_directory}.png"
        UtilsPlot.save_plot(plt, filepath)

    @staticmethod
    def plot_dataset_game_lengths(dataset: UtilsDataset.Dataset):
        if not dataset.complete:
            # TODO: can plot state's number of pieces/turn instead
            print("This dataset doesn't contain complete games, so we cannot plot the lengths of those games")
            return

        game_length = [len(game) for game in dataset.X_game]
        length_counts = Counter(game_length)

        bar_lengths = list(range(min(game_length), max(game_length) + 1))
        bar_counts = [length_counts[length] for length in bar_lengths]
        bar_colors = ["white", "black"] if bar_lengths[0] % 2 == 0 else ["black", "white"]
        bar_width = 0.5

        # Add value labels on top of the bars
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create the bar chart
        bars = ax.bar(bar_lengths, bar_counts, color=bar_colors, width=bar_width)
        ax.set_facecolor("lightblue")

        # Add labels and title
        ax.set_xlabel("Game Length", fontsize=12, fontweight="bold")
        ax.set_ylabel("# of Games", fontsize=12, fontweight="bold")
        ax.set_title(f"Game Lengths in {dataset.name} Mohex Selfplay", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add value labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, str(yval), ha='center', fontsize=10, color='black')

        # Add value labels on top of the bars
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

        # Make the plot look nicer
        plt.tight_layout()

        # Save the plot to a file
        filepath = UtilsPlot.PLOT_GAME_LENGTH_DIR / f"{dataset.name}_games_length.png"
        plt.savefig(filepath, dpi=300)  # Change filename and dpi as needed
        plt.close()  # Close the plot to free resources

    ### RANDOM ONE-TIME METHODS

    @staticmethod
    def _plot_all_search_pattern_variations():
        for pattern_name in UtilsHex.SearchPattern.get_pattern_names():
            for variation in UtilsHex.SearchPattern.get_pattern_variations(pattern_name):
                UtilsPlot.plot_search_pattern(variation)

    @staticmethod
    def _plot_all_search_pattern_match_types(base_name: str):
        # Go through every possible exclude players for template and plot the match type
        base_pattern = UtilsHex.SearchPattern.get_pattern_variations(base_name)[0]

        num_exclude_players = len(base_pattern.exclude_offsets)
        for i, exclude_players in enumerate(itertools.product([-1, 0, 1], repeat=num_exclude_players)):
            # There are too many to look at all the combinations
            # So here's what we do:
            # Any trivial wins are just checked to be correct
            # We then plot everything that isn't trivial
            print(i)

            match_type = UtilsHex.SearchPattern.get_match_type_from_search_pattern_rules(
                base_name, 0, list(exclude_players))

            if all(exclude_player == -1 for exclude_player in exclude_players):
                assert match_type == UtilsHex.SearchPattern.Match.MatchType.EMPTY
                continue

            if base_name == 'span':
                if exclude_players[5] == 0:
                    assert match_type == UtilsHex.SearchPattern.Match.MatchType.WON
                    continue
                if exclude_players[5] == -1:
                    assert match_type != UtilsHex.SearchPattern.Match.MatchType.LOST

            UtilsPlot.plot_search_pattern(base_pattern, exclude_players=list(exclude_players))

UtilsHex.SearchPattern.initialise()
# UtilsPlot._plot_all_search_pattern_variations()
# UtilsPlot._plot_all_search_pattern_match_types('crescent')

# print("Loading Datasets from file...")
# UtilsDataset.load_raw_datasets()
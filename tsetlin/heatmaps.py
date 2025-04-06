# Quick script to mess about with defining heatmaps to be displayed
import numpy as np

from utils import *

dir_heatmaps = Path('plots/heatmaps')
dir_examples = dir_heatmaps / "examples"

def example_random():
    boardsize = 6
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)
    heatmap = UtilsHex.HexGrid.make_empty_heatmap(boardsize)

    # Define a 'random' heatmap
    for row in range(boardsize):
        for col in range(boardsize):
            heatmap[row][col] = random.uniform(0, 1)

    UtilsHex.HexGrid.print_heatmap(heatmap)
    UtilsPlot._plot_hex_grid(hexgrid, dir_examples / "random.png", heatmap=heatmap)

def example_short_diagonal():
    boardsize = 6
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)
    heatmap = UtilsHex.HexGrid.make_empty_heatmap(boardsize, default=0.25)

    # Define the 'short diagonal'
    for row in range(boardsize):
        heatmap[row][boardsize-1-row] = 1.00

    UtilsHex.HexGrid.print_heatmap(heatmap)
    UtilsPlot._plot_hex_grid(hexgrid, dir_examples / "short diagonal.png", heatmap=heatmap)

def dataset_piece_occurrence(dataset: UtilsDataset.Dataset):
    # Make a heatmap for the occurrences of each color in every position they won and lost

    boardsize = dataset.boardsize
    literals_per_player = boardsize**2
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)

    dataset_black_win = [dataset.X[i] for i in range(dataset.num_rows) if dataset.Y[i] == 0]
    dataset_white_win = [dataset.X[i] for i in range(dataset.num_rows) if dataset.Y[i] == 1]

    literals_black = np.array([literals[:literals_per_player] for literals in dataset.X])
    literals_white = np.array([literals[literals_per_player:] for literals in dataset.X])

    literals_black_win_black = np.array([literals[:literals_per_player] for literals in dataset_black_win])
    literals_white_win_white = np.array([literals[literals_per_player:] for literals in dataset_white_win])

    literals_black_win_white = np.array([literals[:literals_per_player] for literals in dataset_white_win])
    literals_white_win_black = np.array([literals[literals_per_player:] for literals in dataset_black_win])

    def process_player_dataset(literals_player, name, is_relative):
        player_literals_counts = np.sum(literals_player, axis=0)
        player_literals_norm = player_literals_counts.astype(float) / len(literals_player)
        player_heatmap = list(player_literals_norm.reshape(boardsize, boardsize))

        name_heatmap = f"{dataset.name}_{name}_heatmap.png"
        path_heatmap = dir_heatmaps / dataset.name / "piece" / name_heatmap
        UtilsHex.HexGrid.print_heatmap(player_heatmap)
        UtilsPlot._plot_hex_grid(hexgrid, path_heatmap, heatmap=player_heatmap, is_heatmap_relative=is_relative)

    process_player_dataset(literals_black, 'black', True)
    process_player_dataset(literals_white, 'white', True)
    process_player_dataset(literals_black_win_black, 'black_win_black', True)
    process_player_dataset(literals_white_win_white, 'white_win_white', True)
    process_player_dataset(literals_black_win_white, 'black_win_white', True)
    process_player_dataset(literals_white_win_black, 'white_win_black', True)

def dataset_matches_occurrence(dataset: UtilsDataset.Dataset):
    # Make a heatmap for the occurrences of matches found for each color in won and lost games
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(dataset.boardsize)
    _, dataset_matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
    pattern_names = UtilsHex.SearchPattern.get_pattern_names()

    # Sort all the matches by who won the game, who played the match, and what type the match is
    game_winner_match_player_match_name_match_type_matrix = [
        [
            [
                [
                    [] for type in range(len(UtilsHex.SearchPattern.Match.MatchType))
                ] for name in pattern_names
            ] for player in [0, 1]
        ] for winner in [0, 1]
    ]
    for i, board_matches in dataset_matches.items():
        for match in board_matches:
            game_winner = dataset.Y[i]
            match_player = match['MatchPlayer']
            match_basename = pattern_names.index(match['MatchBaseName'])
            match_type = match['MatchType']

            game_winner_match_player_match_name_match_type_matrix[game_winner][match_player][match_basename][match_type.value-1].append(match)

    # Get back the matches we want
    desired_match_basename = pattern_names.index('trapezoid')  # ['bridge', 'wheel', 'trapezoid', 'crescent', 'span']
    desired_match_type = UtilsHex.SearchPattern.Match.MatchType.WON
    # e.g. comparing where black matches are when black wins VS black matches when white wins
    matches_A = game_winner_match_player_match_name_match_type_matrix[0][0][desired_match_basename][desired_match_type.value-1]
    matches_B = game_winner_match_player_match_name_match_type_matrix[1][0][desired_match_basename][desired_match_type.value-1]
    num_matches = len(matches_A) + len(matches_B)
    # e.g. comparing white and black matches when black wins
    # matches_A = game_winner_match_player_match_name_match_type_matrix[0][0][desired_match_basename][desired_match_type.value-1]
    # matches_B = game_winner_match_player_match_name_match_type_matrix[0][1][desired_match_basename][desired_match_type.value-1]
    # num_matches = len(matches_A) + len(matches_B)

    # Plot the heatmap
    def process(matches, total, game_winner, match_player):
        heatmap = UtilsHex.HexGrid.make_empty_heatmap(dataset.boardsize)
        for match in matches:
            match_name, match_var = match['MatchBaseName'], match['MatchVarName']
            match_x, match_y = match['MatchX'], match['MatchY']

            # Get the search pattern from the match name
            search_patterns = UtilsHex.SearchPattern.get_pattern_variations(match_name)
            search_pattern = next(sp for sp in search_patterns if match_var == sp.variation_name)

            # Add every position of the template to the heatmap
            for offset in search_pattern.include_offsets:
                position_x = match_x + offset[0]
                position_y = match_y + offset[1]
                heatmap[position_y][position_x] += 1 / total

        name_heatmap = f"" \
                       f"{dataset.name}_" \
                       f"{'black' if game_winner == 0 else 'white'}_win_" \
                       f"{'black' if match_player == 0 else 'white'}_" \
                       f"{pattern_names[desired_match_basename]}_" \
                       f"{desired_match_type.name}_" \
                       f"heatmap.png"
        path_heatmap = dir_heatmaps / dataset.name / "match" / name_heatmap
        UtilsHex.HexGrid.print_heatmap(heatmap)
        UtilsPlot._plot_hex_grid(hexgrid, path_heatmap, heatmap=heatmap, is_heatmap_relative=True)

    process(matches_A, num_matches, 0, 0)
    process(matches_B, num_matches, 1, 0)

def clause_piece_occurrence(dataset: UtilsDataset.Dataset):
    # Make a heatmap for the occurrences of matches found for each color in won and lost games
    boardsize = dataset.boardsize
    literals_per_player = boardsize ** 2
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)

    dir_tm = UtilsTM.Model.dataset_to_default_model_path(dataset)
    clause_datasets = UtilsTM.Model.load_trained_tm_data(dir_tm, dataset.boardsize)
    print(clause_datasets)

    for player, player_name in enumerate(['Black', 'White']):
        for polarity, polarity_name in enumerate(['Negative', 'Positive']):
            clause_dataset: UtilsDataset.Dataset = clause_datasets[player_name][polarity_name]
            clauses, weights = clause_dataset.X, clause_dataset.weights

            clauses = np.array(clauses) * np.abs(np.array(weights))[:, np.newaxis]
            literals_black = np.array([literals[:literals_per_player] for literals in clauses])
            literals_white = np.array([literals[literals_per_player:] for literals in clauses])

            print('fff', len(literals_black), len(literals_white), len(weights))

            def process_player_dataset(literals_player, name, is_relative):
                player_literals_counts = np.sum(literals_player, axis=0)
                player_literals_norm = player_literals_counts.astype(float) / len(literals_player)
                player_heatmap = list(player_literals_norm.reshape(boardsize, boardsize))

                name_heatmap = f"clauses_{player_name}_{polarity_name}_{name}_heatmap.png"
                path_heatmap = dir_tm / "matches" / "heatmaps" / name_heatmap
                UtilsHex.HexGrid.print_heatmap(player_heatmap)
                UtilsPlot._plot_hex_grid(hexgrid, path_heatmap, heatmap=player_heatmap, is_heatmap_relative=is_relative)

            process_player_dataset(literals_black, 'black', True)
            process_player_dataset(literals_white, 'white', True)

def clause_matches_occurrence(dataset: UtilsDataset.Dataset):
    # TODO: take from playground.analyse_matches_in_tm

    # Make a heatmap for the occurrences of matches found for each color in won and lost games
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(dataset.boardsize)

    dir_tm = UtilsTM.Model.dataset_to_default_model_path(dataset)
    clause_datasets = UtilsTM.Model.load_trained_tm_data(dir_tm, dataset.boardsize)
    print(clause_datasets)

    for player, player_name in enumerate(['Black', 'White']):
        for polarity, polarity_name in enumerate(['Negative', 'Positive']):
            clause_dataset: UtilsDataset.Dataset = clause_datasets[player_name][polarity_name]
            clauses, weights = clause_dataset.X, clause_dataset.weights
            path_matches = dir_tm / "matches" / Path(f"{clause_dataset.name}_matches.csv")
            _, matches = UtilsHex.SearchPattern.load_matches_in_dataset(clause_dataset, path_matches)

            # With the clauses, weights, and matches, we can now make the heatmap
            heatmap = UtilsHex.HexGrid.make_empty_heatmap(dataset.boardsize)

            for i in range(len(clauses)):
                # If this clause has no templates, then we ignore it
                if i not in matches:
                    continue

                # This clause has a template, so we want to know more about it

                # Get the information for this clause
                clause = clauses[i]
                clause_weight = weights[i]
                clause_matches = matches[i]

                # We look one by one at each match in the clause
                for match in clause_matches:
                    match_name, match_var = match['MatchBaseName'], match['MatchVarName']
                    match_type = match['MatchType']
                    match_player = match['MatchPlayer']
                    match_x, match_y = match['MatchX'], match['MatchY']

                    # Get the search pattern from the match name
                    search_patterns = UtilsHex.SearchPattern.get_pattern_variations(match_name)
                    search_pattern = next(sp for sp in search_patterns if match_var == sp.variation_name)

                    # Add every position of the template to the heatmap
                    for offset in search_pattern.include_offsets:
                        position_x = match_x + offset[0]
                        position_y = match_y + offset[1]
                        heatmap[position_y][position_x] += abs(clause_weight)  # Either all -ve or +ve, so just take abs

            heatmap = list(np.array(heatmap) / np.max(np.array(heatmap)))
            name_heatmap = f"matches_{player_name}_" \
                           f"{polarity_name}_" \
                           f"ds_" \
                           f"heatmap.png"
            path_heatmap = dir_tm / "matches" / "heatmaps" / name_heatmap
            UtilsHex.HexGrid.print_heatmap(heatmap)
            UtilsPlot._plot_hex_grid(hexgrid, path_heatmap, heatmap=heatmap, is_heatmap_relative=True)



if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()

    # example_random()
    # example_short_diagonal()

    dataset_piece_occurrence(UtilsDataset.X6_BASELINE)
    dataset_matches_occurrence(UtilsDataset.X6_BASELINE)
    clause_piece_occurrence(UtilsDataset.X6_BASELINE)
    clause_matches_occurrence(UtilsDataset.X6_BASELINE)

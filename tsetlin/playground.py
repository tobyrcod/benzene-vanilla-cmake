from utils import *

def analyse_matches_in_tm(dir_tm: Path):
    # Create needed directories
    dir_matches = dir_tm / "matches"
    dir_matches.mkdir(parents=False, exist_ok=True)

    # Calculate matches
    print('---------------------------------------------')
    print('Finding matches...')
    datasets = UtilsTM.Model.load_trained_tm_data(dir_tm, 6)
    for player, player_name in enumerate(['Black', 'White']):
        for polarity, polarity_name in enumerate(['Negative', 'Positive']):
            dataset: UtilsDataset.Dataset = datasets[player_name][polarity_name]
            path_matches = dir_matches / Path(f"{dataset.name}_matches.csv")
            if not path_matches.exists():
                UtilsHex.SearchPattern.calculate_matches_in_dataset(dataset, path_matches)

    # Reload the clauses & matches
    print('---------------------------------------------')
    print('Reloading matches...')
    for player, player_name in enumerate(['Black', 'White']):
        for polarity, polarity_name in enumerate(['Negative', 'Positive']):
            dataset: UtilsDataset.Dataset = datasets[player_name][polarity_name]
            clauses, weights = dataset.X, dataset.weights
            path_matches = dir_matches / Path(f"{dataset.name}_matches.csv")
            _, matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset, path_matches)


            # ------------------------------------------------------------------------------------------------------------------

            # Perform the analysis
            print('---------------------------------------------')
            print(f'Analysing Templates found in {player_name} {polarity_name} clauses')

            # [Lost, Empty, Inconclusive, Won]
            NUM_MATCHES = 0
            MATCH_TYPES = [mt.name for mt in UtilsHex.SearchPattern.Match.MatchType]
            MATCH_TYPE_WEIGHTS = np.array([0.1, 0.1, 0.2, 1.0])
            discrete_clause_discrete_match_matrix = [np.array([0 for m in range(len(MATCH_TYPES))]) for p in range(2)]
            weighted_clause_discrete_match_matrix = [np.array([0 for m in range(len(MATCH_TYPES))]) for p in range(2)]

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
                    NUM_MATCHES += 1
                    match_type = match['MatchType']
                    match_player = match['MatchPlayer']

                    discrete_clause_discrete_match_matrix[match_player][match_type.value-1] += 1
                    weighted_clause_discrete_match_matrix[match_player][match_type.value-1] += clause_weight

            discrete_clause_weighted_match_matrix = discrete_clause_discrete_match_matrix * MATCH_TYPE_WEIGHTS
            weighted_clause_weighted_match_matrix = weighted_clause_discrete_match_matrix * MATCH_TYPE_WEIGHTS

            print('---------------------------------------------')
            print('Results...')
            def analyse_matrix(matrix):
                # print(matrix)
                match_type_sum = np.sum(matrix, axis=0)
                # print(match_type_sum)                               # Total clause weight for each match type
                print(matrix / match_type_sum)                      # Player percentage of clause weight for each match type
                match_type_score = np.sum(matrix, axis=1)
                # print(match_type_score)                             # Total clause weight for each player
                print(match_type_score / sum(match_type_score))     # Player percentage of clause weight

            analyse_matrix(weighted_clause_discrete_match_matrix)


def analyse_matches_in_dataset(dataset: UtilsDataset.Dataset):
    # Reload the clauses & matches
    print('---------------------------------------------')
    print('Reloading matches...')
    _, dataset_matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
    ids_players_win = [{i for i in range(dataset.num_rows) if dataset.Y[i] == winner} for winner in [0, 1]]

    # Perform the analysis
    for player, player_name in enumerate(['black', 'white']):
        print('---------------------------------------------')
        print(f'Analysing Templates found in {player_name} won game states')

        # [Lost, Empty, Inconclusive, Won]
        NUM_MATCHES = 0
        MATCH_TYPES = [mt.name for mt in UtilsHex.SearchPattern.Match.MatchType]
        MATCH_TYPE_WEIGHTS = np.array([0.1, 0.1, 0.2, 1.0])
        discrete_clause_discrete_match_matrix = [np.array([0 for m in range(len(MATCH_TYPES))]) for p in range(2)]

        id_player_win = ids_players_win[player]
        for i in id_player_win:
            # If this clause has no templates, then we ignore it
            if i not in dataset_matches:
                continue

            # This clause has a template, so we want to know more about it

            # Get the information for this clause
            board_matches = dataset_matches[i]

            # We look one by one at each match in the clause
            for match in board_matches:
                NUM_MATCHES += 1
                match_type = match['MatchType']
                match_player = match['MatchPlayer']

                discrete_clause_discrete_match_matrix[match_player][match_type.value - 1] += 1

        discrete_clause_weighted_match_matrix = discrete_clause_discrete_match_matrix * MATCH_TYPE_WEIGHTS

        print('---------------------------------------------')
        print('Results...')

        def analyse_matrix(matrix):
            # print(matrix)
            match_type_sum = np.sum(matrix, axis=0)
            # print(match_type_sum)                               # Total clause weight for each match type
            print(matrix / match_type_sum)  # Player percentage of clause weight for each match type
            match_type_score = np.sum(matrix, axis=1)
            # print(match_type_score)                             # Total clause weight for each player
            print(match_type_score / sum(match_type_score))  # Player percentage of clause weight

        analyse_matrix(discrete_clause_weighted_match_matrix)


if __name__ == '__main__':

    #
    # Calculate templates in blunder models:
    UtilsHex.SearchPattern.initialise()
    analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder0"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder1"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder2"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder3"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder4"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder5"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder10"))

    #
    # Calculate templates in a blunder dataset:
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=0)
    # analyse_matches_in_dataset(UtilsDataset.X6_EQUAL_UNDER)

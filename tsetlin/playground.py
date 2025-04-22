from utils import *
import os

def analyse_matches_in_tm(dir_tm: Path):
    # Create needed directories
    dir_matches = dir_tm / "matches"
    dir_matches.mkdir(parents=False, exist_ok=True)

    print('---------------------------------------------')
    print('---------------------------------------------')
    print('---------------------------------------------')
    print('TM', dir_tm)

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
    winner_scores = []
    for player, player_name in enumerate(['Black', 'White']):
        for polarity, polarity_name in enumerate(['Negative', 'Positive']):
            # player, polarity, winner
            # 0     , 0       , 1
            # 0,    , 1       , 0
            # 1,    , 0       , 0
            # 1,    , 1       , 1
            # therefore, winner = (player == polarity)
            winner = int(player == polarity)
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
                    weighted_clause_discrete_match_matrix[match_player][match_type.value-1] += abs(clause_weight)

            discrete_clause_weighted_match_matrix = discrete_clause_discrete_match_matrix * MATCH_TYPE_WEIGHTS
            weighted_clause_weighted_match_matrix = weighted_clause_discrete_match_matrix * MATCH_TYPE_WEIGHTS

            print('---------------------------------------------')
            print('Results...')
            def analyse_matrix(matrix):
                match_type_sum = np.sum(matrix, axis=0)
                print(match_type_sum)                               # Total clause weight for each match type
                print(matrix / match_type_sum)                      # Player percentage of clause weight for each match type
                match_type_score = np.sum(matrix, axis=1)
                # print(match_type_score)                             # Total clause weight for each player
                player_weight = match_type_score / sum(match_type_score)
                print(player_weight)                                  # Player percentage of clause weight
                winner_score = player_weight[winner]
                winner_scores.append(winner_score)

            # To get the number of each type of template found from 'print(match_type_sum)'
            # analyse_matrix(discrete_clause_discrete_match_matrix)
            # Example: Black Negative: [  448 60784 10515 11142]

            # To get the interesting matrix we look at for trends from 'print(matrix / match_type_sum)'
            analyse_matrix(weighted_clause_discrete_match_matrix)
            # Example: Black Positive:
            # [[0.16131302 0.53504675 0.52664002 0.76642335]
            # [0.83868698 0.46495325 0.47335998 0.23357665]]

            # To get weighted by heuristic for interpretability score from 'print(player_weight)'
            # analyse_matrix(weighted_clause_weighted_match_matrix)
            # Example: Black Positive: [0.66958931 0.33041069]

    # Put it all together and get a final score
    print('---------------------------------------------')
    print(f'Final Templates Score...')
    print(winner_scores)

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

        analyse_matrix(discrete_clause_discrete_match_matrix)


if __name__ == '__main__':

    #
    # Calculate & analyse templates in blunder models:
    UtilsHex.SearchPattern.initialise()
    analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder0"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder1"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder2"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder3"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder4"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder5"))
    # analyse_matches_in_tm(Path("models/tmu/standard/blunder/6x6-blunder10"))

    #
    # Calculate & analyse templates in history models:
    # dir_run = Path("models/tmu/standard/history/grid run")
    # for i in range(18):
    #     dir_tm = dir_run / f"tm{i}"
    #     os.rename(dir_tm / f"tm{i}.pkl", dir_tm / f"tm.pkl")
    #     os.rename(dir_tm / f"tm{i}_data.json", dir_tm / f"tm_data.json")
    # for i in range(18):
    #     dir_tm = dir_run / f"tm{i}"
    #     analyse_matches_in_tm(dir_tm)

    #
    # Calculate templates in a blunder dataset:
    # Load:
    # 6x6
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=0)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=1)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=2)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=3)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=4)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=5)
    # UtilsDataset.load_raw_datasets(boardsize=6, blunder=10)
    # 7x7
    # UtilsDataset.load_raw_datasets(boardsize=7, blunder=0)
    # Calculate:
    # ds = UtilsDataset.EQUAL_UNDER
    # UtilsHex.SearchPattern.calculate_matches_in_dataset(ds)
    # Analyse:
    # analyse_matches_in_dataset(ds)

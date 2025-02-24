import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from tqdm import tqdm
from utils import UtilsHex, UtilsDataset, UtilsPlot
from sklearn.metrics import classification_report
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args


# TODO: Add all the control things here
#  - heuristic search pattern predictor
#  - matches in random / in dataset

# A heuristic model that uses search pattern matches ONLY for winner prediction
class WinnerPredModel:
    # Potentially add hyperparameter tuning here for the weights
    # Potentially add turn indicators to influence the weights

    @staticmethod
    def predict_winners_for_dataset(dataset: UtilsDataset.Dataset, weights=None):
        num_games, matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
        assert num_games == len(dataset.Y)

        all_predictions = []
        for (board_number, board_matches) in matches.items():
            prediction = WinnerPredModel._predict_winner_from_matches(board_matches, weights)
            all_predictions.append(prediction)

        return all_predictions, set(matches.keys())

    @staticmethod
    def _predict_winner_from_matches(matches: List, weights=None):
        # Predict the winner of a game purely by a heuristic of the templates in the state

        if not matches:
            raise NotImplementedError()

        if not weights:
            # [Lost, Empty, Inconclusive, Won]
            weights = [0.0, 0.13, 0.14, 0.84]
        weights = dict(zip(UtilsHex.SearchPattern.Match.MatchType, weights))

        board_score = 0
        for match in matches:
            score = weights.get(match['MatchType'], 0)
            player = match['MatchPlayer']
            if player == 1:
                # Matches for white go against the score
                score *= -1
            board_score += score

        prediction = 0 if board_score >= 0 else 1
        return prediction


if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()
    dataset = UtilsDataset.BASELINE

    # We need to find the best weights, use bayesian optimisation to do this
    space = [Real(low=-1, high=0,  name="lost"),
             Real(low=-0.5, high=0.5, name="empty"),
             Real(low=-0.5, high=0.5, name="inconclusive"),
             Real(low=0, high=1, name="won")]

    # We need to define what we are trying to optimise for
    # In our case it is accuracy (for example)
    @use_named_args(space)
    def objective(**params):
        weights = [params["lost"], params["empty"], params["inconclusive"], params["won"]]
        y_pred, boards_with_matches = WinnerPredModel.predict_winners_for_dataset(dataset, weights)
        y_true = [dataset.Y[i] for i in range(len(dataset.Y)) if i in boards_with_matches]

        report = classification_report(y_true, y_pred, output_dict=True)
        return -report['accuracy']  # Need to negate accuracy as this process minimises the objective

    # initial_weights = [[-100, 20, 10, 100], [-100, 20, 20, 100], [0, 10, 10, 50]]
    result = gp_minimize(
        objective,
        space,
        n_calls=50,
        acq_func='EI',
        verbose=True,
        n_jobs=-1
    )
    best_weights = result.x
    best_accuracy = -result.fun
    print(f"Best accuracy: {best_accuracy:.4f} with weights {best_weights}")

    plt.plot(-result.func_vals)  # Convert negative accuracy back to positive
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Bayesian Optimization Progress")
    UtilsPlot.save_plot(plt, Path("optimisation3.png"))
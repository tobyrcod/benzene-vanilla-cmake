import random
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
from tqdm import tqdm
from utils import UtilsHex, UtilsDataset, UtilsPlot
from sklearn.metrics import classification_report
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from deap import base, creator, tools, algorithms


# TODO: Add all the control things here
#  - heuristic search pattern predictor
#  - matches in random / in dataset

# A heuristic model that uses search pattern matches ONLY for winner prediction
class WinnerPredModel:
    # Potentially add hyperparameter tuning here for the weights
    # Potentially add turn indicators to influence the weights

    # [Lost, Empty, Inconclusive, Won]
    BEST_TYPE_WEIGHTS = [0.0, 0.13, 0.14, 0.84]

    @staticmethod
    def predict_winners_for_dataset(dataset: UtilsDataset.Dataset, name_weights=None, type_weights=None):
        num_games, matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
        assert num_games == len(dataset.Y)

        all_predictions = []
        for (board_number, board_matches) in matches.items():
            prediction = WinnerPredModel._predict_winner_from_matches(board_matches, name_weights, type_weights)
            all_predictions.append(prediction)

        return all_predictions, set(matches.keys())

    @staticmethod
    def _predict_winner_from_matches(matches: List, name_weights=None, type_weights=None):
        # Predict the winner of a game purely by a heuristic of the templates in the state

        if not matches:
            raise NotImplementedError()

        match_names = UtilsHex.SearchPattern.get_pattern_names()
        if not name_weights:
            # TODO: find best weights
            name_weights = [0] * len(match_names)
        name_weights = dict(zip(match_names, name_weights))

        if not type_weights:
            type_weights = WinnerPredModel.BEST_TYPE_WEIGHTS
        type_weights = dict(zip(UtilsHex.SearchPattern.Match.MatchType, type_weights))

        board_score = 0
        for match in matches:
            # TODO: should I add or multiply? - multiply seems to give best results
            match_name_weight = name_weights.get(match['MatchBaseName'], 0)
            match_type_weight = type_weights.get(match['MatchType'], 0)
            match_score = match_name_weight * match_type_weight

            player = match['MatchPlayer']
            if player == 1:
                # Matches for white go against the score
                match_score *= -1
            board_score += match_score

        prediction = 0 if board_score >= 0 else 1
        return prediction

    @staticmethod
    def _find_best_name_weights():
        UtilsDataset.load_raw_datasets()
        UtilsHex.SearchPattern.initialise()
        dataset = UtilsDataset.BASELINE

        # We need to find the best weights, use bayesian optimisation to do this
        space = [Real(low=0, high=1, name="trapezoid"),
                 Real(low=0, high=1, name="bridge"),
                 Real(low=0, high=1, name="wheel"),
                 Real(low=0, high=1, name="span"),
                 Real(low=0, high=1, name="crescent")]

        # We need to define what we are trying to optimise for
        # In our case it is accuracy (for example)
        @use_named_args(space)
        def objective(**params):
            name_weights = [params["trapezoid"], params["bridge"], params["wheel"], params["span"], params["crescent"]]
            y_pred, boards_with_matches = WinnerPredModel.predict_winners_for_dataset(dataset, name_weights, None)
            y_true = [dataset.Y[i] for i in range(len(dataset.Y)) if i in boards_with_matches]

            report = classification_report(y_true, y_pred, output_dict=True)
            return -report['accuracy']  # Need to negate accuracy as this process minimises the objective

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
        UtilsPlot.save_plot(plt, Path("optimisation.png"))

    @staticmethod
    def _find_best_type_weights():
        UtilsDataset.load_raw_datasets()
        UtilsHex.SearchPattern.initialise()
        dataset = UtilsDataset.BASELINE

        # We need to find the best weights, use bayesian optimisation to do this
        space = [Real(low=-1, high=0, name="lost"),
                 Real(low=-0.5, high=0.5, name="empty"),
                 Real(low=-0.5, high=0.5, name="inconclusive"),
                 Real(low=0, high=1, name="won")]

        # We need to define what we are trying to optimise for
        # In our case it is accuracy (for example)
        @use_named_args(space)
        def objective(**params):
            type_weights = [params["lost"], params["empty"], params["inconclusive"], params["won"]]
            y_pred, boards_with_matches = WinnerPredModel.predict_winners_for_dataset(dataset, None, type_weights)
            y_true = [dataset.Y[i] for i in range(len(dataset.Y)) if i in boards_with_matches]

            report = classification_report(y_true, y_pred, output_dict=True)
            return -report['accuracy']  # Need to negate accuracy as this process minimises the objective

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
        UtilsPlot.save_plot(plt, Path("optimisation.png"))

    @staticmethod
    def _find_best_combined_weights():
        # We need to find the best weights, use bayesian optimisation to do this
        space = ([Real(low=0, high=1, name="trapezoid"),
                Real(low=0, high=1, name="bridge"),
                Real(low=0, high=1, name="wheel"),
                Real(low=0, high=1, name="span"),
                Real(low=0, high=1, name="crescent")] +
                 [Real(low=-1, high=0, name="lost"),
                 Real(low=-0.5, high=0.5, name="empty"),
                 Real(low=-0.5, high=0.5, name="inconclusive"),
                 Real(low=0, high=1, name="won")])

        # We need to define what we are trying to optimise for
        # In our case it is accuracy (for example)
        @use_named_args(space)
        def objective(**params):
            name_weights = [params["trapezoid"], params["bridge"], params["wheel"], params["span"], params["crescent"]]
            type_weights = [params["lost"], params["empty"], params["inconclusive"], params["won"]]
            y_pred, boards_with_matches = WinnerPredModel.predict_winners_for_dataset(dataset, name_weights, type_weights)
            y_true = [dataset.Y[i] for i in range(len(dataset.Y)) if i in boards_with_matches]

            report = classification_report(y_true, y_pred, output_dict=True)
            return -report['accuracy']  # Need to negate accuracy as this process minimises the objective

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
        UtilsPlot.save_plot(plt, Path("optimisation.png"))


class WinnerPredGA:
    NUM_WEIGHTS = 9  # 5 name weights + 4 type weights

    @staticmethod
    def evaluate(individual):
        """
        Evaluates an individual (set of weights) by computing the classification accuracy.
        """
        name_weights = individual[:5]
        type_weights = individual[5:]

        dataset = UtilsDataset.BASELINE
        y_pred, boards_with_matches = WinnerPredModel.predict_winners_for_dataset(dataset, name_weights, type_weights)
        y_true = [dataset.Y[i] for i in range(len(dataset.Y)) if i in boards_with_matches]

        report = classification_report(y_true, y_pred, output_dict=True)
        return report['accuracy'],  # Must return as a tuple

    @staticmethod
    def optimize():
        """
        Runs the genetic algorithm to optimize weights.
        """
        # Define the problem (maximize accuracy)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, 0, 1)  # Random float between 0 and 1
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=WinnerPredGA.NUM_WEIGHTS)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Genetic Operators
        toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation
        toolbox.register("select", tools.selTournament, tournsize=3)  # Selection
        toolbox.register("evaluate", WinnerPredGA.evaluate)

        # Initialize population
        population = toolbox.population(n=50)
        NGEN = 40  # Number of generations
        CXPB, MUTPB = 0.5, 0.2  # Probabilities for crossover and mutation

        # Store best individuals for visualization
        best_scores = []

        for gen in tqdm(range(NGEN)):
            offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
            fits = list(map(toolbox.evaluate, offspring))

            for ind, fit in zip(offspring, fits):
                ind.fitness.values = fit

            population = toolbox.select(offspring, k=len(population))

            best_ind = tools.selBest(population, k=1)[0]
            best_scores.append(best_ind.fitness.values[0])

            print(f"Gen {gen + 1}: Best Accuracy = {best_ind.fitness.values[0]:.4f}")

        # Extract best weights
        best_weights = tools.selBest(population, k=1)[0]
        print(f"Best accuracy: {best_scores[-1]:.4f} with weights {best_weights}")

        # Plot accuracy progress
        plt.plot(best_scores)
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title("Genetic Algorithm Optimization Progress")
        UtilsPlot.save_plot(plt, Path("ga_optimization.png"))


if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()
    dataset = UtilsDataset.BASELINE

    WinnerPredGA.optimize()
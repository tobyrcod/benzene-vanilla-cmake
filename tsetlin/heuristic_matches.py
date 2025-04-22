import csv
import itertools
import os
import random

import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
from utils import UtilsHex, UtilsDataset, UtilsPlot
from sklearn.metrics import classification_report
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from joblib import Parallel, delayed
from deap import base, creator, tools, algorithms


# Currently finding best weights and predicting on any boards with a match
# TODO: May get more reliable results if we only consider boards with 2/3 matches
# TODO: need to analyse this distribution of how many matches per board per boardsize

# A heuristic model that uses search pattern matches for winner prediction
class WinnerPredModel:

    running_bests = []

    # [Lost, Empty, Inconclusive, Won]
    # All in 74% to 76% f1-score range

    # X6_BASELINE
    running_bests += [[0.06982960291266704, 0.17516456769004907, 0.42727656445486617, 0.9866716130339614]]
    running_bests += [[-0.03947003652705827, 0.037813562710824256, 0.1629400191565482, 0.8121813048055615]]
    running_bests += [[0.060208099220821154, 0.13861677310231246, 0.2713189559133222, 0.8360831959171893]]

    # X6_EQUAL_UNDER
    running_bests += [[0.1643978774931485, 0.0859056035063468, 0.308019943415063, 0.9774104634072205]]
    running_bests += [[0.13414634823586802, 0.14398294956523916, 0.16695020840543928, 0.950241264687298]]
    running_bests += [[0.07510496449113369, 0.08171303616065129, 0.15069470057666212, 1.0]]

    # X6_EQUAL_OVER
    running_bests += [[0.07354597823037734, 0.08112298500148296, 0.10948571349839109, 1.0]]
    running_bests += [[0.08737416319604652, 0.09685936948923123, 0.1511188829385083, 1.0]]
    running_bests += [[0.12323992368461179, 0.20745282272140916, 0.24173260671095204, 1.0]]

    @staticmethod
    def normalise_weights(weights):
        """Normalizes weights to the range [0, 1]"""
        max_weight = max(weights)
        return [w / max_weight for w in weights]

    @staticmethod
    def predict_winners_for_dataset(dataset, type_weights=None):
        """Predict winners based on type weights."""
        num_games, matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
        assert num_games == len(dataset.Y)

        all_predictions = []
        for board_number, board_matches in matches.items():
            prediction = WinnerPredModel._predict_winner_from_matches(board_matches, type_weights)
            all_predictions.append(prediction)

        return all_predictions, set(matches.keys())

    @staticmethod
    def _predict_winner_from_matches(matches: List, type_weights=None):
        """Predict winner by applying type weights to board matches."""
        if not matches:
            raise NotImplementedError()

        if not type_weights:
            type_weights = WinnerPredModel.BEST_TYPE_WEIGHTS

        type_weights = dict(zip(UtilsHex.SearchPattern.Match.MatchType, type_weights))

        board_score = 0
        for match in matches:
            match_type_weight = type_weights.get(match['MatchType'], 0)

            player = match['MatchPlayer']
            if player == 1:
                match_type_weight *= -1
            board_score += match_type_weight

        prediction = 0 if board_score >= 0 else 1
        return prediction


# ------------------------------------------------------------------------------------------------------------------

# Code to find the best parameters using simple searches
# Parallel code written with ChatGPT

# --------------------------------------

# Speedup by precomputing a matrix
def calculate_dataset_info(dataset: UtilsDataset.Dataset):
    num_games, matches = UtilsHex.SearchPattern.load_matches_in_dataset(dataset)
    num_boards_with_matches = len(matches)
    y_true = [dataset.Y[i] for i in matches]

    # Initialize an empty matrix with 100 rows and 4 columns
    num_match_types = len(UtilsHex.SearchPattern.Match.MatchType)
    type_matrix = np.zeros((num_boards_with_matches, num_match_types), dtype=int)

    # Go through each board with matches and get convert the matches to a vector
    for i, board_matches in enumerate(matches.values()):
        for match in board_matches:
            match_type = match['MatchType']
            player = match['MatchPlayer']

            match_type_player_score = 1 if player == 0 else -1  # 1 for black, -1 for white
            type_matrix[i][match_type.value - 1] += match_type_player_score

    return type_matrix, y_true

def evaluate_weight_matrix_parallel(type_matrix, type_weights, y_true):
    score_matrix = type_matrix * type_weights
    score_vector = np.sum(score_matrix, axis=1)
    y_pred = (score_vector <= 0).astype(int)  # Positive for black, negative for white

    # Get the final score
    report = classification_report(y_true, y_pred, output_dict=True)
    f1 = float(report['macro avg']['f1-score'])
    return f1, type_weights

# --------------------------------------

# GP
def find_best_type_weights_gp(dataset: UtilsDataset.Dataset):
    type_matrix, y_true = calculate_dataset_info(dataset)

    # We need to find the best weights, use bayesian optimisation to do this
    space = [Real(low=-1, high=1, name="lost"),
             Real(low=0, high=1, name="empty"),
             Real(low=0, high=1, name="inconclusive"),
             Real(low=0, high=1, name="won")]

    # We need to define what we are trying to optimise for
    # In our case it is accuracy (for example)
    @use_named_args(space)
    def objective(**params):
        type_weights = [params["lost"], params["empty"], params["inconclusive"], params["won"]]
        f1, _ = evaluate_weight_matrix_parallel(type_matrix, type_weights, y_true)

        return -f1

    x0_random = [[dim.rvs()[0] for dim in space] for _ in range(len(WinnerPredModel.running_bests))]  # random initializations to avoid local minima
    x0 = WinnerPredModel.running_bests + x0_random  # Plus the current bests

    result = gp_minimize(
        objective,
        space,
        x0=x0,
        n_calls=200,
        acq_func='LCB',
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

# Grid Search
def find_best_type_weights_grid(dataset: UtilsDataset.Dataset):
    type_matrix, y_true = calculate_dataset_info(dataset)

    # Now we can multiply this matrix by the type weights to get a prediction
    # Grid search with parallelization, file logging, and resumption capability
    n_jobs = -1
    batch_size = 10
    output_file = "dataset matches/grid_search_results.csv"
    checkpoint_file = "checkpoint.txt"

    lost_range = np.arange(-1.0, 0.3, 0.1)   # [-1, 0.2]
    other_range = np.arange(-0.2, 1.1, 0.1)  # [-0.2, 1]

    # Generate all combinations
    weight_combinations = list(itertools.product(lost_range, other_range, other_range, other_range))

    # Check if checkpoint exists to resume
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            last_processed_batch = int(f.read().strip())
    else:
        last_processed_batch = 0

    # Open CSV file for writing results
    with open(output_file, mode='a', newline='') as file:  # 'a' to append to the file
        writer = csv.writer(file)

        # If it's a new file, write the header
        if os.path.getsize(output_file) == 0:
            writer.writerow(["F1 Score", "Lost", "Empty", "Inconclusive", "Won"])

        # Process in batches
        for i in tqdm(range(last_processed_batch, len(weight_combinations), batch_size)):
            batch = weight_combinations[i:i + batch_size]

            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_weight_matrix_parallel)(type_matrix, weights, y_true) for weights in batch
            )

            # Save results and update best F1
            for f1, weights in results:
                # Write the F1 and weights to the CSV file
                writer.writerow([f1, *weights])

            # Save checkpoint after processing each batch
            with open(checkpoint_file, 'w') as f:
                f.write(str(i + batch_size))  # Save the next batch index to start from

    # Final result
    print(f"Results saved to {output_file}")

# Genetic Algorithm
def find_best_type_weights_ga(dataset: UtilsDataset.Dataset, generations=100, population_size=50, cx_prob=0.7,
                              mut_prob=0.2):

    # Prepare the dataset
    type_matrix, y_true = calculate_dataset_info(dataset)

    # GA setup
    WEIGHT_BOUNDS = [(-1, 1), (0, 1), (0, 1), (0, 1)]  # Bounds for each weight

    # Define fitness evaluation function
    def evaluate(individual):
        f1, _ = evaluate_weight_matrix_parallel(type_matrix, individual, y_true)
        return f1,  # Tuple is required for DEAP compatibility

    # DEAP initialization
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Random initialization within bounds
    for i, (low, high) in enumerate(WEIGHT_BOUNDS):
        toolbox.register(f"attr_{i}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_0, toolbox.attr_1, toolbox.attr_2, toolbox.attr_3), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate)

    # Initialize population
    population = toolbox.population(n=population_size)

    # Run the GA
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=cx_prob, mutpb=mut_prob,
        ngen=generations, stats=stats,
        verbose=True
    )

    # Find the best individual
    best_individual = tools.selBest(population, k=1)[0]
    best_weights = best_individual
    best_accuracy = best_individual.fitness.values[0]

    print(f"Best accuracy: {best_accuracy:.4f} with weights {best_weights}")

    # Plot GA Progress
    gen = logbook.select("gen")
    max_fitness = logbook.select("max")
    avg_fitness = logbook.select("avg")

    plt.figure(figsize=(12, 6))
    plt.plot(gen, max_fitness, label="Max Fitness")
    plt.plot(gen, avg_fitness, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (F1 Score)")
    plt.title("Genetic Algorithm Progress")
    plt.legend()
    UtilsPlot.save_plot(plt, Path("dataset matches/ga_optimisation.png"))

    return best_weights, best_accuracy

# ------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()

    dataset = UtilsDataset.EQUAL_UNDER
    # find_best_type_weights_gp(dataset)
    find_best_type_weights_grid(dataset)
    # find_best_type_weights_ga(dataset)

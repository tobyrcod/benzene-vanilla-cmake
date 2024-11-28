# TODO: turn into a notebook
import random

import numpy as np
import matplotlib.pyplot as plt

from tsetlin.utils import UtilsTM, UtilsTournament, UtilsHex, UtilsPlot
from pathlib import Path
from collections import Counter

tournaments_path = Path("../tournaments")
tournament_name = "6x6-4ply-simple-incomplete"
tournament_path = tournaments_path / tournament_name
plots_path = Path("plots")
templates_path = Path("../../templates")

def get_games_to_explore():
    dataset_path = tournament_path / "dataset.csv"


    # Load the winner prediction dataset into game states (X) and results (Y)
    X, Y = UtilsTM.load_winner_pred_dataset(
        dataset_path=dataset_path,
        augmentation=UtilsTM.Literals.Augmentation.AUG_NONE,
        history_type=UtilsTM.Literals.History.HISTORY_NONE,
        history_size=0
    )

    # Split the dataset into each game
    X_game = []
    Y_game = []
    for i in range(len(X)):
        # If we have reached a new empty board
        if not any(X[i]):
            # Start a new game
            X_game.append([])
            Y_game.append(int(Y[i]))
        # Append this game state to the current game
        X_game[-1].append(X[i])


    # Loader the tournament results file and check some basic game facts
    games, boardsize = UtilsTournament.load_tournament_games(tournament_path)
    assert len(games) == len(X_game) == len(Y_game)                                 #  We have the correct number of games and results
    assert all(Y_game[i] == games[i][0] for i in range(len(games)))                 #  The winner of each game is correct
    assert all(len(X_game[i]) == len(games[i][1]) + 1 for i in range(len(games)))   #  The length of each game is correct

    return X_game, Y_game

X_games, Y_games = get_games_to_explore()

def player_win_rates():
    title = f"Player Win Rates in {tournament_name} Mohex Selfplay"
    title_color = 'black'
    bg_colour = 'lightblue'

    win_counts = Counter(Y_games)
    wedge_labels = [0, 1]
    wedge_values = [win_counts[i] for i in wedge_labels]
    wedge_colors = ['white', 'black']
    wedge_text_color = ['black', 'white']

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
    win_rates_path = plots_path / "win_rates"
    filepath = win_rates_path / f"{tournament_name}_winrate.png"
    plt.savefig(filepath, dpi=300)  # Change filename and dpi as needed
    plt.close()  # Close the plot to free resources

def game_lengths():
    game_length = [len(game) for game in X_games]
    length_counts = Counter(game_length)

    bar_lengths = list(range(min(game_length), max(game_length)+1))
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
    ax.set_title(f"Game Lengths in {tournament_name} Mohex Selfplay", fontsize=14, fontweight="bold")
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
    game_lengths_path = plots_path / "game_lengths"
    filepath = game_lengths_path / f"{tournament_name}_games_length.png"
    plt.savefig(filepath, dpi=300)  # Change filename and dpi as needed
    plt.close()  # Close the plot to free resources

def templates_search():
    templates_path = Path("../../templates")
    UtilsHex.SearchPattern.add_from_templates_directory(templates_path)

    boardsizes = list(range(6, 14))
    boardsize = random.choice(boardsizes)
    template_names = UtilsHex.SearchPattern.get_pattern_names()
    template_name = random.choice(template_names)
    variations = UtilsHex.SearchPattern.get_pattern_variations(template_name)
    search_pattern = random.choice(variations)

    i = 0
    literals = None
    match = None
    while not match:
        literals = UtilsTM.Literals.make_random_board(boardsize)
        match = UtilsHex.SearchPattern.search_literals(search_pattern, literals, boardsize)
        i += 1

    player, position = match
    player = "black" if player == 0 else "white"
    search_file_name = f"{boardsize}x{boardsize}_{search_pattern}_at_{position}_for_{player}"
    template_search_path = plots_path / "templates" / template_name / "searches"
    template_search_path.mkdir(parents=True, exist_ok=True)

    UtilsPlot.plot_literals(literals, boardsize, template_search_path / f"{search_file_name}.png")
    print(search_pattern, match)

if __name__ == "__main__":
    player_win_rates()
    game_lengths()


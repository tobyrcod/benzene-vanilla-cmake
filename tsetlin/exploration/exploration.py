# TODO: turn into a notebook
import random

import numpy as np
import matplotlib.pyplot as plt

from tsetlin.utils import UtilsTM, UtilsTournament, UtilsHex, UtilsPlot, UtilsDataset
from pathlib import Path
from collections import Counter

tournaments_dir = Path("../tournaments")
plots_path = Path("plots")
templates_path = Path("../../templates")

UtilsDataset.load_raw_datasets(tournaments_dir)
DATASET: UtilsDataset.Dataset = UtilsDataset.COMBINED
DATASET = DATASET.oversample()

def player_win_rates(by_state: bool = True):
    title = f"Player Win Rates in {DATASET.name} Mohex Selfplay"
    if by_state:
        title += " States"
    title_color = 'black'
    bg_colour = 'lightblue'

    win_counts = Counter(DATASET.Y if by_state else DATASET.Y_game)
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
    filepath = win_rates_path / f"{DATASET.name}_winrate.png"
    plt.savefig(filepath, dpi=300)  # Change filename and dpi as needed
    plt.close()  # Close the plot to free resources

def game_lengths():
    game_length = [len(game) for game in DATASET.X_game]
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
    ax.set_title(f"Game Lengths in {DATASET.name} Mohex Selfplay", fontsize=14, fontweight="bold")
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
    filepath = game_lengths_path / f"{DATASET.name}_games_length.png"
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

# TODO: turn into a notebook

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle

from tsetlin.utils import UtilsTM, UtilsTournament, UtilsHex
from pathlib import Path
from collections import Counter

tournaments_path = Path("../tournaments")
tournament_name = "6x6-3ply-simple"
tournament_path = tournaments_path / tournament_name
plots_path = Path("plots")

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

def hex_plot(literals: list[int], boardsize: int):
    # Plot a hexagonal grid
    # Later to be used to visualise all good things Hex
    # Hexagonal Grid logic from: https://www.redblobgames.com/grids/hexagons/

    def get_cell_position(x: float, y: float) -> tuple[float, float]:
        pos_x = x * dx + (y * dx / 2)  # Add additional row offset as we go down
        pos_y = (boardsize-1-y) * dy   # Row 0 is at the top, so need to reverse y
        return pos_x, pos_y

    # Get the literals for each player
    literals_per_player = boardsize**2
    black_literals = literals[:literals_per_player]
    white_literals = literals[literals_per_player:]

    # Set up the hex board
    cell_color = 'lightyellow'
    piece_colors = ['black', 'white']
    grid_color = 'black'

    # Set up the figure
    hex_radius = 1
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define the grid properties
    dy = (3 / 2) * hex_radius
    dx = np.sqrt(3) * hex_radius

    # Plot each hex cell at the desired position
    for y in range(boardsize):
        for x in range(boardsize):
            position = get_cell_position(x, y)

            # Plot the grid
            hexagon = RegularPolygon(position, numVertices=6, radius=hex_radius,
                                     orientation=0, edgecolor=grid_color, facecolor=cell_color)
            ax.add_patch(hexagon)

            # Plot any piece
            i = UtilsHex.coordinates_2d_to_1d(x, y, boardsize)
            player = 0 if black_literals[i] else 1 if white_literals[i] else -1
            if player != -1:
                piece = Circle(position, radius=hex_radius * 0.6,
                               edgecolor=grid_color, facecolor=piece_colors[player])
                ax.add_patch(piece)

            # Write the hex position
            hex_pos_text = UtilsHex.index_to_position(i, boardsize)
            hex_pos_text_color = grid_color if player == -1 else piece_colors[1-player]
            ax.text(*position, hex_pos_text, ha='center', va='center', size=7, color=hex_pos_text_color)



    # Plot the axis labels for hex
    text_offset = hex_radius * 1.4
    for i in range(boardsize):
        # Along the bottom
        position = get_cell_position(i, boardsize-1)
        string = UtilsHex._number_to_string(i+1)
        ax.text(position[0], position[1] - text_offset, string.upper(), ha='center', va='center', size=10)

        # Along the top
        position = get_cell_position(i, 0)
        string = UtilsHex._number_to_string(i+1)
        ax.text(position[0], position[1] + text_offset, string.upper(), ha='center', va='center', size=10)

        # Along the left
        string = str(i+1)
        ax.text(*get_cell_position(-1, i), string.upper(), ha='center', va='center', size=10)

        # Along the right
        string = str(i+1)
        ax.text(*get_cell_position(boardsize, i), string.upper(), ha='center', va='center', size=10)


    # Calculate the width and height of the grid
    grid_width = (boardsize + 1) * dx + boardsize * (dx / 2)
    grid_height = (boardsize + 1) * dy


    # Crop the plot to only the area used for the hex board
    grid_center = get_cell_position((boardsize-1)/2, (boardsize-1)/2)
    ax.set_xlim(grid_center[0] - grid_width / 2, grid_center[0] + grid_width / 2)
    ax.set_ylim(grid_center[1] - grid_height / 2, grid_center[1] + grid_height / 2)
    ax.set_aspect('equal')
    plt.axis('off')


    # Save the plot to a file
    hex_path = plots_path / "hex"
    filepath = hex_path / f"hex.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')  # Change filename and dpi as needed
    plt.close()  # Close the plot to free resources


boardsize = 6
hex_plot(UtilsTM.Literals.make_random_board(boardsize), boardsize)

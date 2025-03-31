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

def dataset_occurrence(dataset: UtilsDataset.Dataset):
    # Make a heatmap for the occurrences of each color in every position they won and lost

    boardsize = dataset.boardsize
    literals_per_player = boardsize**2
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)

    dataset_black_win = [dataset.X[i] for i in range(dataset.num_rows) if dataset.Y[i] == 0]
    dataset_white_win = [dataset.X[i] for i in range(dataset.num_rows) if dataset.Y[i] == 1]

    literals_black = np.array([literals[:literals_per_player] for literals in dataset.X])
    literals_white = np.array([literals[:literals_per_player] for literals in dataset.X])

    literals_black_win_black = np.array([literals[:literals_per_player] for literals in dataset_black_win])
    literals_white_win_white = np.array([literals[literals_per_player:] for literals in dataset_white_win])

    literals_black_win_white = np.array([literals[:literals_per_player] for literals in dataset_white_win])
    literals_white_win_black = np.array([literals[literals_per_player:] for literals in dataset_black_win])

    def process_player_dataset(literals_player, name, is_relative):
        player_literals_counts = np.sum(literals_player, axis=0)
        player_literals_norm = player_literals_counts.astype(float) / len(literals_player)
        player_heatmap = list(player_literals_norm.reshape(boardsize, boardsize))

        name_heatmap = f"{dataset.name}_{name}_heatmap.png"
        path_heatmap = dir_heatmaps / dataset.name / name_heatmap
        UtilsHex.HexGrid.print_heatmap(player_heatmap)
        UtilsPlot._plot_hex_grid(hexgrid, path_heatmap, heatmap=player_heatmap, is_heatmap_relative=is_relative)

    process_player_dataset(literals_black, 'black', True)
    process_player_dataset(literals_white, 'white', True)
    process_player_dataset(literals_black_win_black, 'black_win_black', True)
    process_player_dataset(literals_white_win_white, 'white_win_white', True)
    process_player_dataset(literals_black_win_white, 'black_win_white', True)
    process_player_dataset(literals_white_win_black, 'white_win_black', True)


if __name__ == '__main__':
    UtilsDataset.load_raw_datasets()
    UtilsHex.SearchPattern.initialise()

    example_random()
    example_short_diagonal()

    dataset_occurrence(UtilsDataset.X6_BASELINE)

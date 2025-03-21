# Quick script to mess about with defining heatmaps to be displayed
from pathlib import Path

def main():
    boardsize = 6
    hexgrid = UtilsHex.HexGrid.make_empty_hexgrid(boardsize)
    heatmap = UtilsHex.HexGrid.make_empty_heatmap(boardsize, default=0.25)

    # Define the 'short diagonal'
    for row in range(boardsize):
        heatmap[row][boardsize-1-row] = 1.00

    # Define a 'random' heatmap
    # for row in range(boardsize):
    #     for col in range(boardsize):
    #         heatmap[row][col] = random.uniform(0, 1)

    UtilsHex.HexGrid.print_heatmap(heatmap)
    UtilsPlot._plot_hex_grid(hexgrid, Path("heatmap short diagonal.png"), heatmap=heatmap)

if __name__ == '__main__':
    from utils import *
    # UtilsDataset.load_raw_datasets()
    # UtilsHex.SearchPattern.initialise()

    main()

from tsetlin.utils import UtilsTournament
from pathlib import Path

from_tournament_path = Path("6/6x6-blunder1")
to_dataset_path = from_tournament_path / "dataset.csv"

# Loader the tournament results file
games, boardsize = UtilsTournament.load_tournament_games(from_tournament_path)
print(games[0])

# Convert it to a winner prediction dataset
UtilsTournament.games_to_winner_prediction_dataset(games, boardsize, to_dataset_path)

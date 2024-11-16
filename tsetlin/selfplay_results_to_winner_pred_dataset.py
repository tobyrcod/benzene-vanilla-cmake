from utils import UtilsTournament
from pathlib import Path

test_path = Path("tournaments")
from_tournament_path = test_path / "6x6-1ply-simple"
to_dataset_path = from_tournament_path / "dataset.csv"

# Loader the tournament results file
games, boardsize = UtilsTournament.load_tournament_games(from_tournament_path)

# Convert it to a winner prediction dataset
UtilsTournament.games_to_winner_prediction_dataset(games, boardsize, to_dataset_path)

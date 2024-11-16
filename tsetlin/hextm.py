from utils import UtilsTM
from pathlib import Path

tournaments_path = Path("tournaments")
tournament_path = tournaments_path / "6x6-2ply-simple"
dataset_path = tournament_path / "dataset.csv"

dataset = UtilsTM.load_dataset(dataset_path=dataset_path,
                               augmentation=UtilsTM.Literals.Augmentation.AUG_NONE,
                               history_type=UtilsTM.Literals.History.HISTORY_NONE,
                               history_size=0)

print(len(dataset))
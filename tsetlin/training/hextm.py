import numpy as np

from tsetlin.utils import UtilsTM
from pathlib import Path
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


# Define which dataset we want to use
tournaments_path = Path("../tournaments")
tournament_path = tournaments_path / "6x6-2ply-simple"
dataset_path = tournament_path / "dataset.csv"


# Load the dataset into game states (X) and results (Y)
X, Y = UtilsTM.load_winner_pred_dataset(
    dataset_path=dataset_path,
    augmentation=UtilsTM.Literals.Augmentation.AUG_NONE,
    history_type=UtilsTM.Literals.History.HISTORY_NONE,
    history_size=0
)
print(f"X: {X.shape}")
print(f"Y: {Y.shape}")


# Define parameter settings
# from winner prediction paper: Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
tm_number_of_clauses = 10_000
tm_T = 8000
tm_s = 100
tm_train_test_split = 0.67
tm_epochs = 200


# Simple for now splitting of dataset into train and test sets
split = int(len(X) * tm_train_test_split)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Y_train: {Y_train.shape}, Y_test: {Y_test.shape}")


# Create the TM with these settings
tm = MultiClassTsetlinMachine(
    number_of_clauses=tm_number_of_clauses,
    T=tm_T,
    s=tm_s,
    boost_true_positive_feedback=1,
    number_of_state_bits=8,
    indexed=True,
    append_negated= True,
    weighted_clauses=False,
    s_range=False,
    clause_drop_p=0.0,
    literal_drop_p=0.0,
    max_included_literals=None
)


# Train the TM
tm.fit(
    X=X_train,
    Y=Y_train,
    epochs=1,
    incremental=False
)


# See how well the TM performed
print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())

state_1 = [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1]
state_0 = [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
print(len(state_1), len(state_0))
print(tm.predict(np.array([state_1, state_0])))
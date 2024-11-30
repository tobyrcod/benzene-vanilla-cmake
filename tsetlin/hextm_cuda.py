import pickle
from utils import UtilsTM, UtilsDataset
from pathlib import Path
from PyTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
from sklearn.model_selection import train_test_split

# TODO: Use TMU?? to hopefully combine the benefits of regular TM with CUDA
# TODO: hyperparam tune
# TODO: kfold
# TODO: accuracy per class

# Define parameter settings
# from winner prediction paper: Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
tm_number_of_clauses = 10000
tm_T = 8000
tm_s = 100
tm_max_weight = 255
tm_train_test_split = 0.67
tm_epochs = 200


# Create the TM with these settings
tm = MultiClassTsetlinMachine(
    number_of_clauses=tm_number_of_clauses,
    T=tm_T,
    s=tm_s,
    # boost_true_positive_feedback=1,
    # number_of_state_bits=8,
    # append_negated=True,
    max_weight=tm_max_weight,
)


# Define which dataset we want to use
UtilsDataset.load_raw_datasets(Path("tournaments"))
DATASET: UtilsDataset.Dataset = UtilsDataset.BASELINE
# DATASET = DATASET.undersample()

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(DATASET.X, DATASET.Y,
                                                    stratify=DATASET.Y, random_state=42,
                                                    test_size=tm_train_test_split)

# Get the original accuracy
tm.fit(X_train[0:1], Y_train[0:1], epochs=1, incremental=True)
acc_train = 100 * (tm.predict(X_train) == Y_train).mean()
acc_test = 100 * (tm.predict(X_test) == Y_test).mean()
print(f"Epoch: {0}, Acc Train: {acc_train:.4f}, Acc Test: {acc_test:.4f}")

# Train the TM
for i in range(tm_epochs):
    tm.fit(
        X=X_train,
        Y=Y_train,
        epochs=1,
        incremental=True
    )

    # See how well the TM performed
    acc_train = 100 * (tm.predict(X_train) == Y_train).mean()
    acc_test = 100 * (tm.predict(X_test) == Y_test).mean()
    print(f"Epoch: {i+1}, Acc Train: {acc_train:.4f}, Acc Test: {acc_test:.4f}")

# Save the TM
with open(f'{DATASET.name}_model.pkl', 'wb') as f:
    pickle.dump(tm, f)
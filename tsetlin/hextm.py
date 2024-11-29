from utils import UtilsTM, UtilsDataset
from pathlib import Path
from pyTsetlinMachine.tm import MultiClassTsetlinMachine


# Define parameter settings
# from winner prediction paper: Logic-based AI for Interpretable Board Game Winner Prediction with Tsetlin Machine
tm_number_of_clauses = 10_000
tm_T = 8000
tm_s = 100
tm_train_test_split = 0.67
tm_epochs = 200


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


# Define which dataset we want to use
UtilsDataset.load_raw_datasets(Path("tournaments"))
DATASET: UtilsDataset.Dataset = UtilsDataset.COMBINED

# Train the TM
for i in range(5):
    ds = DATASET.oversample()

    tm.fit(
        X=ds.X,
        Y=ds.Y,
        epochs=1,
        incremental=True
    )

    # See how well the TM performed
    print("Accuracy:", 100 * (tm.predict(ds.X) == ds.Y).mean())

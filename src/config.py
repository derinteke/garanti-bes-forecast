"""Project configuration: paths, columns, validation months and model settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
TRAIN_CSV = RAW_DIR / "train.csv"
TEST_CSV = RAW_DIR / "test_input.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SUBMISSIONS_DIR = ROOT / "submissions"

TARGET = "ADDCONTAMNT"   # BES additional contribution amount in the snapshot month
ID = "CUSTNBR"
MONTH = "month"

# Train covers month-end snapshots Mar-Nov 2018; the test is Dec 2018.
# Every model is judged by rolling-origin validation: to score month m, it is
# trained only on months < m. Decisions (features, objective, params) were made
# on the DEV months; November is a holdout that played no part in any choice.
DEV_MONTHS = [6, 7, 8, 9, 10]
HOLDOUT_MONTH = 11

RANDOM_STATE = 42

# Tweedie handles "90% zeros + a long right tail" natively. The variance power
# (1.8) was picked from {1.2, 1.5, 1.8} by pooled DEV RMSE; November played no part.
LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.8,
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
}

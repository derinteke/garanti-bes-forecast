"""Load the competition CSVs into one consistent shape.

Both files become: ``CUSTNBR``, ``month`` (1-12), the raw feature columns and,
for train, the target. ``CONTPAIDAMNT07`` is dropped because the test file
does not have it.
"""
from __future__ import annotations

import pandas as pd

from . import config

DROP_COLS = ["CONTPAIDAMNT07"]


def _check(path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download train.csv and test_input.csv from the "
            "Kaggle competition 'garanti-bbva-data-day-case-study' into data/raw/."
        )


def _tidy(df: pd.DataFrame, date_format: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["TRAN_DATE"], format=date_format)
    df = df.drop(columns=["TRAN_DATE"] + [c for c in DROP_COLS if c in df.columns])
    df.insert(1, config.MONTH, dates.dt.month)
    return df


def load_train() -> pd.DataFrame:
    _check(config.TRAIN_CSV)
    return _tidy(pd.read_csv(config.TRAIN_CSV), "%Y-%m-%d")


def load_test() -> pd.DataFrame:
    _check(config.TEST_CSV)
    return _tidy(pd.read_csv(config.TEST_CSV), "%d.%m.%Y")


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """The anonymised columns as delivered (no id, month or target)."""
    return [c for c in df.columns if c not in (config.ID, config.MONTH, config.TARGET)]

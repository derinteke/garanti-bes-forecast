"""Feature engineering for the LightGBM model.

Two groups on top of the raw columns (NaNs are left as NaN for LightGBM):

* contribution features - summaries of the 11 monthly ``CONTPAIDAMNT`` columns;
* customer history      - what this customer did in *earlier* snapshots. This
                          is my 2024 "override" idea, turned into features so
                          the model decides how much to trust it.

``month`` is not a model feature: December never appears in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import raw_feature_columns

HISTORY_COLS = ["h_n", "h_pos_n", "h_last_y", "h_last_gap", "h_last_pos_y", "h_mean_y", "h_max_y"]


def add_contribution_features(df: pd.DataFrame) -> pd.DataFrame:
    paid_cols = [c for c in df.columns if c.startswith("CONTPAIDAMNT")]
    paid = df[paid_cols]
    out = df.copy()
    out["paid_mean"] = paid.mean(axis=1)
    out["paid_std"] = paid.std(axis=1)
    out["paid_max"] = paid.max(axis=1)
    out["paid_nonzero_months"] = (paid > 0).sum(axis=1).where(paid.notna().any(axis=1))
    out["paid_first3_minus_last3"] = paid[paid_cols[:3]].mean(axis=1) - paid[paid_cols[-3:]].mean(axis=1)
    out["paid_to_plan"] = out["paid_mean"] / out["CONTMONTHAMNT"]
    out["n_missing"] = df[raw_feature_columns(df)].isna().sum(axis=1)
    return out


def add_customer_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Per row, summarise the same customer's rows from strictly earlier months.

    ``panel`` holds labelled history rows plus the rows to predict (target NaN),
    with at most one row per customer and month. A row never sees its own
    target or anything later.
    """
    p = panel.reset_index(drop=True)
    p = p.loc[p.sort_values([config.ID, config.MONTH]).index].copy()
    key = p[config.ID]
    y = p[config.TARGET]
    y0 = y.fillna(0)
    pos0 = (y > 0).astype(float)
    by = p.groupby(key)

    p["h_n"] = by.cumcount()
    p["h_pos_n"] = pos0.groupby(key).cumsum() - pos0
    p["h_last_y"] = by[config.TARGET].shift(1)
    p["h_last_gap"] = p[config.MONTH] - by[config.MONTH].shift(1)
    p["h_last_pos_y"] = y.where(y > 0).groupby(key).shift(1).groupby(key).ffill()
    p["h_mean_y"] = ((y0.groupby(key).cumsum() - y0) / p["h_n"]).where(p["h_n"] > 0)
    p["h_max_y"] = y0.groupby(key).cummax().groupby(key).shift(1)
    return p.sort_index()


def build_matrices(train: pd.DataFrame, target: pd.DataFrame, history: bool = True,
                   contribution: bool = True):
    """Return (X_train, y_train, X_target) with identical columns."""
    panel = pd.concat([train, target.assign(**{config.TARGET: np.nan})], ignore_index=True)
    if history:
        panel = add_customer_history(panel)
    if contribution:
        panel = add_contribution_features(panel)
    feats = [c for c in panel.columns if c not in (config.ID, config.MONTH, config.TARGET)]
    X_train, X_target = panel.iloc[: len(train)][feats], panel.iloc[len(train):][feats]
    return X_train, train[config.TARGET].to_numpy(), X_target

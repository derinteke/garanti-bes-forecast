"""Rolling-origin validation: to score month m, train only on months < m.

This mirrors the real task (history up to November -> predict December) and is
the check my 2024 notebook was missing.
"""
from __future__ import annotations

from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config

Pipeline = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def rmse(y, pred) -> float:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def rolling_folds(df: pd.DataFrame, months: Iterable[int]):
    for m in months:
        yield m, df[df[config.MONTH] < m], df[df[config.MONTH] == m]


def collect_predictions(pipeline: Pipeline, df: pd.DataFrame, months: Iterable[int]) -> dict:
    """{month: (y_true, y_pred)} for every validation month."""
    return {m: (val[config.TARGET].to_numpy(), pipeline(train, val))
            for m, train, val in rolling_folds(df, months)}


def summarise(preds: dict, baseline: dict, dev_months, holdout_month) -> dict:
    """Pooled RMSE over the dev months, the holdout RMSE, and both relative to
    the mean baseline. Also: in how many dev months it beats the baseline, and
    the ROC-AUC of "who contributes at all" (a check that isn't ruled by a
    handful of huge contributions)."""
    y_dev = np.concatenate([preds[m][0] for m in dev_months])
    p_dev = np.concatenate([preds[m][1] for m in dev_months])
    b_dev = np.concatenate([baseline[m][1] for m in dev_months])
    y_ho, p_ho = preds[holdout_month]
    b_ho = baseline[holdout_month][1]

    dev, dev_base = rmse(y_dev, p_dev), rmse(y_dev, b_dev)
    ho, ho_base = rmse(y_ho, p_ho), rmse(y_ho, b_ho)
    auc = roc_auc_score(y_dev > 0, p_dev) if np.ptp(p_dev) > 0 else 0.5
    return {
        "dev_rmse": dev,
        "dev_vs_mean_pct": 100 * (dev / dev_base - 1),
        "dev_months_beating_mean": int(sum(rmse(*preds[m]) < rmse(*baseline[m]) for m in dev_months)),
        "holdout_rmse": ho,
        "holdout_vs_mean_pct": 100 * (ho / ho_base - 1),
        "dev_auc_contributes": float(auc),
    }

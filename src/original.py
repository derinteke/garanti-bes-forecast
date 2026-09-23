"""My May 2024 competition notebook, replayed as a function.

Defaults reproduce the notebook exactly, bugs included; each bug has a switch
so its effect can be measured on its own:

* ``scaler_bug``    - the target month was scaled with ``scaler.fit_transform``,
                      i.e. with *its own* mean/std instead of the training ones.
* ``last_fold_bug`` - predictions came from whichever model the 10-fold CV loop
                      fitted last (90% of the rows), never refit on everything.
* ``override``      - customers with a positive ``ADDCONTAMNT`` somewhere in
                      the history got their model prediction replaced by their
                      most recent *positive* value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from . import config
from .data import raw_feature_columns


def last_positive_contribution(history: pd.DataFrame) -> pd.Series:
    """Per customer: ADDCONTAMNT from the latest month where it was > 0."""
    pos = history[history[config.TARGET] > 0].sort_values(config.MONTH)
    return pos.groupby(config.ID)[config.TARGET].last()


def original_pipeline(
    train: pd.DataFrame,
    target: pd.DataFrame,
    *,
    scaler_bug: bool = True,
    last_fold_bug: bool = True,
    override: bool = True,
    drop: tuple[str, ...] = (),
) -> np.ndarray:
    """Fit on ``train`` and predict ``target`` exactly as the 2024 notebook did."""
    cols = [c for c in raw_feature_columns(train) + [config.MONTH] if c not in drop]
    X = train[cols].fillna(0).to_numpy()
    y = train[config.TARGET].to_numpy()
    X_target = target[cols].fillna(0).to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    if last_fold_bug:
        for fold_train, _ in KFold(10, shuffle=True, random_state=config.RANDOM_STATE).split(X_scaled):
            model.fit(X_scaled[fold_train], y[fold_train])
    else:
        model.fit(X_scaled, y)

    X_target_scaled = (StandardScaler().fit_transform(X_target) if scaler_bug
                       else scaler.transform(X_target))
    pred = np.clip(model.predict(X_target_scaled), 0, None)

    if override:
        last_pos = target[config.ID].map(last_positive_contribution(train)).to_numpy()
        pred = np.where(np.isnan(last_pos), pred, last_pos)
    return pred


def original_kfold_score(train: pd.DataFrame, groups: pd.Series | None = None) -> dict:
    """The CV number the notebook printed (mean of 10 fold RMSEs), next to the
    same CV applied to a predict-the-mean baseline - the comparison I never ran.

    With ``groups`` the folds are grouped by customer instead of shuffled rows.
    """
    from sklearn.model_selection import GroupKFold

    cols = raw_feature_columns(train) + [config.MONTH]
    X = StandardScaler().fit_transform(train[cols].fillna(0).to_numpy())
    y = train[config.TARGET].to_numpy()
    if groups is None:
        splits = KFold(10, shuffle=True, random_state=config.RANDOM_STATE).split(X)
    else:
        splits = GroupKFold(10).split(X, y, groups)

    model_rmse, mean_rmse = [], []
    for tr_idx, va_idx in splits:
        pred = LinearRegression().fit(X[tr_idx], y[tr_idx]).predict(X[va_idx])
        model_rmse.append(np.sqrt(np.mean((y[va_idx] - pred) ** 2)))
        mean_rmse.append(np.sqrt(np.mean((y[va_idx] - y[tr_idx].mean()) ** 2)))
    return {"model": float(np.mean(model_rmse)), "mean_baseline": float(np.mean(mean_rmse))}

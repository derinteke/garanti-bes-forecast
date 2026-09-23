"""The yardstick I skipped in 2024, and the model that replaced linear regression."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config
from .features import build_matrices


def mean_baseline(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Predict the training mean for everyone. Under RMSE this is the number to beat."""
    return np.full(len(target), train[config.TARGET].mean())


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, params: dict | None = None) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1,
                              **(params or config.LGBM_PARAMS))
    return model.fit(X, y)


def lgbm_pipeline(train: pd.DataFrame, target: pd.DataFrame, *, history: bool = True,
                  contribution: bool = True, params: dict | None = None,
                  return_model: bool = False):
    X_train, y_train, X_target = build_matrices(train, target, history, contribution)
    model = fit_lgbm(X_train, y_train, params)
    pred = np.clip(model.predict(X_target), 0, None)
    return (pred, model) if return_model else pred

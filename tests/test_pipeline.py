"""Tests that pin down the claims the README makes.

EN: Built on small synthetic frames, so they run without the competition data.
    They check that history features only look backwards, that validation folds
    are chronological, and that the 2024 bugs really behave as described.
TR: Küçük sentetik tablolarla çalışıyor, yarışma verisi gerekmiyor. Geçmiş
    özelliklerinin yalnızca geriye baktığını, doğrulama katlarının kronolojik
    olduğunu ve 2024 bug'larının README'de anlatıldığı gibi davrandığını doğruluyor.

Run / Koşum:  pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.features import add_customer_history, build_matrices
from src.model import lgbm_pipeline
from src.original import last_positive_contribution, original_pipeline
from src.validation import rolling_folds

T, ID, M = config.TARGET, config.ID, config.MONTH


def make_frame(n_customers=300, months=range(3, 12), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_customers):
        for m in rng.choice(list(months), size=rng.integers(1, 4), replace=False):
            x = rng.normal(100, 20)
            rows.append({ID: float(c), M: int(m), T: max(0.0, 5 * x - 450 + rng.normal(0, 30)),
                         "CONTPAIDAMNT00": rng.choice([0.0, 300.0, np.nan]),
                         "CONTPAIDAMNT01": rng.choice([0.0, 340.0]),
                         "CONTMONTHAMNT": 340.0, "FEATURE_X": x})
    return pd.DataFrame(rows)


def one_customer():
    return pd.DataFrame({ID: [7.0] * 4, M: [3, 5, 7, 9], T: [500.0, 0.0, 1200.0, np.nan],
                         "FEATURE_X": [1.0, 2.0, 3.0, 4.0]})


def test_history_features_summarise_earlier_rows_only():
    h = add_customer_history(one_customer())
    np.testing.assert_array_equal(h["h_n"], [0, 1, 2, 3])
    np.testing.assert_array_equal(h["h_pos_n"], [0, 1, 1, 2])
    np.testing.assert_array_equal(h["h_last_y"], [np.nan, 500, 0, 1200])
    np.testing.assert_array_equal(h["h_last_pos_y"], [np.nan, 500, 500, 1200])
    np.testing.assert_array_equal(h["h_last_gap"], [np.nan, 2, 2, 2])
    np.testing.assert_allclose(h["h_mean_y"], [np.nan, 500, 250, 1700 / 3])
    np.testing.assert_array_equal(h["h_max_y"], [np.nan, 500, 500, 1200])


def test_history_features_do_not_move_when_the_future_changes():
    base = add_customer_history(one_customer())
    changed = one_customer()
    changed.loc[2, T] = 99_999.0          # month 7
    after = add_customer_history(changed)
    cols = ["h_n", "h_pos_n", "h_last_y", "h_last_pos_y", "h_mean_y", "h_max_y"]
    pd.testing.assert_frame_equal(base.loc[:2, cols], after.loc[:2, cols])


def test_history_keeps_input_row_order():
    df = make_frame(50).sample(frac=1, random_state=1).reset_index(drop=True)
    out = add_customer_history(df)
    pd.testing.assert_frame_equal(out[df.columns], df)


def test_rolling_folds_train_strictly_before_validation():
    df = make_frame()
    for m, train, val in rolling_folds(df, [6, 9, 11]):
        assert train[M].max() < m
        assert set(val[M]) == {m}


def test_build_matrices_align_and_exclude_id_month_target():
    df = make_frame()
    train, target = df[df[M] < 11], df[df[M] == 11]
    X_train, y_train, X_target = build_matrices(train, target)
    assert list(X_train.columns) == list(X_target.columns)
    assert not {ID, M, T} & set(X_train.columns)
    assert len(X_train) == len(y_train) == len(train) and len(X_target) == len(target)


def test_scaler_bug_breaks_a_model_that_is_otherwise_right():
    """Target month has a shifted feature distribution. Scaling it with its own
    mean/std (the 2024 bug) erases the shift; the train scaler keeps it."""
    rng = np.random.default_rng(3)
    x_tr, x_te = rng.normal(100, 10, 2000), rng.normal(130, 10, 500)
    train = pd.DataFrame({ID: np.arange(2000.0), M: 5, T: 5 * x_tr, "FEATURE_X": x_tr})
    target = pd.DataFrame({ID: np.arange(2000.0, 2500.0), M: 6, T: 5 * x_te, "FEATURE_X": x_te})

    fixed = original_pipeline(train, target, scaler_bug=False, last_fold_bug=False, override=False)
    buggy = original_pipeline(train, target, scaler_bug=True, last_fold_bug=False, override=False)
    np.testing.assert_allclose(fixed, 5 * x_te, rtol=1e-6)
    assert abs(buggy.mean() - 5 * x_te.mean()) > 100


def test_override_takes_last_positive_value_even_after_a_zero_month():
    history = pd.DataFrame({ID: [1.0, 1.0, 2.0], M: [3, 5, 4], T: [900.0, 0.0, 0.0]})
    lp = last_positive_contribution(history)
    assert lp.to_dict() == {1.0: 900.0}


def test_lgbm_predictions_are_non_negative():
    df = make_frame()
    train, target = df[df[M] < 11], df[df[M] == 11]
    params = dict(config.LGBM_PARAMS, n_estimators=20, min_child_samples=5)
    pred = lgbm_pipeline(train, target, params=params)
    assert pred.shape == (len(target),)
    assert (pred >= 0).all()

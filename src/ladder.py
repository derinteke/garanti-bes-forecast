"""Every version of the model, from what I submitted in 2024 to the fixed one.

Each step changes one thing, so its effect can be read off the results table.
``CHECKS`` are side experiments that answer "was that idea right?" rather than
steps on the way to the final model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from .model import lgbm_pipeline, mean_baseline
from .original import original_pipeline


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    pipeline: object


STEPS = [
    Step("0_mean", "Baseline: predict the training mean", mean_baseline),
    Step("1_as_submitted", "My 2024 notebook, replayed exactly", original_pipeline),
    Step("2_fix_scaler", "+ fix: scale the target month with the TRAIN scaler",
         partial(original_pipeline, scaler_bug=False)),
    Step("3_fix_refit", "+ fix: refit on all rows, not the last CV fold",
         partial(original_pipeline, scaler_bug=False, last_fold_bug=False)),
    Step("4_lgbm_tweedie", "LightGBM (Tweedie), raw features, no override",
         partial(lgbm_pipeline, history=False, contribution=False)),
    Step("5_contribution", "+ contribution-history features (final)",
         partial(lgbm_pipeline, history=False, contribution=True)),
]

# The final model is the step with the lowest pooled dev RMSE; the holdout
# month plays no part in that choice.
CHECKS = [
    Step("check_customer_history", "Final + customer-history features (the override idea as features)",
         partial(lgbm_pipeline, history=True, contribution=True)),
    Step("check_no_override", "2024 notebook (bugs fixed) without the override",
         partial(original_pipeline, scaler_bug=False, last_fold_bug=False, override=False)),
    Step("check_drop_rtrndesvamnt", "2024 notebook (bugs fixed) without RTRNDESVAMNT",
         partial(original_pipeline, scaler_bug=False, last_fold_bug=False, drop=("RTRNDESVAMNT",))),
]

FINAL = STEPS[-1]

"""End-to-end: replay the 2024 notebook, fix it step by step, score every step
with the same rolling validation, then fit the final model and write a submission.

Usage: python train.py
"""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from src import config, figures
from src.data import load_test, load_train
from src.ladder import CHECKS, FINAL, STEPS
from src.model import mean_baseline
from src.original import original_kfold_score, original_pipeline
from src.validation import collect_predictions, summarise

MONTHS = config.DEV_MONTHS + [config.HOLDOUT_MONTH]


def score(steps, df, baseline):
    rows = []
    for step in steps:
        t0 = time.time()
        preds = collect_predictions(step.pipeline, df, MONTHS)
        row = {"key": step.key, "label": step.label,
               **summarise(preds, baseline, config.DEV_MONTHS, config.HOLDOUT_MONTH)}
        rows.append(row)
        print(f"  {step.label:<58} dev {row['dev_rmse']:8.1f} ({row['dev_vs_mean_pct']:+5.1f}%) | "
              f"Nov {row['holdout_rmse']:8.1f} ({row['holdout_vs_mean_pct']:+5.1f}%) | "
              f"AUC {row['dev_auc_contributes']:.3f} | {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    for d in (config.MODELS_DIR, config.FIGURES_DIR, config.SUBMISSIONS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train, test = load_train(), load_test()
    print(f"Train: {len(train):,} rows | {train[config.ID].nunique():,} customers | "
          f"months {train[config.MONTH].min()}-{train[config.MONTH].max()}")
    print(f"Test:  {len(test):,} customers (month {test[config.MONTH].iloc[0]}), "
          f"{test[config.ID].isin(train[config.ID]).sum():,} of them seen in train\n")

    baseline = collect_predictions(mean_baseline, train, MONTHS)

    print("Ladder (rolling validation, train on months < m):")
    results = score(STEPS, train, baseline)
    print("\nChecks:")
    checks = score(CHECKS, train, baseline)

    print("\nHow the 2024 CV fooled me:")
    kfold = original_kfold_score(train)
    grouped = original_kfold_score(train, groups=train[config.ID])
    kfold_pct = 100 * (kfold["model"] / kfold["mean_baseline"] - 1)
    grouped_pct = 100 * (grouped["model"] / grouped["mean_baseline"] - 1)
    rolling_pct = float(results.loc[results["key"] == "1_as_submitted", "dev_vs_mean_pct"].iloc[0])
    print(f"  random 10-fold : model {kfold['model']:.1f} vs mean {kfold['mean_baseline']:.1f} ({kfold_pct:+.1f}%)")
    print(f"  grouped 10-fold: model {grouped['model']:.1f} vs mean {grouped['mean_baseline']:.1f} ({grouped_pct:+.1f}%)")
    print(f"  rolling months : {rolling_pct:+.1f}%")

    # Final model: all labelled months -> December.
    final_pred, model = FINAL.pipeline(train, test, return_model=True)
    replay_pred = original_pipeline(train, test)
    pd.DataFrame({"Id": test[config.ID], "Predicted": final_pred}).to_csv(
        config.SUBMISSIONS_DIR / "submission_final.csv", index=False)
    pd.DataFrame({"Id": test[config.ID], "Predicted": replay_pred}).to_csv(
        config.SUBMISSIONS_DIR / "submission_2024_replay.csv", index=False)
    joblib.dump(model, config.MODELS_DIR / "final_lgbm.joblib")

    # Figures + reports
    figures.ladder(results)
    whales = figures.whale_share(train)
    figures.cv_illusion(kfold_pct, grouped_pct, rolling_pct)
    rates, counts = figures.repeat_signal(train)
    importance = figures.feature_importance(model, model.feature_name_)

    results.to_csv(config.REPORTS_DIR / "results.csv", index=False, float_format="%.3f")
    checks.to_csv(config.REPORTS_DIR / "checks.csv", index=False, float_format="%.3f")
    metrics = {
        "validation": {"dev_months": config.DEV_MONTHS, "holdout_month": config.HOLDOUT_MONTH},
        "ladder": results.set_index("key").drop(columns="label").round(3).to_dict("index"),
        "checks": checks.set_index("key").drop(columns="label").round(3).to_dict("index"),
        "cv_illusion_pct_vs_mean": {"random_kfold": round(kfold_pct, 2),
                                    "grouped_kfold": round(grouped_pct, 2),
                                    "rolling_months": round(rolling_pct, 2)},
        "original_kfold_rmse": round(kfold["model"], 1),
        "top10_share_of_squared_error_by_month": whales.round(3).to_dict(),
        "p_contributes_by_previous_snapshot_pct": {k.replace("\n", " "): round(v, 1) for k, v in rates.items()},
        "final_model": {"step": FINAL.key, "params": config.LGBM_PARAMS,
                        "top_features_gain_pct": importance.sort_values(ascending=False).round(2).to_dict()},
        "test_prediction_mean": {"final": round(float(final_pred.mean()), 1),
                                 "replay_2024": round(float(replay_pred.mean()), 1),
                                 "train_target_mean": round(float(train[config.TARGET].mean()), 1)},
    }
    with open(config.MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    final = results.iloc[-1]
    print(f"\nFinal ({FINAL.label}): dev {final['dev_rmse']:.1f} ({final['dev_vs_mean_pct']:+.1f}% vs mean), "
          f"Nov {final['holdout_rmse']:.1f} ({final['holdout_vs_mean_pct']:+.1f}% vs mean)")
    print(f"Wrote {config.SUBMISSIONS_DIR / 'submission_final.csv'} and figures in {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()

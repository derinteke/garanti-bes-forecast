"""Figures for the README. Static PNGs on a light surface."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"     # categorical slot 1: the rebuild
ORANGE = "#eb6834"   # categorical slot 2: the 2024 notebook

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "xtick.color": MUTED,
    "ytick.color": INK_2,
    "savefig.facecolor": SURFACE,
    "savefig.dpi": 150,
})


def _style(ax, grid_axis="x"):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def ladder(results: pd.DataFrame):
    """% RMSE vs the mean baseline for every step, dev months and holdout side by side."""
    steps = results[results["key"] != "0_mean"].iloc[::-1]
    colors = [ORANGE if k[0] in "123" else BLUE for k in steps["key"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, col, title in [
        (axes[0], "dev_vs_mean_pct", "Dev months (Jun-Oct, pooled)"),
        (axes[1], "holdout_vs_mean_pct", "November holdout (no choice made on it)"),
    ]:
        vals = steps[col].to_numpy()
        y = np.arange(len(steps))
        ax.barh(y, vals, height=0.55, color=colors)
        ax.axvline(0, color=INK_2, linewidth=1)
        lim = max(abs(results[["dev_vs_mean_pct", "holdout_vs_mean_pct"]].to_numpy()).max() * 1.35, 5)
        ax.set_xlim(-lim, lim)
        for yi, v in zip(y, vals):
            ax.text(v + (0.4 if v >= 0 else -0.4), yi, f"{v:+.1f}%", va="center",
                    ha="left" if v >= 0 else "right", color=INK, fontsize=9)
        ax.set_title(title, loc="left")
        ax.set_xlabel("RMSE vs predicting the mean  (left of 0 = better)")
        _style(ax)
    axes[0].set_yticks(np.arange(len(steps)), steps["label"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    fig.legend(handles, ["2024 linear model (+ bug fixes)", "LightGBM rebuild"], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.6, -0.08), labelcolor=INK_2)
    fig.suptitle("Only the rebuild beats the most boring prediction there is", x=0.01, ha="left",
                 fontsize=12.5, fontweight="semibold", color=INK)
    fig.tight_layout()
    return _save(fig, "ladder.png")


def whale_share(df: pd.DataFrame, top: int = 10):
    """Share of each month's squared error (around the month mean) caused by its `top` rows."""
    shares = {}
    for m, g in df.groupby(config.MONTH):
        se = np.sort((g[config.TARGET] - g[config.TARGET].mean()).to_numpy() ** 2)[::-1]
        shares[m] = se[:top].sum() / se.sum()
    s = pd.Series(shares)
    labels = pd.to_datetime([f"2018-{m:02d}-01" for m in s.index]).strftime("%b")
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.bar(labels, s.to_numpy() * 100, width=0.55, color=BLUE)
    for x, v in zip(labels, s.to_numpy() * 100):
        ax.text(x, v + 1.5, f"{v:.0f}%", ha="center", color=INK, fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of the month's squared error")
    ax.set_title(f"In most months, the {top} biggest contributions are most of the RMSE", loc="left")
    _style(ax, "y")
    ax.text(0, -0.2, f"Each month has 15-34k rows. Error measured against the month mean.",
            transform=ax.transAxes, color=MUTED, fontsize=8.5)
    _save(fig, "whale_share.png")
    return s


def cv_illusion(kfold_pct: float, group_pct: float, rolling_pct: float):
    labels = ["Random 10-fold\n(what I ran in 2024)", "10-fold grouped\nby customer",
              "Rolling by month\n(matches the real task)"]
    vals = [kfold_pct, group_pct, rolling_pct]
    fig, ax = plt.subplots(figsize=(8, 3.1))
    y = np.arange(3)[::-1]
    ax.barh(y, vals, height=0.5, color=[MUTED, MUTED, ORANGE])
    ax.axvline(0, color=INK_2, linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v + (0.3 if v >= 0 else -0.3), yi, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    lim = max(abs(np.array(vals))) * 1.4
    ax.set_xlim(-lim, lim)
    ax.set_yticks(y, labels)
    ax.set_xlabel("2024 linear model's RMSE vs predicting the mean  (left of 0 = better)")
    ax.set_title("The same model looks better or worse than the mean depending on how you validate it", loc="left")
    _style(ax)
    return _save(fig, "cv_illusion.png")


def repeat_signal(df: pd.DataFrame):
    """P(contributes this month) by what the same customer did in their previous snapshot."""
    d = df.sort_values([config.ID, config.MONTH])
    prev = d.groupby(config.ID)[config.TARGET].shift(1)
    groups = {
        "Contributed in their\nprevious snapshot": d.loc[prev > 0, config.TARGET],
        "Did not contribute in\ntheir previous snapshot": d.loc[prev == 0, config.TARGET],
        "No earlier snapshot": d.loc[prev.isna(), config.TARGET],
    }
    rates = {k: (v > 0).mean() * 100 for k, v in groups.items()}
    counts = {k: len(v) for k, v in groups.items()}
    fig, ax = plt.subplots(figsize=(8, 2.8))
    y = np.arange(3)[::-1]
    ax.barh(y, list(rates.values()), height=0.5, color=BLUE)
    for yi, (k, v) in zip(y, rates.items()):
        ax.text(v + 1, yi, f"{v:.0f}%  (n={counts[k]:,})", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_yticks(y, list(rates.keys()))
    ax.set_xlabel("% of rows with an additional contribution this month")
    ax.set_title("My 2024 hunch was right: past contributors contribute again", loc="left")
    _style(ax)
    _save(fig, "repeat_signal.png")
    return rates, counts


def feature_importance(model, feature_names, top: int = 15):
    imp = pd.Series(model.booster_.feature_importance("gain"), index=feature_names)
    imp = (imp / imp.sum() * 100).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(imp.index, imp.to_numpy(), height=0.55, color=BLUE)
    for yi, v in enumerate(imp.to_numpy()):
        ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", color=INK, fontsize=8.5)
    ax.set_xlabel("% of total split gain")
    ax.set_title("What the final model leans on", loc="left")
    _style(ax)
    _save(fig, "feature_importance.png")
    return imp

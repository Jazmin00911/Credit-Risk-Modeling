"""Discrimination, calibration, stability and fairness metrics."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

Metric = Callable[[np.ndarray, np.ndarray], float]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p))


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return 2 * auc(y, p) - 1


def ks(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y), np.asarray(p)
    return float(ks_2samp(p[y == 1], p[y == 0]).statistic)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(brier_score_loss(y, p))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. <0.1 stable, 0.1-0.25 monitor, >0.25 material shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def bootstrap_ci(metric: Metric, y: np.ndarray, p: np.ndarray, n: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals: list[float] = []
    while len(vals) < n:
        idx = rng.integers(0, len(y), len(y))
        if 0 < y[idx].sum() < len(idx):  # need both classes
            vals.append(metric(y[idx], p[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    lo, hi = bootstrap_ci(gini, y, p)
    return {"auc": auc(y, p), "gini": gini(y, p), "gini_ci_low": lo, "gini_ci_high": hi,
            "ks": ks(y, p), "brier": brier(y, p)}


def fairness_report(y: np.ndarray, p: np.ndarray, threshold: float, groups: pd.Series) -> pd.DataFrame:
    """Approval and error rates by group, plus adverse-impact ratio vs the best-treated group."""
    df = pd.DataFrame({"y": np.asarray(y), "reject": np.asarray(p) >= threshold, "g": np.asarray(groups)})
    rows = []
    for g, d in df.groupby("g"):
        rows.append({"group": g, "n": len(d), "approval_rate": 1 - d["reject"].mean(),
                     "default_rate": d["y"].mean(),
                     "fn_rate": ((~d["reject"]) & (d["y"] == 1)).sum() / max((d["y"] == 1).sum(), 1),
                     "fp_rate": (d["reject"] & (d["y"] == 0)).sum() / max((d["y"] == 0).sum(), 1)})
    out = pd.DataFrame(rows)
    out["adverse_impact_ratio"] = out["approval_rate"] / out["approval_rate"].max()
    return out

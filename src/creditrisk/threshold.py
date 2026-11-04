"""Cost-sensitive cutoff optimisation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.config import CostMatrix


def cost_at(y: np.ndarray, p: np.ndarray, t: float, costs: CostMatrix,
            exposure: np.ndarray | None = None) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    w = np.ones(len(y)) if exposure is None else np.asarray(exposure, dtype=float)
    reject = p >= t
    fn = costs.c_fn * w[(~reject) & (y == 1)].sum()  # approved a defaulter
    fp = costs.c_fp * w[reject & (y == 0)].sum()     # rejected a good customer
    return {"threshold": float(t), "cost": float(fn + fp), "fn_cost": float(fn), "fp_cost": float(fp),
            "approval_rate": float(1 - reject.mean())}


def optimize_threshold(y: np.ndarray, p: np.ndarray, costs: CostMatrix,
                       exposure: np.ndarray | None = None) -> tuple[float, pd.DataFrame]:
    grid = np.linspace(0.02, 0.98, 97)
    curve = pd.DataFrame([cost_at(y, p, t, costs, exposure) for t in grid])
    best = float(curve.loc[curve["cost"].idxmin(), "threshold"])
    return best, curve

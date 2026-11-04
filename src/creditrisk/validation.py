"""Cross-validated OOF predictions and walk-forward (out-of-time) backtesting."""
from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from creditrisk.metrics import brier, gini, ks, psi

logger = logging.getLogger(__name__)
ModelFactory = Callable[[], BaseEstimator]


def oof_proba(make_model: ModelFactory, X: pd.DataFrame, y: pd.Series, n_splits: int, seed: int) -> np.ndarray:
    """Out-of-fold PDs. The WoE step is refit inside each fold, so there is no leakage."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return cross_val_predict(make_model(), X, y, cv=cv, method="predict_proba")[:, 1]


def repeated_cv_metrics(make_model: ModelFactory, X: pd.DataFrame, y: pd.Series,
                        n_splits: int, repeats: int, seed: int) -> pd.DataFrame:
    rows = []
    for r in range(repeats):
        p = oof_proba(make_model, X, y, n_splits, seed + r)
        yy = y.to_numpy()
        rows.append({"repeat": r, "gini": gini(yy, p), "ks": ks(yy, p), "brier": brier(yy, p)})
    return pd.DataFrame(rows)


def walk_forward(make_model: ModelFactory, X: pd.DataFrame, y: pd.Series, periods: pd.Series,
                 min_train: int = 4, gap: int = 1, window: int | None = None) -> pd.DataFrame:
    """Expanding (or rolling, if `window`) walk-forward backtest.

    `gap` periods are skipped between training and test so only loans with an observed outcome
    are used for fitting (label maturity). Requires a real, dated dataset.
    """
    order = sorted(periods.unique())
    rows = []
    for i in range(min_train + gap, len(order)):
        stop = i - gap
        train_p = order[:stop] if window is None else order[max(0, stop - window):stop]
        tr, te = periods.isin(train_p), periods == order[i]
        model = make_model().fit(X[tr], y[tr])
        p_tr = model.predict_proba(X[tr])[:, 1]
        p_te = model.predict_proba(X[te])[:, 1]
        yt = y[te].to_numpy()
        if yt.min() == yt.max():
            logger.warning("Period %s has a single class; skipped", order[i])
            continue
        rows.append({"test_period": order[i], "n_test": int(te.sum()), "gini": gini(yt, p_te),
                     "ks": ks(yt, p_te), "brier": brier(yt, p_te), "psi_score": psi(p_tr, p_te)})
    return pd.DataFrame(rows)

"""Weight of Evidence / Information Value transformer (sklearn-compatible).

Convention: y = 1 is default. WoE = ln(%good / %bad), so a higher WoE means LOWER risk.
Fit on training folds only; inside a Pipeline this is guaranteed during cross-validation.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

MISSING = "MISSING"


def iv_strength(iv: float) -> str:
    if iv < 0.02:
        return "not useful"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious (check leakage)"


class WoETransformer(BaseEstimator, TransformerMixin):
    """Quantile-bin high-cardinality numerics; treat everything else as categories."""

    def __init__(self, n_bins: int = 5, smoothing: float = 0.5, max_levels: int = 10) -> None:
        self.n_bins = n_bins
        self.smoothing = smoothing
        self.max_levels = max_levels

    def _bucket(self, col: str, s: pd.Series) -> pd.Series:
        if col in self.edges_:
            s = pd.cut(s, self.edges_[col])
        return s.astype(object).where(s.notna(), MISSING)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> WoETransformer:
        yv = pd.Series(np.asarray(y), index=X.index)
        self.columns_: list[str] = list(X.columns)
        self.edges_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, dict] = {}
        iv_rows, bin_rows = [], []
        for col in self.columns_:
            s = X[col]
            if pd.api.types.is_numeric_dtype(s) and s.nunique() > self.max_levels:
                _, edges = pd.qcut(s, self.n_bins, retbins=True, duplicates="drop")
                edges[0], edges[-1] = -np.inf, np.inf
                self.edges_[col] = edges
            g = self._bucket(col, s)
            tab = pd.DataFrame({"g": g, "y": yv}).groupby("g", sort=False)["y"].agg(bad="sum", n="count")
            tab["good"] = tab["n"] - tab["bad"]
            k = len(tab)
            dist_bad = (tab["bad"] + self.smoothing) / (tab["bad"].sum() + self.smoothing * k)
            dist_good = (tab["good"] + self.smoothing) / (tab["good"].sum() + self.smoothing * k)
            woe = np.log(dist_good / dist_bad)
            self.woe_[col] = woe.to_dict()
            iv_rows.append({"feature": col, "iv": float(((dist_good - dist_bad) * woe).sum())})
            for b, row in tab.iterrows():
                bin_rows.append({"feature": col, "bin": str(b), "n": int(row["n"]),
                                 "bad_rate": row["bad"] / row["n"], "woe": float(woe[b])})
        self.iv_table_ = (pd.DataFrame(iv_rows).sort_values("iv", ascending=False)
                          .assign(strength=lambda d: d["iv"].map(iv_strength)).reset_index(drop=True))
        self.woe_table_ = pd.DataFrame(bin_rows)
        logger.debug("Fitted WoE on %d features", len(self.columns_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            out[col] = self._bucket(col, X[col]).map(self.woe_[col]).astype(float).fillna(0.0)  # unseen -> neutral
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.columns_)

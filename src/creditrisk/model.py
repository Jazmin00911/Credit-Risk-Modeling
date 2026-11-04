"""Model construction and the single deployable artifact (CreditModelBundle)."""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from creditrisk.config import Config
from creditrisk.woe import WoETransformer

logger = logging.getLogger(__name__)
ModelFactory = Callable[[], Pipeline]

REASON_TEXT: dict[str, str] = {
    "Age": "Applicant age",
    "Job": "Employment / skill category",
    "Credit amount": "Requested credit amount",
    "Duration": "Loan duration",
    "Housing": "Housing status",
    "Saving accounts": "Savings account balance",
    "Checking account": "Checking account status",
    "Purpose": "Loan purpose",
}


def build_scorecard(cfg: Config, C: float = 0.5) -> Pipeline:
    """Champion: WoE -> logistic regression (transparent, monotone by construction)."""
    return Pipeline([("woe", WoETransformer()),
                     ("clf", LogisticRegression(C=C, max_iter=1000))])


def build_xgb(cfg: Config) -> Pipeline:
    """Challenger: WoE -> shallow XGBoost.

    Every WoE feature is monotone in risk by construction (higher WoE = lower PD), so a
    -1 monotone constraint on all features is valid and keeps the challenger explainable.
    """
    n = len(cfg.features)
    clf = XGBClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                        monotone_constraints=tuple([-1] * n), eval_metric="logloss",
                        random_state=cfg.seed, n_jobs=1)
    return Pipeline([("woe", WoETransformer()), ("clf", clf)])


def scorecard_points(pipe: Pipeline, base_score: float = 600, base_odds: float = 2.0,
                     pdo: float = 20) -> pd.DataFrame:
    """Per-bin points. Higher score = lower risk; +pdo points doubles the good:bad odds."""
    woe: WoETransformer = pipe.named_steps["woe"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    n = len(woe.columns_)
    coef = dict(zip(woe.columns_, clf.coef_[0], strict=True))
    tab = woe.woe_table_.copy()
    tab["coef"] = tab["feature"].map(coef)
    tab["points"] = -factor * (tab["coef"] * tab["woe"] + clf.intercept_[0] / n) + offset / n
    return tab.round({"bad_rate": 3, "woe": 3, "coef": 3, "points": 1})


@dataclass
class CreditModelBundle:
    """Everything needed to score, decide and explain, saved as ONE versioned file."""

    name: str
    pipeline: Pipeline
    threshold: float
    features: list[str]
    background: pd.DataFrame  # small raw-feature sample used by the SHAP explainer
    metadata: dict[str, Any] = field(default_factory=dict)
    _explainer: Any = field(default=None, repr=False, compare=False)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of default."""
        return self.pipeline.predict_proba(X[self.features])[:, 1]

    def decide(self, X: pd.DataFrame) -> np.ndarray:
        return np.where(self.predict_proba(X) >= self.threshold, "DECLINE", "APPROVE")

    @property
    def explainer(self):
        if self._explainer is None:
            from creditrisk.explain import CreditExplainer
            self._explainer = CreditExplainer(self.pipeline, self.background, self.features, REASON_TEXT)
        return self._explainer

    def explain(self, X: pd.DataFrame):
        return self.explainer.explain(X[self.features])

    def reason_codes(self, X: pd.DataFrame, top_k: int = 4) -> list[dict[str, Any]]:
        return self.explainer.reason_codes(X[self.features], top_k)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        cached, self._explainer = self._explainer, None  # SHAP objects are rebuilt lazily
        try:
            joblib.dump(self, path)
        finally:
            self._explainer = cached
        logger.info("Saved bundle %s -> %s", self.name, path)

    @staticmethod
    def load(path: Path) -> CreditModelBundle:
        return joblib.load(path)

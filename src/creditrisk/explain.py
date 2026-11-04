"""SHAP explanations in the original variable space, plus adverse-action style reason codes."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


class CreditExplainer:
    """Explains log-odds of DEFAULT. Positive contribution = pushes the applicant toward decline.

    Because every model input is one WoE column per raw variable, attributions map 1:1 to the
    applicant's original fields (no one-hot fragments to re-aggregate).
    """

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame, features: list[str],
                 reason_text: dict[str, str]) -> None:
        self.pre = pipeline[:-1]
        self.clf = pipeline[-1]
        self.features = features
        self.reason_text = reason_text
        if isinstance(self.clf, XGBClassifier):
            self.exp = shap.TreeExplainer(self.clf)
        else:
            self.exp = shap.LinearExplainer(self.clf, self.pre.transform(background[features]))

    def explain(self, X_raw: pd.DataFrame) -> shap.Explanation:
        sv = self.exp(self.pre.transform(X_raw[self.features]))
        base = np.broadcast_to(np.asarray(sv.base_values).reshape(-1), (len(X_raw),)).copy()
        return shap.Explanation(values=sv.values, base_values=base,
                                data=X_raw[self.features].to_numpy(dtype=object),
                                feature_names=self.features)

    def reason_codes(self, X_raw: pd.DataFrame, top_k: int = 4) -> list[dict[str, Any]]:
        """Top factors pushing ONE applicant toward decline (principal reasons)."""
        e = self.explain(X_raw.iloc[[0]])
        contrib = pd.Series(e.values[0], index=self.features)
        top = contrib[contrib > 0].nlargest(top_k)
        return [{"reason": self.reason_text.get(f, f), "feature": f,
                 "applicant_value": X_raw.iloc[0][f], "log_odds_impact": float(v)}
                for f, v in top.items()]

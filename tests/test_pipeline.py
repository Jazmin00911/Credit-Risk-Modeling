import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from creditrisk.model import CreditModelBundle, build_scorecard, build_xgb
from creditrisk.validation import walk_forward


def test_prepare_keeps_all_rows_and_no_protected_features(dataset, cfg):
    X, y, groups = dataset
    assert len(X) == 1000 and not X.isna().any().any()
    assert set(y.unique()) == {0, 1}
    assert not set(cfg.protected) & set(X.columns)


@pytest.mark.parametrize("builder", [build_scorecard, build_xgb])
def test_shap_additivity_and_reason_codes(dataset, cfg, builder):
    X, y, _ = dataset
    X_tr, X_te, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
    pipe = builder(cfg).fit(X_tr, y_tr)
    bundle = CreditModelBundle("t", pipe, 0.17, cfg.features, X_tr.head(50))
    row = X_te.iloc[[0]]
    e = bundle.explain(row)
    p = bundle.predict_proba(row)[0]
    assert np.isclose(e.base_values[0] + e.values[0].sum(), np.log(p / (1 - p)), atol=1e-3)
    codes = bundle.reason_codes(row, top_k=3)
    assert len(codes) <= 3 and all(c["log_odds_impact"] > 0 for c in codes)


def test_bundle_roundtrip(dataset, cfg, tmp_path):
    X, y, _ = dataset
    b = CreditModelBundle("t", build_scorecard(cfg).fit(X, y), 0.2, cfg.features, X.head(30))
    b.explain(X.head(1))  # populate the explainer cache; save() must still work
    b.save(tmp_path / "m.joblib")
    b2 = CreditModelBundle.load(tmp_path / "m.joblib")
    assert np.allclose(b.predict_proba(X.head(20)), b2.predict_proba(X.head(20)))


def test_walk_forward_row_count_and_signal(cfg):
    rng = np.random.default_rng(0)
    n = 4000
    X = pd.DataFrame({"Job": rng.integers(0, 4, n), "Credit amount": rng.lognormal(8, 0.6, n),
                      "Duration": rng.integers(6, 48, n), "Housing": rng.choice(["own", "rent"], n),
                      "Saving accounts": rng.choice(["little", "rich"], n),
                      "Checking account": rng.choice(["little", "none"], n),
                      "Purpose": rng.choice(["car", "radio/TV"], n)})
    logit = -1.5 + 0.04 * (X["Duration"] - 24) + 0.8 * (X["Saving accounts"] == "little")
    y = pd.Series((rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int))
    periods = pd.Series(np.repeat(np.arange(8), n // 8))
    out = walk_forward(lambda: build_scorecard(cfg), X, y, periods, min_train=3, gap=1)
    assert len(out) == 8 - (3 + 1) and (out["gini"] > 0.1).all()

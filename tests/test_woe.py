import numpy as np
import pandas as pd

from creditrisk.woe import WoETransformer


def toy():
    X = pd.DataFrame({"c": ["a"] * 6 + ["b"] * 4})
    y = np.array([0, 0, 0, 0, 0, 1] + [1, 1, 1, 0])  # a: 1 bad/5 good, b: 3 bad/1 good
    return X, y


def test_known_answer_woe_and_iv():
    X, y = toy()
    t = WoETransformer(smoothing=0.0).fit(X, y)
    # totals: 6 good, 4 bad. a: good 5/6, bad 1/4. b: good 1/6, bad 3/4
    assert np.isclose(t.woe_["c"]["a"], np.log((5 / 6) / (1 / 4)))
    assert np.isclose(t.woe_["c"]["b"], np.log((1 / 6) / (3 / 4)))
    iv = (5 / 6 - 1 / 4) * np.log((5 / 6) / (1 / 4)) + (1 / 6 - 3 / 4) * np.log((1 / 6) / (3 / 4))
    assert np.isclose(t.iv_table_.loc[0, "iv"], iv)


def test_lower_risk_bin_has_higher_woe():
    X, y = toy()
    t = WoETransformer().fit(X, y)
    assert t.woe_["c"]["a"] > t.woe_["c"]["b"]


def test_unseen_category_is_neutral_and_nan_is_own_bucket():
    X, y = toy()
    t = WoETransformer().fit(X, y)
    assert t.transform(pd.DataFrame({"c": ["zzz"]})).iloc[0, 0] == 0.0
    Xn = pd.DataFrame({"n": list(range(50)) + [np.nan] * 5})
    yn = np.r_[np.zeros(25), np.ones(25), np.ones(5)]
    out = WoETransformer().fit(Xn, yn).transform(Xn)
    assert not out.isna().any().any()

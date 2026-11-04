import numpy as np

from creditrisk.config import CostMatrix
from creditrisk.metrics import bootstrap_ci, gini, ks, psi
from creditrisk.stress import SCENARIOS, scenario_z, vasicek_pd
from creditrisk.threshold import optimize_threshold


def test_perfect_separation():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    assert gini(y, p) == 1.0 and ks(y, p) == 1.0


def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 1.5) > 0.25


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    p = rng.uniform(size=600); y = (rng.uniform(size=600) < p).astype(int)
    lo, hi = bootstrap_ci(gini, y, p, n=300)
    assert lo < gini(y, p) < hi


def test_optimal_cutoff_matches_closed_form_when_calibrated():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=200_000); y = (rng.uniform(size=p.size) < p).astype(int)
    costs = CostMatrix(c_fn=5, c_fp=1)
    t, _ = optimize_threshold(y, p, costs)
    assert abs(t - costs.bayes_threshold) < 0.03


def test_stress_identity_and_monotonicity():
    pd_ = np.array([0.05, 0.3, 0.6])
    assert np.allclose(vasicek_pd(pd_, 0.0), pd_)
    assert np.all(vasicek_pd(pd_, -2.0) > pd_)
    z = [scenario_z(s) for s in SCENARIOS.values()]
    assert z == sorted(z, reverse=True)

"""Vasicek single-factor stress overlay. All macro loadings are ASSUMPTIONS, not estimates."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

# z < 0 is a downturn. Loadings translate macro shocks into the systematic factor.
SCENARIOS: dict[str, dict[str, float]] = {
    "baseline": {"unemployment_pp": 0.0, "gdp_pp": 0.0},
    "adverse": {"unemployment_pp": 3.0, "gdp_pp": -2.5},
    "severely_adverse": {"unemployment_pp": 6.0, "gdp_pp": -5.0},
}
LOADINGS: dict[str, float] = {"unemployment_pp": -0.25, "gdp_pp": 0.30}


def scenario_z(shocks: dict[str, float]) -> float:
    return float(sum(LOADINGS[k] * v for k, v in shocks.items()))


def vasicek_pd(pd_base: np.ndarray, z: float, rho: float = 0.10) -> np.ndarray:
    """PD conditional on systematic factor z, with the model PD taken as the median-state PD.

    PD(z) = Phi( Phi^-1(PD) - sqrt(rho / (1 - rho)) * z ). z = 0 returns the input PD,
    z = -2.33 is roughly a 1-in-100 downturn.
    """
    p = np.clip(np.asarray(pd_base, dtype=float), 1e-6, 1 - 1e-6)
    return norm.cdf(norm.ppf(p) - np.sqrt(rho / (1 - rho)) * z)


def portfolio_stress(pd_hat: np.ndarray, ead: np.ndarray, threshold: float,
                     lgd: float, rho: float) -> pd.DataFrame:
    rows = []
    for name, shocks in SCENARIOS.items():
        z = scenario_z(shocks)
        stressed = vasicek_pd(pd_hat, z, rho)
        rows.append({"scenario": name, "z": z, "mean_pd": stressed.mean(),
                     "expected_loss": float(np.sum(stressed * lgd * ead)),
                     "share_above_cutoff": float((stressed >= threshold).mean())})
    out = pd.DataFrame(rows)
    out["el_multiple_vs_baseline"] = out["expected_loss"] / out.loc[0, "expected_loss"]
    return out

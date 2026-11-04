"""Training entry point:  python -m creditrisk.train

Produces two deployable bundles (scorecard champion, XGBoost challenger) and a reports/ folder.
Holdout test rows are touched exactly once per model, after the threshold has been fixed on OOF data.
"""
from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split

from creditrisk import __version__
from creditrisk.config import Config
from creditrisk.data import category_levels, load_raw, prepare
from creditrisk.logging_config import setup_logging
from creditrisk.metrics import fairness_report, psi, summarize
from creditrisk.model import CreditModelBundle, ModelFactory, build_scorecard, build_xgb, scorecard_points
from creditrisk.stress import portfolio_stress
from creditrisk.threshold import cost_at, optimize_threshold
from creditrisk.validation import oof_proba

logger = logging.getLogger("creditrisk.train")


def data_audit(raw: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """Quantify what complete-case deletion (`dropna()`) would discard from the raw file."""
    body = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    kept = body.dropna()
    dropped = body.drop(index=kept.index)
    bad = lambda d: float((d[cfg.target] == "bad").mean())  # noqa: E731
    return {"rows_raw": len(body), "rows_after_dropna": len(kept),
            "pct_rows_lost": 100 * (1 - len(kept) / len(body)),
            "default_rate_raw": bad(body), "default_rate_after_dropna": bad(kept),
            "default_rate_of_dropped_rows": bad(dropped)}


def evaluate(name: str, make: ModelFactory, X_tr: pd.DataFrame, y_tr: pd.Series,
             X_te: pd.DataFrame, y_te: pd.Series, cfg: Config) -> dict[str, Any]:
    oof = oof_proba(make, X_tr, y_tr, cfg.n_splits, cfg.seed)
    thr, curve = optimize_threshold(y_tr.to_numpy(), oof, cfg.costs)
    p_te = make().fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    yt = y_te.to_numpy()
    policies = {"approve_all": 1.01, "cutoff_0.50": 0.5,
                f"bayes_{cfg.costs.bayes_threshold:.3f}": cfg.costs.bayes_threshold, "optimized": thr}
    cost = {k: cost_at(yt, p_te, t, cfg.costs) for k, t in policies.items()}
    for v in cost.values():
        v["cost_per_applicant"] = v["cost"] / len(yt)
    logger.info("%-9s | CV gini %.3f | test gini %.3f | thr %.2f | cost/applicant: all-approve %.3f, "
                "0.5 cutoff %.3f, optimized %.3f", name, summarize(y_tr.to_numpy(), oof)["gini"],
                summarize(yt, p_te)["gini"], thr, cost["approve_all"]["cost_per_applicant"],
                cost["cutoff_0.50"]["cost_per_applicant"], cost["optimized"]["cost_per_applicant"])
    return {"threshold": thr, "cv_oof": summarize(y_tr.to_numpy(), oof), "test": summarize(yt, p_te),
            "test_calibration": {"mean_pd": float(p_te.mean()), "observed_default_rate": float(yt.mean())},
            "psi_oof_vs_test": psi(oof, p_te), "test_cost_by_policy": cost, "_curve": curve}


def deploy(name: str, make: ModelFactory, X: pd.DataFrame, y: pd.Series, groups: pd.DataFrame,
           cfg: Config, results: dict[str, Any]) -> CreditModelBundle:
    """Refit on all data; cutoff from all-data OOF predictions."""
    oof = oof_proba(make, X, y, cfg.n_splits, cfg.seed)
    thr, curve = optimize_threshold(y.to_numpy(), oof, cfg.costs)
    pipe = make().fit(X, y)
    pd_hat = pipe.predict_proba(X)[:, 1]
    bundle = CreditModelBundle(
        name=name, pipeline=pipe, threshold=thr, features=cfg.features,
        background=X.sample(100, random_state=cfg.seed).reset_index(drop=True),
        metadata={"version": __version__, "trained_at": datetime.now(timezone.utc).isoformat(),
                  "n_rows": len(X), "default_rate": float(y.mean()),
                  "libs": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                           "xgboost": xgboost.__version__, "pandas": pd.__version__},
                  "costs": {"c_fn": cfg.costs.c_fn, "c_fp": cfg.costs.c_fp},
                  "lgd": cfg.lgd, "rho": cfg.rho, "holdout": results[name],
                  "categories": category_levels(X, cfg),
                  "numeric_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in cfg.numeric_features}})
    bundle.save(cfg.artifact_dir / f"{name}_v1.joblib")
    # --- reports ---
    cfg.report_dir.mkdir(exist_ok=True)
    curve.to_csv(cfg.report_dir / f"cost_curve_{name}.csv", index=False)
    age_band = pd.cut(groups["Age"], [0, 29, 49, 120], labels=["<30", "30-49", "50+"]).astype(str)
    fair = pd.concat([fairness_report(y.to_numpy(), oof, thr, groups["Sex"]).assign(attribute="Sex"),
                      fairness_report(y.to_numpy(), oof, thr, age_band).assign(attribute="Age band")])
    fair.round(3).to_csv(cfg.report_dir / f"fairness_{name}.csv", index=False)
    stress = portfolio_stress(pd_hat, X["Credit amount"].to_numpy(), thr, cfg.lgd, cfg.rho)
    stress.round(4).to_csv(cfg.report_dir / f"stress_{name}.csv", index=False)
    if name == "scorecard":
        woe = pipe.named_steps["woe"]
        woe.iv_table_.round(4).to_csv(cfg.report_dir / "iv_table.csv", index=False)
        scorecard_points(pipe).to_csv(cfg.report_dir / "scorecard_points.csv", index=False)
    return bundle


def main(cfg: Config | None = None) -> dict[str, Any]:
    setup_logging()
    cfg = cfg or Config()
    raw = load_raw(cfg.data_path)
    audit = data_audit(raw, cfg)
    logger.info("Complete-case deletion would discard %.1f%% of rows and move the default rate %.1f%% -> %.1f%%",
                audit["pct_rows_lost"], 100 * audit["default_rate_raw"], 100 * audit["default_rate_after_dropna"])
    X, y, groups = prepare(raw, cfg)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    factories: dict[str, ModelFactory] = {"scorecard": lambda: build_scorecard(cfg), "xgb": lambda: build_xgb(cfg)}
    results = {n: evaluate(n, f, X_tr, y_tr, X_te, y_te, cfg) for n, f in factories.items()}
    for n, f in factories.items():
        deploy(n, f, X, y, groups, cfg, results)
    for r in results.values():
        r.pop("_curve")
    out = {"data_audit": audit, "models": results, "config": {"seed": cfg.seed, "test_size": cfg.test_size}}
    (cfg.report_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    main()

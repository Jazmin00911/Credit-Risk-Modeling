"""Independent validation test battery.   python -m creditrisk.validate

Re-derives every figure quoted in VALIDATION_MEMO.md and writes reports/validation_results.json.
It imports the production modules but re-estimates models from scratch, so it does not rely on
the stored artifacts.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from creditrisk.config import Config
from creditrisk.data import load_raw, prepare
from creditrisk.logging_config import setup_logging
from creditrisk.metrics import brier, fairness_report, gini, ks, psi
from creditrisk.model import build_scorecard, build_xgb
from creditrisk.stress import SCENARIOS, portfolio_stress, vasicek_pd
from creditrisk.threshold import cost_at, optimize_threshold
from creditrisk.validation import oof_proba
from creditrisk.woe import WoETransformer

logger = logging.getLogger("creditrisk.validate")


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def status(v: float) -> str:
    return "GREEN" if v < 0.10 else ("AMBER" if v < 0.25 else "RED")


def calibration(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    lr = LogisticRegression(C=1e9, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    df = pd.DataFrame({"y": y, "p": p})
    df["bin"] = pd.qcut(df["p"], 10, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))
    hl = float((g["n"] * (g["observed"] - g["predicted"]) ** 2 / (g["predicted"] * (1 - g["predicted"]))).sum())
    dof = len(g) - 2
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0]), "mean_pd": float(p.mean()),
            "observed_rate": float(y.mean()), "hl_chi2": hl, "hl_dof": dof,
            "hl_p": float(1 - stats.chi2.cdf(hl, dof)), "deciles": g.round(4).reset_index(drop=True).to_dict("records")}


def csi_table(woe: WoETransformer, expected: pd.DataFrame, actual: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for c in woe.columns_:
        a = woe._bucket(c, expected[c]).value_counts(normalize=True)
        b = woe._bucket(c, actual[c]).value_counts(normalize=True)
        idx = a.index.union(b.index, sort=False)
        a, b = a.reindex(idx, fill_value=0).clip(1e-6), b.reindex(idx, fill_value=0).clip(1e-6)
        v = float(((b - a) * np.log(b / a)).sum())
        rows.append({"feature": c, "csi": v, "status": status(v)})
    return rows


def psi_null(p: np.ndarray, n_e: int, n_a: int, reps: int = 1000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        perm = rng.permutation(len(p))
        vals.append(psi(p[perm[:n_e]], p[perm[n_e:n_e + n_a]]))
    return {"mean": float(np.mean(vals)), "p95": float(np.percentile(vals, 95)), "p99": float(np.percentile(vals, 99))}


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000, seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d: list[float] = []
    while len(d) < n:
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(gini(y[i], p_b[i]) - gini(y[i], p_a[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"mean_diff": float(np.mean(d)), "ci_low": float(lo), "ci_high": float(hi), "share_positive": float(np.mean(np.array(d) > 0))}


def main() -> dict[str, Any]:
    setup_logging()
    cfg = Config()
    raw = load_raw(cfg.data_path)
    X, y, groups = prepare(raw, cfg)
    yv = y.to_numpy()
    R: dict[str, Any] = {}
    makers = {"scorecard": lambda: build_scorecard(cfg), "xgb": lambda: build_xgb(cfg)}

    # ---- 1. Data quality ------------------------------------------------------------------
    body = raw.drop(columns=[c for c in raw.columns if c.startswith("Unnamed")])
    def tukey(s: pd.Series) -> dict[str, float]:
        q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
        return {"n_above_fence": int((s > q3 + 1.5 * iqr).sum()), "pct": float((s > q3 + 1.5 * iqr).mean()),
                "p50": float(s.median()), "p99": float(s.quantile(0.99)), "max": float(s.max()), "skew": float(s.skew())}
    levels = {c: {str(k): {"n": int(v["n"]), "default_rate": round(float(v["dr"]), 3)}
                  for k, v in pd.DataFrame({"g": X[c], "y": yv}).groupby("g")["y"].agg(n="size", dr="mean").iterrows()}
              for c in cfg.categorical_features + ("Job",)}
    R["data"] = {"n": len(X), "duplicates": int(body.duplicated().sum()),
                 "missing_raw": {c: int(body[c].isna().sum()) for c in body.columns if body[c].isna().any()},
                 "default_rate": float(yv.mean()), "credit_amount": tukey(X["Credit amount"]),
                 "duration": tukey(X["Duration"]), "duration_range": [int(X["Duration"].min()), int(X["Duration"].max())],
                 "amount_range": [int(X["Credit amount"].min()), int(X["Credit amount"].max())],
                 "levels": levels, "levels_below_30": [f"{c}={k}" for c, d in levels.items() for k, v in d.items() if v["n"] < 30]}

    # ---- 2. Performance, overfitting, calibration -----------------------------------------
    perf, oof_main = {}, {}
    for name, mk in makers.items():
        reps = []
        for r in range(5):
            p = oof_proba(mk, X, y, cfg.n_splits, cfg.seed + r)
            reps.append({"gini": gini(yv, p), "ks": ks(yv, p), "brier": brier(yv, p), "logloss": float(log_loss(yv, p))})
            if r == 0:
                oof_main[name] = p
        df = pd.DataFrame(reps)
        insample = mk().fit(X, y).predict_proba(X)[:, 1]
        perf[name] = {"repeated_cv_mean": df.mean().to_dict(), "repeated_cv_sd": df.std().to_dict(),
                      "insample_gini": gini(yv, insample), "calibration_oof": calibration(yv, oof_main[name])}
        perf[name]["overfit_gap_gini"] = perf[name]["insample_gini"] - perf[name]["repeated_cv_mean"]["gini"]
    R["performance"] = perf
    R["no_skill_brier"] = float(yv.mean() * (1 - yv.mean()))

    # ---- 3. Stability: PSI, noise floor, CSI, simulated drift -----------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.seed)
    sc = build_scorecard(cfg)
    oof_tr = oof_proba(lambda: build_scorecard(cfg), X_tr, y_tr, cfg.n_splits, cfg.seed)
    sc.fit(X_tr, y_tr)
    p_te = sc.predict_proba(X_te)[:, 1]
    woe = sc.named_steps["woe"]
    base_psi = psi(oof_tr, p_te)
    shocks = {"duration_plus_12m": lambda d: d.assign(Duration=d["Duration"] + 12),
              "credit_amount_x1.5": lambda d: d.assign(**{"Credit amount": d["Credit amount"] * 1.5}),
              "no_checking_to_little": lambda d: d.assign(**{"Checking account": d["Checking account"].replace("none", "little")}),
              "all_savings_little": lambda d: d.assign(**{"Saving accounts": "little"})}
    drift = {}
    for n, f in shocks.items():
        Xs = f(X_te.copy())
        v = psi(oof_tr, sc.predict_proba(Xs)[:, 1])
        csi = csi_table(woe, X_tr, Xs)
        top = max(csi, key=lambda r: r["csi"])
        drift[n] = {"score_psi": v, "status": status(v), "max_csi_feature": top["feature"], "max_csi": top["csi"],
                    "mean_pd": float(sc.predict_proba(Xs)[:, 1].mean())}
    R["stability"] = {"psi_oof_vs_holdout": base_psi, "status": status(base_psi), "n_expected": len(X_tr), "n_actual": len(X_te),
                      "psi_null_random_split": psi_null(oof_main["scorecard"], len(X_tr), len(X_te)),
                      "csi_holdout_vs_train": csi_table(woe, X_tr, X_te), "simulated_drift": drift,
                      "holdout_mean_pd": float(p_te.mean())}

    # ---- 4. Challenger benchmarking --------------------------------------------------------
    d_oof = paired_bootstrap(yv, oof_main["scorecard"], oof_main["xgb"])
    xgb_tr = build_xgb(cfg).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    R["challenger"] = {"paired_bootstrap_oof_gini_xgb_minus_scorecard": d_oof,
                       "paired_bootstrap_holdout_gini_xgb_minus_scorecard": paired_bootstrap(y_te.to_numpy(), p_te, xgb_tr),
                       "xgb_trees": 200, "xgb_depth": 2, "scorecard_parameters": len(X.columns) + 1}

    # ---- 5. Methodology tests: leakage, imbalance, hyper-parameters, collinearity ----------
    cv = StratifiedKFold(cfg.n_splits, shuffle=True, random_state=cfg.seed)
    leak = {}
    Zfull = WoETransformer().fit(X, y).transform(X)
    for name, est in {"scorecard": LogisticRegression(C=0.5, max_iter=1000), "xgb": build_xgb(cfg).named_steps["clf"]}.items():
        p_leaky = cross_val_predict(est, Zfull, y, cv=cv, method="predict_proba")[:, 1]
        leak[name] = {"gini_leaky_woe": gini(yv, p_leaky), "gini_proper": gini(yv, oof_main[name]),
                      "optimism": gini(yv, p_leaky) - gini(yv, oof_main[name])}
    R["leakage_test"] = leak

    imb = {}
    ratio = float((yv == 0).sum() / (yv == 1).sum())
    for w in (1.0, ratio):
        def mk_w(w: float = w) -> Any:
            return build_xgb(cfg).set_params(clf__scale_pos_weight=w)

        p = oof_proba(mk_w, X, y, cfg.n_splits, cfg.seed)
        cal = calibration(yv, p)
        imb[f"scale_pos_weight_{w:.2f}"] = {"mean_pd": cal["mean_pd"], "observed": cal["observed_rate"], "slope": cal["slope"],
                                           "brier": brier(yv, p), "gini": gini(yv, p),
                                           "cost_at_bayes": cost_at(yv, p, cfg.costs.bayes_threshold, cfg.costs)["cost"] / len(yv),
                                           "approval_at_bayes": cost_at(yv, p, cfg.costs.bayes_threshold, cfg.costs)["approval_rate"]}
    R["imbalance_test"] = imb

    hp: dict[str, Any] = {"scorecard_C": {}, "xgb_max_depth": {}}
    for c in (0.05, 0.1, 0.5, 1.0, 5.0, 100.0):
        def mk(c: float = c) -> Any:
            return build_scorecard(cfg, C=c)

        hp["scorecard_C"][str(c)] = {"oof_gini": gini(yv, oof_proba(mk, X, y, cfg.n_splits, cfg.seed)),
                                    "insample_gini": gini(yv, mk().fit(X, y).predict_proba(X)[:, 1])}
    for d in (1, 2, 3, 4, 6):
        def mk_xgb(d: int = d) -> Any:
            return build_xgb(cfg).set_params(clf__max_depth=d)

        hp["xgb_max_depth"][str(d)] = {"oof_gini": gini(yv, oof_proba(mk_xgb, X, y, cfg.n_splits, cfg.seed)),
                                      "insample_gini": gini(yv, mk_xgb().fit(X, y).predict_proba(X)[:, 1])}
    R["hyperparameter_sensitivity"] = hp

    full_sc = build_scorecard(cfg).fit(X, y)
    wfull = full_sc.named_steps["woe"]
    corr = Zfull.corr()
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    vif = dict(zip(Zfull.columns, np.diag(np.linalg.inv(corr.values)), strict=True))
    coef = dict(zip(Zfull.columns, full_sc.named_steps["clf"].coef_[0], strict=True))
    def ordered_bad_rate(col: str) -> list[float]:
        b = pd.cut(X[col], wfull.edges_[col])
        return [round(float(v), 3) for v in pd.DataFrame({"b": b, "y": yv}).groupby("b", observed=True)["y"].mean()]
    R["features"] = {"iv": wfull.iv_table_.round(4).to_dict("records"), "coefficients": coef,
                     "all_coefficients_negative": bool(all(v < 0 for v in coef.values())),
                     "max_abs_woe_correlation": float(off.max().max()), "vif": {k: float(v) for k, v in vif.items()},
                     "bad_rate_by_ordered_bin": {c: ordered_bad_rate(c) for c in ("Duration", "Credit amount")}}

    # ---- 5b. Leave-one-characteristic-out ablation and age-proxy screen ----------------------
    def mean_gini(frame: pd.DataFrame, reps: int = 3) -> float:
        return float(np.mean([gini(yv, oof_proba(lambda: build_scorecard(cfg), frame, y, cfg.n_splits, cfg.seed + r))
                              for r in range(reps)]))
    base3 = mean_gini(X)
    abl = {f: {"gini_without": mean_gini(X.drop(columns=[f]))} for f in X.columns}
    for f in abl:
        abl[f]["delta"] = abl[f]["gini_without"] - base3
    R["ablation_scorecard"] = {"gini_all_features": base3, "leave_one_out": abl}
    R["age_ablation"] = {"gini_with_age": mean_gini(X.assign(Age=groups["Age"]), 5), "gini_without_age": mean_gini(X, 5)}
    R["age_proxy_corr_with_woe"] = {c: float(np.corrcoef(Zfull[c], groups["Age"])[0, 1]) for c in Zfull.columns}

    # ---- 6. Deployed cutoffs, cutoff stability -------------------------------------------
    thr: dict[str, dict[str, Any]] = {}
    for name in makers:
        t, curve = optimize_threshold(yv, oof_main[name], cfg.costs)
        near = curve[curve["cost"] <= curve["cost"].min() * 1.02]["threshold"]
        rng = np.random.default_rng(7)
        bs = []
        for _ in range(300):
            i = rng.integers(0, len(yv), len(yv))
            bs.append(optimize_threshold(yv[i], oof_main[name][i], cfg.costs)[0])
        thr[name] = {"deployed_threshold": t, "approval_rate": float(1 - (oof_main[name] >= t).mean()),
                     "within_2pct_of_min_cost": [float(near.min()), float(near.max())],
                     "bootstrap_threshold_median": float(np.median(bs)),
                     "bootstrap_threshold_p5_p95": [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]}
    R["cutoff"] = thr

    # ---- 7. Sensitivity (deployed models, fit on all data) --------------------------------
    sens: dict[str, Any] = {}
    mono: dict[str, Any] = {}
    fitted = {n: mk().fit(X, y) for n, mk in makers.items()}
    cuts: dict[str, float] = {n: float(thr[n]["deployed_threshold"]) for n in makers}
    shock_set = {"baseline": lambda d: d, "duration_plus_12m": lambda d: d.assign(Duration=d["Duration"] + 12),
                 "duration_x1.5": lambda d: d.assign(Duration=(d["Duration"] * 1.5).round()),
                 "credit_amount_x1.5": lambda d: d.assign(**{"Credit amount": d["Credit amount"] * 1.5}),
                 "credit_amount_x2": lambda d: d.assign(**{"Credit amount": d["Credit amount"] * 2}),
                 "checking_all_little": lambda d: d.assign(**{"Checking account": "little"}),
                 "savings_all_little": lambda d: d.assign(**{"Saving accounts": "little"}),
                 "out_of_range_duration_120m": lambda d: d.assign(Duration=120)}
    for n, m in fitted.items():
        sens[n] = {}
        for s, f in shock_set.items():
            p = m.predict_proba(f(X.copy()))[:, 1]
            sens[n][s] = {"mean_pd": float(p.mean()), "decline_rate": float((p >= cuts[n]).mean())}
        med = {"Job": 2, "Housing": "own", "Saving accounts": "little", "Checking account": "little", "Purpose": "car",
               "Credit amount": float(X["Credit amount"].median())}
        grid = pd.DataFrame([{**med, "Duration": d} for d in range(int(X["Duration"].min()), int(X["Duration"].max()) + 1, 4)])
        pg = m.predict_proba(grid[X.columns])[:, 1]
        mono[n] = {"duration_grid_pd": [round(float(v), 3) for v in pg], "monotone_non_decreasing": bool(np.all(np.diff(pg) >= -1e-9))}
    R["sensitivity"] = {"oat": sens, "duration_monotonicity": mono}

    # ---- 8. Stress --------------------------------------------------------------------------
    ead = X["Credit amount"].to_numpy()
    pd_oof, pd_in = oof_main["scorecard"], fitted["scorecard"].predict_proba(X)[:, 1]
    st = portfolio_stress(pd_oof, ead, cuts["scorecard"], cfg.lgd, cfg.rho)
    rho_grid = {}
    for rv in (0.05, 0.10, 0.20, 0.30):
        s = portfolio_stress(pd_oof, ead, cuts["scorecard"], cfg.lgd, rv).set_index("scenario")
        rho_grid[str(rv)] = {k: float(s.loc[k, "el_multiple_vs_baseline"]) for k in ("adverse", "severely_adverse")}
    z_half = brentq(lambda z: vasicek_pd(pd_oof, z, cfg.rho).mean() - 0.5, -12, 0)
    R["stress"] = {"oof_based": st.round(4).to_dict("records"), "el_baseline_in_sample": float(np.sum(pd_in * cfg.lgd * ead)),
                   "el_baseline_oof": float(np.sum(pd_oof * cfg.lgd * ead)), "rho_sensitivity_el_multiple": rho_grid,
                   "z_for_mean_pd_50pct": float(z_half), "scenarios": SCENARIOS}

    # ---- 9. Fairness (OOF predictions, deployed cutoff) -----------------------------------
    band = pd.cut(groups["Age"], [0, 29, 49, 120], labels=["<30", "30-49", "50+"]).astype(str)
    fair = {}
    for attr, g in {"Sex": groups["Sex"], "Age band": band}.items():
        rep = fairness_report(yv, oof_main["scorecard"], cuts["scorecard"], g).round(3)
        rep["within_group_gini"] = [round(gini(yv[(g == k).to_numpy()], oof_main["scorecard"][(g == k).to_numpy()]), 3) for k in rep["group"]]
        fair[attr] = rep.to_dict("records")
    R["fairness"] = fair

    cfg.report_dir.mkdir(exist_ok=True)
    (cfg.report_dir / "validation_results.json").write_text(json.dumps(R, indent=2, default=float))
    logger.info("Wrote %s", cfg.report_dir / "validation_results.json")
    return R


if __name__ == "__main__":
    main()

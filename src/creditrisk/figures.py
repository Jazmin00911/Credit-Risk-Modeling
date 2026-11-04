"""README figures built from the generated reports.   python -m creditrisk.figures"""
from __future__ import annotations

import json
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from creditrisk.config import ROOT, Config  # noqa: E402
from creditrisk.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger("creditrisk.figures")
OUT = ROOT / "docs" / "figures"
BLUE, GREY, RED = "#1f4e79", "#8c8c8c", "#b03a2e"


def _style(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def information_value(cfg: Config) -> None:
    iv = pd.read_csv(cfg.report_dir / "iv_table.csv").sort_values("iv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(iv["feature"], iv["iv"], color=BLUE)
    ax.axvline(0.02, color=GREY, ls=":", lw=1)
    ax.axvline(0.5, color=RED, ls="--", lw=1)
    ax.text(0.5, -0.9, " review level 0.50", color=RED, fontsize=8, va="center")
    ax.set_xlabel("Information value")
    ax.set_title("Information value by characteristic (training data)", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "information_value.png", dpi=160)
    plt.close(fig)


def calibration(cfg: Config) -> None:
    v = json.loads((cfg.report_dir / "validation_results.json").read_text())
    dec = pd.DataFrame(v["performance"]["scorecard"]["calibration_oof"]["deciles"])
    cal = v["performance"]["scorecard"]["calibration_oof"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 0.8], [0, 0.8], color=GREY, ls="--", lw=1, label="Perfect calibration")
    ax.plot(dec["predicted"], dec["observed"], "o-", color=BLUE, label="Scorecard (out-of-fold deciles)")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.set_title(f"Calibration: slope {cal['slope']:.2f}, Hosmer-Lemeshow p = {cal['hl_p']:.2f}", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "calibration.png", dpi=160)
    plt.close(fig)


def cost_curve(cfg: Config) -> None:
    v = json.loads((cfg.report_dir / "validation_results.json").read_text())
    cut = v["cutoff"]["scorecard"]
    curve = pd.read_csv(cfg.report_dir / "cost_curve_scorecard.csv")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(curve["threshold"], curve["cost"] / 1000, color=BLUE)
    lo, hi = cut["within_2pct_of_min_cost"]
    ax.axvspan(lo, hi, color=BLUE, alpha=0.12, label=f"Within 2% of minimum cost ({lo:.2f} to {hi:.2f})")
    bayes = cfg.costs.bayes_threshold
    ax.axvline(bayes, color=RED, ls="--", lw=1, label=f"Closed-form cutoff {bayes:.3f}")
    ax.axvline(cut["deployed_threshold"], color="black", lw=1, label=f"Deployed cutoff {cut['deployed_threshold']:.2f}")
    ax.set_xlabel("Decline if predicted PD is at or above")
    ax.set_ylabel("Total misclassification cost (thousands)")
    ax.set_title("Expected cost by cutoff, 5:1 cost matrix (out-of-fold, n = 1,000)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "cost_curve.png", dpi=160)
    plt.close(fig)


def main() -> None:
    setup_logging()
    cfg = Config()
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (information_value, calibration, cost_curve):
        fn(cfg)
    logger.info("Figures written to %s", OUT)


if __name__ == "__main__":
    main()

"""Central configuration. Change assumptions here, not inside functions."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CostMatrix:
    """Misclassification costs. y=1 means default ("bad")."""

    c_fn: float = 5.0  # approve a borrower who defaults
    c_fp: float = 1.0  # reject a borrower who would have repaid

    @property
    def bayes_threshold(self) -> float:
        """Optimal PD cutoff if probabilities are calibrated: c_fp / (c_fp + c_fn)."""
        return self.c_fp / (self.c_fp + self.c_fn)


@dataclass(frozen=True)
class Config:
    data_path: Path = ROOT / "data" / "raw" / "german_credit_data.csv"
    artifact_dir: Path = ROOT / "artifacts"
    report_dir: Path = ROOT / "reports"
    seed: int = 42
    test_size: float = 0.25
    n_splits: int = 5
    target: str = "Risk"
    numeric_features: tuple[str, ...] = ("Job", "Credit amount", "Duration")
    categorical_features: tuple[str, ...] = ("Housing", "Saving accounts", "Checking account", "Purpose")
    account_columns: tuple[str, ...] = ("Saving accounts", "Checking account")
    protected: tuple[str, ...] = ("Sex", "Age")  # audited for fairness, never model inputs (ablation: Age adds no Gini)
    costs: CostMatrix = field(default_factory=CostMatrix)
    lgd: float = 0.45  # assumption
    rho: float = 0.10  # assumed asset correlation for the Vasicek stress

    @property
    def features(self) -> list[str]:
        return list(self.numeric_features + self.categorical_features)

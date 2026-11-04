"""Loading and schema handling. All target/label conventions live here."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from creditrisk.config import Config

logger = logging.getLogger(__name__)

NO_ACCOUNT = "none"


def load_raw(path: Path) -> pd.DataFrame:
    logger.info("Reading %s", path)
    return pd.read_csv(path)


def prepare(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return (X, y, protected). y = 1 for default ("bad").

    Missing values in the account columns mean the applicant has no such account. That is
    information, so it becomes an explicit category instead of dropping the row.
    """
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")]).copy()
    required = set(cfg.features) | set(cfg.protected) | {cfg.target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in cfg.account_columns:
        n = int(df[col].isna().sum())
        df[col] = df[col].fillna(NO_ACCOUNT)
        logger.info("%s: %d missing values recoded to '%s'", col, n, NO_ACCOUNT)
    y = (df[cfg.target] == "bad").astype(int).rename("default")
    logger.info("Rows: %d | default rate: %.1f%%", len(df), 100 * y.mean())
    return df[cfg.features], y, df[list(cfg.protected)]


def category_levels(X: pd.DataFrame, cfg: Config) -> dict[str, list]:
    return {c: sorted(X[c].unique().tolist()) for c in cfg.categorical_features}

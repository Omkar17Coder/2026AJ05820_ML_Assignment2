"""Loading, cleaning, splitting and validating the shopper-session data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg

_TRUTHY = {"true", "1", "yes", "y", "t"}
_FALSY = {"false", "0", "no", "n", "f"}


class SchemaError(ValueError):
    """Raised when an uploaded csv does not look like the training data."""


@dataclass
class SchemaReport:
    missing_features: list[str]
    extra_columns: list[str]
    has_target: bool
    n_rows: int

    @property
    def ok(self) -> bool:
        return not self.missing_features and self.n_rows > 0


def to_binary(series: pd.Series) -> pd.Series:
    """Map TRUE/FALSE, True/False, 1/0, yes/no ... to 0/1 integers."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.astype(float) > 0).astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    unknown = sorted(set(lowered) - _TRUTHY - _FALSY)
    if unknown:
        raise SchemaError(f"column '{series.name}' has non-boolean values: {unknown[:5]}")
    return lowered.isin(_TRUTHY).astype(int)


def load_raw(path=cfg.RAW_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning shared by training and the app.

    * the UCI file writes June as "June" while every other month is 3 letters
    * Weekend / Revenue arrive as TRUE/FALSE strings or bools -> 0/1
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if "Month" in df:
        df["Month"] = df["Month"].astype(str).str.strip().str[:3].str.title()
    for col in cfg.BOOL_COLS + [cfg.TARGET]:
        if col in df:
            df[col] = to_binary(df[col])
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """125 sessions are exact copies (mostly single-page bounces) - keep one of each."""
    return df.drop_duplicates().reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified 80/20 split so both halves keep the ~15% purchase rate."""
    return train_test_split(
        df, test_size=cfg.TEST_SIZE, stratify=df[cfg.TARGET], random_state=cfg.RNG
    )


def features_target(df: pd.DataFrame):
    return df[cfg.FEATURE_COLS], df[cfg.TARGET].astype(int)


def check_schema(df: pd.DataFrame) -> SchemaReport:
    cols = set(df.columns)
    return SchemaReport(
        missing_features=[c for c in cfg.FEATURE_COLS if c not in cols],
        extra_columns=sorted(cols - set(cfg.FEATURE_COLS) - {cfg.TARGET}),
        has_target=cfg.TARGET in cols,
        n_rows=len(df),
    )


def prepare_uploaded(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Validate and clean a user csv; returns (X, y) with y=None if no target column."""
    report = check_schema(df)
    if not report.ok:
        if report.n_rows == 0:
            raise SchemaError("the uploaded file has no rows")
        raise SchemaError("missing required feature columns: " + ", ".join(report.missing_features))

    df = clean(df)
    y = df[cfg.TARGET].astype(int) if report.has_target else None
    return df[cfg.FEATURE_COLS], y

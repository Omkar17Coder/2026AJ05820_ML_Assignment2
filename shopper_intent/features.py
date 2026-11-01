"""Feature engineering + the preprocessing ColumnTransformer.

Both are fitted inside each model's Pipeline, so a saved .joblib file takes
raw csv rows and returns predictions - no separate scaler file to keep in sync.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)

from . import config as cfg
from .data import clean

ENGINEERED_LOG_COLS = ["TotalPages", "TotalDuration", "AvgSecondsPerPage"]
ENGINEERED_FLAG_COLS = ["HasPageValue"]

# skewed >= 2 in EDA -> these get the skew-reducing transform before scaling
LOG_SCALE_COLS = cfg.COUNT_COLS + cfg.DURATION_COLS + cfg.VALUE_COLS + ENGINEERED_LOG_COLS
PLAIN_SCALE_COLS = cfg.RATE_COLS + [cfg.SPECIAL_DAY_COL]
PASSTHROUGH_COLS = cfg.BOOL_COLS + ENGINEERED_FLAG_COLS
ONE_HOT_COLS = cfg.TEXT_CATEGORICAL_COLS + cfg.CODED_CATEGORICAL_COLS

# browser / traffic-type codes seen fewer times than this are pooled into one "rare" column
RARE_CATEGORY_MIN_COUNT = 30


class SessionFeatureBuilder(BaseEstimator, TransformerMixin):
    """Adds a few session-level aggregates on top of the 17 raw features.

    * TotalPages / TotalDuration - overall engagement, independent of page type
    * AvgSecondsPerPage          - browsing pace (fast skimmers vs careful readers)
    * HasPageValue               - PageValues is zero for ~78% of sessions; whether
                                   the shopper reached a page with value at all is
                                   the strongest single purchase signal in EDA
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = clean(X)  # idempotent - lets the saved pipeline accept raw UCI rows too
        for col in cfg.COUNT_COLS + cfg.DURATION_COLS + cfg.VALUE_COLS:
            X[col] = X[col].astype(float).clip(lower=0)

        X["TotalPages"] = X[cfg.COUNT_COLS].sum(axis=1)
        X["TotalDuration"] = X[cfg.DURATION_COLS].sum(axis=1)
        X["AvgSecondsPerPage"] = X["TotalDuration"] / X["TotalPages"].replace(0, np.nan)
        X["AvgSecondsPerPage"] = X["AvgSecondsPerPage"].fillna(0.0)
        X["HasPageValue"] = (X["PageValues"] > 0).astype(int)

        for col in cfg.CODED_CATEGORICAL_COLS:
            X[col] = X[col].astype(str)  # treat codes as labels, not magnitudes
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(input_features) if input_features is not None else list(cfg.FEATURE_COLS)
        return np.array(base + ENGINEERED_LOG_COLS + ENGINEERED_FLAG_COLS)


def _log_then(scaler) -> Pipeline:
    return Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", scaler),
    ])


# name -> (transform for skewed columns, transform for bounded columns).
# The trainer treats this choice like a hyper-parameter, per model.
SCALERS = {
    "log_standard": lambda: (_log_then(StandardScaler()), StandardScaler()),
    "log_robust": lambda: (_log_then(RobustScaler()), RobustScaler()),
    "log_minmax": lambda: (_log_then(MinMaxScaler()), MinMaxScaler()),
    "yeo_johnson": lambda: (PowerTransformer(method="yeo-johnson"), PowerTransformer(method="yeo-johnson")),
}
SCALER_LABELS = {
    "log_standard": "log1p + StandardScaler",
    "log_robust": "log1p + RobustScaler",
    "log_minmax": "log1p + MinMaxScaler (normalisation)",
    "yeo_johnson": "Yeo-Johnson power transform",
}
DEFAULT_SCALER = "log_standard"


def build_preprocessor(scaler: str = DEFAULT_SCALER) -> Pipeline:
    skewed_transform, bounded_transform = SCALERS[scaler]()
    one_hot = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=RARE_CATEGORY_MIN_COUNT,
        sparse_output=False,  # GaussianNB needs dense input
    )
    columns = ColumnTransformer(
        [
            ("log_scaled", skewed_transform, LOG_SCALE_COLS),
            ("scaled", bounded_transform, PLAIN_SCALE_COLS),
            ("flags", "passthrough", PASSTHROUGH_COLS),
            ("one_hot", one_hot, ONE_HOT_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("engineer", SessionFeatureBuilder()), ("columns", columns)])

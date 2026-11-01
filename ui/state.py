"""Cached loaders and the 'active dataset' shared by every tab.

Models are loaded once per server process (cache_resource); evaluation results
are cached per (model, file hash, threshold) so switching tabs is instant.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from shopper_intent import artifacts
from shopper_intent import config as cfg
from shopper_intent.data import SchemaError, prepare_uploaded
from shopper_intent.evaluation import Metrics, compute_metrics, positive_scores, predict_at
from shopper_intent.models import MODELS


@dataclass
class ActiveData:
    name: str            # file name shown in the UI
    digest: str          # content hash, used as a cache key
    raw: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series | None  # None when the csv has no Revenue column
    content: bytes       # original file, replayed by the Pipeline tab

    @property
    def labelled(self) -> bool:
        return self.y is not None


@st.cache_resource(show_spinner="Loading saved models…")
def get_models() -> dict:
    return artifacts.load_all_models()


@st.cache_data
def get_saved_metrics() -> dict:
    return artifacts.load_metrics()


@st.cache_data
def get_metadata() -> dict:
    return artifacts.load_metadata()


@st.cache_data
def get_train_frame() -> pd.DataFrame:
    return pd.read_csv(cfg.TRAIN_DATA_PATH)


@st.cache_data
def bundled_test_bytes() -> bytes:
    return cfg.TEST_DATA_PATH.read_bytes()


@st.cache_data(show_spinner=False)
def _parse(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


def load_active(name: str, content: bytes) -> ActiveData:
    """Parse + validate a csv. Raises SchemaError with a readable message."""
    try:
        raw = _parse(content)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise SchemaError(f"could not read the file as csv ({exc.__class__.__name__})") from exc
    X, y = prepare_uploaded(raw)
    return ActiveData(name=name, digest=hashlib.md5(content).hexdigest(), raw=raw, X=X, y=y, content=content)


@st.cache_data(show_spinner=False)
def scores_for(model_key: str, digest: str, _X: pd.DataFrame) -> np.ndarray:
    """P(purchase) for every row. `digest` keys the cache; `_X` is not hashed."""
    return positive_scores(get_models()[model_key], _X)


@st.cache_data(show_spinner=False)
def predictions_for(model_key: str, digest: str, _X: pd.DataFrame) -> np.ndarray:
    return get_models()[model_key].predict(_X)


def evaluate_active(model_key: str, data: ActiveData, threshold: float | None = None):
    """(Metrics, y_pred, scores) for the active dataset.

    threshold=None -> the model's own predict(), i.e. exactly what the README reports.
    """
    scores = scores_for(model_key, data.digest, data.X)
    if threshold is None:
        y_pred = predictions_for(model_key, data.digest, data.X)
    else:
        y_pred = predict_at(scores, threshold)
    metrics = compute_metrics(data.y, y_pred, scores) if data.labelled else None
    return metrics, y_pred, scores


def evaluate_all(data: ActiveData) -> dict[str, Metrics]:
    return {spec.name: evaluate_active(key, data)[0] for key, spec in MODELS.items()}

"""Reading/writing everything under model/: the five pipelines and two json files."""

from __future__ import annotations

import json

import joblib

from . import config as cfg
from .models import MODELS, model_path


def save_model(key: str, pipeline) -> None:
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path(key), compress=3)


def load_model(key: str):
    path = model_path(key)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - run `python -m shopper_intent.train` first")
    return joblib.load(path)


def load_all_models() -> dict:
    return {key: load_model(key) for key in MODELS}


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))


def read_json(path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_metrics() -> dict:
    return read_json(cfg.METRICS_PATH)


def load_metadata() -> dict:
    return read_json(cfg.METADATA_PATH)

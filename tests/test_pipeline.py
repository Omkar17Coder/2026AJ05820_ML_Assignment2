"""Quick checks: cleaning rules, schema validation and that the saved models
reproduce the numbers written to model/metrics.json (= the README table).

    python -m pytest -q
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from shopper_intent import artifacts
from shopper_intent import config as cfg
from shopper_intent.data import SchemaError, clean, prepare_uploaded
from shopper_intent.evaluation import evaluate
from shopper_intent.features import SCALERS
from shopper_intent.models import MODELS, build_pipeline
from shopper_intent.trace import trace_pipeline


@pytest.fixture(scope="module")
def test_df():
    return pd.read_csv(cfg.TEST_DATA_PATH)


def test_clean_normalises_month_and_booleans():
    raw = pd.DataFrame({"Month": ["June", "Feb"], "Weekend": ["TRUE", "FALSE"], "Revenue": [False, True]})
    out = clean(raw)
    assert out["Month"].tolist() == ["Jun", "Feb"]
    assert out["Weekend"].tolist() == [1, 0]
    assert out["Revenue"].tolist() == [0, 1]


def test_clean_is_idempotent(test_df):
    once = clean(test_df)
    pd.testing.assert_frame_equal(once, clean(once))


def test_missing_feature_column_is_reported(test_df):
    with pytest.raises(SchemaError, match="PageValues"):
        prepare_uploaded(test_df.drop(columns=["PageValues"]))


def test_csv_without_target_still_predicts(test_df):
    X, y = prepare_uploaded(test_df.drop(columns=[cfg.TARGET]))
    assert y is None
    model = artifacts.load_model("logistic_regression")
    assert len(model.predict(X)) == len(test_df)


def test_pipeline_accepts_raw_uci_rows():
    """The saved pipeline should cope with the original file's 'June' / 'TRUE' spellings."""
    raw = pd.read_csv(cfg.RAW_DATA_PATH).head(50)
    model = artifacts.load_model("decision_tree")
    assert len(model.predict(raw[cfg.FEATURE_COLS])) == 50


@pytest.mark.parametrize("key", list(MODELS))
def test_saved_model_reproduces_reported_metrics(key, test_df):
    X, y = prepare_uploaded(test_df)
    metrics, _, _ = evaluate(artifacts.load_model(key), X, y)
    reported = artifacts.load_metrics()[MODELS[key].name]
    for name, value in metrics.as_dict().items():
        assert value == pytest.approx(reported[name], abs=1e-9), name


@pytest.mark.parametrize("scaler", list(SCALERS))
def test_every_scaler_fits_and_pickles(scaler, test_df):
    X, y = prepare_uploaded(test_df.head(300))
    pipe = build_pipeline("naive_bayes", scaler=scaler).fit(X, y)
    restored = pickle.loads(pickle.dumps(pipe))
    assert np.array_equal(restored.predict(X), pipe.predict(X))


@pytest.mark.parametrize("key", list(MODELS))
def test_trace_matches_model_predict(key):
    """The Pipeline tab must report what the saved model really does."""
    content = cfg.TEST_DATA_PATH.read_bytes()
    model = artifacts.load_model(key)
    steps = {step.title: step for step in trace_pipeline(model, content)}
    X, _ = prepare_uploaded(pd.read_csv(cfg.TEST_DATA_PATH))
    assert np.array_equal(steps["Predict"].extras["y_pred"], model.predict(X))
    reported = artifacts.load_metrics()[MODELS[key].name]
    assert steps["Compute metrics"].extras["metrics"].mcc == pytest.approx(reported["mcc"], abs=1e-9)


def test_trace_without_target_stops_at_predict(test_df):
    content = test_df.drop(columns=[cfg.TARGET]).to_csv(index=False).encode()
    titles = [step.title for step in trace_pipeline(artifacts.load_model("knn"), content)]
    assert titles[-1] == "Predict"

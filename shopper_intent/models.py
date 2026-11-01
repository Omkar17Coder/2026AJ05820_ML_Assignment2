"""The five classifiers, their tuning grids and a one-line description of each.

`MODELS` is the single registry the trainer, the notebook and the app read from.
Grid keys are prefixed with `clf__` because each estimator sits inside a Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from . import config as cfg
from .features import DEFAULT_SCALER, SCALERS, build_preprocessor


@dataclass(frozen=True)
class ModelSpec:
    key: str                                  # file name stem in model/
    name: str                                 # display name used in README + app
    short: str                                # compact label for charts
    factory: Callable[..., ClassifierMixin]
    grid: dict[str, list[Any]]
    blurb: str
    # wrap the final model in CalibratedClassifierCV to get predict_proba.
    # Only SVC needs it, and it is 5x slower, so grid search runs uncalibrated.
    calibrate: bool = False
    # preprocessing options to try; trees split on rank order, so scaling can't change them
    scalers: tuple[str, ...] = tuple(SCALERS)


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in [
        ModelSpec(
            key="logistic_regression",
            name="Logistic Regression",
            short="LogReg",
            factory=lambda **kw: LogisticRegression(max_iter=2000, random_state=cfg.RNG, **kw),
            grid={
                "clf__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
            blurb="Linear decision boundary on the scaled features; outputs calibrated-ish probabilities.",
        ),
        ModelSpec(
            key="decision_tree",
            name="Decision Tree",
            short="Tree",
            factory=lambda **kw: DecisionTreeClassifier(random_state=cfg.RNG, **kw),
            grid={
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 5, 20],
                "clf__criterion": ["gini", "entropy"],
                "clf__class_weight": [None, "balanced"],
            },
            blurb="Axis-aligned if/else splits; easy to read, but prone to overfitting when deep.",
            scalers=(DEFAULT_SCALER,),
        ),
        ModelSpec(
            key="knn",
            name="KNN",
            short="KNN",
            factory=lambda **kw: KNeighborsClassifier(**kw),
            grid={
                "clf__n_neighbors": [5, 11, 21, 31, 51],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
            blurb="Votes among the k most similar past sessions; stores the whole training set instead of learning weights.",
        ),
        ModelSpec(
            key="naive_bayes",
            name="Naive Bayes",
            short="NB",
            factory=lambda **kw: GaussianNB(**kw),
            grid={"clf__var_smoothing": list(np.logspace(-9, 1, 11))},
            blurb="Gaussian NB - assumes features are independent given the class; very fast baseline.",
        ),
        ModelSpec(
            key="svm",
            name="Support Vector Machine",
            short="SVM",
            factory=lambda **kw: SVC(kernel="rbf", random_state=cfg.RNG, **kw),
            grid={
                "clf__C": [0.5, 1.0, 4.0, 8.0],
                "clf__gamma": ["scale", 0.01],
                "clf__class_weight": [None, "balanced"],
            },
            blurb="RBF-kernel maximum-margin classifier; strong on scaled data, slowest to train and predict.",
            calibrate=True,
        ),
    ]
}


def build_pipeline(key: str, scaler: str = DEFAULT_SCALER, calibrate: bool = False, **clf_params) -> Pipeline:
    """Fresh, unfitted `preprocess -> classifier` pipeline for one registry entry."""
    clf = MODELS[key].factory(**clf_params)
    if calibrate:
        # ensemble=False: calibrate on cross-validated scores, then keep one SVC fitted on all rows
        clf = CalibratedClassifierCV(clf, cv=cfg.CV_FOLDS, ensemble=False)
    return Pipeline([("prep", build_preprocessor(scaler)), ("clf", clf)])


def model_path(key: str):
    return cfg.MODEL_DIR / f"{key}.joblib"

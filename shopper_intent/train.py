"""Train, tune and save all five classifiers.

    python -m shopper_intent.train            # full grid search (~10 min)
    python -m shopper_intent.train --quick    # default hyper-parameters, seconds

Steps: clean -> dedupe -> stratified 80/20 split -> write train/test csv ->
5-fold GridSearchCV per model and preprocessing option (scored on MCC) -> evaluate on the held-out 20% ->
save model/*.joblib, model/metrics.json, model/metadata.json.
"""

from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime, timezone

import sklearn
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from . import config as cfg
from .artifacts import save_model, write_json
from .data import clean, drop_duplicates, features_target, load_raw, split
from .evaluation import Metrics, evaluate, leaderboard
from .features import DEFAULT_SCALER
from .models import MODELS, build_pipeline

# MCC uses all four confusion-matrix cells, so it can't be gamed by
# predicting "No purchase" for everyone the way accuracy can (84.5% for free)
MCC_SCORER = make_scorer(matthews_corrcoef)


def prepare_splits(verbose: bool = True):
    raw = load_raw()
    df = drop_duplicates(clean(raw))
    train_df, test_df = split(df)

    train_df.to_csv(cfg.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(cfg.TEST_DATA_PATH, index=False)
    if verbose:
        print(f"raw {raw.shape} -> deduped {df.shape}")
        print(f"train {train_df.shape}  test {test_df.shape}  "
              f"purchase rate train={train_df[cfg.TARGET].mean():.3f} test={test_df[cfg.TARGET].mean():.3f}")
    return train_df, test_df


def _plain(value):
    """np.float64(1.0) -> 1.0 so best params print and serialise cleanly."""
    return value.item() if hasattr(value, "item") else value


def tune(key: str, X, y, quick: bool):
    spec = MODELS[key]
    if quick:
        pipe = build_pipeline(key, calibrate=spec.calibrate).fit(X, y)
        return pipe, DEFAULT_SCALER, {}, float("nan"), {}

    # preprocessing is chosen like a hyper-parameter, using the training folds only
    searches = {}
    for scaler in spec.scalers:
        searches[scaler] = GridSearchCV(
            build_pipeline(key, scaler=scaler),
            spec.grid,
            scoring=MCC_SCORER,
            cv=StratifiedKFold(cfg.CV_FOLDS, shuffle=True, random_state=cfg.RNG),
            n_jobs=-1,
        ).fit(X, y)

    scaler_scores = {name: search.best_score_ for name, search in searches.items()}
    scaler = max(scaler_scores, key=scaler_scores.get)
    search = searches[scaler]
    best = {k.removeprefix("clf__"): _plain(v) for k, v in search.best_params_.items()}

    # refit on the full training split, calibrated if the model needs it
    pipe = build_pipeline(key, scaler=scaler, calibrate=spec.calibrate, **best).fit(X, y)
    return pipe, scaler, best, search.best_score_, scaler_scores


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip grid search, use defaults")
    args = parser.parse_args(argv)

    train_df, test_df = prepare_splits()
    X_train, y_train = features_target(train_df)
    X_test, y_test = features_target(test_df)

    metrics, model_meta = {}, {}
    for key, spec in MODELS.items():
        start = time.perf_counter()
        pipe, scaler, best_params, cv_mcc, scaler_scores = tune(key, X_train, y_train, args.quick)
        fit_seconds = time.perf_counter() - start

        test_metrics, _, _ = evaluate(pipe, X_test, y_test)
        save_model(key, pipe)

        metrics[spec.name] = test_metrics.as_dict()
        model_meta[key] = {
            "name": spec.name,
            "scaler": scaler,
            "best_params": best_params,
            "cv_mcc": cv_mcc,
            "cv_mcc_by_scaler": scaler_scores,
            "fit_seconds": round(fit_seconds, 2),
        }
        print(f"  {spec.name:<24} cv_mcc={cv_mcc:.4f}  test_mcc={test_metrics.mcc:.4f}  "
              f"({fit_seconds:.1f}s)  {scaler}  {best_params}")

    write_json(cfg.METRICS_PATH, metrics)
    write_json(cfg.METADATA_PATH, {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": "quick" if args.quick else "grid_search",
        "cv_folds": cfg.CV_FOLDS,
        "random_state": cfg.RNG,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "models": model_meta,
    })

    print("\nHeld-out test set (sorted by MCC):")
    board = leaderboard({name: Metrics(**m) for name, m in metrics.items()})
    print(board.round(4).to_string())


if __name__ == "__main__":
    main()

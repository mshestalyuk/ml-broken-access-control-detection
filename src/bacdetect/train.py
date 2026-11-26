"""Training: hyper-parameter search -> calibration -> threshold on valid -> one-shot test.

Spec steps:
  1. Pipeline (features + classifier)
  2. RandomizedSearchCV, 5-fold stratified CV on train
  3. threshold on valid: recall >= 0.70 with max precision
  4. CalibratedClassifierCV (risk score shown in PR comment)
  5. single evaluation on test
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from bacdetect import __version__
from bacdetect.config import RANDOM_STATE
from bacdetect.data import Dataset
from bacdetect.evaluate import Metrics, compute_metrics, select_threshold
from bacdetect.models import build_pipeline, param_distributions


@dataclass
class TrainResult:
    model_name: str
    estimator: object
    threshold: float
    best_params: dict
    valid: Metrics
    test: Metrics
    seconds: float


def train_model(
    name: str,
    ds: Dataset,
    n_iter: int = 20,
    cv: int = 5,
    max_features: int = 50_000,
    calibrate: bool = True,
) -> TrainResult:
    start = time.perf_counter()
    search = RandomizedSearchCV(
        build_pipeline(name, max_features),
        param_distributions(name),
        n_iter=n_iter,
        scoring="f1",  # recall is enforced later by the threshold
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score="raise",
    )
    search.fit(ds.train.code, ds.train.label)
    estimator = search.best_estimator_

    if calibrate or name == "svm":  # SVC has no predict_proba without calibration
        estimator = CalibratedClassifierCV(estimator, method="sigmoid", cv=min(cv, 3))
        estimator.fit(ds.train.code, ds.train.label)

    valid_proba = estimator.predict_proba(ds.valid.code)[:, 1]
    threshold = select_threshold(ds.valid.label, valid_proba)
    test_proba = estimator.predict_proba(ds.test.code)[:, 1]

    return TrainResult(
        model_name=name,
        estimator=estimator,
        threshold=threshold,
        best_params={k: str(v) for k, v in search.best_params_.items()},
        valid=compute_metrics(ds.valid.label, valid_proba, threshold),
        test=compute_metrics(ds.test.label, test_proba, threshold),
        seconds=time.perf_counter() - start,
    )


def save_artifact(result: TrainResult, dataset_name: str, out_dir: Path) -> Path:
    """Versioned model artifact + human-readable metadata next to it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model_name": result.model_name,
        "dataset": dataset_name,
        "threshold": result.threshold,
        "best_params": result.best_params,
        "valid": result.valid.to_dict(),
        "test": result.test.to_dict(),
        "train_seconds": round(result.seconds, 1),
        "bacdetect_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    path = out_dir / f"{result.model_name}.joblib"
    joblib.dump({"estimator": result.estimator, "meta": meta}, path)
    (out_dir / f"{result.model_name}.json").write_text(json.dumps(meta, indent=2))
    return path


def load_artifact(path: Path) -> dict:
    # joblib == pickle: only load artifacts we produced ourselves (CI release assets).
    return joblib.load(path)

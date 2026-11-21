"""End-to-end smoke test on the seed dataset (tiny, so metrics are not meaningful)."""

import numpy as np
import pytest

from bacdetect.evaluate import compute_metrics, mcnemar, select_threshold
from bacdetect.extract import functions_in_source
from bacdetect.models import MODEL_NAMES, build_pipeline
from bacdetect.scan import render_markdown, score
from bacdetect.train import load_artifact, save_artifact, train_model
from tests.conftest import FIXED, VULN


def test_split_has_no_project_leakage(seed_dataset):
    train, test = set(seed_dataset.train.group), set(seed_dataset.test.group)
    assert not train & test


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipelines_fit_and_predict(name, seed_dataset):
    pipe = build_pipeline(name, max_features=2000)
    pipe.fit(seed_dataset.train.code, seed_dataset.train.label)
    assert len(pipe.predict([VULN, FIXED])) == 2


def test_train_save_load_scan(seed_dataset, tmp_path):
    result = train_model("lr", seed_dataset, n_iter=2, cv=2, max_features=2000)
    assert 0.0 <= result.threshold <= 1.0
    path = save_artifact(result, seed_dataset.name, tmp_path)
    artifact = load_artifact(path)

    fns = functions_in_source(VULN, "app/views.py")
    findings = score(fns, artifact)
    assert len(findings) == 1 and 0.0 <= findings[0].risk <= 1.0
    md = render_markdown(findings, artifact["meta"], seconds=0.1)
    assert "bacdetect-report" in md and "1** changed function" in md


def test_threshold_reaches_target_recall():
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.6, 0.7])
    t = select_threshold(y, p, min_recall=0.75)
    assert compute_metrics(y, p, t).recall >= 0.75


def test_mcnemar_identical_models():
    y = [0, 1, 1, 0]
    assert mcnemar(y, y, y)["p_value"] == 1.0

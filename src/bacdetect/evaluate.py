"""Metrics, success-criteria check and McNemar test"""

from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from bacdetect.config import CRITERIA, SuccessCriteria


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    baseline_accuracy: float
    threshold: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def majority_baseline(y_true) -> float:
    """Accuracy of always predicting the most frequent class (~54% on CodeXGLUE)."""
    counts = Counter(y_true)
    return max(counts.values()) / len(y_true)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    auc = roc_auc_score(y_true, proba) if len(set(y_true)) > 1 else float("nan")
    return Metrics(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=auc,
        baseline_accuracy=majority_baseline(y_true),
        threshold=threshold,
    )


def select_threshold(y_true, proba, min_recall: float = CRITERIA.recall) -> float:
    """Highest-precision threshold that still keeps recall >= min_recall (on valid set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # precision/recall have len(thresholds)+1 entries; the last one has no threshold
    ok = np.where(recall[:-1] >= min_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())  # cannot reach target recall: flag everything risky
    best = ok[np.argmax(precision[ok])]
    return float(thresholds[best])


def check_criteria(m: Metrics, c: SuccessCriteria = CRITERIA) -> dict[str, bool]:
    return {
        f"recall >= {c.recall}": m.recall >= c.recall,
        f"precision >= {c.precision}": m.precision >= c.precision,
        f"f1 >= {c.f1}": m.f1 >= c.f1,
        f"roc_auc >= {c.roc_auc}": m.roc_auc >= c.roc_auc,
        f"accuracy >= baseline + {c.accuracy_over_baseline:.0%}": (
            m.accuracy >= m.baseline_accuracy + c.accuracy_over_baseline
        ),
    }


def mcnemar(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test: do models A and B make significantly different errors?"""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right = pred_a == y_true
    b_right = pred_b == y_true
    b = int(np.sum(a_right & ~b_right))  # A right, B wrong
    c = int(np.sum(~a_right & b_right))  # A wrong, B right
    p = 1.0 if b + c == 0 else binomtest(b, b + c, 0.5).pvalue
    return {"a_only_correct": b, "b_only_correct": c, "p_value": float(p)}


def format_report(name: str, m: Metrics) -> str:
    lines = [f"## {name}", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in m.to_dict().items()]
    lines += ["", "| criterion | pass |", "|---|---|"]
    lines += [f"| {k} | {'yes' if ok else 'no'} |" for k, ok in check_criteria(m).items()]
    return "\n".join(lines)

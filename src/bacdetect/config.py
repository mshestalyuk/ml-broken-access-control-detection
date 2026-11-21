"""Project-wide constants: paths, labels and success criteria from the project spec."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SEED_DATASET = DATA_DIR / "bac_idor" / "seed.jsonl"

BENIGN, VULNERABLE = 0, 1
LABEL_NAMES = {BENIGN: "BENIGN", VULNERABLE: "VULNERABLE"}

RANDOM_STATE = 42

# Marker used to find (and update) our own PR comment instead of spamming new ones.
COMMENT_MARKER = "<!-- bacdetect-report -->"


@dataclass(frozen=True)
class SuccessCriteria:
    """Thresholds from the 'Kryteria sukcesu' table. To be confirmed after experiment 1."""

    recall: float = 0.70
    precision: float = 0.50
    f1: float = 0.58
    roc_auc: float = 0.65
    accuracy_over_baseline: float = 0.04
    bac_idor_recall: float = 0.60
    max_pr_seconds: float = 60.0


CRITERIA = SuccessCriteria()

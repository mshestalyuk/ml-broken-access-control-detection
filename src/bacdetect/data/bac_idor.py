import json
from pathlib import Path

from sklearn.model_selection import StratifiedGroupKFold

from bacdetect.config import RANDOM_STATE
from bacdetect.data import Dataset, Split

BAC_CWES = {"CWE-284", "CWE-285", "CWE-639", "CWE-862", "CWE-863"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _subset(records: list[dict], idx) -> Split:
    return Split(
        code=[records[i]["code"] for i in idx],
        label=[int(records[i]["label"]) for i in idx],
        group=[records[i].get("project", "unknown") for i in idx],
    )


def load_bac_idor(path: Path, n_folds: int = 5) -> Dataset:
    """Grouped + stratified split: 1 fold -> test, 1 fold -> valid, rest -> train."""
    records = read_jsonl(path)
    labels = [int(r["label"]) for r in records]
    groups = [r.get("project", "unknown") for r in records]

    n_groups = len(set(groups))
    n_folds = max(3, min(n_folds, n_groups))
    folds = list(
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE).split(
            records, labels, groups
        )
    )
    test_idx = folds[0][1]
    valid_idx = folds[1][1]
    held = set(test_idx) | set(valid_idx)
    train_idx = [i for i in range(len(records)) if i not in held]

    return Dataset(
        name=path.stem,
        train=_subset(records, train_idx),
        valid=_subset(records, valid_idx),
        test=_subset(records, test_idx),
    )

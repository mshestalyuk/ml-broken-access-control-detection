import hashlib

from bacdetect.data import Dataset, Split
from bacdetect.preprocessing import clean_code

HF_DATASET = "google/code_x_glue_cc_defect_detection"


def _fingerprint(code: str) -> str:
    return hashlib.sha1(clean_code(code).encode(), usedforsecurity=False).hexdigest()


def deduplicate(train: Split, others: list[Split]) -> Split:
    """Drop train functions that also appear (after cleaning) in valid/test.

    Train/test duplicates inflate scores, as noted in the spec.
    """
    held_out = {_fingerprint(c) for s in others for c in s.code}
    seen: set[str] = set()
    keep = []
    for i, code in enumerate(train.code):
        fp = _fingerprint(code)
        if fp in held_out or fp in seen:
            continue
        seen.add(fp)
        keep.append(i)
    return Split(
        code=[train.code[i] for i in keep],
        label=[train.label[i] for i in keep],
        group=[train.group[i] for i in keep] if train.group else [],
    )


def load_codexglue(limit: int | None = None) -> Dataset:
    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install the data extra first: uv sync --extra data") from e

    raw = load_dataset(HF_DATASET)

    def to_split(name: str) -> Split:
        part = raw[name] if limit is None else raw[name].select(range(min(limit, len(raw[name]))))
        return Split(
            code=list(part["func"]),
            label=[int(t) for t in part["target"]],
            group=list(part["project"]),
        )

    valid, test = to_split("validation"), to_split("test")
    train = deduplicate(to_split("train"), [valid, test])
    return Dataset(name="codexglue", train=train, valid=valid, test=test)

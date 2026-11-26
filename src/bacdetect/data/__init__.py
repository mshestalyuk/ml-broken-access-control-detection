"""Dataset loaders. Every loader returns a Dataset with train/valid/test splits."""

from dataclasses import dataclass, field


@dataclass
class Split:
    code: list[str]
    label: list[int]
    group: list[str] = field(default_factory=list)  # project / repo, for grouped splits

    def __len__(self) -> int:
        return len(self.code)


@dataclass
class Dataset:
    name: str
    train: Split
    valid: Split
    test: Split

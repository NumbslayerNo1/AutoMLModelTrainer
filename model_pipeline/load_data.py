"""Load Parquet tables and validate feature / label columns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class LoadedData:
    """Container returned by :func:`load_dataset`."""

    df: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_column: str


def load_dataset(
    path: str | Path,
    *,
    label_column: str,
    feature_columns: list[str],
    exclude_columns: Iterable[str] | None = None,
) -> LoadedData:
    """Read Parquet from file or directory (partitioned dataset) and validate columns.

    No hard-coded filesystem roots — *path* is always caller-supplied.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    df = pd.read_parquet(p)

    if label_column not in df.columns:
        raise KeyError(f"label_column {label_column!r} not found in data")

    feats = list(feature_columns)
    if exclude_columns:
        ex = set(exclude_columns)
        feats = [c for c in feats if c not in ex]

    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"feature_columns not in data: {missing[:10]!r}")

    use_feats = [c for c in feats if c in df.columns]
    if not use_feats:
        raise ValueError("No usable feature columns after validation")

    return LoadedData(
        df=df,
        feature_columns=tuple(use_feats),
        label_column=label_column,
    )

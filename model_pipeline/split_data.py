"""Train/validation splits: random (sklearn) and time-window (inclusive bounds)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


class SplitStrategy(str, Enum):
    RANDOM = "random"
    TIME_WINDOW = "time_window"


def split_data(
    df: pd.DataFrame,
    strategy: str | SplitStrategy,
    *,
    label_column: Optional[str] = None,
    test_size: float = 0.25,
    random_state: Optional[int] = 30,
    stratify: bool = False,
    date_column: Optional[str] = None,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train_df, val_df)``.

    * **random** — ``sklearn.model_selection.train_test_split`` on ``df``.
    * **time_window** — train rows with ``date_column`` in ``[train_start, train_end]`` (string
      comparison, same semantics as notebook date strings ``YYYY-MM-DD``); validation is the complement
      within ``df``.
    """
    strat = SplitStrategy(strategy) if isinstance(strategy, str) else strategy

    if strat == SplitStrategy.RANDOM:
        stratify_labels = None
        if stratify:
            if not label_column:
                raise ValueError("label_column is required when stratify=True")
            if label_column not in df.columns:
                raise KeyError(f"label_column {label_column!r} not in DataFrame")
            stratify_labels = df[label_column]
        if not 0 < test_size < 1:
            raise ValueError("test_size must be strictly between 0 and 1")
        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        return train_df, val_df

    if strat == SplitStrategy.TIME_WINDOW:
        if not date_column or date_column not in df.columns:
            raise KeyError("time_window requires date_column present in df")
        if train_start is None or train_end is None:
            raise ValueError("time_window requires train_start and train_end")
        col = df[date_column].astype(str)
        mask = (col >= str(train_start)) & (col <= str(train_end))
        train_df = df.loc[mask].copy()
        val_df = df.loc[~mask].copy()
        return train_df, val_df

    raise ValueError(f"Unknown strategy: {strategy!r}")

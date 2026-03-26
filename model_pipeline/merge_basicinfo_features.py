"""Merge basicinfo-style Parquet with a raw feature Parquet for model evaluation.

Uses LightGBM model feature names to select columns from the feature table, inner-joins on
``trace_id`` with basicinfo (dropping overlapping non-key columns from basicinfo so the
feature table wins for model inputs).
"""

from __future__ import annotations

import errno
from pathlib import Path
from typing import Optional, Union

import lightgbm as lgb
import pandas as pd


def merge_basicinfo_features(
    model_file: Union[str, Path],
    basicinfo_parquet: Union[str, Path],
    features_parquet: Union[str, Path],
    merged_out: Union[str, Path],
    *,
    max_rows: Optional[int] = None,
) -> Path:
    """Inner-merge basicinfo with feature columns; return path to written Parquet."""
    booster = lgb.Booster(model_file=str(model_file))
    feature_names = list(booster.feature_name())

    print(f"Loading features: {features_parquet}", flush=True)
    import pyarrow.parquet as pq

    features_path = Path(features_parquet)
    avail = set(pq.ParquetFile(features_path).schema_arrow.names)
    read_cols = ["trace_id"] + [c for c in feature_names if c in avail]
    missing_model_feats = [c for c in feature_names if c not in avail]
    if missing_model_feats:
        raise KeyError(
            f"features parquet missing {len(missing_model_feats)} model columns "
            f"(showing first 10): {missing_model_feats[:10]!r}"
        )

    features_df = pd.read_parquet(features_path, columns=read_cols)
    print(f"features columns read: {len(read_cols) - 1} features, rows={len(features_df)}", flush=True)

    print(f"Loading basicinfo: {basicinfo_parquet}", flush=True)
    base_df = pd.read_parquet(basicinfo_parquet)
    print(f"basicinfo rows={len(base_df)}, cols={len(base_df.columns)}", flush=True)

    overlap = (set(base_df.columns) & set(features_df.columns)) - {"trace_id"}
    if overlap:
        print(
            f"Dropping {len(overlap)} overlapping columns from basicinfo "
            f"(feature table wins for model inputs). Example: {list(overlap)[:5]!r}",
            flush=True,
        )
        base_df = base_df.drop(columns=list(overlap), errors="ignore")

    merged = base_df.merge(features_df, on="trace_id", how="inner")
    print(f"merged shape: {merged.shape}", flush=True)

    if max_rows is not None and max_rows > 0 and len(merged) > max_rows:
        merged = merged.iloc[:max_rows].copy()
        print(f"trimmed to max_rows={max_rows}: {merged.shape}", flush=True)

    merged_out = Path(merged_out).resolve()
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged.to_parquet(merged_out, index=False)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                "No space left for merged Parquet. Free disk space or write to a larger volume.",
            ) from e
        raise
    print(f"Wrote {merged_out}", flush=True)
    return merged_out

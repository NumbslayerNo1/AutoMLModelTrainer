"""Load trained models and score feature matrices (LGB first)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from model_pipeline.train_model import ModelType


def predict_scores(
    model_path: str | Path,
    predict_df: pd.DataFrame,
    *,
    model_type: ModelType = ModelType.LGB,
    feature_order: Optional[list[str]] = None,
    num_threads: Optional[int] = None,
) -> np.ndarray:
    import lightgbm as lgb

    if model_type != ModelType.LGB:
        raise NotImplementedError(f"predict_scores not implemented for {model_type!r}")
    booster = lgb.Booster(model_file=str(model_path))
    names = list(feature_order) if feature_order is not None else booster.feature_name()
    missing = [c for c in names if c not in predict_df.columns]
    if missing:
        raise KeyError(f"predict_df missing feature columns: {missing[:20]}")
    kwargs = {}
    if num_threads is not None:
        kwargs["num_threads"] = int(num_threads)
    return booster.predict(predict_df[names], **kwargs)


def predict_and_save_parquet(
    model_path: str | Path,
    predict_df: pd.DataFrame,
    *,
    id_column: str,
    score_column: str,
    output_path: str | Path,
    model_type: ModelType = ModelType.LGB,
    feature_order: Optional[list[str]] = None,
    num_threads: Optional[int] = None,
) -> None:
    scores = predict_scores(
        model_path,
        predict_df,
        model_type=model_type,
        feature_order=feature_order,
        num_threads=num_threads,
    )
    out = pd.DataFrame({id_column: predict_df[id_column], score_column: scores})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

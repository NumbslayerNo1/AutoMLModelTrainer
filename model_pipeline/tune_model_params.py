"""Hyperparameter grids and tuning (LGB-first)."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from model_pipeline.metrics import compute_metric, resolve_metric
from model_pipeline.train_model import ModelType, TuningReport, train


def build_param_grid(
    model_type: ModelType, grid_spec: dict[str, list | tuple]
) -> list[dict[str, Any]]:
    """Cartesian product of lists in *grid_spec* (keys = param names)."""
    if not grid_spec:
        return []
    keys = list(grid_spec)
    vals = [list(grid_spec[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def tune(
    train_df: pd.DataFrame,
    *,
    model_type: ModelType = ModelType.LGB,
    label: str,
    features: list[str],
    base_params: dict[str, Any],
    grid_spec: dict[str, list | tuple],
    metric: str | Callable[..., float] = "auc",
    n_splits: int = 1,
    val_df: Optional[pd.DataFrame] = None,
    random_state: int = 42,
    num_boost_round: int = 80,
    workdir: str | Path = ".",
    is_early_stopping: bool = False,
) -> TuningReport:
    """Evaluate *grid_spec* variants; higher metric is better.

    * ``n_splits == 1`` and *val_df* set → single holdout score per param dict.
    * ``n_splits > 1`` → stratified K-fold on *train_df* (ignores *val_df*).
    """
    if model_type != ModelType.LGB:
        raise NotImplementedError(f"tune not implemented for {model_type!r}")
    _ = resolve_metric(metric) if isinstance(metric, str) else metric

    combos = build_param_grid(model_type, grid_spec)
    if not combos:
        empty = pd.DataFrame()
        return TuningReport(
            model_type=model_type,
            results=empty,
            best_params={},
            best_score=float("nan"),
        )

    if n_splits == 1 and val_df is None:
        raise ValueError("n_splits=1 requires val_df")

    import lightgbm as lgb

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    if n_splits == 1:
        for i, delta in enumerate(combos):
            params = {**base_params, **delta}
            mp = workdir / f"tune_model_{i}.txt"
            ip = workdir / f"tune_importance_{i}.csv"
            train(
                train_df,
                val_df,
                features,
                label,
                params,
                model_type=model_type,
                model_path=str(mp),
                importance_path=str(ip),
                is_early_stopping=is_early_stopping,
                num_boost_round=num_boost_round,
                log_evaluation_period=10**9,
                refit_on_full_data=False,
            )
            booster = lgb.Booster(model_file=str(mp))
            pred = booster.predict(val_df[features])
            score = compute_metric(metric, val_df[label].values, pred)
            row = {**delta, "metric_mean": score, "metric_std": 0.0}
            rows.append(row)
    else:
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        y = train_df[label].values
        for i, delta in enumerate(combos):
            params = {**base_params, **delta}
            fold_scores: list[float] = []
            for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
                tr = train_df.iloc[tr_idx]
                va = train_df.iloc[va_idx]
                mp = workdir / f"tune_model_{i}_fold{fold}.txt"
                ip = workdir / f"tune_imp_{i}_fold{fold}.csv"
                train(
                    tr,
                    va,
                    features,
                    label,
                    params,
                    model_type=model_type,
                    model_path=str(mp),
                    importance_path=str(ip),
                    is_early_stopping=is_early_stopping,
                    num_boost_round=num_boost_round,
                    log_evaluation_period=10**9,
                    refit_on_full_data=False,
                )
                booster = lgb.Booster(model_file=str(mp))
                pred = booster.predict(va[features])
                fold_scores.append(
                    compute_metric(metric, va[label].values, pred)
                )
            row = {
                **delta,
                "metric_mean": float(np.mean(fold_scores)),
                "metric_std": float(np.std(fold_scores)),
            }
            rows.append(row)

    results = pd.DataFrame(rows)
    best_idx = int(results["metric_mean"].idxmax())
    best_row = results.iloc[best_idx]
    param_keys = list(grid_spec.keys())
    best_params = {k: best_row[k] for k in param_keys}
    best_score = float(best_row["metric_mean"])
    return TuningReport(
        model_type=model_type,
        results=results,
        best_params=best_params,
        best_score=best_score,
    )

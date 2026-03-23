"""Multi-backend training entry (LGB implemented; XGB reserved)."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from model_pipeline.split_data import SplitStrategy, split_data


class ModelType(Enum):
    LGB = "lgb"
    XGB = "xgb"


@dataclass
class TrainResult:
    model_type: ModelType
    model_path: str
    importance_path: str
    mid_model_path: Optional[str] = None
    mid_importance_path: Optional[str] = None
    best_iteration: Optional[int] = None
    final_num_boost_round: Optional[int] = None


@dataclass
class TuningReport:
    model_type: ModelType
    results: pd.DataFrame
    best_params: dict[str, Any]
    best_score: float


def _effective_val(val_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if val_df is None or val_df.empty:
        return None
    return val_df


def _train_lgb_once(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    features: list[str],
    label: str,
    params: dict[str, Any],
    *,
    model_path: str,
    importance_path: str,
    is_early_stopping: bool,
    num_boost_round: int,
    log_evaluation_period: int,
    early_stopping_rounds: int,
    num_threads: Optional[int],
) -> Any:
    import lightgbm as lgb
    from lightgbm import early_stopping, log_evaluation

    p = dict(params)
    if num_threads is not None:
        p["num_threads"] = int(num_threads)

    val_df = _effective_val(val_df)

    if val_df is not None:
        lgtrain = lgb.Dataset(
            train_df[features], label=train_df[label], feature_name=features
        )
        lgval = lgb.Dataset(val_df[features], label=val_df[label], feature_name=features)
        callbacks = [log_evaluation(period=log_evaluation_period)]
        if is_early_stopping:
            callbacks.append(early_stopping(stopping_rounds=early_stopping_rounds))
        valid_sets = [lgtrain, lgval]
    else:
        lgtrain = lgb.Dataset(
            train_df[features], label=train_df[label], feature_name=features
        )
        callbacks = [log_evaluation(period=log_evaluation_period)]
        valid_sets = [lgtrain]

    model = lgb.train(
        p,
        lgtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    out_m = Path(model_path)
    out_m.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_m))

    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    imp_df = pd.DataFrame(
        {"feature_name": feat_names, "importance": importance}
    ).sort_values("importance", ascending=False)
    out_i = Path(importance_path)
    out_i.parent.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(out_i, index=False)

    gc.collect()
    return model


def train(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame],
    features: list[str],
    label: str,
    params: dict[str, Any],
    *,
    model_type: ModelType = ModelType.LGB,
    model_path: str,
    importance_path: str,
    is_early_stopping: bool = True,
    num_boost_round: int = 5000,
    log_evaluation_period: int = 600,
    early_stopping_rounds: int = 100,
    refit_on_full_data: bool = False,
    boost_round_multiplier: float = 1.1,
    mid_model_path: Optional[str] = None,
    mid_importance_path: Optional[str] = None,
    holdout_test_size: float = 0.25,
    holdout_random_state: int = 30,
    num_threads: Optional[int] = None,
) -> TrainResult:
    if model_type == ModelType.XGB:
        raise NotImplementedError(
            "ModelType.XGB is not implemented in this milestone; use ModelType.LGB."
        )
    if model_type != ModelType.LGB:
        raise NotImplementedError(f"Unsupported model_type: {model_type!r}")

    import lightgbm as lgb

    val_df = _effective_val(val_df)
    work_train = train_df
    work_val = val_df

    if work_val is None and is_early_stopping:
        work_train, work_val = split_data(
            train_df,
            SplitStrategy.RANDOM,
            label_column=label,
            test_size=holdout_test_size,
            random_state=holdout_random_state,
            stratify=False,
        )

    if not refit_on_full_data:
        model = _train_lgb_once(
            work_train,
            work_val,
            features,
            label,
            params,
            model_path=model_path,
            importance_path=importance_path,
            is_early_stopping=is_early_stopping,
            num_boost_round=num_boost_round,
            log_evaluation_period=log_evaluation_period,
            early_stopping_rounds=early_stopping_rounds,
            num_threads=num_threads,
        )
        best_it = getattr(model, "best_iteration", None)
        return TrainResult(
            model_type=model_type,
            model_path=str(model_path),
            importance_path=str(importance_path),
            best_iteration=best_it,
            final_num_boost_round=model.num_trees(),
        )

    stem = Path(model_path).stem
    parent = Path(model_path).parent
    mid_mp = mid_model_path or str(parent / f"{stem}_mid.txt")
    mid_ip = mid_importance_path or str(parent / f"{stem}_mid_importance.csv")

    _train_lgb_once(
        work_train,
        work_val,
        features,
        label,
        params,
        model_path=mid_mp,
        importance_path=mid_ip,
        is_early_stopping=is_early_stopping,
        num_boost_round=num_boost_round,
        log_evaluation_period=log_evaluation_period,
        early_stopping_rounds=early_stopping_rounds,
        num_threads=num_threads,
    )

    mid_booster = lgb.Booster(model_file=str(mid_mp))
    if is_early_stopping:
        final_rounds = max(1, int(mid_booster.num_trees() * boost_round_multiplier))
    else:
        final_rounds = num_boost_round

    if work_val is None:
        full_df = work_train.copy()
    else:
        full_df = pd.concat([work_train, work_val], ignore_index=True)
    _train_lgb_once(
        full_df,
        None,
        features,
        label,
        params,
        model_path=model_path,
        importance_path=importance_path,
        is_early_stopping=False,
        num_boost_round=final_rounds,
        log_evaluation_period=log_evaluation_period,
        early_stopping_rounds=early_stopping_rounds,
        num_threads=num_threads,
    )

    final_model = lgb.Booster(model_file=str(model_path))
    return TrainResult(
        model_type=model_type,
        model_path=str(model_path),
        importance_path=str(importance_path),
        mid_model_path=str(mid_mp),
        mid_importance_path=str(mid_ip),
        best_iteration=getattr(mid_booster, "best_iteration", None),
        final_num_boost_round=final_model.num_trees(),
    )

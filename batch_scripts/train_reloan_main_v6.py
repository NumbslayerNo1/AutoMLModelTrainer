#!/usr/bin/env python3
"""Load mex reloan main V6 train parquet and run ``model_pipeline.train`` like ``model_train_predict.ipynb``.

Hyperparameters match the notebook cell: 主模型 V6 with ``is_early_stoppping = True``,
``train_model_with_all_data`` (early stopping then refit on train+val with ``1.1 ×`` trees).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Repo root on sys.path when run as ``python batch_scripts/train_mex_reloan_main_v6.py``
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from model_pipeline.load_data import load_dataset
from model_pipeline.train_model import train

DATA_PATH = Path("/data1/mex_reloan_data/mex_reloan_main_v6_train_data")
VAL_DATA_PATH = Path("/data1/mex_reloan_data/mex_reloan_main_v6_validation_data")
FEATURE_DESC_CSV = Path("/data1/mex_reloan_data/复贷主模型V6入模特征描述.csv")
LABEL_COLUMN = "1pd7"

# Notebook: ``num_boost_round = 30000`` (主模型 V6 + early stopping + 全量二次训练)
NOTEBOOK_NUM_BOOST_ROUND = 30_000
NOTEBOOK_LOG_EVAL_PERIOD = 600
NOTEBOOK_EARLY_STOPPING_ROUNDS = 100

DEFAULT_OUTPUT_DIR = Path("/data1/bogeng/data_output/pipeline_test")


def notebook_lgb_params(scale_pos_weight: float) -> dict[str, Any]:
    """Same keys as ``model_train_predict.ipynb`` (CPU ``device_type`` block)."""
    num_threads = int(os.environ.get("MEX_RELOAN_NUM_THREADS", "80"))
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.02,
        "bagging_fraction": 0.8,
        "bagging_freq": 100,
        "min_child_weight": 1000,
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 4,
        "num_leaves": 15,
        "reg_alpha": 0.5,
        "reg_lambda": 20,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "min_data_in_bin": 2000,
        "bin_construct_sample_cnt": 100_000,
        "num_threads": num_threads,
        "device_type": "cpu",
        "random_state": 123,
    }


def _feature_columns_from_csv(path: Path) -> list[str]:
    desc = pd.read_csv(path, sep=",")
    if "etl_feature_name" not in desc.columns:
        raise KeyError(
            f"Expected column 'etl_feature_name' in {path}, got {list(desc.columns)}"
        )
    return desc["etl_feature_name"].astype(str).tolist()


def _load_val_if_available(path: Path, features: list[str], label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    val_df = pd.read_parquet(path)
    need = list(dict.fromkeys(list(features) + [label]))
    missing = [c for c in need if c not in val_df.columns]
    if missing:
        raise KeyError(f"val data missing columns: {missing[:15]!r}")
    return val_df


def main() -> None:
    feat_csv = Path(os.environ.get("MEX_RELOAN_FEATURE_DESC_CSV", FEATURE_DESC_CSV))
    data_path = Path(os.environ.get("MEX_RELOAN_TRAIN_DATA", DATA_PATH))
    val_path = Path(os.environ.get("MEX_RELOAN_VAL_DATA", VAL_DATA_PATH))

    features = _feature_columns_from_csv(feat_csv)
    loaded = load_dataset(
        data_path,
        label_column=LABEL_COLUMN,
        feature_columns=features,
    )
    df = loaded.df
    features = list(loaded.feature_columns)
    label = loaded.label_column

    pos_rate = float(df[label].mean())
    scale_pos_weight = round((1.0 - pos_rate) / pos_rate, 2) if pos_rate > 0 else 1.0
    print(f"scale_pos_weight: {scale_pos_weight}", flush=True)

    params = notebook_lgb_params(scale_pos_weight)

    val_df = _load_val_if_available(val_path, features, label)

    output_dir = Path(os.environ.get("MEX_RELOAN_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"output_dir: {output_dir}", flush=True)
    model_path = output_dir / "model.txt"
    importance_path = output_dir / "importance.csv"

    result = train(
        df,
        val_df,
        features,
        label,
        params,
        model_path=str(model_path),
        importance_path=str(importance_path),
        is_early_stopping=True,
        num_boost_round=NOTEBOOK_NUM_BOOST_ROUND,
        log_evaluation_period=NOTEBOOK_LOG_EVAL_PERIOD,
        early_stopping_rounds=NOTEBOOK_EARLY_STOPPING_ROUNDS,
        refit_on_full_data=True,
        boost_round_multiplier=1.1,
    )
    print(result, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tune LightGBM hyperparameters (validation AUC) from oracle V6 init, then train best model.

Uses :func:`model_pipeline.tune_model_params.tune` with ``init_params`` from
``oracle_train_reloan_v6.notebook_lgb_params``. After search, retrains once with the
oracle notebook recipe (early stopping + refit on train+val) and writes:

* ``model_best.txt`` — LightGBM model
* ``feature_importance_best.txt`` — feature importance (CSV lines, ``.txt`` suffix as requested)
* ``config_best.txt`` — JSON with metric, best validation score during search, and full param dict

Each trial writes under ``<output_dir>/tune_workspace/``: ``trial_<n>_model.txt``,
``trial_<n>_feature_importance.csv``, and ``trial_<n>_trial_config.json``. By default the
run resumes from those configs (skip completed params, continue from best metric). New
bests are copied to ``<output_dir>/validation_record/improvement_*``. After tuning,
``best_of_all_*`` files are written under ``<output_dir>``. Use ``--clean-run`` to delete
**everything** under ``<output_dir>`` (then recreate ``tune_workspace`` and
``validation_record``); confirmation if non-empty. Use
``--sample-frac`` (e.g. ``0.01``) to subsample train and validation rows for dry runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_oracle_module():
    path = Path(__file__).resolve().parent / "oracle_train_reloan_v6.py"
    spec = importlib.util.spec_from_file_location("oracle_train_reloan_v6", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load oracle module from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


import pandas as pd

from model_pipeline.load_data import load_dataset
from model_pipeline.train_model import train
from model_pipeline.tune_model_params import (
    BEST_OF_ALL_CONFIG,
    BEST_OF_ALL_FEATURE_IMPORTANCE,
    BEST_OF_ALL_MODEL,
    tune,
)

DEFAULT_OUTPUT_DIR = Path("/data1/bogeng/data_output/validation_03272026")


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune LightGBM on validation, then train the best config."
    )
    parser.add_argument(
        "--clean-run",
        action="store_true",
        help=(
            "Remove all files and folders under output_dir (BG_TUNING_OUTPUT_DIR), then "
            "run validation from scratch (prompt if non-empty; non-interactive: "
            "TUNE_CONFIRM_CLEAN=1). Default: resume from trial_*_trial_config.json."
        ),
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=1.0,
        metavar="F",
        help=(
            "After load, randomly keep this fraction of rows in train and validation "
            "(0 < F ≤ 1). Default 1.0 uses full data. Seed: BG_SAMPLE_SEED (default 42)."
        ),
    )
    args = parser.parse_args()

    oracle = _load_oracle_module()

    feat_csv = Path(os.environ.get("MEX_RELOAN_FEATURE_DESC_CSV", oracle.FEATURE_DESC_CSV))
    data_path = Path(os.environ.get("MEX_RELOAN_TRAIN_DATA", oracle.DATA_PATH))
    val_path = Path(os.environ.get("MEX_RELOAN_VAL_DATA", oracle.VAL_DATA_PATH))
    output_dir = Path(os.environ.get("BG_TUNING_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)
    tune_workspace = output_dir / "tune_workspace"
    tune_workspace.mkdir(parents=True, exist_ok=True)

    features = oracle._feature_columns_from_csv(feat_csv)
    loaded = load_dataset(
        data_path,
        label_column=oracle.LABEL_COLUMN,
        feature_columns=features,
    )
    train_df = loaded.df
    features = list(loaded.feature_columns)
    label = loaded.label_column

    val_df = oracle._load_val_if_available(val_path, features, label)
    if val_df is None or len(val_df) == 0:
        raise FileNotFoundError(
            f"Validation data required for tuning; missing or empty: {val_path}"
        )

    sf = float(args.sample_frac)
    if sf <= 0 or sf > 1.0:
        raise ValueError("--sample-frac must satisfy 0 < F ≤ 1")
    if sf < 1.0:
        sample_seed = int(os.environ.get("BG_SAMPLE_SEED", "42"))
        n_tr, n_va = len(train_df), len(val_df)
        train_df = train_df.sample(frac=sf, random_state=sample_seed).reset_index(
            drop=True
        )
        val_df = val_df.sample(frac=sf, random_state=sample_seed + 1).reset_index(
            drop=True
        )
        print(
            f"sample_frac={sf}: train {n_tr}->{len(train_df)}, val {n_va}->{len(val_df)}",
            flush=True,
        )

    pos_rate = float(train_df[label].mean())
    scale_pos_weight = (
        round((1.0 - pos_rate) / pos_rate, 2) if pos_rate > 0 else 1.0
    )
    print(f"scale_pos_weight: {scale_pos_weight}", flush=True)

    init_params = oracle.notebook_lgb_params(scale_pos_weight)
    metric = os.environ.get("BG_TUNE_METRIC", "auc")

    tune_num_rounds = int(os.environ.get("TUNE_NUM_BOOST_ROUND", "8000"))
    tune_max_expansion = int(os.environ.get("TUNE_MAX_EXPANSION_ROUNDS", "8"))

    print(f"output_dir: {output_dir}", flush=True)
    print(f"tune_workspace: {tune_workspace}", flush=True)
    print(f"Tuning (metric={metric}, num_boost_round={tune_num_rounds})...", flush=True)

    report = tune(
        train_df,
        val_df,
        label=label,
        features=features,
        init_params=init_params,
        metric=metric,
        max_expansion_rounds=tune_max_expansion,
        workdir=tune_workspace,
        output_dir=output_dir,
        is_early_stopping=True,
        num_boost_round=tune_num_rounds,
        clean_run=args.clean_run,
    )

    print(f"Best {metric}: {report.best_score}", flush=True)
    print(f"Best delta (tuned keys): {report.best_params}", flush=True)
    print(
        f"Search best artifacts: {output_dir / BEST_OF_ALL_MODEL}, "
        f"{output_dir / BEST_OF_ALL_FEATURE_IMPORTANCE}, "
        f"{output_dir / BEST_OF_ALL_CONFIG}",
        flush=True,
    )

    full_best_params = {**init_params, **report.best_params}
    config_path = output_dir / "config_best.txt"
    model_path = output_dir / "model_best.txt"
    importance_path = output_dir / "feature_importance_best.txt"

    payload = {
        "metric": metric,
        "best_validation_score_search": report.best_score,
        "lgb_parameters": full_best_params,
    }
    config_path.write_text(
        json.dumps(payload, indent=2, default=_json_default, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("Training final model (oracle recipe: early stop + refit)...", flush=True)
    result = train(
        train_df,
        val_df,
        features,
        label,
        full_best_params,
        model_path=str(model_path),
        importance_path=str(importance_path),
        is_early_stopping=True,
        num_boost_round=oracle.NOTEBOOK_NUM_BOOST_ROUND,
        log_evaluation_period=oracle.NOTEBOOK_LOG_EVAL_PERIOD,
        early_stopping_rounds=oracle.NOTEBOOK_EARLY_STOPPING_ROUNDS,
        refit_on_full_data=True,
        boost_round_multiplier=1.1,
    )
    print(result, flush=True)
    print(f"Wrote {model_path}", flush=True)
    print(f"Wrote {importance_path}", flush=True)
    print(f"Wrote {config_path}", flush=True)


if __name__ == "__main__":
    main()

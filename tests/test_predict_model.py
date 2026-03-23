from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_pipeline.predict_model import predict_and_save_parquet, predict_scores
from model_pipeline.train_model import train


def _train_small_model(tmp_path, binary_df):
    tr = binary_df.iloc[:120]
    va = binary_df.iloc[120:150]
    mp = tmp_path / "m.txt"
    ip = tmp_path / "i.csv"
    train(
        tr,
        va,
        ["f1", "f2"],
        "y",
        {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "seed": 0,
        },
        model_path=str(mp),
        importance_path=str(ip),
        is_early_stopping=False,
        num_boost_round=40,
        log_evaluation_period=10**9,
    )
    return mp


def test_predict_scores_shape(tmp_path, binary_df):
    mp = _train_small_model(tmp_path, binary_df)
    te = binary_df.iloc[150:].copy()
    s = predict_scores(mp, te)
    assert len(s) == len(te)


def test_predict_missing_columns(tmp_path, binary_df):
    mp = _train_small_model(tmp_path, binary_df)
    bad = pd.DataFrame({"trace_id": [1], "f1": [0.0]})
    with pytest.raises(KeyError, match="missing feature"):
        predict_scores(mp, bad)


def test_predict_and_save_parquet_roundtrip(tmp_path, binary_df):
    mp = _train_small_model(tmp_path, binary_df)
    te = binary_df.iloc[150:].copy()
    out = tmp_path / "scores.parquet"
    predict_and_save_parquet(
        mp,
        te,
        id_column="trace_id",
        score_column="score",
        output_path=out,
    )
    assert out.is_file()
    back = pd.read_parquet(out)
    assert set(back.columns) == {"trace_id", "score"}
    assert len(back) == len(te)

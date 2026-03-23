"""US1: parquet path → load_dataset → shape & columns."""

from pathlib import Path

import pandas as pd

from model_pipeline.load_data import load_dataset


def test_acceptance_load_parquet_file(tmp_path):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
            "target": [0, 1],
        }
    )
    p = tmp_path / "t.parquet"
    df.to_parquet(p, index=False)
    ld = load_dataset(p, label_column="target", feature_columns=["a", "b"])
    assert ld.df.shape == (2, 3)
    assert list(ld.feature_columns) == ["a", "b"]


def test_acceptance_partitioned_like_directory(tmp_path):
    d = tmp_path / "ds"
    d.mkdir()
    pd.DataFrame({"x": [1.0], "y": [0]}).to_parquet(d / "f1.parquet", index=False)
    pd.DataFrame({"x": [2.0], "y": [1]}).to_parquet(d / "f2.parquet", index=False)
    ld = load_dataset(d, label_column="y", feature_columns=["x"])
    assert len(ld.df) == 2

from pathlib import Path

import pandas as pd
import pytest

from model_pipeline.load_data import load_dataset


def test_load_dataset_roundtrip(tmp_path, binary_df):
    fp = tmp_path / "d.parquet"
    binary_df.to_parquet(fp, index=False)
    ld = load_dataset(
        fp,
        label_column="y",
        feature_columns=["f1", "f2"],
    )
    assert ld.df.shape[0] == len(binary_df)
    assert set(ld.feature_columns) == {"f1", "f2"}
    assert ld.label_column == "y"


def test_load_dataset_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.parquet", label_column="y", feature_columns=["a"])


def test_load_dataset_bad_label(tmp_path, binary_df):
    fp = tmp_path / "d.parquet"
    binary_df.to_parquet(fp, index=False)
    with pytest.raises(KeyError, match="label_column"):
        load_dataset(fp, label_column="missing", feature_columns=["f1"])


def test_load_dataset_no_features(tmp_path, binary_df):
    fp = tmp_path / "d.parquet"
    binary_df.to_parquet(fp, index=False)
    with pytest.raises(ValueError, match="No usable feature"):
        load_dataset(fp, label_column="y", feature_columns=[])


def test_load_dataset_partitioned_dir(tmp_path, binary_df):
    part = tmp_path / "part"
    part.mkdir()
    binary_df.iloc[:100].to_parquet(part / "p0.parquet", index=False)
    binary_df.iloc[100:].to_parquet(part / "p1.parquet", index=False)
    ld = load_dataset(part, label_column="y", feature_columns=["f1", "f2"])
    assert len(ld.df) == len(binary_df)

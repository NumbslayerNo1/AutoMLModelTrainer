import numpy as np
import pandas as pd
import pytest

from model_pipeline.split_data import SplitStrategy, split_data


def test_random_split_shape(binary_df):
    tr, va = split_data(
        binary_df,
        SplitStrategy.RANDOM,
        label_column="y",
        test_size=0.25,
        random_state=0,
        stratify=False,
    )
    assert len(tr) + len(va) == len(binary_df)
    assert len(va) == pytest.approx(50, abs=2)


def test_random_stratify_rates(binary_df):
    tr, va = split_data(
        binary_df,
        "random",
        label_column="y",
        test_size=0.25,
        random_state=0,
        stratify=True,
    )
    r_tr = tr["y"].mean()
    r_va = va["y"].mean()
    assert abs(r_tr - r_va) < 0.08


def test_random_bad_test_size(binary_df):
    with pytest.raises(ValueError, match="test_size"):
        split_data(binary_df, "random", test_size=1.0)


def test_random_stratify_requires_label(binary_df):
    with pytest.raises(ValueError, match="label_column"):
        split_data(binary_df, "random", stratify=True)


def test_time_window_masks():
    df = pd.DataFrame(
        {
            "dt": ["2024-01-01", "2024-06-01", "2025-01-01"],
            "v": [1, 2, 3],
        }
    )
    tr, va = split_data(
        df,
        SplitStrategy.TIME_WINDOW,
        date_column="dt",
        train_start="2024-01-01",
        train_end="2024-12-31",
    )
    assert set(tr["v"]) == {1, 2}
    assert list(va["v"]) == [3]


def test_time_window_empty_train():
    df = pd.DataFrame({"dt": ["2020-01-01"], "v": [1]})
    tr, va = split_data(
        df,
        "time_window",
        date_column="dt",
        train_start="2024-01-01",
        train_end="2024-12-31",
    )
    assert len(tr) == 0
    assert len(va) == 1


def test_time_window_missing_column():
    df = pd.DataFrame({"x": [1]})
    with pytest.raises(KeyError):
        split_data(
            df,
            "time_window",
            date_column="dt",
            train_start="2024-01-01",
            train_end="2024-12-31",
        )

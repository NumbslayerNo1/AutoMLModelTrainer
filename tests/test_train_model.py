from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_pipeline.train_model import ModelType, train


@pytest.fixture
def lgb_params():
    return {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.05,
        "num_leaves": 8,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "seed": 1,
    }


def test_train_lgb_writes_artifacts(tmp_path, binary_df, lgb_params):
    features = ["f1", "f2"]
    label = "y"
    tr = binary_df.iloc[:160]
    va = binary_df.iloc[160:]
    mp = tmp_path / "m.txt"
    ip = tmp_path / "imp.csv"
    res = train(
        tr,
        va,
        features,
        label,
        lgb_params,
        model_path=str(mp),
        importance_path=str(ip),
        is_early_stopping=False,
        num_boost_round=30,
        log_evaluation_period=10**9,
    )
    assert res.model_type == ModelType.LGB
    assert mp.is_file()
    assert ip.is_file()


def test_train_xgb_not_implemented(tmp_path, binary_df, lgb_params):
    with pytest.raises(NotImplementedError, match="XGB"):
        train(
            binary_df.iloc[:50],
            None,
            ["f1", "f2"],
            "y",
            lgb_params,
            model_type=ModelType.XGB,
            model_path=str(tmp_path / "x.txt"),
            importance_path=str(tmp_path / "x.csv"),
            is_early_stopping=False,
            num_boost_round=5,
            log_evaluation_period=10**9,
        )


def test_internal_holdout_uses_train_labels_only(tmp_path, binary_df, lgb_params, monkeypatch):
    """Regression: labels for internal split must come from train_df, not a stray global."""
    called = {}

    def fake_split_data(df, strategy, **kwargs):
        called["label_col"] = kwargs.get("label_column")
        assert "y" in df.columns
        n = len(df)
        return df.iloc[: n // 2].copy(), df.iloc[n // 2 :].copy()

    monkeypatch.setattr("model_pipeline.train_model.split_data", fake_split_data)

    train(
        binary_df,
        None,
        ["f1", "f2"],
        "y",
        lgb_params,
        model_path=str(tmp_path / "m.txt"),
        importance_path=str(tmp_path / "i.csv"),
        is_early_stopping=True,
        num_boost_round=20,
        log_evaluation_period=10**9,
    )
    assert called["label_col"] == "y"


def test_refit_creates_mid_and_final(tmp_path, binary_df, lgb_params):
    tr = binary_df.iloc[:160]
    va = binary_df.iloc[160:]
    mp = tmp_path / "final.txt"
    ip = tmp_path / "final.csv"
    res = train(
        tr,
        va,
        ["f1", "f2"],
        "y",
        lgb_params,
        model_path=str(mp),
        importance_path=str(ip),
        is_early_stopping=True,
        num_boost_round=80,
        log_evaluation_period=10**9,
        refit_on_full_data=True,
        boost_round_multiplier=1.1,
    )
    assert res.mid_model_path and Path(res.mid_model_path).is_file()
    assert mp.is_file()
    assert res.final_num_boost_round is not None

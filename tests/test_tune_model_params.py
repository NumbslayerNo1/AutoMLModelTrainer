import pytest

from model_pipeline.train_model import ModelType
from model_pipeline.tune_model_params import build_param_grid, tune


def test_build_param_grid_cartesian():
    g = build_param_grid(
        ModelType.LGB,
        {"num_leaves": [8, 16], "learning_rate": [0.05, 0.1]},
    )
    assert len(g) == 4
    lrs = {x["learning_rate"] for x in g}
    assert lrs == {0.05, 0.1}


def test_build_param_grid_empty():
    assert build_param_grid(ModelType.LGB, {}) == []


def test_tune_empty_grid(tmp_path, binary_df):
    tr = binary_df.iloc[:100]
    va = binary_df.iloc[100:130]
    rep = tune(
        tr,
        model_type=ModelType.LGB,
        label="y",
        features=["f1", "f2"],
        base_params={
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 8,
            "seed": 0,
        },
        grid_spec={},
        val_df=va,
        workdir=tmp_path / "tune_empty",
        num_boost_round=20,
    )
    assert rep.results.empty


def test_tune_requires_val_when_n_splits_1(tmp_path, binary_df):
    with pytest.raises(ValueError, match="val_df"):
        tune(
            binary_df.iloc[:50],
            model_type=ModelType.LGB,
            label="y",
            features=["f1", "f2"],
            base_params={"objective": "binary", "verbosity": -1, "seed": 0},
            grid_spec={"learning_rate": [0.1]},
            n_splits=1,
            val_df=None,
            workdir=tmp_path,
        )


def test_tune_bad_metric(tmp_path, binary_df):
    with pytest.raises(KeyError):
        tune(
            binary_df.iloc[:80],
            model_type=ModelType.LGB,
            label="y",
            features=["f1", "f2"],
            base_params={"objective": "binary", "verbosity": -1, "seed": 0},
            grid_spec={"learning_rate": [0.1]},
            metric="not_real",
            val_df=binary_df.iloc[80:120],
            workdir=tmp_path / "bm",
        )


def test_tune_xgb_not_implemented(tmp_path, binary_df):
    with pytest.raises(NotImplementedError):
        tune(
            binary_df.iloc[:80],
            model_type=ModelType.XGB,
            label="y",
            features=["f1", "f2"],
            base_params={},
            grid_spec={"a": [1]},
            val_df=binary_df.iloc[80:120],
            workdir=tmp_path,
        )

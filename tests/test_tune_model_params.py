import pandas as pd
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
        va,
        label="y",
        features=["f1", "f2"],
        init_params={
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "num_leaves": 8,
            "seed": 0,
        },
        tune_keys=[],
        workdir=tmp_path / "tune_empty",
        num_boost_round=20,
    )
    assert rep.results.empty


def test_tune_validate_df_empty_raises(tmp_path, binary_df):
    empty_va = pd.DataFrame(columns=binary_df.columns)
    with pytest.raises(ValueError, match="validate_df"):
        tune(
            binary_df.iloc[:50],
            empty_va,
            label="y",
            features=["f1", "f2"],
            init_params={"objective": "binary", "verbosity": -1, "seed": 0},
            param_lists={"learning_rate": [0.1]},
            tune_keys=["learning_rate"],
            max_expansion_rounds=0,
            workdir=tmp_path,
        )


def test_tune_bad_metric(tmp_path, binary_df):
    with pytest.raises(KeyError):
        tune(
            binary_df.iloc[:80],
            binary_df.iloc[80:120],
            label="y",
            features=["f1", "f2"],
            init_params={"objective": "binary", "verbosity": -1, "seed": 0},
            param_lists={"learning_rate": [0.1]},
            tune_keys=["learning_rate"],
            max_expansion_rounds=0,
            metric="not_real",
            workdir=tmp_path / "bm",
        )


def test_tune_xgb_not_implemented(tmp_path, binary_df):
    with pytest.raises(NotImplementedError):
        tune(
            binary_df.iloc[:80],
            binary_df.iloc[80:120],
            label="y",
            features=["f1", "f2"],
            init_params={},
            param_lists={"a": [1]},
            tune_keys=["a"],
            model_type=ModelType.XGB,
            max_expansion_rounds=0,
            workdir=tmp_path,
        )


def test_tune_running_history_written_and_resume_skips_rerun(tmp_path, binary_df):
    work = tmp_path / "tune_ws"
    tr = binary_df.iloc[:120]
    va = binary_df.iloc[120:]
    kwargs = dict(
        label="y",
        features=["f1", "f2"],
        init_params={
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 11,
            "learning_rate": 0.1,
            "num_leaves": 12,
        },
        param_lists={
            "learning_rate": [0.05, 0.2],
            "num_leaves": [8, 16],
        },
        tune_keys=["learning_rate", "num_leaves"],
        metric="auc",
        max_expansion_rounds=0,
        workdir=work,
        num_boost_round=40,
    )
    rep1 = tune(tr, va, **kwargs, clean_run=True, output_dir=work.parent)
    cfgs = sorted(work.glob("trial_*_trial_config.json"))
    assert len(cfgs) == len(rep1.results)

    rep2 = tune(tr, va, **kwargs, clean_run=False, output_dir=work.parent)
    assert len(rep2.results) == len(rep1.results)
    assert rep2.best_score == rep1.best_score
    assert len(sorted(work.glob("trial_*_trial_config.json"))) == len(rep1.results)


def test_tune_clean_run_non_empty_workdir_requires_confirm(
    tmp_path, binary_df, monkeypatch
):
    work = tmp_path / "dirty"
    work.mkdir()
    (work / "old.txt").write_text("x", encoding="utf-8")
    monkeypatch.delenv("TUNE_CONFIRM_CLEAN", raising=False)
    with pytest.raises(RuntimeError, match="TUNE_CONFIRM_CLEAN"):
        tune(
            binary_df.iloc[:80],
            binary_df.iloc[80:120],
            label="y",
            features=["f1", "f2"],
            init_params={
                "objective": "binary",
                "metric": "auc",
                "verbosity": -1,
                "seed": 0,
                "learning_rate": 0.1,
            },
            param_lists={"learning_rate": [0.05, 0.2]},
            tune_keys=["learning_rate"],
            max_expansion_rounds=0,
            workdir=work,
            num_boost_round=20,
            clean_run=True,
        )


def test_tune_clean_run_non_empty_workdir_ok_with_env(
    tmp_path, binary_df, monkeypatch
):
    work = tmp_path / "dirty2"
    work.mkdir()
    (work / "old.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("TUNE_CONFIRM_CLEAN", "1")
    rep = tune(
        binary_df.iloc[:80],
        binary_df.iloc[80:120],
        label="y",
        features=["f1", "f2"],
        init_params={
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 0,
            "learning_rate": 0.1,
        },
        param_lists={"learning_rate": [0.05, 0.2]},
        tune_keys=["learning_rate"],
        max_expansion_rounds=0,
        workdir=work,
        num_boost_round=20,
        clean_run=True,
    )
    assert len(rep.results) == 2


def test_tune_require_clean_confirmation_false_skips_prompt(
    tmp_path, binary_df, monkeypatch
):
    work = tmp_path / "dirty3"
    work.mkdir()
    (work / "old.txt").write_text("x", encoding="utf-8")
    monkeypatch.delenv("TUNE_CONFIRM_CLEAN", raising=False)
    rep = tune(
        binary_df.iloc[:80],
        binary_df.iloc[80:120],
        label="y",
        features=["f1", "f2"],
        init_params={
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "seed": 0,
            "learning_rate": 0.1,
        },
        param_lists={"learning_rate": [0.05]},
        tune_keys=["learning_rate"],
        max_expansion_rounds=0,
        workdir=work,
        num_boost_round=20,
        clean_run=True,
        require_clean_confirmation=False,
    )
    assert len(rep.results) == 1

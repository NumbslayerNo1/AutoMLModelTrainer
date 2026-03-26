"""US5: 2×2 grid on small data returns TuningReport with best_params."""

from model_pipeline.train_model import ModelType
from model_pipeline.tune_model_params import tune


def test_acceptance_small_grid(tmp_path, binary_df):
    tr = binary_df.iloc[:120]
    va = binary_df.iloc[120:]
    rep = tune(
        tr,
        va,
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
        workdir=tmp_path / "tune_run",
        num_boost_round=40,
    )
    assert len(rep.results) == 4
    assert set(rep.best_params.keys()) == {"learning_rate", "num_leaves"}
    assert rep.best_score == rep.results["metric_mean"].max()

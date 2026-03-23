"""US3: real LightGBM train → model file + importance CSV + booster trees."""

from pathlib import Path

from model_pipeline.train_model import ModelType, train


def test_acceptance_train_and_reload_booster(tmp_path, binary_df):
    import lightgbm as lgb

    tr = binary_df.iloc[:160]
    va = binary_df.iloc[160:]
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "seed": 42,
    }
    mp = tmp_path / "model.txt"
    ip = tmp_path / "fi.csv"
    train(
        tr,
        va,
        ["f1", "f2"],
        "y",
        params,
        model_type=ModelType.LGB,
        model_path=str(mp),
        importance_path=str(ip),
        is_early_stopping=True,
        num_boost_round=200,
        log_evaluation_period=10**9,
        early_stopping_rounds=20,
    )
    assert mp.is_file()
    assert ip.is_file()
    booster = lgb.Booster(model_file=str(mp))
    assert booster.num_trees() >= 1

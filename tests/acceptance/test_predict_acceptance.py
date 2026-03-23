"""US4: load trained booster, predict, AUC is finite."""

import numpy as np
from sklearn import metrics

from model_pipeline.predict_model import predict_scores
from model_pipeline.train_model import train


def test_acceptance_predict_auc_finite(tmp_path, binary_df):
    tr = binary_df.iloc[:140]
    va = binary_df.iloc[140:170]
    te = binary_df.iloc[170:].copy()
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": 0.1,
        "num_leaves": 12,
        "seed": 99,
    }
    mp = tmp_path / "mdl.txt"
    ip = tmp_path / "imp.csv"
    train(
        tr,
        va,
        ["f1", "f2"],
        "y",
        params,
        model_path=str(mp),
        importance_path=str(ip),
        is_early_stopping=False,
        num_boost_round=60,
        log_evaluation_period=10**9,
    )
    pred = predict_scores(mp, te)
    assert len(pred) == len(te)
    # Ensure both classes present so AUC is well-defined
    assert te["y"].nunique() == 2
    auc = metrics.roc_auc_score(te["y"], pred)
    assert not np.isnan(auc)

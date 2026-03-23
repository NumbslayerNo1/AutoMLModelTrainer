import pytest

from model_pipeline.metrics import compute_metric, resolve_metric
from model_pipeline.train_model import ModelType, TrainResult, TuningReport


def test_model_type_values():
    assert ModelType.LGB.value == "lgb"
    assert ModelType.XGB.value == "xgb"


def test_train_result_fields():
    r = TrainResult(
        model_type=ModelType.LGB,
        model_path="/tmp/m.txt",
        importance_path="/tmp/i.csv",
    )
    assert r.mid_model_path is None


def test_tuning_report_minimal():
    import pandas as pd

    rep = TuningReport(
        model_type=ModelType.LGB,
        results=pd.DataFrame({"metric_mean": [0.5]}),
        best_params={},
        best_score=0.5,
    )
    assert rep.best_score == 0.5


def test_resolve_metric_auc():
    fn = resolve_metric("auc")
    assert callable(fn)


def test_resolve_metric_unknown():
    with pytest.raises(KeyError, match="Unknown metric"):
        resolve_metric("not_a_real_metric")


def test_compute_metric_string():
    import numpy as np

    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.7, 0.9])
    v = compute_metric("auc", y, s)
    assert v == 1.0


def test_register_metric():
    from sklearn import metrics

    from model_pipeline.metrics import register_metric, resolve_metric

    register_metric("logloss_alias", metrics.log_loss)
    assert callable(resolve_metric("logloss_alias"))

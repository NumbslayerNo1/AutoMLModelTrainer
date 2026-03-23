import numpy as np
import pandas as pd
import pytest


def pytest_collection_modifyitems(config, items):
    try:
        import lightgbm  # noqa: F401
    except OSError:
        skip = pytest.mark.skip(
            reason="LightGBM native library failed to load (e.g. install Homebrew libomp)."
        )
        needle = (
            "test_train_model",
            "test_predict_model",
            "test_tune_model_params",
            "test_train_model_acceptance",
            "test_predict_acceptance",
            "test_tune_acceptance",
        )
        for item in items:
            p = str(item.fspath)
            if any(n in p for n in needle):
                item.add_marker(skip)


@pytest.fixture
def binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.5 * x1 -0.3 * x2)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"f1": x1, "f2": x2, "y": y, "trace_id": np.arange(n)})


@pytest.fixture
def tiny_train_val(binary_df):
    tr = binary_df.iloc[:140].reset_index(drop=True)
    va = binary_df.iloc[140:].reset_index(drop=True)
    return tr, va

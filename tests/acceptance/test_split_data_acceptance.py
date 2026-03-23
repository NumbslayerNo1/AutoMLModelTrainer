"""US2: time window boundaries + stratified random rates."""

import pandas as pd

from model_pipeline.split_data import split_data


def test_acceptance_time_window_inclusive_bounds():
    df = pd.DataFrame(
        {
            "month": ["2024-01", "2024-02", "2024-03", "2025-01"],
            "trace_id": range(4),
        }
    )
    tr, _ = split_data(
        df,
        "time_window",
        date_column="month",
        train_start="2024-01",
        train_end="2024-03",
    )
    assert tr["month"].min() >= "2024-01"
    assert tr["month"].max() <= "2024-03"
    assert len(tr) == 3


def test_acceptance_stratified_random_stable():
    rng = __import__("numpy").random.default_rng(7)
    y = rng.integers(0, 2, size=400)
    df = pd.DataFrame({"y": y, "f": rng.normal(size=400)})
    tr, va = split_data(
        df,
        "random",
        label_column="y",
        test_size=0.25,
        random_state=123,
        stratify=True,
    )
    assert abs(tr["y"].mean() - va["y"].mean()) < 0.05

# Quickstart: `model_pipeline`

Minimal offline flow: **load → split → train → predict** (LGB backend).

## Setup

```bash
cd /path/to/TrainLGBMexico
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

On macOS, if `import lightgbm` fails with `libomp.dylib`, install OpenMP (e.g. `brew install libomp`).

## Example

```python
from pathlib import Path

from model_pipeline import (
    ModelType,
    load_dataset,
    predict_scores,
    split_data,
    train,
)

data_path = Path("/your/data.parquet")  # file or directory of parquet parts
ld = load_dataset(
    data_path,
    label_column="y",
    feature_columns=["f1", "f2"],
)
df = ld.df
features = list(ld.feature_columns)
label = ld.label_column

train_df, val_df = split_data(
    df,
    "random",
    label_column=label,
    test_size=0.25,
    random_state=42,
    stratify=True,
)

params = {
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "seed": 42,
}

out_dir = Path("./artifacts")
out_dir.mkdir(exist_ok=True)
train(
    train_df,
    val_df,
    features,
    label,
    params,
    model_type=ModelType.LGB,
    model_path=str(out_dir / "model.txt"),
    importance_path=str(out_dir / "importance.csv"),
    is_early_stopping=True,
    num_boost_round=500,
)

te = train_df.head(10)  # or your OOT frame
scores = predict_scores(out_dir / "model.txt", te)
```

For hyperparameter search, use `build_param_grid` and `tune` from `model_pipeline.tune_model_params`.

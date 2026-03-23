# Spec: Model training pipeline refactor

## Summary

Extract logic from `model_train_predict.ipynb` into importable Python modules for loading data, splitting, **training models** (multi-backend API; **LGB default and only implementation in the first iteration**), predicting, and cross-validating hyperparameters.

## Requirements

1. **load_data.py** — Read features and labels from a user-supplied path (parquet/directory as in notebook).
2. **split_data.py** — Split by configurable ratio and strategy: time-window (e.g. date column range), random (`train_test_split`-style), and extensible hooks for future strategies.
3. **train_model.py** — Train from explicit inputs: **`train_df`**, optional **`val_df`**, **`features`** (column list), **`label`** (column name), **`params`** (backend hyperparameters dict), **`model_path`** / **`importance_path`** (or directory + stems), flags **`is_early_stopping`**, **`num_boost_round`**, optional **`refit_on_full_data`** / **`boost_round_multiplier`**, plus **`model_type` (backend) selector** defaulting to **LGB**; support **XGB** and **other frameworks later**; **phase 1 implements only LGB**. Preserve optional two-phase flow (mid model + full-data refit) for the LGB path as in `train_model_with_all_data`.
4. **predict_model.py** — Score rows from a loaded model and prediction feature matrix (optional save path / column name); **`model_type` defaults to LGB**, other backends when added use the same argument.
5. **tune_model_params.py** — Build candidate hyperparameter grids/windows **per `model_type`** and run cross-validation on train/validation splits with a chosen metric (LGB first; same registry as `train_model`).

## Success criteria

- Notebook (or a thin script) can orchestrate the same workflows using the modules without duplicating core training/predict logic.
- Split and CV strategies are selectable by name or enum, not hard-coded paths.

## Assumptions

- **Training API** is backend-agnostic at the call site (`model_type` with default **LGB**); **XGB and others** are specified in design and reached through the same entry points when implemented.
- **First implementation milestone** trains and saves **only LGB** models; XGB parity (save format, feature importance, early stopping, refit) comes in a later milestone.
- Data on disk remains **Parquet** (single file or dataset directory) unless extended in `load_data`.

## Terminology

- **LGB** 指由 **`lightgbm`** 包实现的后端；规格、计划与对外 API 默认值均用 **LGB** 表述。实现代码使用包名 **`lightgbm`**（例如 `import lightgbm as lgb`）。

## Out of scope

- Replacing `modelEvaluation` / `psiCalculation` or notebook plotting.
- Deployment or REST APIs.

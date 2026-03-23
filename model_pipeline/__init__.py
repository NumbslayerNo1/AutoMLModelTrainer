"""Model training pipeline package (LGB-first, multi-backend API)."""

from model_pipeline.load_data import LoadedData, load_dataset
from model_pipeline.metrics import compute_metric, register_metric, resolve_metric
from model_pipeline.predict_model import predict_and_save_parquet, predict_scores
from model_pipeline.split_data import SplitStrategy, split_data
from model_pipeline.train_model import ModelType, TrainResult, TuningReport, train
from model_pipeline.tune_model_params import build_param_grid, tune

__all__ = [
    "LoadedData",
    "ModelType",
    "SplitStrategy",
    "TrainResult",
    "TuningReport",
    "build_param_grid",
    "compute_metric",
    "load_dataset",
    "predict_and_save_parquet",
    "predict_scores",
    "register_metric",
    "resolve_metric",
    "split_data",
    "train",
    "tune",
]

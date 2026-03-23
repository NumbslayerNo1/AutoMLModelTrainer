"""Metric name → sklearn-compatible callables (y_true, y_score)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn import metrics

MetricFn = Callable[..., float]

_REGISTRY: dict[str, MetricFn] = {
    "auc": metrics.roc_auc_score,
}


def register_metric(name: str, fn: MetricFn) -> None:
    _REGISTRY[name.lower()] = fn


def resolve_metric(name: str) -> MetricFn:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def compute_metric(metric: str | MetricFn, y_true: Any, y_score: Any) -> float:
    fn = resolve_metric(metric) if isinstance(metric, str) else metric
    return float(fn(y_true, y_score))

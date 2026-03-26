"""Hyperparameter search for LightGBM around an initial setting, scored on a validation set.

Strategy (high level)
---------------------
* **Local grid**: For each tuned key, build a *small* candidate list around ``init_params``
  (multiplicative steps for rates/regularization, integer neighbors for tree depth/leaves,
  clipped perturbations for subsampling). This matches common LGB practice: the most
  sensitive knobs are ``learning_rate``, tree size (``num_leaves`` / ``max_depth``), and
  L1/L2 regularization; subsample / column sample are secondary.

* **Validation metric**: Each candidate = ``{**init_params, **delta}`` trained on
  ``train_df``, scored on ``validate_df`` with ``metric`` (default ``auc``). Higher is better.

* **Boundary expansion**: If the best point lies on the **min or max** of any tuned
  dimension, that axis is extended (e.g. lower learning rates, larger ``num_leaves``) and
  only **new** combinations (not previously evaluated) are tried. Rounds stop when there
  are no boundary hits, ``max_expansion_rounds`` is reached, or the best validation metric
  does **not improve** after an expansion round (plateau / decline).
"""

from __future__ import annotations

import math
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from model_pipeline.metrics import compute_metric, resolve_metric
from model_pipeline.train_model import ModelType, TuningReport, train

# Keys commonly tuned for binary LGB; subset is used if missing from init_params.
_DEFAULT_TUNE_KEYS: Tuple[str, ...] = (
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_weight",
    "reg_alpha",
    "reg_lambda",
    "colsample_bytree",
    "bagging_fraction",
    "bagging_freq",
)


def build_param_grid(
    model_type: ModelType, grid_spec: dict[str, list | tuple]
) -> list[dict[str, Any]]:
    """Cartesian product of lists in *grid_spec* (keys = param names)."""
    if not grid_spec:
        return []
    keys = list(grid_spec)
    vals = [list(grid_spec[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def _finite_sorted_unique(values: Sequence[Any]) -> List[Any]:
    out: List[Any] = []
    seen: Set[Any] = set()
    for v in values:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return sorted(out, key=lambda x: (isinstance(x, str), x))


def suggest_lgb_param_lists(
    init_params: dict[str, Any],
    tune_keys: Optional[Sequence[str]] = None,
) -> dict[str, list]:
    """Build small candidate lists *around* ``init_params`` for LightGBM.

    Uses bounded multiplicative grids for continuous params and neighbor integers for
    tree hyperparameters — a standard local search pattern before wider search or BO.
    """
    keys = list(tune_keys) if tune_keys is not None else list(_DEFAULT_TUNE_KEYS)
    keys = [k for k in keys if k in init_params]
    out: dict[str, list] = {}

    for k in keys:
        v0 = init_params[k]
        if k == "learning_rate":
            lr = float(v0)
            facs = (0.5, 0.75, 1.0, 1.25, 1.5)
            cand = [max(1e-4, min(0.5, lr * f)) for f in facs]
            out[k] = _finite_sorted_unique(cand)
        elif k == "num_leaves":
            n = int(v0)
            cand = {max(2, n - 8), max(2, n - 4), n, n + 4, n + 8}
            out[k] = sorted(cand)
        elif k == "max_depth":
            d = int(v0)
            cand = {max(3, d - 2), max(3, d - 1), d, d + 1, min(16, d + 2)}
            out[k] = sorted(cand)
        elif k == "min_child_weight":
            m = float(v0)
            if m <= 0:
                m = 1.0
            mults = (0.5, 0.75, 1.0, 1.5, 2.0)
            cand = [max(1e-6, m * mu) for mu in mults]
            out[k] = _finite_sorted_unique(cand)
        elif k in ("reg_alpha", "reg_lambda"):
            r = float(v0)
            base = max(r, 1e-8)
            mults = (0.25, 0.5, 1.0, 2.0, 4.0)
            cand = [max(1e-8, base * mu) for mu in mults]
            out[k] = _finite_sorted_unique(cand)
        elif k in ("colsample_bytree", "feature_fraction", "bagging_fraction"):
            x = float(v0)
            cand = [
                max(0.05, min(1.0, x - 0.15)),
                max(0.05, min(1.0, x - 0.05)),
                x,
                max(0.05, min(1.0, x + 0.05)),
                max(0.05, min(1.0, x + 0.15)),
            ]
            out[k] = _finite_sorted_unique(cand)
        elif k == "bagging_freq":
            b = int(v0)
            cand = sorted({0, max(0, b - 50), b, b + 50, b + 100})
            out[k] = cand
        else:
            if isinstance(v0, (int, float)):
                if isinstance(v0, int):
                    out[k] = sorted({max(1, v0 - 2), v0 - 1, v0, v0 + 1, v0 + 2})
                else:
                    f = float(v0)
                    out[k] = _finite_sorted_unique(
                        [f * 0.5, f * 0.75, f, f * 1.25, f * 1.5]
                    )
            else:
                out[k] = [v0]

    return out


def _combo_frozen(delta: dict[str, Any], keys: Sequence[str]) -> Tuple[Any, ...]:
    return tuple((k, delta[k]) for k in sorted(keys))


def _find_boundary_keys(
    param_lists: dict[str, list],
    best_delta: dict[str, Any],
    tune_keys: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Return (at_min, at_max) key lists where best sits on grid edge."""
    at_low: List[str] = []
    at_high: List[str] = []
    for k in tune_keys:
        if k not in best_delta or k not in param_lists:
            continue
        vals = sorted(set(param_lists[k]), key=lambda x: (isinstance(x, str), x))
        if not vals:
            continue
        bv = best_delta[k]
        if bv == vals[0]:
            at_low.append(k)
        if bv == vals[-1]:
            at_high.append(k)
    return at_low, at_high


def _expand_param_list(
    key: str,
    values: List[Any],
    best_val: Any,
    at_low: bool,
    at_high: bool,
    init_params: dict[str, Any],
) -> List[Any]:
    """Append new candidates beyond current min/max for one hyperparameter."""
    s = list(values)
    base = init_params.get(key, best_val)

    if key == "learning_rate":
        lo = float(min(s))
        hi = float(max(s))
        if at_low:
            s.extend([max(1e-4, lo / 1.5), max(1e-4, lo / 1.25)])
        if at_high:
            s.extend([min(0.5, hi * 1.25), min(0.5, hi * 1.5)])
        return _finite_sorted_unique(s)
    if key == "num_leaves":
        lo, hi = int(min(s)), int(max(s))
        if at_low:
            s.extend([max(2, lo // 2), max(2, lo - 4)])
        if at_high:
            s.extend([min(1024, hi + 8), min(1024, hi + 16)])
        return sorted({int(x) for x in s})
    if key == "max_depth":
        lo, hi = int(min(s)), int(max(s))
        if at_low:
            s.extend([max(3, lo - 2)])
        if at_high:
            s.extend([min(20, hi + 2)])
        return sorted({int(x) for x in s})
    if key == "min_child_weight":
        lo = float(min(s))
        hi = float(max(s))
        if at_low:
            s.extend([max(1e-6, lo * 0.5), max(1e-6, lo * 0.75)])
        if at_high:
            s.extend([hi * 1.5, hi * 2.0])
        return _finite_sorted_unique(s)
    if key in ("reg_alpha", "reg_lambda"):
        lo = float(min(s))
        hi = float(max(s))
        if at_low:
            s.extend([max(1e-8, lo * 0.25), max(1e-8, lo * 0.5)])
        if at_high:
            s.extend([hi * 2.0, hi * 4.0])
        return _finite_sorted_unique(s)
    if key in ("colsample_bytree", "feature_fraction", "bagging_fraction"):
        lo = float(min(s))
        hi = float(max(s))
        if at_low:
            s.extend([max(0.05, lo - 0.1), max(0.05, lo - 0.05)])
        if at_high:
            s.extend([min(1.0, hi + 0.05), min(1.0, hi + 0.1)])
        return _finite_sorted_unique(s)
    if key == "bagging_freq":
        lo, hi = int(min(s)), int(max(s))
        if at_low:
            s.extend([max(0, lo - 25)])
        if at_high:
            s.extend([hi + 150, hi + 300])
        return sorted({int(x) for x in s})

    if isinstance(base, int):
        lo, hi = int(min(s)), int(max(s))
        if at_low:
            s.append(max(1, lo - 1))
        if at_high:
            s.append(hi + 1)
        return sorted({int(x) for x in s})
    if isinstance(base, float):
        lo, hi = float(min(s)), float(max(s))
        if at_low:
            s.append(lo * 0.8)
        if at_high:
            s.append(hi * 1.2)
        return _finite_sorted_unique(s)
    return sorted(set(s), key=lambda x: (isinstance(x, str), x))


def tune(
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    *,
    label: str,
    features: list[str],
    init_params: dict[str, Any],
    model_type: ModelType = ModelType.LGB,
    tune_keys: Optional[Sequence[str]] = None,
    param_lists: Optional[dict[str, list]] = None,
    metric: str | Callable[..., float] = "auc",
    max_expansion_rounds: int = 8,
    workdir: str | Path = ".",
    is_early_stopping: bool = False,
    num_boost_round: int = 80,
) -> TuningReport:
    """Hyperparameter search on ``validate_df`` (default metric: ``auc``).

    Candidate grids default to :func:`suggest_lgb_param_lists` around ``init_params``.
    If the best point lies on the edge of any tuned axis, that axis is extended and new
    combinations are evaluated until the score stops improving, there are no edges, or
    ``max_expansion_rounds`` is hit.
    """
    if model_type != ModelType.LGB:
        raise NotImplementedError(f"tune not implemented for {model_type!r}")
    _ = resolve_metric(metric) if isinstance(metric, str) else metric

    if validate_df is None or len(validate_df) == 0:
        raise ValueError("validate_df must be a non-empty DataFrame")

    if tune_keys is not None and len(list(tune_keys)) == 0:
        keys: List[str] = []
    else:
        keys = (
            list(tune_keys)
            if tune_keys is not None
            else [k for k in _DEFAULT_TUNE_KEYS if k in init_params]
        )
        keys = [k for k in keys if k in init_params]
        if not keys:
            raise ValueError(
                "No tune_keys intersect init_params; pass tune_keys=[] to skip search, "
                "or add keys to init_params."
            )

    if not keys:
        return TuningReport(
            model_type=model_type,
            results=pd.DataFrame(),
            best_params={},
            best_score=float("nan"),
        )

    base = dict(init_params)
    lists: dict[str, list] = (
        {k: list(param_lists[k]) for k in keys}
        if param_lists is not None
        else suggest_lgb_param_lists(base, keys)
    )
    for k in keys:
        if k not in lists or not lists[k]:
            lists[k] = [base[k]]

    import lightgbm as lgb

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    seen: Set[Tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    best_global = float("-inf")

    for round_idx in range(max_expansion_rounds + 1):
        grid_spec = {k: lists[k] for k in keys}
        combos = build_param_grid(model_type, grid_spec)
        new_deltas: list[dict[str, Any]] = []
        for delta in combos:
            key = _combo_frozen(delta, keys)
            if key not in seen:
                seen.add(key)
                new_deltas.append(delta)

        if not new_deltas:
            break

        best_before_round = best_global

        for i, delta in enumerate(new_deltas):
            params = {**base, **delta}
            mp = workdir / f"tune_r{round_idx}_{i}.txt"
            ip = workdir / f"tune_imp_r{round_idx}_{i}.csv"
            train(
                train_df,
                validate_df,
                features,
                label,
                params,
                model_type=model_type,
                model_path=str(mp),
                importance_path=str(ip),
                is_early_stopping=is_early_stopping,
                num_boost_round=num_boost_round,
                log_evaluation_period=10**9,
                refit_on_full_data=False,
            )
            booster = lgb.Booster(model_file=str(mp))
            pred = booster.predict(validate_df[features])
            score = compute_metric(metric, validate_df[label].values, pred)
            row = {
                **delta,
                "metric_mean": score,
                "metric_std": 0.0,
                "expansion_round": round_idx,
            }
            rows.append(row)
            prev_best = best_global
            best_global = max(best_global, score)
            if score > prev_best + 1e-15:
                metric_label = metric if isinstance(metric, str) else "metric"
                print(
                    f"[tune] new best validation {metric_label}={score:.6f} "
                    f"(expansion_round={round_idx}) params={params}",
                    flush=True,
                )

        if round_idx == max_expansion_rounds:
            break

        if round_idx > 0 and best_global <= best_before_round + 1e-12:
            break

        results_df = pd.DataFrame(rows)
        best_idx = int(results_df["metric_mean"].idxmax())
        best_delta = {k: results_df.iloc[best_idx][k] for k in keys}
        at_low, at_high = _find_boundary_keys(lists, best_delta, keys)
        if not at_low and not at_high:
            break

        expanded_any = False
        for k in keys:
            low = k in at_low
            high = k in at_high
            if not low and not high:
                continue
            old_vals = set(lists[k])
            lists[k] = _expand_param_list(
                k, list(lists[k]), best_delta[k], low, high, base
            )
            if set(lists[k]) != old_vals:
                expanded_any = True

        if not expanded_any:
            break

    results = pd.DataFrame(rows)
    if results.empty:
        return TuningReport(
            model_type=model_type,
            results=results,
            best_params={},
            best_score=float("nan"),
        )

    best_idx = int(results["metric_mean"].idxmax())
    best_row = results.iloc[best_idx]
    best_params = {k: best_row[k] for k in keys}
    best_score = float(best_row["metric_mean"])
    return TuningReport(
        model_type=model_type,
        results=results,
        best_params=best_params,
        best_score=best_score,
    )


def tune_cv(
    train_df: pd.DataFrame,
    *,
    model_type: ModelType = ModelType.LGB,
    label: str,
    features: list[str],
    base_params: dict[str, Any],
    grid_spec: dict[str, list | tuple],
    metric: str | Callable[..., float] = "auc",
    n_splits: int = 3,
    random_state: int = 42,
    num_boost_round: int = 80,
    workdir: str | Path = ".",
    is_early_stopping: bool = False,
) -> TuningReport:
    """Stratified K-fold on *train_df* only (no holdout validate_df)."""
    from sklearn.model_selection import StratifiedKFold

    if model_type != ModelType.LGB:
        raise NotImplementedError(f"tune_cv not implemented for {model_type!r}")
    _ = resolve_metric(metric) if isinstance(metric, str) else metric
    if n_splits < 2:
        raise ValueError("tune_cv requires n_splits >= 2")

    combos = build_param_grid(model_type, grid_spec)
    if not combos:
        return TuningReport(
            model_type=model_type,
            results=pd.DataFrame(),
            best_params={},
            best_score=float("nan"),
        )

    import lightgbm as lgb

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    y = train_df[label].values
    rows: list[dict[str, Any]] = []

    for i, delta in enumerate(combos):
        params = {**base_params, **delta}
        fold_scores: list[float] = []
        for fold, (tr_idx, va_idx) in enumerate(skf.split(train_df, y)):
            tr = train_df.iloc[tr_idx]
            va = train_df.iloc[va_idx]
            mp = workdir / f"tune_cv_{i}_fold{fold}.txt"
            ip = workdir / f"tune_cv_imp_{i}_fold{fold}.csv"
            train(
                tr,
                va,
                features,
                label,
                params,
                model_type=model_type,
                model_path=str(mp),
                importance_path=str(ip),
                is_early_stopping=is_early_stopping,
                num_boost_round=num_boost_round,
                log_evaluation_period=10**9,
                refit_on_full_data=False,
            )
            booster = lgb.Booster(model_file=str(mp))
            pred = booster.predict(va[features])
            fold_scores.append(compute_metric(metric, va[label].values, pred))
        row = {
            **delta,
            "metric_mean": float(np.mean(fold_scores)),
            "metric_std": float(np.std(fold_scores)),
            "expansion_round": -1,
        }
        rows.append(row)

    results = pd.DataFrame(rows)
    best_idx = int(results["metric_mean"].idxmax())
    best_row = results.iloc[best_idx]
    param_keys = list(grid_spec.keys())
    best_params = {k: best_row[k] for k in param_keys}
    best_score = float(best_row["metric_mean"])
    return TuningReport(
        model_type=model_type,
        results=results,
        best_params=best_params,
        best_score=best_score,
    )

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

import json
import math
import os
import shutil
import sys
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from model_pipeline.metrics import compute_metric, resolve_metric
from model_pipeline.train_model import ModelType, TuningReport, train

# Per-trial JSON next to model / feature-importance files under ``workdir`` (``tune_workspace``).
TRIAL_CONFIG_VERSION = 1
# Human-readable tuning messages (terminal + append to this file under ``workdir``).
TUNE_LOG_FILENAME = "log.txt"
# Best-so-far snapshots under ``output_dir / VALIDATION_RECORD_DIRNAME``.
VALIDATION_RECORD_DIRNAME = "validation_record"
# Root-of-search outputs under ``output_dir`` after a successful tune.
BEST_OF_ALL_MODEL = "best_of_all_model.txt"
BEST_OF_ALL_FEATURE_IMPORTANCE = "best_of_all_feature_importance.csv"
BEST_OF_ALL_CONFIG = "best_of_all_config.json"


def _tune_log_line(workdir: Path, *parts: Any, sep: str = " ", flush: bool = True) -> None:
    """Print to stdout and append the same line to ``workdir / TUNE_LOG_FILENAME``."""
    msg = sep.join(str(p) for p in parts)
    print(msg, flush=flush)
    log_path = workdir / TUNE_LOG_FILENAME
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        if flush:
            f.flush()


def _clear_directory_contents(path: Path) -> None:
    """Remove all children of *path*; leave *path* as an empty directory."""
    if not path.is_dir():
        return
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink(missing_ok=True)
        elif child.is_dir():
            shutil.rmtree(child)


def _workdir_has_children(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(path.iterdir())


def _confirm_clean_workdir(workdir: Path, *, require_confirmation: bool) -> None:
    """Prompt or require env before clearing a non-empty *workdir* (``clean_run=True``)."""
    if not _workdir_has_children(workdir):
        return
    if not require_confirmation:
        return
    if sys.stdin.isatty():
        reply = input(
            f"About to delete all files in {workdir!s}. Continue? [y/N]: "
        ).strip().lower()
        if reply not in ("y", "yes"):
            raise SystemExit("Aborted: tune_workspace was not cleared.")
        return
    env = os.environ.get("TUNE_CONFIRM_CLEAN", "").strip().lower()
    if env in ("1", "yes", "true"):
        return
    raise RuntimeError(
        "Refusing to clear non-empty workdir in non-interactive mode. Set "
        "environment variable TUNE_CONFIRM_CLEAN=1 to confirm, or run from a "
        "terminal, or pass require_clean_confirmation=False to tune()."
    )


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


def _jsonable_param_value(v: Any) -> Any:
    if hasattr(v, "item"):
        try:
            return v.item()
        except Exception:
            return v
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def _delta_json_for_log(delta: dict[str, Any], keys: Sequence[str]) -> str:
    ordered = sorted(keys, key=lambda x: (isinstance(x, str), x))
    return json.dumps(
        {k: _jsonable_param_value(delta[k]) for k in ordered},
        sort_keys=False,
        ensure_ascii=False,
    )


def _trial_asset_names(trial_index: int) -> Tuple[str, str, str]:
    stem = f"trial_{trial_index:06d}"
    return (
        f"{stem}_model.txt",
        f"{stem}_feature_importance.csv",
        f"{stem}_trial_config.json",
    )


def _trial_config_record(
    *,
    trial_index: int,
    expansion_round: int,
    trial_index_in_round: int,
    validation_metric: float,
    metric_name: str | None,
    trial_params: dict[str, Any],
    tune_keys: Sequence[str],
    model_file: str,
    feature_importance_file: str,
) -> dict[str, Any]:
    sorted_param_keys = sorted(trial_params.keys(), key=lambda x: (isinstance(x, str), x))
    return {
        "version": TRIAL_CONFIG_VERSION,
        "trial_index": int(trial_index),
        "expansion_round": int(expansion_round),
        "trial_index_in_round": int(trial_index_in_round),
        "validation_metric": float(validation_metric),
        "metric_name": metric_name,
        "trial_params": {
            k: _jsonable_param_value(trial_params[k]) for k in sorted_param_keys
        },
        "tune_keys": sorted(tune_keys, key=lambda x: (isinstance(x, str), x)),
        "model_file": model_file,
        "feature_importance_file": feature_importance_file,
    }


def _write_trial_config(workdir: Path, record: dict[str, Any]) -> None:
    cfg_name = f"trial_{int(record['trial_index']):06d}_trial_config.json"
    (workdir / cfg_name).write_text(
        json.dumps(record, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )


def _resolve_output_dir(workdir: Path, output_dir: Optional[str | Path]) -> Path:
    if output_dir is not None:
        return Path(output_dir).resolve()
    return workdir.resolve().parent


def _load_trial_configs_from_workdir(
    workdir: Path,
    keys: Sequence[str],
    metric: str | Callable[..., float],
    *,
    log_workdir: Path,
) -> Tuple[Set[Tuple[Any, ...]], List[dict[str, Any]], float, int]:
    """Load prior trials from ``trial_*_trial_config.json``; return seen, rows, best_score, max_trial_index."""
    seen: Set[Tuple[Any, ...]] = set()
    rows: List[dict[str, Any]] = []
    best_global = float("-inf")
    max_trial_index = -1
    key_set = set(keys)
    metric_filter: str | None = metric if isinstance(metric, str) else None

    if not workdir.is_dir():
        return seen, rows, best_global, max_trial_index

    for path in sorted(workdir.glob("trial_*_trial_config.json")):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("version") != TRIAL_CONFIG_VERSION:
            continue
        tk = rec.get("tune_keys")
        if not isinstance(tk, list) or set(tk) != key_set:
            continue
        if metric_filter is not None and rec.get("metric_name") != metric_filter:
            continue
        if metric_filter is None and rec.get("metric_name") is not None:
            continue
        tp = rec.get("trial_params")
        if not isinstance(tp, dict):
            continue
        try:
            delta = {k: tp[k] for k in keys}
        except KeyError:
            continue
        try:
            mm = float(rec["validation_metric"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isnan(mm) or math.isinf(mm):
            continue
        try:
            ti = int(rec["trial_index"])
        except (KeyError, TypeError, ValueError):
            continue
        model_file = rec.get("model_file")
        imp_file = rec.get("feature_importance_file")
        if not isinstance(model_file, str) or not isinstance(imp_file, str):
            continue
        if not (workdir / model_file).is_file() or not (workdir / imp_file).is_file():
            continue
        fp = _combo_frozen(delta, keys)
        if fp in seen:
            continue
        seen.add(fp)
        max_trial_index = max(max_trial_index, ti)
        try:
            er = int(rec.get("expansion_round", 0))
        except (TypeError, ValueError):
            er = 0
        try:
            tir = int(rec.get("trial_index_in_round", -1))
        except (TypeError, ValueError):
            tir = -1
        try:
            ms = float(rec.get("metric_std", 0.0))
        except (TypeError, ValueError):
            ms = 0.0
        row = {
            **delta,
            "metric_mean": mm,
            "metric_std": ms,
            "expansion_round": er,
            "trial_index": ti,
            "trial_index_in_round": tir,
            "model_file": model_file,
            "feature_importance_file": imp_file,
        }
        rows.append(row)
        best_global = max(best_global, mm)
        delta_log = _delta_json_for_log(delta, keys)
        metric_label = metric_filter if metric_filter is not None else "<callable_metric>"
        _tune_log_line(
            log_workdir,
            "[tune] resume: loaded trial_config — "
            f"trial_index={ti} params={delta_log}; metric={metric_label} "
            f"validation_metric={mm:.6f} expansion_round={er}",
        )

    if metric_filter is None and rows:
        _tune_log_line(
            log_workdir,
            "[tune] resume: metric is callable; loaded "
            f"{len(rows)} trial config(s) with no metric name filter.",
        )
    elif rows:
        _tune_log_line(
            log_workdir,
            f"[tune] resume: loaded {len(rows)} trial(s) from {workdir}",
        )
    return seen, rows, best_global, max_trial_index


def _copy_best_of_all(
    workdir: Path,
    output_dir: Path,
    best_row: pd.Series,
    keys: Sequence[str],
    metric_name: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ti = int(best_row["trial_index"])
    cfg_path = workdir / f"trial_{ti:06d}_trial_config.json"
    model_rel = str(best_row["model_file"])
    imp_rel = str(best_row["feature_importance_file"])
    mp = workdir / model_rel
    ip = workdir / imp_rel
    if not mp.is_file() or not ip.is_file():
        _tune_log_line(
            workdir,
            f"[tune] warning: best trial files missing ({model_rel!r} / {imp_rel!r}); "
            "skipping best_of_all copy.",
        )
        return
    trial_params: dict[str, Any]
    if cfg_path.is_file():
        try:
            src = json.loads(cfg_path.read_text(encoding="utf-8"))
            tp = src.get("trial_params")
            trial_params = dict(tp) if isinstance(tp, dict) else {k: best_row[k] for k in keys}
        except (json.JSONDecodeError, OSError, TypeError):
            trial_params = {k: best_row[k] for k in keys}
    else:
        trial_params = {k: best_row[k] for k in keys}
    shutil.copy2(mp, output_dir / BEST_OF_ALL_MODEL)
    shutil.copy2(ip, output_dir / BEST_OF_ALL_FEATURE_IMPORTANCE)
    rec = _trial_config_record(
        trial_index=ti,
        expansion_round=int(best_row["expansion_round"]),
        trial_index_in_round=int(best_row.get("trial_index_in_round", -1)),
        validation_metric=float(best_row["metric_mean"]),
        metric_name=metric_name,
        trial_params=trial_params,
        tune_keys=keys,
        model_file=BEST_OF_ALL_MODEL,
        feature_importance_file=BEST_OF_ALL_FEATURE_IMPORTANCE,
    )
    (output_dir / BEST_OF_ALL_CONFIG).write_text(
        json.dumps(rec, indent=2, ensure_ascii=True, default=str) + "\n",
        encoding="utf-8",
    )
    _tune_log_line(
        workdir,
        f"[tune] wrote {output_dir / BEST_OF_ALL_MODEL}, "
        f"{output_dir / BEST_OF_ALL_FEATURE_IMPORTANCE}, "
        f"{output_dir / BEST_OF_ALL_CONFIG}",
    )


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
    clean_run: bool = False,
    output_dir: Optional[str | Path] = None,
    require_clean_confirmation: bool = True,
) -> TuningReport:
    """Hyperparameter search on ``validate_df`` (default metric: ``auc``).

    Candidate grids default to :func:`suggest_lgb_param_lists` around ``init_params``.
    If the best point lies on the edge of any tuned axis, that axis is extended and new
    combinations are evaluated until the score stops improving, there are no edges, or
    ``max_expansion_rounds`` is hit.

    Each completed trial writes under ``workdir``: a model file, a feature-importance
    file, and ``trial_<index>_trial_config.json`` recording trial index, validation
    metric, full ``trial_params``, and artifact basenames.

    When ``clean_run`` is false (default), existing ``trial_*_trial_config.json`` files
    in ``workdir`` are read (matching ``tune_keys`` and metric name): those parameter
    sets are skipped, the best validation metric seeds the search, and only unseen grid
    points are trained.

    When ``clean_run`` is true, ``workdir`` and ``output_dir / validation_record`` are
    cleared (then a full search from scratch). ``workdir`` must not be cwd. Confirmation
    rules match ``require_clean_confirmation`` / ``TUNE_CONFIRM_CLEAN``.

    On each new best validation score, artifacts are copied under
    ``output_dir / validation_record`` as ``improvement_<n>_*.`` At the end, the overall
    best trial is copied to ``output_dir`` as ``best_of_all_model.txt``,
    ``best_of_all_feature_importance.csv``, and ``best_of_all_config.json``.

    ``output_dir`` defaults to ``workdir.parent`` when omitted.

    The same ``[tune] ...`` messages are appended to ``workdir / TUNE_LOG_FILENAME``.
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
    out_root = _resolve_output_dir(workdir, output_dir)
    validation_record_dir = out_root / VALIDATION_RECORD_DIRNAME

    if clean_run:
        try:
            if workdir.resolve() == Path.cwd().resolve():
                raise ValueError(
                    "workdir must not be the current working directory when clean_run=True "
                    "(output folder would be cleared). Use a dedicated subdirectory."
                )
        except OSError:
            pass
        if workdir.is_dir():
            _confirm_clean_workdir(
                workdir, require_confirmation=require_clean_confirmation
            )
            _clear_directory_contents(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if validation_record_dir.is_dir():
            _clear_directory_contents(validation_record_dir)
        validation_record_dir.mkdir(parents=True, exist_ok=True)
    else:
        workdir.mkdir(parents=True, exist_ok=True)
        validation_record_dir.mkdir(parents=True, exist_ok=True)

    metric_name = metric if isinstance(metric, str) else None

    seen: Set[Tuple[Any, ...]] = set()
    rows: list[dict[str, Any]] = []
    best_global = float("-inf")
    next_trial_seq = 0
    improvement_n = 0

    if not clean_run:
        seen, loaded_rows, best_global, max_trial = _load_trial_configs_from_workdir(
            workdir, keys, metric, log_workdir=workdir
        )
        rows.extend(loaded_rows)
        next_trial_seq = max_trial + 1 if max_trial >= 0 else 0
        if loaded_rows:
            metric_label = metric if isinstance(metric, str) else "metric"
            _tune_log_line(
                workdir,
                f"[tune] resume: {len(loaded_rows)} prior trial(s) from configs; "
                f"best validation {metric_label}={best_global:.6f}; "
                f"next trial_index={next_trial_seq}; training only unseen parameter sets.",
            )

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
            t_idx = next_trial_seq
            next_trial_seq += 1
            model_file, imp_file, _cfg_name = _trial_asset_names(t_idx)
            mp = workdir / model_file
            ip = workdir / imp_file
            delta_log = _delta_json_for_log(delta, keys)
            _tune_log_line(
                workdir,
                f"[tune] parameter params={delta_log} is under training "
                f"(expansion_round={round_idx}, trial_index_in_round={i}, trial_index={t_idx})",
            )
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
            cfg_rec = _trial_config_record(
                trial_index=t_idx,
                expansion_round=round_idx,
                trial_index_in_round=i,
                validation_metric=score,
                metric_name=metric_name,
                trial_params=params,
                tune_keys=keys,
                model_file=model_file,
                feature_importance_file=imp_file,
            )
            _write_trial_config(workdir, cfg_rec)
            row = {
                **delta,
                "metric_mean": score,
                "metric_std": 0.0,
                "expansion_round": round_idx,
                "trial_index": t_idx,
                "trial_index_in_round": i,
                "model_file": model_file,
                "feature_importance_file": imp_file,
            }
            rows.append(row)
            prev_best = best_global
            best_global = max(best_global, score)
            if score > prev_best + 1e-15:
                metric_label = metric if isinstance(metric, str) else "metric"
                _tune_log_line(
                    workdir,
                    f"[tune] new best validation {metric_label}={score:.6f} "
                    f"(trial_index={t_idx}, expansion_round={round_idx}) params={params}",
                )
                improvement_n += 1
                m_dst = (
                    validation_record_dir
                    / f"improvement_{improvement_n:04d}_model.txt"
                )
                i_dst = (
                    validation_record_dir
                    / f"improvement_{improvement_n:04d}_feature_importance.csv"
                )
                c_dst = (
                    validation_record_dir
                    / f"improvement_{improvement_n:04d}_config.json"
                )
                shutil.copy2(mp, m_dst)
                shutil.copy2(ip, i_dst)
                imp_rec = _trial_config_record(
                    trial_index=t_idx,
                    expansion_round=round_idx,
                    trial_index_in_round=i,
                    validation_metric=score,
                    metric_name=metric_name,
                    trial_params=params,
                    tune_keys=keys,
                    model_file=m_dst.name,
                    feature_importance_file=i_dst.name,
                )
                c_dst.write_text(
                    json.dumps(imp_rec, indent=2, ensure_ascii=True, default=str)
                    + "\n",
                    encoding="utf-8",
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
    _copy_best_of_all(workdir, out_root, best_row, keys, metric_name)
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


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser(
        description=(
            "LightGBM hyperparameter search helpers. Import and call tune() from your "
            "training script; default clean_run=False resumes from trial_*_trial_config.json "
            "files under workdir."
        )
    )
    _ap.add_argument(
        "--clean-run",
        action="store_true",
        help=(
            "Not used when this file is run directly; pass clean_run=True to tune() for a "
            "fresh search (confirmation if workdir is non-empty)."
        ),
    )
    _ap.parse_args()
    _ap.print_help()

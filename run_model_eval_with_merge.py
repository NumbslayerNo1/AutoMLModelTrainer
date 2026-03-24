#!/usr/bin/env python3
"""Merge basicinfo + v6features (like ``model_eval.ipynb``), then run ``model_pipeline.model_eval``.

The notebook evaluates on data that has **business / label / portrait** columns from the
basicinfo table **and** raw **ETL feature** columns for LightGBM. The v6features Parquet
holds the latter; basicinfo Parquet holds the former. This script:

1. Loads feature names from the trained **LightGBM** model file.
2. Reads **v6features** with ``trace_id`` + those columns only.
3. Reads **basicinfo** (full row set), drops any columns that would duplicate v6features
   (except ``trace_id``), then **inner-joins** on ``trace_id``.
4. Writes ``merged_eval_input.parquet`` under ``--data-output-path``.
5. Calls ``model_pipeline.model_eval.run_evaluation`` so AUC / curves / lift / heatmaps
   can run (needs ``loan_account_id``, ``1pd7``, etc. from basicinfo).

Example (defaults use ``/data1/bogeng/data_output`` for model + eval output)::

    PYTHONPATH=. python3 run_model_eval_with_merge.py \\
      --model-file /data1/bogeng/data_output/model.txt \\
      --basicinfo-parquet /data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_basicinfo_and_v6modelscore \\
      --v6features-parquet /data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_v6features \\
      --data-output-path /data1/bogeng/data_output/model_eval_test_run

Large merges need enough RAM for two wide frames + join result.

Use **python3** (not ``python`` on systems where ``python`` is 2.x).
"""

import argparse
import errno
import sys
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_MODEL = Path("/data1/bogeng/data_output/model.txt")
DEFAULT_BASICINFO = Path(
    "/data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_basicinfo_and_v6modelscore"
)
DEFAULT_V6FEATURES = Path(
    "/data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_v6features"
)
DEFAULT_OUTPUT_DIR = Path("/data1/bogeng/data_output/model_eval_test_run")
MERGED_NAME = "merged_eval_input.parquet"


def merge_basicinfo_v6features(
    model_file: Path,
    basicinfo_parquet: Path,
    v6features_parquet: Path,
    merged_out: Path,
    *,
    max_rows: Optional[int] = None,
) -> Path:
    """Inner-merge basicinfo with v6 feature columns; return path to written Parquet."""
    booster = lgb.Booster(model_file=str(model_file))
    feature_names = list(booster.feature_name())

    print(f"Loading v6features: {v6features_parquet}", flush=True)
    import pyarrow.parquet as pq

    avail = set(pq.ParquetFile(v6features_parquet).schema_arrow.names)
    read_cols = ["trace_id"] + [c for c in feature_names if c in avail]
    missing_model_feats = [c for c in feature_names if c not in avail]
    if missing_model_feats:
        raise KeyError(
            f"v6features missing {len(missing_model_feats)} model columns "
            f"(showing first 10): {missing_model_feats[:10]!r}"
        )

    feat_df = pd.read_parquet(v6features_parquet, columns=read_cols)
    print(f"v6features columns read: {len(read_cols) - 1} features, rows={len(feat_df)}", flush=True)

    print(f"Loading basicinfo: {basicinfo_parquet}", flush=True)
    base_df = pd.read_parquet(basicinfo_parquet)
    print(f"basicinfo rows={len(base_df)}, cols={len(base_df.columns)}", flush=True)

    overlap = (set(base_df.columns) & set(feat_df.columns)) - {"trace_id"}
    if overlap:
        print(
            f"Dropping {len(overlap)} overlapping columns from basicinfo "
            f"(v6features wins for model inputs). Example: {list(overlap)[:5]!r}",
            flush=True,
        )
        base_df = base_df.drop(columns=list(overlap), errors="ignore")

    merged = base_df.merge(feat_df, on="trace_id", how="inner")
    print(f"merged shape: {merged.shape}", flush=True)

    if max_rows is not None and max_rows > 0 and len(merged) > max_rows:
        merged = merged.iloc[:max_rows].copy()
        print(f"trimmed to --max-rows={max_rows}: {merged.shape}", flush=True)

    merged_out = merged_out.resolve()
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged.to_parquet(merged_out, index=False)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise OSError(
                errno.ENOSPC,
                "No space left for merged Parquet. Free disk space, set "
                "--merged-parquet to a directory on a larger volume (e.g. under /data1), "
                "and/or use --max-rows to write fewer rows.",
            ) from e
        raise
    print(f"Wrote {merged_out}", flush=True)
    return merged_out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge basicinfo + v6features, then run model_pipeline.model_eval."
    )
    p.add_argument(
        "--model-file",
        type=Path,
        default=DEFAULT_MODEL,
        help="LightGBM model .txt (feature names taken from booster)",
    )
    p.add_argument(
        "--basicinfo-parquet",
        type=Path,
        default=DEFAULT_BASICINFO,
        help="Notebook-style basicinfo + labels + portrait (e.g. …basicinfo_and_v6modelscore)",
    )
    p.add_argument(
        "--v6features-parquet",
        type=Path,
        default=DEFAULT_V6FEATURES,
        help="Parquet with trace_id + raw ETL features",
    )
    p.add_argument(
        "--data-output-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for model_eval artifacts (log, figures) and default merged parquet",
    )
    p.add_argument(
        "--merged-parquet",
        type=Path,
        default=None,
        help=f"Where to write merged input (default: <data-output-path>/{MERGED_NAME})",
    )
    p.add_argument(
        "--skip-merge",
        action="store_true",
        help=f"If {MERGED_NAME} already exists, skip merge and only run model_eval",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        metavar="N",
        help="After merge, keep only the first N rows (smaller Parquet; full merge still uses RAM)",
    )
    args = p.parse_args()

    out_dir = args.data_output_path.expanduser().resolve()
    merged_path = (
        args.merged_parquet.expanduser().resolve()
        if args.merged_parquet
        else out_dir / MERGED_NAME
    )

    if args.skip_merge and merged_path.is_file():
        print(f"Using existing merged file (--skip-merge): {merged_path}", flush=True)
    else:
        merge_basicinfo_v6features(
            args.model_file.expanduser().resolve(),
            args.basicinfo_parquet.expanduser().resolve(),
            args.v6features_parquet.expanduser().resolve(),
            merged_path,
            max_rows=args.max_rows,
        )

    from model_pipeline.model_eval import run_evaluation

    run_evaluation(
        args.model_file.expanduser().resolve(),
        merged_path,
        out_dir,
    )


if __name__ == "__main__":
    main()

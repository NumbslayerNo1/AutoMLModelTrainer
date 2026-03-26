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

Example (from repo root; ``sys.path`` is set automatically)::

    python3 batch_scripts/model_eval_with_merge.py \\
      --model-file /data1/bogeng/data_output/model.txt \\
      --basicinfo-parquet /data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_basicinfo_and_v6modelscore \\
      --v6features-parquet /data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_v6features \\
      --data-output-path /data1/bogeng/data_output/model_eval_test_run

Large merges need enough RAM for two wide frames + join result.

Use **python3** (not ``python`` on systems where ``python`` is 2.x).
"""

import argparse
import sys
from pathlib import Path

# Repo root (parent of ``batch_scripts/``) so ``import model_pipeline`` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_pipeline.merge_basicinfo_features import merge_basicinfo_features

DEFAULT_MODEL = Path("/data1/bogeng/data_output/pipeline_test/model.txt")
DEFAULT_BASICINFO = Path(
    "/data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_basicinfo_and_v6modelscore"
)
DEFAULT_V6FEATURES = Path(
    "/data1/mex_reloan_data/mex_reloan_trace_test_data_s20250601_e20260228_v6features"
)
DEFAULT_OUTPUT_DIR = Path("/data1/bogeng/data_output/model_eval_test_run")
DEFAULT_LABEL_FILE = Path(
    "/data1/mex_reloan_data/mex_reloan_trace_payout_test_data_s20230101_e20260217_v6features_label_basicinfo"
)
MERGED_NAME = "merged_eval_input.parquet"


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
        "--label-file",
        type=Path,
        default=DEFAULT_LABEL_FILE,
        help="Parquet for BOOT tagging in model_eval (get_threeGroup_tag); 3rd arg to run_evaluation",
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
        merge_basicinfo_features(
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
        args.label_file.expanduser().resolve(),
        out_dir,
    )


if __name__ == "__main__":
    main()

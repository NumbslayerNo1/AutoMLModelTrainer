#!/usr/bin/env python3
"""Offline model evaluation mirroring ``model_eval.ipynb`` main evaluation cell.

Depends on ``<repo>/model_eval_metric/`` (``modelEvaluation``, ``modelEvaluation2``,
``psiCalculation``), or on ``MODEL_EVAL_CODE_DIR`` when set. Same Python stack as the notebook
(pandas, sklearn, seaborn, matplotlib, plotnine, shap, etc.).

Scores ``eval_data_file`` with the LightGBM model at ``model_file``, then runs the same
reports as ``model_eval.ipynb`` when the Parquet schema allows it.

* **Full notebook parity** (AUC, curves, lift, heatmap): use eval data merged like the
  notebook (basicinfo + labels + ``trace_month``, ``loan_account_id``, etc.).
* **v6features-only** Parquet: scoring + §1 correlation always run; §5 distribution runs if
  ``trace_month`` and ``trace_id`` exist; other sections are skipped with ``[skip …]`` lines
  in ``log.txt`` (no crash).

All ``print`` output and captured tables go to ``log.txt``; each ``plt.show()`` becomes a
PNG under ``data_output_path``. The same stream, ``display()`` tables, and figures are also
written as ``model_eval_run.ipynb`` (nbformat 4.5-style outputs, similar to
``orig_notebook_files/notebook_model_eval.ipynb``).

Run from the repository root (editable install or ``PYTHONPATH``)::

    PYTHONPATH=. python model_pipeline/model_eval.py <model.txt> <eval.parquet> [label.parquet] <out_dir>
    python -m model_pipeline.model_eval <model.txt> <eval.parquet> [label.parquet] <out_dir>

``label.parquet`` is optional; if omitted, the built-in default label Parquet path is used.
"""

import argparse
import io
import os
import sys
import time
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root (parent of ``model_pipeline``) so ``import model_pipeline.*`` works.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CODE_DIR = Path(
    os.environ.get("MODEL_EVAL_CODE_DIR", str(_REPO_ROOT / "model_eval_metric"))
).expanduser().resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

EVAL_MODEL_SCORE_COLUMN = "eval_lgb_model_score"


class Tee(io.TextIOBase):
    """Write to multiple text streams (e.g. console + log file)."""

    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


class TeeWithNotebook(io.TextIOBase):
    """Like ``Tee`` but also feeds stdout/stderr text into a ``NotebookOutputRecorder``."""

    def __init__(self, *streams: Any, recorder: Optional[Any] = None):
        self.streams = streams
        self._recorder = recorder

    def write(self, s: str) -> int:
        if self._recorder is not None and s:
            self._recorder.feed_stream(s)
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def _install_display_and_show_hooks(
    out_dir: Path,
    me: ModuleType,
    me2: ModuleType,
    *,
    recorder: Optional[Any] = None,
) -> None:
    """Capture notebook ``display`` and matplotlib ``show`` into ``out_dir`` and optional notebook."""
    from pandas.io.formats.style import Styler

    display_counter: List[int] = [0]
    figure_counter: List[int] = [0]

    def display_fn(obj: Any) -> None:
        display_counter[0] += 1
        n = display_counter[0]
        base = out_dir / f"display_{n:04d}"
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(f"{base}.csv", index=False)
            if recorder is not None:
                recorder.add_dataframe(obj)
        elif hasattr(obj, "data") and isinstance(getattr(obj, "data"), pd.DataFrame):
            getattr(obj, "data").to_csv(f"{base}_styler_data.csv", index=False)
            if recorder is not None:
                if isinstance(obj, Styler):
                    recorder.add_styler(obj)
                else:
                    recorder.add_dataframe(getattr(obj, "data"))
        else:
            with open(f"{base}.txt", "w", encoding="utf-8") as fh:
                fh.write(repr(obj))
            if recorder is not None:
                recorder.add_repr_display(obj)

    def show_fn(*_a: Any, **_kw: Any) -> None:
        figure_counter[0] += 1
        n = figure_counter[0]
        path = out_dir / f"figure_{n:04d}.png"
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=120)
        png = buf.getvalue()
        path.write_bytes(png)
        if recorder is not None:
            recorder.add_png_bytes(png)
        plt.close()

    me.display = display_fn  # type: ignore[attr-defined]
    me2.display = display_fn  # type: ignore[attr-defined]
    plt.show = show_fn  # type: ignore[assignment]


def _maybe_boot_tag(
    test_data: pd.DataFrame,
    *,
    label_parquet: Optional[Path],
) -> pd.DataFrame:
    """Match notebook ``get_threeGroup_tag`` when possible; else ``boot_main_model_v6`` = 1.

    ``get_threeGroup_tag`` merges on ``loan_account_id``. Pure v6features Parquet often has no
    that column — then real BOOT tagging is skipped and all rows are treated as in-BOOT.
    """
    import modelEvaluation as me

    if label_parquet is None or not label_parquet.exists():
        out = test_data.copy()
        out["boot_main_model_v6"] = 1
        return out

    if "loan_account_id" not in test_data.columns:
        print(
            "eval data has no column 'loan_account_id'; skipping get_threeGroup_tag. "
            "Merge basicinfo (or any table with loan_account_id) for notebook-faithful BOOT. "
            "Using boot_main_model_v6 = 1 for all rows.",
            flush=True,
        )
        out = test_data.copy()
        out["boot_main_model_v6"] = 1
        return out

    label_name = os.environ.get("MODEL_EVAL_LABEL_NAME", "current_456overdue7_ever")
    split_key = os.environ.get(
        "MODEL_EVAL_LABEL_SPLIT_KEY", "current_payout_order_payout_date"
    )
    split_start = os.environ.get("MODEL_EVAL_LABEL_START", "2024-01-01")
    split_end = os.environ.get("MODEL_EVAL_LABEL_END", "2025-05-31")

    cols = [
        "trace_id",
        "loan_account_id",
        split_key,
        label_name,
    ]
    label_data = pd.read_parquet(label_parquet, columns=cols)
    label_data = label_data[
        (label_data[split_key] >= split_start)
        & (label_data[split_key] <= split_end)
        & (label_data[label_name] >= 0)
    ]
    label_data = label_data.astype({"loan_account_id": np.int64})
    return me.get_threeGroup_tag(
        test_data=test_data,
        train_data=label_data,
        label=label_name,
        suffix="main_model_v6",
    )


def _has_columns(df: pd.DataFrame, names: List[str]) -> bool:
    c = df.columns
    return all(n in c for n in names)


def run_evaluation(
    model_file: Union[str, Path],
    eval_data_file: Union[str, Path],
    label_parquet: Optional[Union[str, Path]],
    data_output_path: Union[str, Path],
) -> None:
    """Load data, score with LGB model, run notebook-equivalent evaluation."""
    from model_pipeline.predict_model import predict_scores
    from model_pipeline.train_model import ModelType

    import modelEvaluation as me
    import modelEvaluation2 as me2

    from model_pipeline.model_eval_nb import NotebookOutputRecorder, format_run_header_markdown

    out = Path(data_output_path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.txt"
    log_f = log_path.open("w", encoding="utf-8")
    run_meta: Dict[str, str] = {}
    nb_rec = NotebookOutputRecorder()
    tee = TeeWithNotebook(sys.__stdout__, log_f, recorder=nb_rec)
    prev_stdout, prev_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = tee, tee

    try:
        warnings.filterwarnings("ignore")
        _install_display_and_show_hooks(out, me, me2, recorder=nb_rec)

        model_path = Path(model_file).expanduser().resolve()
        eval_path = Path(eval_data_file).expanduser().resolve()

        label_path: Optional[Path] = None
        if label_parquet is not None:
            label_path = Path(label_parquet).expanduser().resolve()
        if label_path is None and os.environ.get("MODEL_EVAL_LABEL_DATA"):
            label_path = Path(os.environ["MODEL_EVAL_LABEL_DATA"]).expanduser()

        run_meta["model"] = str(model_path)
        run_meta["eval"] = str(eval_path)
        run_meta["label"] = str(label_path) if label_path is not None else ""
        run_meta["out"] = str(out)

        print(f"model_file: {model_path}", flush=True)
        print(f"eval_data_file: {eval_path}", flush=True)
        print(f"label_parquet: {label_path!r}", flush=True)
        print(f"data_output_path: {out}", flush=True)

        test_data = pd.read_parquet(eval_path)
        scores = predict_scores(model_path, test_data, model_type=ModelType.LGB)
        test_data = test_data.copy()
        test_data[EVAL_MODEL_SCORE_COLUMN] = scores
        print(f"test_data shape: {test_data.shape}", flush=True)
        print(f"scores shape: {scores.shape}", flush=True)

        score_parquet = out / "eval_data_with_model_score.parquet"
        test_data.to_parquet(score_parquet, index=False)
        print(f"Wrote scored data: {score_parquet}", flush=True)

        test_data = _maybe_boot_tag(test_data, label_parquet=label_path)
        print(f"After boot tagging, shape: {test_data.shape}", flush=True)

        model_list = [EVAL_MODEL_SCORE_COLUMN]
        sc = EVAL_MODEL_SCORE_COLUMN
        risk_labels = ["1pd7", "4pd7", "6pd7"]
        unpaid_cols = ["1pd7_unpaid", "4pd7_unpaid", "6pd7_unpaid"]

        # Notebook ``model_eval.ipynb`` assumes basicinfo + labels merged with model scores.
        # v6features-only Parquet has scores' raw features but often omits labels / portrait fields.
        do_auc = _has_columns(test_data, risk_labels + ["trace_month"])
        do_curves_head = _has_columns(test_data, risk_labels)
        do_curves_amt = do_curves_head and _has_columns(
            test_data, unpaid_cols + ["current_payout_order_principal"]
        )
        label_lift = "6pd7"
        index_dic_keys = [
            "is_order",
            "is_payout",
            "first_order_credit_usage_rate",
            "bf_credit",
            "af_credit",
            "bf_max_overdue_days",
            "bf_loan_times",
            "bf_inloan_order_cnt",
            "bf_latest_payout_order_paid_terms",
        ]
        do_lift = _has_columns(
            test_data,
            ["trace_id", "current_payout_order_id", label_lift, sc] + index_dic_keys,
        )
        do_distribution = _has_columns(test_data, ["trace_month", "trace_id"])
        heatmap_static = [
            "1pd7",
            "trace_month",
            "current_payout_order_id",
            sc,
            "first_payout_credit_usage_rate",
            "first_payout_order_principal",
            "first_payout_order_credit",
            "bf_credit",
            "af_credit",
        ]
        do_heatmap = _has_columns(test_data, heatmap_static) and _has_columns(
            test_data, ["4pd7", "6pd7"]
        )

        t1 = time.time()

        print(
            "************************************************* 0 评估模型列表 ********************************************************",
            flush=True,
        )
        print(model_list, flush=True)
        print(flush=True)
        print(flush=True)
        if not do_auc:
            print(
                "[skip 2.x AUC] need labels + trace_month (merge basicinfo like the notebook).",
                flush=True,
            )
        if not do_curves_head:
            print("[skip 3.1 curves] need 1pd7/4pd7/6pd7.", flush=True)
        if not do_curves_amt:
            print(
                "[skip 3.2 amount curves] need *_unpaid + current_payout_order_principal.",
                flush=True,
            )
        if not do_lift:
            print("[skip 4.x lift] need 6pd7, order fields, and portrait columns.", flush=True)
        if not do_distribution:
            print("[skip 5.x distribution] need trace_month and trace_id.", flush=True)
        if not do_heatmap:
            print("[skip 6.x heatmap] need full basicinfo + risk labels.", flush=True)
        print(flush=True)

        # 1、模型相关性 (notebook: same corr table for 1.1 / 1.2; here one model → 1×1 matrix)
        print("1、模型相关性", flush=True)
        print(
            "************************************************* 1.1 一次风控（全量客群）********************************************************",
            flush=True,
        )
        me.display(test_data[model_list].corr())  # type: ignore[attr-defined]

        print(
            "************************************************* 1.2 成功放款客群  *************************************************************",
            flush=True,
        )
        me.display(test_data[model_list].corr())  # type: ignore[attr-defined]

        print(flush=True)
        print(flush=True)

        # 2、AUC
        print("2、AUC", flush=True)
        if do_auc:
            print("trace_month 时间分组", flush=True)
            print("", flush=True)
            print(
                "********************************************** 2.1 成功放款客群：OOT:本笔label (新增风险)***********************************************************",
                flush=True,
            )
            for label in risk_labels:
                print(label, flush=True)
                me.get_auc_groups(
                    test_data[(test_data[label].notnull())],
                    model_list,
                    label,
                    "trace_month",
                )

            print(flush=True)
            print(
                "********************************************** 2.2 成功放款客群：BOOT: 本笔label (新增风险)***********************************************************",
                flush=True,
            )
            for label in risk_labels:
                print(label, flush=True)
                me.get_auc_groups(
                    test_data[
                        (test_data[label].notnull())
                        & (test_data["boot_main_model_v6"] == 1)
                    ],
                    model_list,
                    label,
                    "trace_month",
                )

        print(flush=True)
        print(flush=True)

        # 3、人头 / 金额逾期曲线
        print("3、人头逾期和金额逾期曲线", flush=True)
        if do_curves_head:
            print(
                "********************************************** 3.1 本笔订单人头逾期（term1、term4、term6）***********************************************************",
                flush=True,
            )

            for i, m in zip([1, 4, 6], ["202506~202602", "202506~202601", "202506~202512"]):
                eval_data = test_data[
                    (test_data[f"{i}pd7"].notna()) & (test_data[sc].notna())
                ].copy()
                eval_data["tag"] = 1
                term = i
                ipd7 = f"{i}pd7"
                unpaid = f"{i}pd7"
                principal = "tag"
                title = f"{i}pd7"
                month = f"{m} AllTerms"
                score_list = model_list.copy()
                model_name_list = model_list.copy()
                me2.get_cumulative_bad_debt_rate_plot(
                    eval_data,
                    term,
                    ipd7,
                    unpaid,
                    principal,
                    title,
                    month,
                    score_list,
                    model_name_list,
                    p=0.0,
                )

        if do_curves_amt:
            print(flush=True)
            print(
                "********************************************** 3.2 本笔订单金额逾期（term1、term4、term6）***********************************************************",
                flush=True,
            )

            for i, m in zip([1, 4, 6], ["202506~202602", "202506~202601", "202506~202512"]):
                eval_data = test_data[
                    (test_data[f"{i}pd7"].notna()) & (test_data[sc].notna())
                ]
                term = i
                ipd7 = f"{i}pd7"
                unpaid = f"{i}pd7_unpaid"
                principal = "current_payout_order_principal"
                title = f"{i}pd7_unpaid"
                month = f"{m} AllTerms"
                score_list = model_list.copy()
                model_name_list = model_list.copy()
                me2.get_cumulative_bad_debt_rate_plot(
                    eval_data,
                    term,
                    ipd7,
                    unpaid,
                    principal,
                    title,
                    month,
                    score_list,
                    model_name_list,
                    p=0.0,
                )

        print(flush=True)
        print(flush=True)

        # 4、分bin
        print("4、分bin-风险Lift和基础画像", flush=True)
        basic_info = {
            "trace_id": "count",
            "current_payout_order_id": "count",
        }
        index_dic = {
            "is_order": "mean",
            "is_payout": "mean",
            "first_order_credit_usage_rate": "mean",
            "bf_credit": "mean",
            "af_credit": "mean",
            "bf_max_overdue_days": "mean",
            "bf_loan_times": "mean",
            "bf_inloan_order_cnt": "mean",
            "bf_latest_payout_order_paid_terms": "mean",
        }

        fmt_dict = {
            "num_ratio": "{:.3f}",
            "good_num": "{:.0f}",
            "bad_num": "{:.0f}",
            "bad_ratio": "{:.4f}",
            "good_cumsum": "{:.4f}",
            "bad_cumsum": "{:.4f}",
            "ks": "{:.4f}",
            "acc_auc": "{:.4f}",
            "bin_auc": "{:.4f}",
            "bin_lift": "{:.3f}",
            "trace_id": "{:.0f}",
            "current_payout_order_id": "{:.0f}",
            "bf_credit": "{:.0f}",
            "af_credit": "{:.0f}",
            "is_order": "{:.4f}",
            "is_payout": "{:.4f}",
            "first_order_credit_usage_rate": "{:.3f}",
            "bf_max_overdue_days": "{:.4f}",
            "bf_loan_times": "{:.4f}",
            "bf_inloan_order_cnt": "{:.4f}",
            "bf_latest_payout_order_paid_terms": "{:.4f}",
        }

        if do_lift:
            print(
                "********************************************** 4.1 成功放款客群 分bin-风险Lift和基础画像 - 6pd7长期风险标签 ***********************************************************",
                flush=True,
            )
            for model in model_list:
                eval_data = test_data[
                    (test_data.current_payout_order_id.notna())
                    & (test_data[model].notna())
                    & (test_data[label_lift].notna())
                ][
                    list(basic_info.keys())
                    + [model]
                    + [label_lift]
                    + list(index_dic.keys())
                ]
                result = me2.cal_stats_part(
                    eval_data,
                    model,
                    basic_info,
                    label_lift,
                    index_dic,
                    q=10,
                )
                print(model, flush=True)
                styled = result.style.format(fmt_dict).bar(
                    subset=["bin_lift"], color="#8B0000", vmin=0
                )
                me.display(styled)  # type: ignore[attr-defined]

        print(flush=True)
        print(flush=True)

        # 5、模型分布
        print("5、模型分布", flush=True)
        time_group = "trace_month"
        start_time_group = "2025-06"
        if do_distribution:
            print(
                "********************************************** 5.1 全量客群模型分布-图示版 ***********************************************************",
                flush=True,
            )
            time_group_list = sorted(set(test_data[time_group].tolist()))
            me2.get_model_distribution(
                test_data,
                time_group,
                start_time_group,
                model_list,
                time_group_list,
                n_cols=3,
            )
            print(flush=True)
            print(
                "********************************************** 5.2 全量客群模型分布-分bin比例版 ***********************************************************",
                flush=True,
            )
            for model in model_list:
                print(f"model : {model}", flush=True)
                me2.distribution_bin(
                    test_data,
                    time_group,
                    start_time_group,
                    model,
                    time_group_list,
                    q=10,
                )

        print(flush=True)
        print(flush=True)

        # 6、交叉 heatmap（单模型：横纵均为同一列；notebook uses two online scores)
        print("5、交叉评估heatmap", flush=True)
        if do_heatmap:
            print(
                "********************************************** 6.1 成功放款客群 逐月长短风险模型heatmap ***********************************************************",
                flush=True,
            )
            print(
                "成功放款客群：single eval score cross itself (same column for x/y)",
                flush=True,
            )

            x_score = y_score = sc
            x_bins = "eval_model_bin5_x"
            y_bins = "eval_model_bin5_y"
            pred = "eval_lgb_model"
            credit_usage_rate = "first_payout_credit_usage_rate"
            order_principal = "first_payout_order_principal"
            order_credit = "first_payout_order_credit"
            bf_credit = "bf_credit"
            af_credit = "af_credit"
            term1_risk_label = "1pd7"

            for month, long_vintage_index in zip(
                ["2025-10", "2025-11", "2025-12"], [6, 6, 4]
            ):
                maximum_observable_term_risk_label = f"{long_vintage_index}pd7"
                if maximum_observable_term_risk_label not in test_data.columns:
                    print(
                        f"[skip heatmap {month}] missing column {maximum_observable_term_risk_label!r}",
                        flush=True,
                    )
                    continue
                print(f"trace month: {month}", flush=True)
                eval_data = test_data[
                    (test_data[term1_risk_label].notna())
                    & (test_data.trace_month.isin([month]))
                    & (test_data.current_payout_order_id.notna())
                    & (test_data[maximum_observable_term_risk_label].notna())
                    & (test_data[sc].notna())
                ]
                print(f"test_data shape: {test_data.shape}", flush=True)
                print(f"eval_data shape: {eval_data.shape}", flush=True)
                me2.cross_heatmap(
                    eval_data,
                    x_score,
                    y_score,
                    x_bins,
                    y_bins,
                    maximum_observable_term_risk_label,
                    term1_risk_label,
                    pred,
                    credit_usage_rate,
                    order_principal,
                    order_credit,
                    bf_credit,
                    af_credit,
                    bins=5,
                )
                print(flush=True)
                print(flush=True)

        t2 = time.time()
        print(f"报告产出耗时：{(t2 - t1) / 60} 分钟", flush=True)
    finally:
        sys.stdout, sys.stderr = prev_stdout, prev_stderr
        log_f.close()

    try:
        nb_rec.write_ipynb(
            out / "model_eval_run.ipynb",
            title_md=format_run_header_markdown(
                model_path=run_meta.get("model", str(model_file)),
                eval_path=run_meta.get("eval", str(eval_data_file)),
                label_path=run_meta.get("label", str(label_parquet)),
                out_dir=run_meta.get("out", str(out)),
            ),
        )
        print(f"Wrote notebook: {out / 'model_eval_run.ipynb'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Warning: could not write model_eval_run.ipynb ({exc}).",
            file=sys.stderr,
            flush=True,
        )


def main() -> None:
    _default_label = Path(
        "/data1/mex_reloan_data/mex_reloan_trace_payout_test_data_s20230101_e20260217_v6features_label_basicinfo"
    )
    p = argparse.ArgumentParser(
        description="Run model_eval.ipynb-style reports on eval data with one LGB model."
    )
    p.add_argument("model_file", help="LightGBM model file (.txt)")
    p.add_argument("eval_data_file", help="Parquet file of rows to score and evaluate")
    p.add_argument("label_file", help="Parquet for BOOT tagging (get_threeGroup_tag). Default: Mexico v6 label+basicinfo path.")
    p.add_argument(
        "data_output_path",
        help="Output directory for log.txt, figures, tables, scored parquet",
    )
    args = p.parse_args()
    run_evaluation(
        args.model_file,
        args.eval_data_file,
        args.label_file.expanduser().resolve(),
        args.data_output_path,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a compact PDF from ``model_eval`` outputs (``display_*.csv``, ``figure_*.png``).

Example::

    PYTHONPATH=. python -m model_pipeline.generate_report /data1/.../model_eval_test_run
    PYTHONPATH=. python -m model_pipeline.generate_report /path/to/run --pdf-name my_report.pdf
"""

import argparse
import re
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _configure_cjk_font() -> None:
    """Prefer fonts that render Chinese text in table cells (best-effort)."""
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _artifact_sort_key(path: Path) -> Tuple[int, int, int]:
    name = path.name
    m = re.match(r"^figure_(\d+)\.png$", name)
    if m:
        return (1, int(m.group(1)), 0)
    m = re.match(r"^display_(\d+)_styler_data\.csv$", name)
    if m:
        return (0, int(m.group(1)), 1)
    m = re.match(r"^display_(\d+)\.csv$", name)
    if m:
        return (0, int(m.group(1)), 0)
    return (99, 0, 0)


def _describe_artifact(path: Path) -> Tuple[str, str]:
    name = path.name
    m = re.match(r"^figure_(\d+)\.png$", name)
    if m:
        n = m.group(1)
        return (
            f"Figure {n}",
            "Chart from evaluation (bad-rate curve, distribution, heatmap, etc.).",
        )
    m = re.match(r"^display_(\d+)_styler_data\.csv$", name)
    if m:
        n = m.group(1)
        return (
            f"Table {n} (styler data)",
            "Numeric data behind a styled lift/bin table.",
        )
    m = re.match(r"^display_(\d+)\.csv$", name)
    if m:
        n = m.group(1)
        return (
            f"Table {n}",
            "Tabular display output (correlation, AUC, metrics, etc.).",
        )
    return (path.stem, "Evaluation artifact.")


def _collect_artifacts(result_dir: Path) -> Tuple[List[Path], List[Path]]:
    out_dir = result_dir.expanduser().resolve()
    all_paths: List[Path] = []
    all_paths.extend(out_dir.glob("display_*.csv"))
    all_paths.extend(out_dir.glob("figure_*.png"))
    all_paths = [p for p in all_paths if p.is_file()]
    all_paths.sort(key=_artifact_sort_key)
    all_paths = [p for p in all_paths if _artifact_sort_key(p)[0] != 99]
    csvs = [p for p in all_paths if p.suffix.lower() == ".csv"]
    pngs = [p for p in all_paths if p.suffix.lower() == ".png"]
    return csvs, pngs


def _chunks(seq: Sequence[Path], size: int) -> Iterable[List[Path]]:
    for i in range(0, len(seq), size):
        yield list(seq[i : i + size])


def _save_figure_page(fig, pdf: PdfPages) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _page_png_grid(paths: List[Path], nrows: int, ncols: int, pdf: PdfPages) -> None:
    if not paths:
        return
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 8.5))
    axes_arr = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes_arr):
        if idx < len(paths):
            p = paths[idx]
            title, _ = _describe_artifact(p)
            ax.imshow(plt.imread(str(p)))
            ax.set_title(f"{title}\n{p.name}", fontsize=7)
            ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle("Evaluation figures (grid)", fontsize=11, fontweight="bold", y=0.995)
    plt.subplots_adjust(top=0.92, hspace=0.35, wspace=0.12)
    _save_figure_page(fig, pdf)


def _prepare_csv_df(
    path: Path,
    max_rows: int,
    max_cols: int,
) -> Tuple[Optional[pd.DataFrame], str]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)
    notes: List[str] = []
    n_r, n_c = df.shape
    if n_c > max_cols:
        df = df.iloc[:, :max_cols]
        notes.append(f"cols {max_cols}/{n_c}")
    if n_r > max_rows:
        df = df.iloc[:max_rows, :]
        notes.append(f"rows {max_rows}/{n_r}")
    note = f" ({', '.join(notes)})" if notes else ""
    return df, note


def _page_csv_batch(
    paths: List[Path],
    pdf: PdfPages,
    *,
    max_rows: int,
    max_cols: int,
) -> None:
    if not paths:
        return
    n = len(paths)
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(n, 1, hspace=0.45, top=0.94, bottom=0.03, left=0.04, right=0.98)
    for i, path in enumerate(paths):
        title, desc = _describe_artifact(path)
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        df, err_or_note = _prepare_csv_df(path, max_rows, max_cols)
        if df is None:
            ax.set_title(f"{title} — {path.name}", fontsize=8, loc="left")
            ax.text(0.02, 0.5, f"Read error: {err_or_note}", fontsize=8, transform=ax.transAxes)
            continue
        cap = f"{title} — {path.name}\n{desc}{err_or_note}"
        ax.set_title(cap, fontsize=6, loc="left")
        cell_text = df.astype(str).values.tolist()
        col_labels = [str(c)[:18] for c in df.columns]
        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(4)
        tbl.scale(1, 1.05)
    fig.suptitle("Evaluation tables (batch)", fontsize=11, fontweight="bold", y=0.99)
    _save_figure_page(fig, pdf)


def build_eval_pdf(
    result_path: Union[str, Path],
    *,
    pdf_name: str = "eval_report.pdf",
    png_per_page: int = 6,
    csv_per_page: int = 3,
    max_csv_rows: int = 28,
    max_csv_cols: int = 12,
) -> Path:
    """Write ``pdf_name`` under ``result_path`` with batched PNG grids and stacked CSV tables.

    Up to ``png_per_page`` images share one page (grid with at most 4 columns).
    Up to ``csv_per_page`` truncated tables are stacked per page.
    """
    _configure_cjk_font()
    out_dir = Path(result_path).expanduser().resolve()
    csvs, pngs = _collect_artifacts(out_dir)
    pdf_path = out_dir / pdf_name

    pp = max(1, int(png_per_page))
    cp = max(1, int(csv_per_page))
    ncols = max(1, min(4, int(np.ceil(np.sqrt(pp)))))
    nrows = max(1, int(np.ceil(pp / ncols)))

    with PdfPages(pdf_path) as pdf:
        # Cover
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.74, "Model evaluation report", ha="center", fontsize=18, fontweight="bold")
        fig.text(0.5, 0.64, str(out_dir), ha="center", fontsize=9, family="monospace")
        fig.text(
            0.5,
            0.50,
            f"{len(csvs)} CSV + {len(pngs)} PNG — compact layout "
            f"(≤{cp} tables / ≤{pp} figures per page).",
            ha="center",
            fontsize=11,
        )
        _save_figure_page(fig, pdf)

        for batch in _chunks(csvs, cp):
            _page_csv_batch(
                batch,
                pdf,
                max_rows=max_csv_rows,
                max_cols=max_csv_cols,
            )

        for batch in _chunks(pngs, pp):
            _page_png_grid(batch, nrows, ncols, pdf)

    return pdf_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge model_eval display_*.csv and figure_*.png into eval_report.pdf."
    )
    p.add_argument(
        "result_path",
        type=Path,
        help="Directory containing display_*.csv, figure_*.png (e.g. model_eval output folder)",
    )
    p.add_argument(
        "--pdf-name",
        default="eval_report.pdf",
        help="PDF filename written inside result_path (default: eval_report.pdf)",
    )
    p.add_argument(
        "--png-per-page",
        type=int,
        default=6,
        metavar="N",
        help="Max PNGs per page (default: 6, arranged in a square grid)",
    )
    p.add_argument(
        "--csv-per-page",
        type=int,
        default=3,
        metavar="N",
        help="Max CSV tables stacked per page (default: 3)",
    )
    p.add_argument(
        "--max-csv-rows",
        type=int,
        default=28,
        help="Max rows shown per CSV on the PDF (default: 28)",
    )
    p.add_argument(
        "--max-csv-cols",
        type=int,
        default=12,
        help="Max columns shown per CSV on the PDF (default: 12)",
    )
    args = p.parse_args()
    path = build_eval_pdf(
        args.result_path,
        pdf_name=args.pdf_name,
        png_per_page=max(1, args.png_per_page),
        csv_per_page=max(1, args.csv_per_page),
        max_csv_rows=max(1, args.max_csv_rows),
        max_csv_cols=max(1, args.max_csv_cols),
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

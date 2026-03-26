"""Serialize offline ``model_eval`` captures as a Jupyter notebook (nbformat 4.5-style JSON).

Mimics ``orig_notebook_files/notebook_model_eval.ipynb``: alternating ``stream`` (stdout) and
``display_data`` (HTML/plain tables, PNG figures).
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.io.formats.style import Styler


def _html_mime(s: str) -> List[str]:
    if not s:
        return [""]
    lines = s.splitlines(keepends=True)
    return lines if lines else [s]


def format_run_header_markdown(
    *,
    model_path: str,
    eval_path: str,
    label_path: str,
    out_dir: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        "# Model evaluation (offline run)\n\n"
        f"- **model:** `{model_path}`\n"
        f"- **eval data:** `{eval_path}`\n"
        f"- **label (BOOT):** `{label_path}`\n"
        f"- **output dir:** `{out_dir}`\n\n"
        f"Captured at {ts}. Outputs mirror `notebook_model_eval.ipynb` "
        f"(`stream` + `display_data` for tables and figures).\n"
    )


class NotebookOutputRecorder:
    """Collects nbformat outputs for one code cell."""

    def __init__(self) -> None:
        self.outputs: List[Dict[str, Any]] = []
        self._stream_chunks: List[str] = []

    def feed_stream(self, s: str) -> None:
        if s:
            self._stream_chunks.append(s)

    def flush_stream(self) -> None:
        if not self._stream_chunks:
            return
        text = "".join(self._stream_chunks)
        self._stream_chunks.clear()
        if not text:
            return
        lines = text.splitlines(keepends=True)
        if not lines:
            lines = [text]
        self.outputs.append({"name": "stdout", "output_type": "stream", "text": lines})

    def add_dataframe(self, df: pd.DataFrame) -> None:
        self.flush_stream()
        html = df.to_html()
        plain = df.to_string()
        self.outputs.append(
            {
                "data": {
                    "text/html": _html_mime(html),
                    "text/plain": _html_mime(plain),
                },
                "metadata": {},
                "output_type": "display_data",
            }
        )

    def add_styler(self, styler: Styler) -> None:
        self.flush_stream()
        html = styler.to_html()
        plain = styler.data.to_string()
        self.outputs.append(
            {
                "data": {
                    "text/html": _html_mime(html),
                    "text/plain": _html_mime(plain),
                },
                "metadata": {},
                "output_type": "display_data",
            }
        )

    def add_repr_display(self, obj: Any) -> None:
        self.flush_stream()
        plain = repr(obj)
        self.outputs.append(
            {
                "data": {"text/plain": _html_mime(plain)},
                "metadata": {},
                "output_type": "display_data",
            }
        )

    def add_png_bytes(self, png: bytes) -> None:
        self.flush_stream()
        b64 = base64.b64encode(png).decode("ascii")
        self.outputs.append(
            {
                "data": {
                    "image/png": b64,
                    "text/plain": ["<matplotlib.figure.Figure> (PNG capture from offline model_eval)\n"],
                },
                "metadata": {"needs_background": "light"},
                "output_type": "display_data",
            }
        )

    def write_ipynb(self, path: Path, *, title_md: str) -> None:
        self.flush_stream()
        nb = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "id": str(uuid.uuid4()),
                    "metadata": {},
                    "source": [title_md],
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "id": str(uuid.uuid4()),
                    "metadata": {},
                    "outputs": self.outputs,
                    "source": [
                        "# model_eval offline run\n",
                        "# Captured print / display / plt.show outputs appear below.\n",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "pygments_lexer": "ipython3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

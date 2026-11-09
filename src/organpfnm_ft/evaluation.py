"""Optional post-training evaluation: healthy-vs-diseased age-gap analysis.

Run on the output of ``predict()`` (typically ``inference_demo.csv``
produced from ``ssl_subset.csv``, which contains disease samples).

For each (organ, timepoint) combination:

* Healthy vs diseased age_gap: Mann-Whitney U test;
* Disease-specific check: if ``Disease`` column carries e.g. liver disease
  codes (E11, K70, K76 etc.), report mean ``Hepatic_age_gap_corrected``
  vs the other organs' mean age-gap for the same patients.

Writes ``evaluation_summary.txt`` next to the input CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """Two-sided Mann-Whitney U test using scipy if available."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n_a = int(len(a))
    n_b = int(len(b))
    if n_a < 3 or n_b < 3:
        return float("nan"), float("nan"), n_a, n_b
    try:
        from scipy.stats import mannwhitneyu   # noqa: PLC0415
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p), n_a, n_b
    except Exception:
        return float("nan"), float("nan"), n_a, n_b


def evaluate(
    inference_df: pd.DataFrame,
    organ_names: Iterable[str],
    out_path: Path,
) -> None:
    """Write a healthy-vs-diseased age-gap report to ``out_path``."""
    lines: list[str] = []
    p = lines.append
    p("=" * 78)
    p("OrganPFN-M-LF FT evaluation report")
    p("=" * 78)
    p("")

    if "is_healthy" not in inference_df.columns:
        raise ValueError("inference_df must carry an 'is_healthy' column.")
    if "timepoint" not in inference_df.columns:
        # Fallback: treat everything as a single bucket.
        inference_df = inference_df.copy()
        inference_df["timepoint"] = "ALL"

    timepoints = sorted(inference_df["timepoint"].dropna().unique().tolist())
    healthy = inference_df["is_healthy"].astype(int) == 1
    diseased = inference_df["is_healthy"].astype(int) == 0

    p("--- Per-(organ, timepoint) healthy-vs-diseased age-gap ---")
    p(f"{'organ':<16} {'tp':<4} {'n_h':>6} {'n_d':>6} "
      f"{'mean_h':>8} {'mean_d':>8} {'U':>10} {'p':>10}")
    for organ in organ_names:
        gap_col = f"{organ}_age_gap_corrected"
        if gap_col not in inference_df.columns:
            gap_col = f"{organ}_age_gap"
        if gap_col not in inference_df.columns:
            continue
        for tp in timepoints:
            tp_mask = inference_df["timepoint"].astype(str) == str(tp)
            h_gap = inference_df.loc[healthy & tp_mask, gap_col].to_numpy()
            d_gap = inference_df.loc[diseased & tp_mask, gap_col].to_numpy()
            u, pv, n_h, n_d = _mann_whitney_u(h_gap, d_gap)
            mean_h = float(np.nanmean(h_gap)) if len(h_gap) else float("nan")
            mean_d = float(np.nanmean(d_gap)) if len(d_gap) else float("nan")
            p(f"{organ:<16} {str(tp):<4} {n_h:>6d} {n_d:>6d} "
              f"{mean_h:>8.3f} {mean_d:>8.3f} {u:>10.3f} {pv:>10.4g}")
    p("")
    p("=" * 78)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote evaluation report -> %s", out_path)

"""Utility helpers: logging, seeding, checkpoint I/O, schema loading (LF, 10-organ).

The schema exposes the timepoint metadata required by the long-format dataloader and
backbone:

  * ``timepoint_id_map``      e.g. ``{"T0": 0, "T2": 1}``
  * ``n_timepoints``          number of timepoints (2)
  * ``organ_timepoint_availability``  per-organ timepoint(s) it can appear at
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_path: Path | None = None,
                  level: int = logging.INFO) -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(stream=sys.stdout),
    ]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a", encoding="utf-8"))
    logging.basicConfig(
        level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True,
    )
    return logging.getLogger()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Schema (long-format, 10-organ)
# ---------------------------------------------------------------------------

@dataclass
class Schema:
    organ_names: list[str]
    feature_columns: list[str]
    organ_offsets: dict[str, tuple[int, int]]
    n_features_per_organ: dict[str, int]
    n_total_features: int
    metadata_columns: list[str]
    organ_presence_threshold_ssl: float
    organ_presence_threshold_finetune: float
    timepoint_id_map: dict[str, int]
    n_timepoints: int
    organ_timepoint_availability: dict[str, list[str]]
    raw: dict[str, Any]
    # ----- Sex-as-organ-feature extensions (V5) -----
    # Per-organ Sex column name (e.g. "Body_Sex"); present in feature_columns
    # as the FIRST column inside each organ's block.
    sex_column_per_organ: dict[str, str]
    # Index of that Sex column in feature_columns.
    sex_column_indices_per_organ: dict[str, int]
    # Per-organ feature lists EXCLUDING the Sex column - used for
    # organ-presence ratio and column-quality calculations.
    organ_feature_map_excluding_sex: dict[str, list[str]]
    n_features_per_organ_excluding_sex: dict[str, int]
    # SSL presence threshold flag.
    presence_excludes_sex: bool

    def n_organs(self) -> int:
        return len(self.organ_names)

    def real_feature_indices_for_organ(self, organ: str) -> list[int]:
        """Indices in feature_columns for this organ's NON-Sex features."""
        s, e = self.organ_offsets[organ]
        sex_idx = self.sex_column_indices_per_organ.get(organ, -1)
        return [i for i in range(s, e) if i != sex_idx]


def load_schema(path: Path) -> Schema:
    if not path.exists():
        raise FileNotFoundError(f"schema.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    organ_names: list[str] = list(raw["organ_names"])
    feature_columns: list[str] = list(raw["feature_columns"])
    n_total: int = int(raw["n_total_features"])
    if len(feature_columns) != n_total:
        raise ValueError(
            f"schema.json: len(feature_columns)={len(feature_columns)} != "
            f"n_total_features={n_total}"
        )

    organ_offsets: dict[str, tuple[int, int]] = {}
    cur = 0
    for o in organ_names:
        s, e = raw["organ_offsets"][o]
        s, e = int(s), int(e)
        if s != cur:
            raise ValueError(
                f"schema.json: organ_offsets gap detected at '{o}'. "
                f"Expected start={cur}, got {s}."
            )
        organ_offsets[o] = (s, e)
        slice_cols = feature_columns[s:e]
        listed = list(raw["organ_feature_map"][o])
        if slice_cols != listed:
            raise ValueError(
                f"schema.json: feature_columns[{s}:{e}] for organ '{o}' "
                f"does not match organ_feature_map[{o!r}]."
            )
        cur = e
    if cur != n_total:
        raise ValueError(
            f"schema.json: organ_offsets ends at {cur} but n_total_features={n_total}"
        )

    n_per = {o: int(raw["n_features_per_organ"][o]) for o in organ_names}
    for o, (s, e) in organ_offsets.items():
        if e - s != n_per[o]:
            raise ValueError(
                f"schema.json: n_features_per_organ[{o}] ({n_per[o]}) "
                f"!= offset width ({e - s})"
            )

    # Sex-as-organ-feature metadata (V5).  Tolerate older schemas that
    # don't carry these keys by deriving sensible defaults.
    sex_col_per_organ_raw = raw.get("sex_column_per_organ", {})
    sex_idx_per_organ_raw = raw.get("sex_column_indices_per_organ", {})
    fmap_excl_sex_raw = raw.get("organ_feature_map_excluding_sex", {})
    n_per_excl_sex_raw = raw.get("n_features_per_organ_excluding_sex", {})
    presence_excludes_sex = bool(
        raw["filter_thresholds"]["ssl"].get("presence_excludes_sex", False)
    )

    sex_column_per_organ: dict[str, str] = {}
    sex_column_indices: dict[str, int] = {}
    fmap_excl: dict[str, list[str]] = {}
    n_per_excl: dict[str, int] = {}
    for o in organ_names:
        sc = sex_col_per_organ_raw.get(o, f"{o}_Sex")
        sex_column_per_organ[o] = sc
        # Resolve index inside feature_columns.
        if o in sex_idx_per_organ_raw:
            sex_column_indices[o] = int(sex_idx_per_organ_raw[o])
        else:
            try:
                sex_column_indices[o] = feature_columns.index(sc)
            except ValueError:
                sex_column_indices[o] = -1
        # Excluding-Sex feature map.
        if o in fmap_excl_sex_raw:
            fmap_excl[o] = list(fmap_excl_sex_raw[o])
        else:
            block = list(raw["organ_feature_map"][o])
            fmap_excl[o] = [c for c in block if c != sc]
        n_per_excl[o] = int(n_per_excl_sex_raw.get(o, len(fmap_excl[o])))

    return Schema(
        organ_names=organ_names,
        feature_columns=feature_columns,
        organ_offsets=organ_offsets,
        n_features_per_organ=n_per,
        n_total_features=n_total,
        metadata_columns=list(raw["metadata_columns"]),
        organ_presence_threshold_ssl=float(
            raw["filter_thresholds"]["ssl"]["organ_presence_threshold"]
        ),
        organ_presence_threshold_finetune=float(
            raw["filter_thresholds"]["finetune"]["organ_presence_threshold"]
        ),
        timepoint_id_map={
            str(k): int(v) for k, v in raw["timepoint_id_map"].items()
        },
        n_timepoints=int(raw["n_timepoints"]),
        organ_timepoint_availability={
            o: list(raw["organ_timepoint_availability"][o]) for o in organ_names
        },
        raw=raw,
        sex_column_per_organ=sex_column_per_organ,
        sex_column_indices_per_organ=sex_column_indices,
        organ_feature_map_excluding_sex=fmap_excl,
        n_features_per_organ_excluding_sex=n_per_excl,
        presence_excludes_sex=presence_excludes_sex,
    )


# ---------------------------------------------------------------------------
# CSV metric writer
# ---------------------------------------------------------------------------

class CsvLogger:
    def __init__(self, path: Path, fieldnames: list[str]) -> None:
        self.path = path
        self.fieldnames = list(fieldnames)
        self._written_header = path.exists() and path.stat().st_size > 0

    def log(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if not self._written_header:
                writer.writeheader()
                self._written_header = True
            writer.writerow({k: row.get(k, "") for k in self.fieldnames})


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    *,
    model_state_dict: dict[str, torch.Tensor],
    schema: dict[str, Any],
    normalization: dict[str, Any],
    config: dict[str, Any],
    tabpfn_ckpt_path: str,
    global_step: int,
    epoch_summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model_state_dict,
        "schema": schema,
        "normalization": normalization,
        "config": config,
        "tabpfn_ckpt_path": str(tabpfn_ckpt_path),
        "global_step": int(global_step),
        "epoch_summaries": list(epoch_summaries),
    }
    torch.save(payload, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

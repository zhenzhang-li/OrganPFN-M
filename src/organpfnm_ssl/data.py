"""Dataset, organ-presence, normalization, and DataLoader (long-format, 10-organ).

Conventions
-----------
Each row in the CSV is one (Participant_ID, timepoint) observation.  The
metadata columns (per schema.json's ``metadata_columns``) are::

    Participant_ID | Sex | timepoint | timepoint_id | Age | Disease | is_healthy

followed by the measured feature columns. Cells that were not measured at that
timepoint (e.g. Brain features on a T0 row) appear as NaN.

The released table has one sample-level ``Sex`` column. For compatibility
with the trained architecture, the loader broadcasts it to one implementation
slot at the start of each organ block. These slots represent the same variable,
not ten distinct covariates.

Each sample exposes eight tensors / scalars (see __getitem__).

Important invariants
--------------------
* NaNs in raw feature columns are preserved up to standardization; we
  never mean-impute, MICE-impute, or otherwise fabricate values.
* Standardization statistics (per-feature mu/sigma) are computed on the
  healthy subset only across BOTH timepoints (the timepoint embedding
  inside the model is responsible for distinguishing T0 vs T2).
* ``organ_presence`` is computed at ``__getitem__`` time using the
  threshold read from ``schema.json``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import config as _cfg
from .utils import Schema

logger = logging.getLogger(__name__)


def _encode_sex(values: pd.Series) -> np.ndarray:
    normalized = values.astype("string").str.strip().str.upper()
    mapping = {"M": 0.0, "MALE": 0.0, "0": 0.0,
               "F": 1.0, "FEMALE": 1.0, "1": 1.0}
    encoded = normalized.map(mapping)
    if encoded.isna().any():
        bad = sorted(normalized.loc[encoded.isna()].dropna().unique().tolist())
        raise ValueError(f"Sex contains missing or unsupported values: {bad[:5]}")
    return encoded.to_numpy(dtype=np.float32)


def materialize_model_features(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Broadcast the one sample-level Sex value to internal model slots."""
    if "Sex" not in df.columns:
        raise ValueError("DataFrame is missing the sample-level 'Sex' column.")
    missing_measurements = [
        col
        for cols in schema.organ_feature_map_excluding_sex.values()
        for col in cols
        if col not in df.columns
    ]
    if missing_measurements:
        raise ValueError(f"Missing measurement columns: {missing_measurements[:10]}")
    sex = _encode_sex(df["Sex"])
    out = df.copy()
    for organ in schema.organ_names:
        column = schema.sex_column_per_organ.get(organ)
        if column and column not in out.columns:
            out[column] = sex
    return out


# ---------------------------------------------------------------------------
# Per-organ standardization statistics
# ---------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    """Per-feature mean/std on healthy samples (across both timepoints)."""

    mu: np.ndarray
    sigma: np.ndarray
    feature_columns: list[str]
    use_healthy_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "feature_columns": list(self.feature_columns),
            "use_healthy_only": bool(self.use_healthy_only),
        }


def compute_normalization_stats(
    df: pd.DataFrame,
    schema: Schema,
    *,
    sigma_floor: float = 1e-6,
) -> NormalizationStats:
    df = materialize_model_features(df, schema)
    if "is_healthy" not in df.columns:
        raise ValueError("DataFrame is missing required column 'is_healthy'.")
    mask = df["is_healthy"].fillna(0).astype(int) == 1
    if not mask.any():
        raise ValueError(
            "No healthy samples (is_healthy==1) found - cannot compute "
            "normalization statistics."
        )
    healthy = df.loc[mask, schema.feature_columns].to_numpy(dtype=np.float32)

    with np.errstate(invalid="ignore"):
        mu = np.nanmean(healthy, axis=0).astype(np.float32)
        sigma = np.nanstd(healthy, axis=0).astype(np.float32)

    bad_mu = ~np.isfinite(mu)
    bad_sigma = ~np.isfinite(sigma) | (sigma < sigma_floor)
    mu = np.where(bad_mu, 0.0, mu).astype(np.float32)
    sigma = np.where(bad_sigma, 1.0, sigma).astype(np.float32)

    n_bad = int(bad_sigma.sum())
    if n_bad:
        logger.warning(
            "Normalization: %d / %d features had sigma<%.2e on healthy "
            "samples; clamped sigma=1 for those.",
            n_bad, len(sigma), sigma_floor,
        )
    return NormalizationStats(
        mu=mu, sigma=sigma,
        feature_columns=list(schema.feature_columns),
        use_healthy_only=True,
    )


# ---------------------------------------------------------------------------
# Organ presence
# ---------------------------------------------------------------------------

def compute_organ_presence(
    feature_values: np.ndarray,
    organ_offsets: dict[str, tuple[int, int]],
    threshold: float,
    *,
    organ_names: list[str] | None = None,
    sex_column_indices_per_organ: dict[str, int] | None = None,
) -> np.ndarray:
    """0/1 organ-presence vector for ONE sample.

    Sex columns (one per organ, located at
    ``sex_column_indices_per_organ[organ]``) are EXCLUDED from both
    numerator and denominator of the per-organ non-NaN ratio - because
    every sample has a global Sex value and we don't want it to affect
    the presence decision.

    An organ whose real (non-Sex) feature count is zero is treated as
    absent (presence = 0) regardless of threshold; this preserves the
    "no real measurement => no signal" semantics.
    """
    if organ_names is None:
        organ_names = list(organ_offsets.keys())
    sex_idx_map = sex_column_indices_per_organ or {}
    n_organs = len(organ_names)
    presence = np.zeros(n_organs, dtype=np.float32)
    for o_idx, name in enumerate(organ_names):
        s, e = organ_offsets[name]
        if e == s:
            continue
        sex_idx = sex_idx_map.get(name, -1)
        # Build a boolean mask over [s, e) excluding the Sex position.
        if s <= sex_idx < e:
            # Two-segment slice: [s, sex_idx) and [sex_idx+1, e).
            left = feature_values[s:sex_idx]
            right = feature_values[sex_idx + 1: e]
            n_total = left.size + right.size
            if n_total == 0:
                continue
            n_non_nan = int(np.sum(~np.isnan(left))) + int(np.sum(~np.isnan(right)))
        else:
            organ_slice = feature_values[s:e]
            n_total = organ_slice.size
            if n_total == 0:
                continue
            n_non_nan = int(np.sum(~np.isnan(organ_slice)))
        ratio = n_non_nan / n_total
        presence[o_idx] = 1.0 if ratio >= threshold else 0.0
    return presence


# ---------------------------------------------------------------------------
# Per-organ feature-group mapping (for the model's organ-id embedding)
# ---------------------------------------------------------------------------

def build_feature_to_organ_map(schema: Schema) -> np.ndarray:
    F = schema.n_total_features
    out = np.full(F, -1, dtype=np.int64)
    for o_idx, name in enumerate(schema.organ_names):
        s, e = schema.organ_offsets[name]
        out[s:e] = o_idx
    if (out < 0).any():
        bad = np.where(out < 0)[0].tolist()[:5]
        raise ValueError(
            f"build_feature_to_organ_map: features {bad} have no organ assigned."
        )
    return out


def build_group_organ_membership(
    schema: Schema, features_per_group: int,
) -> tuple[np.ndarray, np.ndarray]:
    F = schema.n_total_features
    feat_to_organ = build_feature_to_organ_map(schema)
    F_padded = ((F + features_per_group - 1) // features_per_group) * features_per_group
    G = F_padded // features_per_group

    feat_to_organ_padded = np.concatenate([
        feat_to_organ,
        np.full(F_padded - F, -1, dtype=np.int64),
    ])
    grouped = feat_to_organ_padded.reshape(G, features_per_group)

    n_organs = schema.n_organs()
    organ_in_group = np.zeros((n_organs, G), dtype=bool)
    majority = np.full(G, -1, dtype=np.int64)
    for g in range(G):
        members = grouped[g]
        real = members[members >= 0]
        if real.size == 0:
            majority[g] = 0
            continue
        vals, counts = np.unique(real, return_counts=True)
        majority[g] = int(vals[counts.argmax()])
        for o in vals:
            organ_in_group[int(o), g] = True
    return majority, organ_in_group


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrganTabularDataset(Dataset):
    """Tabular Dataset over the SSL / fine-tune CSV (long-format, 10-organ)."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: Schema,
        norm: NormalizationStats,
        *,
        organ_presence_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.norm = norm
        self.organ_presence_threshold = (
            organ_presence_threshold
            if organ_presence_threshold is not None
            else schema.organ_presence_threshold_ssl
        )
        self._n_organs = schema.n_organs()

        df = materialize_model_features(df, schema)

        for col in (*schema.metadata_columns, *schema.feature_columns):
            if col not in df.columns:
                raise ValueError(f"DataFrame is missing column {col!r}")

        # Filter by min_organs (= 1 by default).  Sex columns are excluded
        # from the presence ratio computation per schema.
        min_organs = int(schema.raw["filter_thresholds"]["ssl"]["min_organs"])
        feat_arr = df[schema.feature_columns].to_numpy(dtype=np.float32, copy=False)
        n_present = np.zeros(feat_arr.shape[0], dtype=np.int64)
        for i in range(feat_arr.shape[0]):
            pres = compute_organ_presence(
                feat_arr[i], schema.organ_offsets,
                self.organ_presence_threshold,
                organ_names=schema.organ_names,
                sex_column_indices_per_organ=schema.sex_column_indices_per_organ,
            )
            n_present[i] = int(pres.sum())
        keep = n_present >= min_organs
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info(
                "SSL dataset filter: dropped %d / %d rows below min_organs=%d "
                "(threshold=%.2f).",
                n_drop, len(df), min_organs, self.organ_presence_threshold,
            )
            df = df.iloc[keep].reset_index(drop=True)

        # Pre-extract raw feature matrix.
        self._raw = df[schema.feature_columns].to_numpy(dtype=np.float32, copy=True)
        self._missing_mask = np.isnan(self._raw)
        std = (self._raw - norm.mu[None, :]) / norm.sigma[None, :]
        std = np.where(self._missing_mask, 0.0, std).astype(np.float32)
        self._std = std

        self._age = df["Age"].to_numpy(dtype=np.float32, copy=True)
        self._is_healthy = df["is_healthy"].fillna(0).astype(np.int64).to_numpy(copy=True)
        try:
            self._pid = df["Participant_ID"].astype(np.int64).to_numpy(copy=True)
        except (TypeError, ValueError):
            codes, _ = pd.factorize(df["Participant_ID"], sort=False)
            self._pid = codes.astype(np.int64)
        self._sex = _encode_sex(df["Sex"]).astype(np.int64)

        # Timepoint id (long format).
        if "timepoint_id" in df.columns:
            self._timepoint_id = df["timepoint_id"].astype(np.int64).to_numpy(copy=True)
        elif "timepoint" in df.columns:
            tp_map = schema.timepoint_id_map
            self._timepoint_id = (
                df["timepoint"].astype(str).map(tp_map).fillna(0)
                .astype(np.int64).to_numpy(copy=True)
            )
        else:
            raise ValueError(
                "DataFrame is missing both 'timepoint_id' and 'timepoint' columns."
            )

    def __len__(self) -> int:
        return self._raw.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        raw_row = self._raw[idx]
        std_row = self._std[idx]
        miss_row = self._missing_mask[idx]
        organ_presence = compute_organ_presence(
            raw_row, self.schema.organ_offsets,
            self.organ_presence_threshold,
            organ_names=self.schema.organ_names,
            sex_column_indices_per_organ=self.schema.sex_column_indices_per_organ,
        )
        return {
            "features": std_row,
            "missing_mask": miss_row,
            "organ_presence": organ_presence,
            "timepoint_id": int(self._timepoint_id[idx]),
            "age": np.float32(self._age[idx]),
            "is_healthy": int(self._is_healthy[idx]),
            "pid": int(self._pid[idx]),
            "sex": int(self._sex[idx]),
        }


# ---------------------------------------------------------------------------
# Collate + DataLoader
# ---------------------------------------------------------------------------

def organ_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    feats = np.stack([b["features"] for b in batch], axis=0)
    miss = np.stack([b["missing_mask"] for b in batch], axis=0)
    presence = np.stack([b["organ_presence"] for b in batch], axis=0)
    timepoint_id = np.array([b["timepoint_id"] for b in batch], dtype=np.int64)
    age = np.array([b["age"] for b in batch], dtype=np.float32)
    is_healthy = np.array([b["is_healthy"] for b in batch], dtype=np.int64)
    pid = np.array([b["pid"] for b in batch], dtype=np.int64)
    sex = np.array([b["sex"] for b in batch], dtype=np.int64)

    return {
        "features": torch.from_numpy(feats),
        "missing_mask": torch.from_numpy(miss),
        "organ_presence": torch.from_numpy(presence),
        "timepoint_id": torch.from_numpy(timepoint_id),
        "age": torch.from_numpy(age),
        "age_valid": torch.from_numpy(np.isfinite(age)),
        "is_healthy": torch.from_numpy(is_healthy),
        "pid": torch.from_numpy(pid),
        "sex": torch.from_numpy(sex),
    }


def make_balanced_sampler(
    is_healthy: np.ndarray, timepoint_id: np.ndarray,
) -> WeightedRandomSampler:
    """Stratified sampler balancing (healthy, T0), (healthy, T2), (disease, T0),
    (disease, T2)."""
    is_healthy = np.asarray(is_healthy).astype(np.int64)
    tp = np.asarray(timepoint_id).astype(np.int64)
    strata = 2 * is_healthy + (tp == 1).astype(np.int64)
    weights = np.zeros_like(strata, dtype=np.float64)

    unique, counts = np.unique(strata, return_counts=True)
    n_classes = len(unique)
    if n_classes == 0:
        return WeightedRandomSampler(
            weights=torch.ones(len(is_healthy), dtype=torch.double),
            num_samples=len(is_healthy), replacement=True,
        )
    per_class = 1.0 / n_classes
    for u, c in zip(unique, counts):
        if c > 0:
            weights[strata == u] = per_class / c
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(is_healthy), replacement=True,
    )


def build_dataloaders(
    schema: Schema,
    *,
    batch_size: int,
    num_workers: int = 0,
    nrows: int | None = None,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, NormalizationStats]:
    ssl_path = _cfg.ssl_csv_path()
    if not ssl_path.exists():
        raise FileNotFoundError(f"ssl_subset.csv not found at {ssl_path}")
    logger.info("Loading SSL CSV from %s (nrows=%s)", ssl_path, nrows)
    ssl_df = pd.read_csv(ssl_path, nrows=nrows)

    norm = compute_normalization_stats(ssl_df, schema)
    logger.info(
        "Computed normalization stats: mu range=[%.4f, %.4f], "
        "sigma range=[%.4f, %.4f]",
        float(np.min(norm.mu)), float(np.max(norm.mu)),
        float(np.min(norm.sigma)), float(np.max(norm.sigma)),
    )

    ssl_ds = OrganTabularDataset(ssl_df, schema, norm)
    ssl_loader = DataLoader(
        ssl_ds, batch_size=batch_size,
        sampler=make_balanced_sampler(ssl_ds._is_healthy, ssl_ds._timepoint_id),
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=organ_collate, drop_last=True,
    )

    # FT csv is used ONLY for the periodic linear-probe sanity check.
    # V2 layout puts each threshold in its own sub-dir; the SSL config
    # points at ft_thr_0p50 by default (override via SSL_LINEAR_PROBE_FT_TAG).
    # Missing FT file is treated as a hard error: the linear probe is part
    # of the spec and we refuse to silently skip it.
    ft_path = _cfg.finetune_csv_path()
    if not ft_path.exists():
        raise FileNotFoundError(
            f"finetune_subset.csv not found at {ft_path}.  V2 layout: the SSL "
            f"linear-probe FT csv lives under data_10org_v2/ft_<tag>/.  Check "
            f"that '两阶段数据合并训练_v2.py' has been run and that the "
            f"ft_thr_0p50/ (or whatever SSL_LINEAR_PROBE_FT_TAG points at) "
            f"directory was copied to the server."
        )

    logger.info("Loading finetune CSV from %s (nrows=%s)", ft_path, nrows)
    ft_df = pd.read_csv(ft_path, nrows=nrows)
    if "is_healthy" in ft_df.columns:
        before = len(ft_df)
        ft_df = ft_df.loc[ft_df["is_healthy"].fillna(0).astype(int) == 1].copy()
        if len(ft_df) != before:
            logger.info(
                "Filtered finetune_subset to healthy: %d -> %d rows",
                before, len(ft_df),
            )

    ft_ds = OrganTabularDataset(
        ft_df, schema, norm,
        organ_presence_threshold=schema.organ_presence_threshold_finetune,
    )
    ft_loader = DataLoader(
        ft_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        collate_fn=organ_collate, drop_last=False,
    )
    return ssl_loader, ft_loader, norm

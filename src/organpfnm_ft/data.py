"""Dataset, organ-presence, paired batch sampler, modality dropout, and
by-Participant_ID train/val split (long-format, 10 organs).

Important contract with self-supervision
----------------------------------------
The standardization statistics (mu/sigma) are loaded from the SSL
checkpoint -- we **never** recompute them.  This guarantees the FT model
sees the same input distribution as the SSL backbone was trained on, so
the loaded SSL weights remain meaningful.

The organ-presence threshold for FT (default ``0.50``, vs. SSL's
``0.30``) is read from ``schema.json`` and used to compute the per-
sample organ-presence vector at ``__getitem__`` time.  The threshold can
be overridden at construction time for the ablation experiment - see
:class:`FineTuneDataset.__init__`'s ``organ_presence_threshold`` kwarg.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, Subset

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
    """Broadcast one sample-level Sex value to internal organ-block slots.

    The resulting columns are implementation slots required by the released
    checkpoint schema. They all contain the same sample-level variable and are
    excluded from organ-presence and feature-missingness calculations.
    """
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
# Normalization (loaded from SSL ckpt; never recomputed)
# ---------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    """Wrap the (mu, sigma) loaded from the SSL checkpoint."""

    mu: np.ndarray
    sigma: np.ndarray
    feature_columns: list[str]
    use_healthy_only: bool = True

    @classmethod
    def from_ssl_checkpoint(cls, ssl_norm: dict[str, Any]) -> "NormalizationStats":
        if not all(k in ssl_norm for k in ("mu", "sigma", "feature_columns")):
            raise KeyError(
                "SSL checkpoint['normalization'] is missing required keys; "
                f"got: {sorted(ssl_norm.keys())}"
            )
        mu = np.asarray(ssl_norm["mu"], dtype=np.float32)
        sigma = np.asarray(ssl_norm["sigma"], dtype=np.float32)
        return cls(
            mu=mu, sigma=sigma,
            feature_columns=list(ssl_norm["feature_columns"]),
            use_healthy_only=bool(ssl_norm.get("use_healthy_only", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
            "feature_columns": list(self.feature_columns),
            "use_healthy_only": bool(self.use_healthy_only),
        }


# ---------------------------------------------------------------------------
# Organ presence (FT-stage threshold)
# ---------------------------------------------------------------------------

def compute_organ_presence(
    feature_values: np.ndarray,
    organ_offsets: dict[str, tuple[int, int]],
    threshold: float,
    *,
    organ_names: list[str] | None = None,
    sex_column_indices_per_organ: dict[str, int] | None = None,
) -> np.ndarray:
    """Identical algorithm to ``organpfnm_ssl.data.compute_organ_presence``.

    The threshold differs between stages (FT default = 0.50, SSL = 0.30)
    and is read from ``schema.json["filter_thresholds"]["finetune"]``.  For
    the ablation runner the threshold is overridden at the Dataset level.

    Sex columns (one per organ at ``sex_column_indices_per_organ[organ]``)
    are EXCLUDED from both numerator and denominator of the per-organ
    non-NaN ratio.  An organ whose real (non-Sex) feature count is zero is
    treated as absent (presence = 0).
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
        if s <= sex_idx < e:
            left = feature_values[s:sex_idx]
            right = feature_values[sex_idx + 1: e]
            n_total = left.size + right.size
            if n_total == 0:
                continue
            n_non_nan = int(np.sum(~np.isnan(left))) + int(np.sum(~np.isnan(right)))
        else:
            sl = feature_values[s:e]
            n_total = sl.size
            if n_total == 0:
                continue
            n_non_nan = int(np.sum(~np.isnan(sl)))
        ratio = n_non_nan / n_total
        presence[o_idx] = 1.0 if ratio >= threshold else 0.0
    return presence


def compute_per_organ_effective_samples(
    df: pd.DataFrame,
    schema: Schema,
    threshold: float,
) -> dict[str, int]:
    """Count rows where each organ's presence ratio >= threshold.

    Sex columns are EXCLUDED from the per-organ feature list before the
    ratio is computed (V5).
    """
    out: dict[str, int] = {}
    for organ in schema.organ_names:
        real_cols = [c for c in schema.organ_feature_map_excluding_sex.get(organ, [])
                     if c in df.columns]
        if not real_cols:
            out[organ] = 0
            continue
        block = df[real_cols].to_numpy(dtype=np.float32)
        ratio = (~np.isnan(block)).mean(axis=1)
        out[organ] = int((ratio >= threshold).sum())
    return out


# ---------------------------------------------------------------------------
# Dataset (long-format)
# ---------------------------------------------------------------------------

class FineTuneDataset(Dataset):
    """Tabular dataset over the long-format finetune subset.

    Each ``__getitem__`` returns::

        features         : (F,) float32, standardized + zero-filled at NaN
        missing_mask     : (F,) bool, True where raw was NaN
        organ_presence   : (n_organs,) float32, computed at FT threshold
        timepoint_id     : int (0 = T0, 1 = T2)
        age              : float32 (the supervised target)
        is_healthy       : int 0/1
        sex              : int 0/1
        pid              : int64 Participant_ID

    By default ``min_organs = schema["filter_thresholds"]["finetune"]["min_organs"]``
    (= 1) is applied as a dataset filter: rows with zero present organs at
    the chosen threshold are dropped because they contribute no usable
    signal to the loss.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        schema: Schema,
        norm: NormalizationStats,
        *,
        organ_presence_threshold: float | None = None,
        min_organs: int | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.norm = norm
        self._n_organs = schema.n_organs()
        df = materialize_model_features(df, schema)
        self._threshold = (
            float(organ_presence_threshold)
            if organ_presence_threshold is not None
            else float(schema.organ_presence_threshold_finetune)
        )
        ft_thresholds = schema.raw["filter_thresholds"]["finetune"]
        self._min_organs = int(
            min_organs if min_organs is not None
            else ft_thresholds.get("min_organs", 1)
        )

        for col in (*schema.metadata_columns, *schema.feature_columns):
            if col not in df.columns:
                raise ValueError(f"DataFrame missing column {col!r}")

        if list(norm.feature_columns) != list(schema.feature_columns):
            raise ValueError(
                "Normalization feature_columns mismatch schema feature_columns. "
                "SSL ckpt may not match the data schema."
            )

        df = df.reset_index(drop=True).copy()

        # ---- Filter by min_organs at the chosen threshold ----
        # Sex columns are excluded from the presence-ratio calculation (V5).
        feat_arr = df[schema.feature_columns].to_numpy(dtype=np.float32)
        n_present_per_row = np.zeros(len(df), dtype=np.int64)
        for i in range(len(df)):
            pres = compute_organ_presence(
                feat_arr[i], schema.organ_offsets, self._threshold,
                organ_names=schema.organ_names,
                sex_column_indices_per_organ=schema.sex_column_indices_per_organ,
            )
            n_present_per_row[i] = int(pres.sum())
        keep = n_present_per_row >= self._min_organs
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info(
                "FT dataset filter: dropped %d / %d rows below min_organs=%d "
                "(threshold=%.2f).",
                n_drop, len(df), self._min_organs, self._threshold,
            )
            df = df.loc[keep].reset_index(drop=True)

        # Pre-extract feature matrix + standardize.
        self._raw = df[schema.feature_columns].to_numpy(dtype=np.float32, copy=True)
        self._missing_mask = np.isnan(self._raw)
        std = (self._raw - norm.mu[None, :]) / norm.sigma[None, :]
        std = np.where(self._missing_mask, 0.0, std).astype(np.float32)
        self._std = std

        # Metadata.
        self._age = df["Age"].to_numpy(dtype=np.float32, copy=True)
        self._is_healthy = df["is_healthy"].fillna(0).astype(np.int64).to_numpy(copy=True)
        # Participant_ID may not be a pure int; preserve the original column
        # for the by-Participant split, and store a numeric form for the
        # batch tensor.
        self._pid_str = df["Participant_ID"].astype(str).to_numpy(copy=True)
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
            raw_row, self.schema.organ_offsets, self._threshold,
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
            "sex": int(self._sex[idx]),
            "pid": int(self._pid[idx]),
        }


# ---------------------------------------------------------------------------
# Modality dropout (training augmentation; anti-degeneracy #3)
# ---------------------------------------------------------------------------

def apply_modality_dropout(
    organ_presence_BO: torch.Tensor,
    *,
    dropout_prob: float,
    min_drop: int,
    max_drop: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Probabilistically drop ``[min_drop, max_drop]`` present organs per sample."""
    if dropout_prob <= 0:
        return organ_presence_BO
    B, O = organ_presence_BO.shape
    out = organ_presence_BO.clone()
    device = out.device
    for b in range(B):
        if torch.rand(1, generator=generator, device=device).item() >= dropout_prob:
            continue
        present_idx = torch.nonzero(out[b] > 0).flatten()
        if present_idx.numel() <= 1:
            continue
        max_drop_eff = min(max_drop, present_idx.numel() - 1)
        min_drop_eff = min(min_drop, max_drop_eff)
        if max_drop_eff < min_drop_eff:
            continue
        n_drop = torch.randint(
            min_drop_eff, max_drop_eff + 1, (1,),
            device=device, generator=generator,
        ).item()
        if n_drop <= 0:
            continue
        perm = torch.randperm(present_idx.numel(), device=device, generator=generator)
        chosen = present_idx[perm[:n_drop]]
        out[b, chosen] = 0.0
    return out


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def ft_collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    feats = np.stack([b["features"] for b in batch], axis=0)
    miss = np.stack([b["missing_mask"] for b in batch], axis=0)
    presence = np.stack([b["organ_presence"] for b in batch], axis=0)
    tp = np.array([b["timepoint_id"] for b in batch], dtype=np.int64)
    age = np.array([b["age"] for b in batch], dtype=np.float32)
    is_healthy = np.array([b["is_healthy"] for b in batch], dtype=np.int64)
    pid = np.array([b["pid"] for b in batch], dtype=np.int64)
    sex = np.array([b["sex"] for b in batch], dtype=np.int64)
    return {
        "features": torch.from_numpy(feats),
        "missing_mask": torch.from_numpy(miss),
        "organ_presence": torch.from_numpy(presence),
        "timepoint_id": torch.from_numpy(tp),
        "age": torch.from_numpy(age),
        "age_valid": torch.from_numpy(np.isfinite(age)),
        "is_healthy": torch.from_numpy(is_healthy),
        "pid": torch.from_numpy(pid),
        "sex": torch.from_numpy(sex),
    }


# ---------------------------------------------------------------------------
# By-Participant_ID train/val split
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    train_indices: np.ndarray
    val_indices: np.ndarray
    train_pids: set[str]
    val_pids: set[str]
    paired_pairs_in_train: list[tuple[int, int]]  # (idx_T0, idx_T2) into full_dataset
    paired_pairs_in_val: list[tuple[int, int]]


def by_participant_split(
    full_dataset: FineTuneDataset,
    pairs_df: pd.DataFrame | None,
    *,
    val_ratio: float,
    seed: int,
    val_pids_override: set | None = None,
) -> SplitResult:
    """Split rows into train/val keeping all rows of each Participant
    together.

    ``val_pids_override`` (NEW, backward-compatible): when given, those
    Participant_IDs become the validation fold and everyone else is train
    (used by the K-fold cross-validation runners).  ``val_ratio``/``seed``
    are ignored in that case.

    Steps:
      1. Collect unique Participant_IDs from the dataset.
      2. Shuffle with the given seed.
      3. Take the first ``val_ratio`` fraction into val; remainder into
         train.
      4. Map back to row indices.
      5. From ``participant_pairs.csv``, materialize the list of
         (T0 row index, T2 row index) within each split using the
         dataset's row order.  Pairs whose participant has only one row
         in the dataset (e.g. because the other row was filtered out)
         are silently dropped.
    """
    pid_str = full_dataset._pid_str  # type: ignore[attr-defined]
    tp_id = full_dataset._timepoint_id  # type: ignore[attr-defined]
    n = len(full_dataset)
    unique_pids = np.unique(pid_str)
    if val_pids_override is not None:
        _vset = {str(p) for p in val_pids_override}
        val_pids = {p for p in unique_pids.tolist() if str(p) in _vset}
        train_pids = {p for p in unique_pids.tolist() if str(p) not in _vset}
    else:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(unique_pids))
        n_val = int(round(len(unique_pids) * float(val_ratio)))
        val_pids = set(unique_pids[perm[:n_val]].tolist())
        train_pids = set(unique_pids[perm[n_val:]].tolist())

    train_idx = np.array(
        [i for i in range(n) if pid_str[i] in train_pids], dtype=np.int64,
    )
    val_idx = np.array(
        [i for i in range(n) if pid_str[i] in val_pids], dtype=np.int64,
    )

    # Build pid -> {T0_row, T2_row} from the dataset's row order.
    rows_by_pid: dict[str, dict[int, int]] = {}
    for i in range(n):
        rows_by_pid.setdefault(str(pid_str[i]), {})[int(tp_id[i])] = i

    def _pairs_in(pid_set: set[str]) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for pid in pid_set:
            d = rows_by_pid.get(pid, {})
            if 0 in d and 1 in d:
                out.append((d[0], d[1]))
        return out

    pairs_train = _pairs_in(train_pids)
    pairs_val = _pairs_in(val_pids)
    logger.info(
        "by-Participant split: train pids=%d (rows=%d, pairs=%d), "
        "val pids=%d (rows=%d, pairs=%d)",
        len(train_pids), len(train_idx), len(pairs_train),
        len(val_pids), len(val_idx), len(pairs_val),
    )
    return SplitResult(
        train_indices=train_idx,
        val_indices=val_idx,
        train_pids=train_pids,
        val_pids=val_pids,
        paired_pairs_in_train=pairs_train,
        paired_pairs_in_val=pairs_val,
    )


# ---------------------------------------------------------------------------
# Paired batch sampler
# ---------------------------------------------------------------------------

class PairedBatchSampler(Sampler[list[int]]):
    """Yield batches where ``paired_quota`` rows come from K paired
    (T0, T2) participants and the remainder is random unpaired rows.

    Concretely: each batch of ``batch_size`` contains
    ``2 * paired_quota`` rows from ``paired_quota`` distinct Participants
    plus ``batch_size - 2 * paired_quota`` random rows drawn from the
    remaining unpaired pool (with replacement if the pool is small).
    Within-batch row order is shuffled so the model does not see paired
    samples in any particular order.

    The sampler yields **subset-local indices**: when wrapping a
    ``Subset(full_dataset, train_indices)`` the values returned here are
    in ``[0, len(train_indices))``.  Pass paired pairs *already
    translated* to subset-local indices.
    """

    def __init__(
        self,
        n_total: int,
        paired_pairs: list[tuple[int, int]],
        *,
        batch_size: int,
        paired_quota: int,
        seed: int,
        drop_last: bool = True,
    ) -> None:
        self.n_total = int(n_total)
        self.batch_size = int(batch_size)
        self.paired_quota = int(paired_quota)
        if self.paired_quota * 2 > self.batch_size:
            raise ValueError(
                f"paired_quota * 2 ({self.paired_quota * 2}) cannot exceed "
                f"batch_size ({self.batch_size})."
            )
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self._epoch = 0

        # Build the unpaired pool (any row not appearing in any pair).
        in_pair = np.zeros(self.n_total, dtype=bool)
        for a, b in paired_pairs:
            in_pair[a] = True
            in_pair[b] = True
        self.pair_rows = list(paired_pairs)
        self.unpaired_pool = np.where(~in_pair)[0].astype(np.int64)
        if len(self.unpaired_pool) == 0:
            # Fall back to a full pool (with possible double-counting).
            self.unpaired_pool = np.arange(self.n_total, dtype=np.int64)

    def __len__(self) -> int:
        n_batches = self.n_total // self.batch_size
        return n_batches if self.drop_last else (n_batches + (1 if self.n_total % self.batch_size else 0))

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1

        # Shuffle the pair list each epoch.
        pair_perm = rng.permutation(len(self.pair_rows))
        unpair_n_per_batch = self.batch_size - 2 * self.paired_quota
        n_batches = len(self)

        pair_cursor = 0
        for _ in range(n_batches):
            batch: list[int] = []
            # 1) paired-rows quota
            for _q in range(self.paired_quota):
                if len(self.pair_rows) == 0:
                    break
                if pair_cursor >= len(pair_perm):
                    pair_perm = rng.permutation(len(self.pair_rows))
                    pair_cursor = 0
                a, b = self.pair_rows[int(pair_perm[pair_cursor])]
                batch.extend([int(a), int(b)])
                pair_cursor += 1
            # 2) random unpaired remainder
            if unpair_n_per_batch > 0:
                pool = self.unpaired_pool
                if len(pool) >= unpair_n_per_batch:
                    pick = rng.choice(pool, size=unpair_n_per_batch, replace=False)
                else:
                    pick = rng.choice(pool, size=unpair_n_per_batch, replace=True)
                batch.extend(int(x) for x in pick.tolist())
            rng.shuffle(batch)
            yield batch


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    full_dataset: FineTuneDataset,
    split: SplitResult,
    *,
    batch_size: int,
    paired_quota: int,
    num_workers: int,
    seed: int,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader, list[tuple[int, int]], list[tuple[int, int]]]:
    """Return (train_loader, val_loader, train_pairs_local, val_pairs_local).

    ``train_pairs_local`` / ``val_pairs_local`` are the pair tuples
    translated into the SUBSET-local index space so the training loop can
    map each ``(idx_T0, idx_T2)`` directly into the batched tensor.  But
    because PairedBatchSampler shuffles within batch, the consumer locates
    pairs in each batch dynamically using ``Participant_ID + timepoint_id``
    rather than relying on positions.
    """
    train_subset = Subset(full_dataset, split.train_indices.tolist())
    val_subset = Subset(full_dataset, split.val_indices.tolist())

    full_to_local_train = {int(full): int(loc) for loc, full in enumerate(split.train_indices)}
    full_to_local_val = {int(full): int(loc) for loc, full in enumerate(split.val_indices)}

    train_pairs_local = [
        (full_to_local_train[a], full_to_local_train[b])
        for a, b in split.paired_pairs_in_train
        if a in full_to_local_train and b in full_to_local_train
    ]
    val_pairs_local = [
        (full_to_local_val[a], full_to_local_val[b])
        for a, b in split.paired_pairs_in_val
        if a in full_to_local_val and b in full_to_local_val
    ]

    # Train: paired batch sampler.
    train_batch_sampler = PairedBatchSampler(
        n_total=len(train_subset),
        paired_pairs=train_pairs_local,
        batch_size=batch_size,
        paired_quota=paired_quota,
        seed=seed,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_subset,
        batch_sampler=train_batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ft_collate,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=ft_collate,
        drop_last=False,
    )
    return train_loader, val_loader, train_pairs_local, val_pairs_local


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_finetune_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"finetune_subset.csv not found at {path}")
    logger.info("Loading FT CSV: %s (nrows=%s)", path, nrows)
    return pd.read_csv(path, nrows=nrows)


def load_participant_pairs(
    path: Path, df: pd.DataFrame,
) -> pd.DataFrame:
    """Load participant_pairs.csv and validate that the listed pairs make
    sense against the in-memory finetune DataFrame.  Pairs whose
    Participant_ID is absent from ``df`` are dropped with a warning."""
    if not path.exists():
        logger.warning("participant_pairs.csv not found at %s; "
                       "longitudinal consistency loss will be effectively "
                       "disabled (no pairs).", path)
        return pd.DataFrame(columns=["Participant_ID", "Age_T0", "Age_T2", "Age_diff"])
    pairs = pd.read_csv(path)
    pairs["Participant_ID"] = pairs["Participant_ID"].astype(str)
    pids = set(df["Participant_ID"].astype(str).unique().tolist())
    n_before = len(pairs)
    pairs = pairs[pairs["Participant_ID"].isin(pids)].reset_index(drop=True)
    if len(pairs) < n_before:
        logger.info(
            "participant_pairs filter: kept %d / %d pairs whose IDs are in "
            "the FT subset.", len(pairs), n_before,
        )
    return pairs

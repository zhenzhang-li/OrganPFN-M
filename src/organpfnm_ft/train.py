"""FT training loop (long-format, 10 organs, by-Participant split).

Implements the supervised procedure documented in `docs/METHODS_TO_CODE.md`:

  * by-Participant train/val split (no row-level leakage);
  * paired batch sampler (at least K paired (T0, T2) samples per batch);
  * modality dropout (anti-degeneracy #3);
  * full FT loss = main Huber + 4 anti-degeneracy regularizers +
    longitudinal consistency;
  * grouped LR (early backbone / late backbone / top backbone / heads);
  * per-(organ, timepoint) MAE + Pearson + calibration in val;
  * anti-degeneracy diagnostics (inter-organ std + cosine similarity);
  * longitudinal consistency diagnostic;
  * per-organ validation prediction CSV (one CSV per organ:
    Participant_ID, Sex, true_age, pred_age, timepoint).

The ``train_run`` function is parameterized by ``organ_presence_threshold``
so the ablation runner can launch the SAME training with different
thresholds.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]

from . import config as _cfg
from .data import (
    FineTuneDataset,
    NormalizationStats,
    SplitResult,
    apply_modality_dropout,
    build_dataloaders,
    by_participant_split,
    compute_per_organ_effective_samples,
    load_finetune_csv,
    load_participant_pairs,
)
from .losses import (
    FTLossOutput,
    compute_ft_loss,
    compute_inverse_frequency_weights,
    compute_val_metrics_per_organ_timepoint,
)
from .model import OPFNMLF_FT, build_ft_model_from_ssl_checkpoint
from .utils import (
    CsvLogger,
    Schema,
    load_schema,
    save_ft_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(
    model: OPFNMLF_FT,
    *,
    train_cfg: _cfg.TrainConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Layer-specific learning-rate schedule used in fine-tuning."""
    backbone = model.backbone
    layers = backbone.tabpfn.transformer_encoder.layers
    n_layers = len(layers)
    half = n_layers // 2
    top_n = max(2, n_layers // 6)

    early_params: list[torch.nn.Parameter] = []
    late_params: list[torch.nn.Parameter] = []
    top_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    # Embedder + first half: early.
    early_modules = [
        backbone.tabpfn.encoder,
        backbone.tabpfn.y_encoder,
        getattr(backbone.tabpfn, "feature_positional_embedding_embeddings", None),
        getattr(backbone.tabpfn, "add_thinking_tokens", None),
    ]
    for m in early_modules:
        if m is None:
            continue
        for p in m.parameters():
            if p.requires_grad:
                early_params.append(p)

    for i, layer in enumerate(layers):
        if i < half:
            bucket = early_params
        elif i >= n_layers - top_n:
            bucket = top_params
        else:
            bucket = late_params
        for p in layer.parameters():
            if p.requires_grad:
                bucket.append(p)

    # OrganPFNMBackbone's own organ_id_embedding + timepoint_embedding -> heads.
    for p in backbone.organ_id_embedding.parameters():
        if p.requires_grad:
            head_params.append(p)
    for p in backbone.timepoint_embedding.parameters():
        if p.requires_grad:
            head_params.append(p)

    # FT-side: orthogonal projector + heads + log_sigma.
    for p in model.orthogonal_projector.parameters():
        if p.requires_grad:
            head_params.append(p)
    for p in model.heads.parameters():
        if p.requires_grad:
            head_params.append(p)
    head_params.append(model.log_sigma_per_organ)

    param_groups = [
        {"params": early_params, "lr": train_cfg.lr_backbone_early,
         "weight_decay": train_cfg.weight_decay_backbone, "name": "backbone_early"},
        {"params": late_params, "lr": train_cfg.lr_backbone_late,
         "weight_decay": train_cfg.weight_decay_backbone, "name": "backbone_late"},
        {"params": top_params, "lr": train_cfg.lr_backbone_top,
         "weight_decay": train_cfg.weight_decay_backbone, "name": "backbone_top"},
        {"params": head_params, "lr": train_cfg.lr_heads,
         "weight_decay": train_cfg.weight_decay_heads, "name": "heads"},
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))

    warmup = int(train_cfg.warmup_steps)
    total_steps = max(1, total_steps)

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = max(0.0, min(1.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: OPFNMLF_FT,
    val_loader,
    schema: Schema,
    *,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, torch.Tensor]]:
    """Run one full pass; return per-(organ, timepoint) metrics + raw tensors."""
    model.eval()
    all_pred: list[torch.Tensor] = []
    all_summary: list[torch.Tensor] = []
    all_age: list[torch.Tensor] = []
    all_pres: list[torch.Tensor] = []
    all_healthy: list[torch.Tensor] = []
    all_pid: list[torch.Tensor] = []
    all_sex: list[torch.Tensor] = []
    all_tp: list[torch.Tensor] = []

    for batch in val_loader:
        feats = batch["features"].to(device, non_blocking=True)
        presence = batch["organ_presence"].to(device, non_blocking=True)
        tp = batch["timepoint_id"].to(device, non_blocking=True)
        out = model(feats, presence, tp)
        all_pred.append(out["per_organ_pred"].cpu())
        all_summary.append(out["per_organ_summary"].cpu())
        all_age.append(batch["age"])
        all_pres.append(batch["organ_presence"])
        all_healthy.append(batch["is_healthy"])
        all_pid.append(batch["pid"])
        all_sex.append(batch["sex"])
        all_tp.append(batch["timepoint_id"])

    pred_BOk = torch.cat(all_pred, dim=0)
    summary_BOE = torch.cat(all_summary, dim=0)
    age_B = torch.cat(all_age, dim=0)
    pres_BO = torch.cat(all_pres, dim=0)
    healthy_B = torch.cat(all_healthy, dim=0)
    pid_B = torch.cat(all_pid, dim=0)
    sex_B = torch.cat(all_sex, dim=0)
    tp_B = torch.cat(all_tp, dim=0)

    metrics = compute_val_metrics_per_organ_timepoint(
        pred_BOk, age_B, pres_BO, tp_B,
        organ_names=schema.organ_names,
        organ_timepoint_availability=schema.organ_timepoint_availability,
        timepoint_id_map=schema.timepoint_id_map,
    )
    raw = {
        "pred_BOk": pred_BOk,
        "summary_BOE": summary_BOE,
        "age_B": age_B,
        "presence_BO": pres_BO,
        "is_healthy_B": healthy_B,
        "pid_B": pid_B,
        "sex_B": sex_B,
        "timepoint_B": tp_B,
    }
    return metrics, raw


# ---------------------------------------------------------------------------
# Per-organ validation prediction CSVs
# ---------------------------------------------------------------------------

_SEX_INT_TO_LETTER = {0: "M", 1: "F"}


@torch.no_grad()
def run_single_organ_validation(
    model: OPFNMLF_FT,
    val_loader,
    schema: Schema,
    *,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run validation 10 separate times, once per organ, with all OTHER
    organ_presence flags forced to 0.

    For each organ ``O`` we re-run the validation loop with the input
    ``organ_presence`` rewritten to a one-hot vector that activates only
    ``O`` (and only on rows where ``O`` is actually present at the active
    FT threshold; otherwise the row contributes nothing).  This isolates
    the organ's predictive signal from cross-organ context.

    Returns a dict keyed by organ name with the same raw tensor layout as
    :func:`run_validation`'s second return value.
    """
    model.eval()
    n_organs = schema.n_organs()
    per_organ_raw: dict[str, dict[str, list[torch.Tensor]]] = {
        o: {"pred": [], "summary": [], "age": [], "pres": [],
            "healthy": [], "pid": [], "sex": [], "tp": []}
        for o in schema.organ_names
    }
    for batch in val_loader:
        feats = batch["features"].to(device, non_blocking=True)
        full_pres = batch["organ_presence"].to(device, non_blocking=True)
        tp = batch["timepoint_id"].to(device, non_blocking=True)
        for o_idx, organ in enumerate(schema.organ_names):
            single = torch.zeros_like(full_pres)
            # Only activate organ O for rows where it was actually present.
            single[:, o_idx] = full_pres[:, o_idx]
            out = model(feats, single, tp)
            per_organ_raw[organ]["pred"].append(out["per_organ_pred"].cpu())
            per_organ_raw[organ]["summary"].append(out["per_organ_summary"].cpu())
            per_organ_raw[organ]["age"].append(batch["age"])
            per_organ_raw[organ]["pres"].append(single.cpu())
            per_organ_raw[organ]["healthy"].append(batch["is_healthy"])
            per_organ_raw[organ]["pid"].append(batch["pid"])
            per_organ_raw[organ]["sex"].append(batch["sex"])
            per_organ_raw[organ]["tp"].append(batch["timepoint_id"])

    out: dict[str, dict[str, torch.Tensor]] = {}
    for organ in schema.organ_names:
        d = per_organ_raw[organ]
        out[organ] = {
            "pred_BOk": torch.cat(d["pred"], dim=0),
            "summary_BOE": torch.cat(d["summary"], dim=0),
            "age_B": torch.cat(d["age"], dim=0),
            "presence_BO": torch.cat(d["pres"], dim=0),
            "is_healthy_B": torch.cat(d["healthy"], dim=0),
            "pid_B": torch.cat(d["pid"], dim=0),
            "sex_B": torch.cat(d["sex"], dim=0),
            "timepoint_B": torch.cat(d["tp"], dim=0),
        }
    return out


def write_per_organ_val_predictions_single(
    out_dir: Path,
    single_raw: dict[str, dict[str, torch.Tensor]],
    schema: Schema,
    *,
    filename_prefix: str = "val_predictions_singleorgan",
) -> dict[str, Path]:
    """Write 10 CSVs, one per organ X, listing organ X's prediction made
    with ONLY X visible (other organ_presence flags forced to 0).

    Output schema is identical to ``write_per_organ_val_predictions`` so
    downstream analysis can compare them row-for-row.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    tp_map_inv = {v: k for k, v in schema.timepoint_id_map.items()}
    for o_idx, organ in enumerate(schema.organ_names):
        raw = single_raw[organ]
        pred_BOk = raw["pred_BOk"]
        pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
        age_B = raw["age_B"].numpy()
        pres_BO = raw["presence_BO"].numpy()
        pid_B = raw["pid_B"].numpy()
        sex_B = raw["sex_B"].numpy()
        tp_B = raw["timepoint_B"].numpy()
        sex_letters = [_SEX_INT_TO_LETTER.get(int(s), "U") for s in sex_B.tolist()]
        tp_letters = [tp_map_inv.get(int(t), str(t)) for t in tp_B.tolist()]
        # Only rows where this organ was originally present.
        present_mask = (pres_BO[:, o_idx] > 0) & np.isfinite(pred_mean_BO[:, o_idx])
        path = out_dir / f"{filename_prefix}_{organ}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Participant_ID", "Sex", "true_age", "pred_age", "timepoint"]
            )
            for i in range(len(age_B)):
                if not present_mask[i]:
                    continue
                writer.writerow([
                    int(pid_B[i]),
                    sex_letters[i],
                    f"{float(age_B[i]):.6g}",
                    f"{float(pred_mean_BO[i, o_idx]):.6g}",
                    tp_letters[i],
                ])
        logger.info(
            "Wrote %d single-organ val predictions for %s -> %s",
            int(present_mask.sum()), organ, path,
        )
        paths[organ] = path
    return paths


def write_per_organ_val_predictions(
    out_dir: Path,
    val_raw: dict[str, torch.Tensor],
    schema: Schema,
    *,
    filename_prefix: str = "val_predictions",
) -> dict[str, Path]:
    """Write one CSV per organ with (Participant_ID, Sex, true_age,
    pred_age, timepoint).

    Only rows where the organ is *present* in the sample (organ_presence
    == 1 at the active FT threshold) are written; absent organs would
    contribute meaningless predictions because the architectural latent
    zero-gate sets their summary to zero.

    Returns ``{organ_name: csv_path}``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_BOk = val_raw["pred_BOk"]
    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    age_B = val_raw["age_B"].numpy()
    pres_BO = val_raw["presence_BO"].numpy()
    pid_B = val_raw["pid_B"].numpy()
    sex_B = val_raw["sex_B"].numpy()
    tp_B = val_raw["timepoint_B"].numpy()

    sex_letters = [_SEX_INT_TO_LETTER.get(int(s), "U") for s in sex_B.tolist()]
    tp_map_inv = {v: k for k, v in schema.timepoint_id_map.items()}
    tp_letters = [tp_map_inv.get(int(t), str(t)) for t in tp_B.tolist()]

    paths: dict[str, Path] = {}
    for o_idx, organ in enumerate(schema.organ_names):
        present_mask = (pres_BO[:, o_idx] > 0) & np.isfinite(pred_mean_BO[:, o_idx])
        path = out_dir / f"{filename_prefix}_{organ}.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["Participant_ID", "Sex", "true_age", "pred_age", "timepoint"]
            )
            for i in range(len(age_B)):
                if not present_mask[i]:
                    continue
                writer.writerow([
                    int(pid_B[i]),
                    sex_letters[i],
                    f"{float(age_B[i]):.6g}",
                    f"{float(pred_mean_BO[i, o_idx]):.6g}",
                    tp_letters[i],
                ])
        logger.info(
            "Wrote %d val predictions for %s -> %s",
            int(present_mask.sum()), organ, path,
        )
        paths[organ] = path
    return paths


def weighted_avg_val_mae(
    metrics: dict[str, dict[str, float]],
    weight_by: str = "n",
) -> float:
    """Weighted average of per-(organ, timepoint) val MAEs."""
    total_w = 0.0
    total = 0.0
    for key, md in metrics.items():
        if not (md and math.isfinite(md.get("mae", float("nan")))
                and (md.get("n", 0) or 0) > 0):
            continue
        w = float(md["n"]) if weight_by == "n" else 1.0
        total_w += w
        total += w * md["mae"]
    if total_w == 0:
        return float("nan")
    return total / total_w


# ---------------------------------------------------------------------------
# Age-bias linear correction (per organ × timepoint)
# ---------------------------------------------------------------------------

def fit_age_bias_correction(
    val_raw: dict[str, torch.Tensor],
    schema: Schema,
) -> dict[str, dict[str, float]]:
    """Fit ``gap = a * age + b`` per (organ, timepoint) on healthy val.

    Output keys are ``"<organ>__<tp_name>"`` (e.g. ``"Body__T0"``); the
    inference utility uses the (organ, timepoint) of each sample to pick
    the right correction coefficients.
    """
    pred_BOk = val_raw["pred_BOk"]
    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    age = val_raw["age_B"].numpy()
    presence = val_raw["presence_BO"].numpy() > 0
    healthy = val_raw["is_healthy_B"].numpy() == 1
    tp_np = val_raw["timepoint_B"].numpy()

    out: dict[str, dict[str, float]] = {}
    for o_idx, organ in enumerate(schema.organ_names):
        avail = list(schema.organ_timepoint_availability.get(organ, []))
        for tp_name in avail:
            tp_id = int(schema.timepoint_id_map[tp_name])
            m = (healthy
                 & presence[:, o_idx]
                 & (tp_np == tp_id)
                 & np.isfinite(age)
                 & np.isfinite(pred_mean_BO[:, o_idx]))
            key = f"{organ}__{tp_name}"
            if m.sum() < 5:
                out[key] = {"a": 0.0, "b": 0.0, "n_used": int(m.sum())}
                continue
            gap = pred_mean_BO[m, o_idx] - age[m]
            x = age[m]
            a, b = np.polyfit(x, gap, deg=1)
            out[key] = {"a": float(a), "b": float(b), "n_used": int(m.sum())}
    return out


# ---------------------------------------------------------------------------
# Anti-degeneracy diagnostic
# ---------------------------------------------------------------------------

def anti_degeneracy_diagnostic(
    val_raw: dict[str, torch.Tensor],
    schema: Schema,
) -> dict[str, Any]:
    """Compute inter-organ std (years) and inter-organ cosine similarity
    on the healthy val subset.

    Target ranges used for diagnostics:
        mean std            in [2, 5] years
        mean cosine sim     < 0.3
    """
    pred_BOk = val_raw["pred_BOk"]
    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    summary_BOE = val_raw["summary_BOE"].numpy()
    presence = val_raw["presence_BO"].numpy() > 0
    healthy = val_raw["is_healthy_B"].numpy() == 1

    # Inter-organ std per row.
    per_row_std: list[float] = []
    cos_sims: list[float] = []
    for b in np.where(healthy)[0]:
        present_idx = np.where(presence[b])[0]
        if len(present_idx) < 2:
            continue
        preds_present = pred_mean_BO[b, present_idx]
        per_row_std.append(float(np.std(preds_present)))
        # Cosine similarity over pairs of present organ summaries.
        sums_present = summary_BOE[b, present_idx]  # (n_present, E)
        # Normalize and compute pairwise cosine.
        norms = np.linalg.norm(sums_present, axis=-1, keepdims=True) + 1e-12
        normed = sums_present / norms
        sim_mat = normed @ normed.T  # (n_present, n_present)
        # Take upper-triangle (excluding diagonal).
        iu = np.triu_indices(len(present_idx), k=1)
        cos_sims.extend(sim_mat[iu].tolist())

    mean_std = float(np.mean(per_row_std)) if per_row_std else float("nan")
    mean_cos = float(np.mean(cos_sims)) if cos_sims else float("nan")
    return {
        "healthy_val_inter_organ_mean_std_years": mean_std,
        "healthy_val_inter_organ_mean_cosine_similarity": mean_cos,
        "n_healthy_rows_used": int(np.sum(healthy)),
        "n_pairs_used_for_cosine": int(len(cos_sims)),
    }


# ---------------------------------------------------------------------------
# Longitudinal-consistency diagnostic
# ---------------------------------------------------------------------------

def longitudinal_consistency_diagnostic(
    val_raw: dict[str, torch.Tensor],
    schema: Schema,
    organs: Sequence[str] = ("Body", "Immune", "Lungs"),
) -> dict[str, dict[str, float]]:
    """Compare healthy paired (T0, T2) ``pred_diff`` mean vs ``age_diff`` mean.

    A well-trained model has both means within 2 years of each other
    This diagnostic is reported for paired visits.
    """
    pred_BOk = val_raw["pred_BOk"]
    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    age = val_raw["age_B"].numpy()
    presence = val_raw["presence_BO"].numpy() > 0
    healthy = val_raw["is_healthy_B"].numpy() == 1
    pid = val_raw["pid_B"].numpy()
    tp = val_raw["timepoint_B"].numpy()

    # Group rows by pid; restrict to healthy ones.
    pid_to_rows: dict[int, dict[int, int]] = {}
    for i in range(len(pid)):
        if not healthy[i]:
            continue
        pid_to_rows.setdefault(int(pid[i]), {})[int(tp[i])] = i

    organ_to_idx = {o: i for i, o in enumerate(schema.organ_names)}
    out: dict[str, dict[str, float]] = {}
    for organ in organs:
        if organ not in organ_to_idx:
            continue
        o_idx = organ_to_idx[organ]
        age_diffs: list[float] = []
        pred_diffs: list[float] = []
        for d in pid_to_rows.values():
            if 0 not in d or 1 not in d:
                continue
            i1, i2 = d[0], d[1]
            if not (presence[i1, o_idx] and presence[i2, o_idx]):
                continue
            age_diffs.append(float(age[i2] - age[i1]))
            pred_diffs.append(float(pred_mean_BO[i2, o_idx] - pred_mean_BO[i1, o_idx]))
        n = len(age_diffs)
        out[organ] = {
            "n_pairs": int(n),
            "mean_age_diff": float(np.mean(age_diffs)) if n else float("nan"),
            "mean_pred_diff": float(np.mean(pred_diffs)) if n else float("nan"),
            "mean_abs_residual": (
                float(np.mean(np.abs(np.array(pred_diffs) - np.array(age_diffs))))
                if n else float("nan")
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Single training run (parameterized by organ_presence_threshold)
# ---------------------------------------------------------------------------

def train_run(
    *,
    output_dir: Path,
    organ_presence_threshold: float | None = None,
    smoke_test: bool = False,
    nrows: int | None = None,
    num_workers: int = 0,
    gpu_id: int = 0,
    override_n_epochs: int | None = None,
    val_pids_override: set | None = None,
    age_reweight: tuple | None = None,
    asym: tuple | None = None,
    ccc_weight: float = 0.0,
) -> dict[str, Any]:
    """One full training run, returning the best-epoch summary.

    NEW (backward-compatible) cross-validation / de-biasing hooks:
      * ``val_pids_override``: set of Participant_IDs to use as the
        validation fold (K-fold CV).  ``None`` -> random 80/20 as before.
      * ``age_reweight``: ``(bin_edges, bin_weights)`` (each a 1-D sequence;
        ``len(weights) == len(edges)+1``) for Method-1 age-bin re-weighting
        of the Huber loss.  ``None`` -> unweighted as before.
      * ``asym``: ``(lam, age_lo, age_hi)`` for the Method-2 age-conditional
        asymmetric Huber (heavier penalty on high-age under-prediction).
        ``None`` -> symmetric as before.

    Args:
        output_dir: All artifacts go under here.
        organ_presence_threshold: Override
            ``schema.organ_presence_threshold_finetune``.  When ``None``,
            use the value from schema.json (= 0.50).  Set by the ablation
            runner to {0.9, 0.7, 0.5, 0.3, 0.1}.
        smoke_test: One mini-batch + one epoch; bypass early stop.
        nrows: CSV row cap for fast local debug.
        num_workers: DataLoader workers.
        gpu_id: CUDA index when CUDA is available.
        override_n_epochs: Override TRAIN_CFG.n_epochs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / _cfg.TRAIN_CFG.train_log_name)

    train_cfg = _cfg.TRAIN_CFG
    model_cfg = _cfg.MODEL_CFG
    if override_n_epochs is not None:
        train_cfg.n_epochs = int(override_n_epochs)
    set_seed(train_cfg.seed)

    schema = load_schema(_cfg.schema_path())
    # The public schema is authoritative. This keeps the loader and model
    # compatible with synthetic smoke tests while preserving the study's
    # 776 measurement + 10 internal Sex-slot configuration in full runs.
    model_cfg.n_total_features = schema.n_total_features
    model_cfg.n_organs = schema.n_organs()
    model_cfg.n_timepoints = schema.n_timepoints
    threshold = (
        float(organ_presence_threshold)
        if organ_presence_threshold is not None
        else float(schema.organ_presence_threshold_finetune)
    )

    logger.info("============================================================")
    logger.info("OrganPFN-M-LF FT (env=%s)", _cfg.ENV)
    logger.info("DATA_DIR     : %s", _cfg.DATA_DIR)
    logger.info("SSL_CKPT     : %s", _cfg.SSL_CKPT_PATH)
    logger.info("OUTPUT_DIR   : %s", output_dir)
    logger.info("organ_presence_threshold = %.2f", threshold)
    logger.info("smoke=%s nrows=%s gpu_id=%s", smoke_test, nrows, gpu_id)
    logger.info("============================================================")

    if not _cfg.SSL_CKPT_PATH.exists():
        raise FileNotFoundError(
            f"SSL checkpoint not found at {_cfg.SSL_CKPT_PATH}"
        )

    # Read SSL norm here so we can build the dataset.
    ssl_ckpt = torch.load(_cfg.SSL_CKPT_PATH, map_location="cpu", weights_only=False)
    norm_dict = ssl_ckpt["normalization"]
    norm = NormalizationStats.from_ssl_checkpoint(norm_dict)
    logger.info("Loaded SSL normalization: mu range=[%.3f, %.3f], "
                "sigma range=[%.3f, %.3f]",
                float(np.min(norm.mu)), float(np.max(norm.mu)),
                float(np.min(norm.sigma)), float(np.max(norm.sigma)))

    # Load FT csv + pairs.
    ft_df = load_finetune_csv(_cfg.finetune_csv_path(), nrows=nrows)
    pairs_df = load_participant_pairs(_cfg.participant_pairs_csv_path(), ft_df)

    # Recompute per-organ effective samples at the active threshold
    # (inverse-frequency weights depend on this).
    per_organ_effective = compute_per_organ_effective_samples(
        ft_df, schema, threshold,
    )
    inv_freq = compute_inverse_frequency_weights(
        per_organ_effective,
        organ_names=schema.organ_names,
        use_sqrt=train_cfg.inverse_freq_use_sqrt,
    )
    logger.info(
        "Per-organ effective samples @ threshold=%.2f: %s",
        threshold,
        {o: per_organ_effective[o] for o in schema.organ_names},
    )
    logger.info(
        "Inverse-frequency weights (sqrt=%s): %s",
        train_cfg.inverse_freq_use_sqrt,
        {o: round(float(inv_freq[i]), 3)
         for i, o in enumerate(schema.organ_names)},
    )

    # Dataset.
    full_dataset = FineTuneDataset(
        ft_df, schema, norm,
        organ_presence_threshold=threshold,
    )

    # By-Participant split (val_pids_override -> K-fold CV fold).
    split = by_participant_split(
        full_dataset, pairs_df,
        val_ratio=train_cfg.val_split_ratio,
        seed=train_cfg.seed,
        val_pids_override=val_pids_override,
    )
    if val_pids_override is not None:
        logger.info("CV fold mode: %d val participants supplied via override.",
                    len(val_pids_override))

    # Method-1 (LDS) age re-weighting LUT -> device tensors (or None).
    _age_edges_t = _age_w_t = None
    if age_reweight is not None:
        _edges, _weights = age_reweight
        logger.info("Age-reweight (LDS) ENABLED: %d bins, weight range=[%.3f, %.3f].",
                    len(_weights), float(min(_weights)), float(max(_weights)))
    if asym is not None:
        _lo, _hi = asym[1], asym[2]
        _desc = (f"per-organ ranges (n={len(_lo)})" if hasattr(_lo, "__len__")
                 else f"age_lo={float(_lo):.1f}, age_hi={float(_hi):.1f}")
        logger.info("Asymmetric Huber (Method-2) ENABLED: lam=%.2f, %s.", float(asym[0]), _desc)
    if ccc_weight and ccc_weight > 0:
        logger.info("CCC accuracy loss (Method-3) ENABLED: ccc_weight=%.3f.", float(ccc_weight))

    # DataLoaders (PairedBatchSampler on train, plain on val).
    train_loader, val_loader, train_pairs_local, val_pairs_local = build_dataloaders(
        full_dataset, split,
        batch_size=train_cfg.batch_size,
        paired_quota=train_cfg.paired_samples_per_batch,
        num_workers=num_workers,
        seed=train_cfg.seed,
        pin_memory=False,  # set True if device is cuda; safe default
    )
    logger.info(
        "Train loader: %d batches/epoch (batch_size=%d, paired_quota=%d).",
        len(train_loader), train_cfg.batch_size, train_cfg.paired_samples_per_batch,
    )

    # Model.
    model, _ssl_norm, report = build_ft_model_from_ssl_checkpoint(
        ssl_ckpt_path=_cfg.SSL_CKPT_PATH,
        schema=schema, model_cfg=model_cfg,
    )
    logger.info("[load report] %s", report)

    if torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n_cuda:
            raise ValueError(f"Requested gpu_id={gpu_id} but only {n_cuda} GPU(s) available.")
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        logger.info("Device: %s (%s)", device, torch.cuda.get_device_name(gpu_id))
    else:
        device = torch.device("cpu")
        logger.info("Device: cpu")
    model = model.to(device)

    if age_reweight is not None:                  # Method-1 LDS weight tensors
        _age_edges_t = torch.as_tensor(_edges, dtype=torch.float32, device=device)
        _age_w_t = torch.as_tensor(_weights, dtype=torch.float32, device=device)

    # Optimizer + scheduler.
    total_steps = max(1, len(train_loader)) * int(train_cfg.n_epochs)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, train_cfg=train_cfg, total_steps=total_steps,
    )
    scaler = GradScaler(
        device="cuda",
        enabled=(device.type == "cuda" and train_cfg.mixed_precision),
    )

    # CSV loggers.
    train_csv = CsvLogger(
        output_dir / train_cfg.train_metrics_csv,
        fieldnames=[
            "global_step", "epoch", "lr_early", "lr_late", "lr_top", "lr_heads",
            "loss_total", "huber_avg", "contrastive", "divergence",
            "age_bias_penalty", "longitudinal",
            *[f"huber_{o}" for o in schema.organ_names],
        ],
    )
    val_fields = ["epoch", "weighted_val_mae"]
    for o in schema.organ_names:
        for tp_name in schema.organ_timepoint_availability.get(o, []):
            key = f"{o}__{tp_name}"
            val_fields += [f"mae_{key}", f"pearson_{key}", f"cal_{key}", f"n_{key}"]
    val_fields += [f"log_sigma_{o}" for o in schema.organ_names]
    val_fields += [
        "ad_mean_std_years", "ad_mean_cosine_similarity",
    ]
    val_csv = CsvLogger(output_dir / train_cfg.val_metrics_csv, fieldnames=val_fields)

    # Training loop.
    inv_freq = inv_freq.to(device)
    best_w_mae = float("inf")
    best_epoch = -1
    epochs_since_best = 0
    history: list[dict[str, Any]] = []
    best_val_metrics: dict[str, dict[str, float]] = {}
    global_step = 0
    rng = torch.Generator(device=device).manual_seed(int(train_cfg.seed))

    for epoch in range(int(train_cfg.n_epochs)):
        model.train()
        epoch_t0 = time.time()
        run_total = 0.0
        run_huber = 0.0
        run_contra = 0.0
        run_div = 0.0
        run_bias = 0.0
        run_long = 0.0
        run_per_organ = np.zeros(schema.n_organs(), dtype=np.float64)
        n_batches = 0

        for batch in train_loader:
            feats = batch["features"].to(device, non_blocking=True)
            presence = batch["organ_presence"].to(device, non_blocking=True)
            tp = batch["timepoint_id"].to(device, non_blocking=True)
            age = batch["age"].to(device, non_blocking=True)
            age_valid = batch["age_valid"].to(device, non_blocking=True)
            healthy_batch = batch["is_healthy"].to(device, non_blocking=True)
            pid_batch = batch["pid"].to(device, non_blocking=True)
            if not bool(age_valid.all()):
                presence = presence * age_valid.unsqueeze(-1).float()

            # Modality dropout (training only).
            presence = apply_modality_dropout(
                presence,
                dropout_prob=train_cfg.modality_dropout_prob,
                min_drop=train_cfg.modality_dropout_min_drop,
                max_drop=train_cfg.modality_dropout_max_drop,
                generator=rng,
            )

            sw = None                              # Method-1 per-sample age weight
            if _age_w_t is not None:
                sw = _age_w_t[torch.bucketize(age, _age_edges_t)]

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                out = model(feats, presence, tp)
                loss_out: FTLossOutput = compute_ft_loss(
                    per_organ_pred_BOk=out["per_organ_pred"],
                    per_organ_summary_BOE=out["per_organ_summary"],
                    target_age_B=age,
                    organ_presence_BO=presence,
                    log_sigma_O=model.log_sigma_per_organ,
                    inv_freq_weights_O=inv_freq,
                    is_healthy_B=healthy_batch,
                    timepoint_id_B=tp,
                    pid_B=pid_batch,
                    organ_names=schema.organ_names,
                    huber_delta=train_cfg.huber_delta,
                    age_bias_reg_weight=model_cfg.age_bias_reg_weight,
                    gamma_contrastive=train_cfg.gamma_contrastive,
                    gamma_divergence=train_cfg.gamma_divergence,
                    gamma_long=train_cfg.gamma_long,
                    contrastive_margin=train_cfg.contrastive_margin,
                    prediction_min_std=train_cfg.prediction_min_std,
                    prediction_max_std=train_cfg.prediction_max_std,
                    long_loss_organs=train_cfg.long_loss_organs,
                    sample_weight_B=sw,
                    asym=asym,
                    ccc_weight=ccc_weight,
                )
            loss = loss_out.total

            did_step = True
            if scaler.is_enabled():
                prev_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip_max_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= prev_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip_max_norm,
                )
                optimizer.step()
            if did_step:
                scheduler.step()
            global_step += 1
            n_batches += 1

            run_total += float(loss.detach().cpu())
            run_huber += float(loss_out.per_organ_huber.mean().cpu())
            run_contra += float(loss_out.contrastive.cpu())
            run_div += float(loss_out.divergence.cpu())
            run_bias += float(loss_out.age_bias_penalty.cpu())
            run_long += float(loss_out.longitudinal.cpu())
            run_per_organ += loss_out.per_organ_huber.cpu().numpy().astype(np.float64)

            if global_step % train_cfg.log_every_n_steps == 0:
                lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
                row = {
                    "global_step": global_step, "epoch": epoch,
                    "lr_early": lrs.get("backbone_early", 0.0),
                    "lr_late": lrs.get("backbone_late", 0.0),
                    "lr_top": lrs.get("backbone_top", 0.0),
                    "lr_heads": lrs.get("heads", 0.0),
                    "loss_total": float(loss.detach().cpu()),
                    "huber_avg": float(loss_out.per_organ_huber.mean().cpu()),
                    "contrastive": float(loss_out.contrastive.cpu()),
                    "divergence": float(loss_out.divergence.cpu()),
                    "age_bias_penalty": float(loss_out.age_bias_penalty.cpu()),
                    "longitudinal": float(loss_out.longitudinal.cpu()),
                }
                for o_idx, organ in enumerate(schema.organ_names):
                    row[f"huber_{organ}"] = float(loss_out.per_organ_huber[o_idx].cpu())
                train_csv.log(row)
                logger.info(
                    "[step %d / epoch %d] total=%.4f huber_avg=%.4f "
                    "contra=%.4f div=%.4f bias=%.4f long=%.4f",
                    global_step, epoch,
                    row["loss_total"], row["huber_avg"],
                    row["contrastive"], row["divergence"],
                    row["age_bias_penalty"], row["longitudinal"],
                )

            if smoke_test:
                break

        # ---- end of epoch: validation ----
        run_total /= max(1, n_batches)
        run_huber /= max(1, n_batches)
        run_contra /= max(1, n_batches)
        run_div /= max(1, n_batches)
        run_bias /= max(1, n_batches)
        run_long /= max(1, n_batches)

        val_metrics, val_raw = run_validation(
            model, val_loader, schema, device=device,
        )
        w_mae = weighted_avg_val_mae(val_metrics)
        ad_diag = anti_degeneracy_diagnostic(val_raw, schema)
        long_diag = longitudinal_consistency_diagnostic(val_raw, schema)
        epoch_dur = time.time() - epoch_t0

        logger.info(
            "[epoch %d] done in %.1fs.  avg train_total=%.4f.  "
            "weighted_val_mae=%.3f.  AD mean_std=%.3f years cos=%.3f.",
            epoch, epoch_dur, run_total, w_mae,
            ad_diag["healthy_val_inter_organ_mean_std_years"],
            ad_diag["healthy_val_inter_organ_mean_cosine_similarity"],
        )
        for key, md in sorted(val_metrics.items()):
            logger.info(
                "[epoch %d / val] %-30s mae=%.3f pearson=%.3f cal=%.3f n=%d",
                epoch, key, md["mae"], md["pearson_r"],
                md["calibration_spearman"], md["n"],
            )
        for organ, md in long_diag.items():
            logger.info(
                "[epoch %d / long] %s n=%d mean_age_diff=%.2f "
                "mean_pred_diff=%.2f abs_resid=%.2f",
                epoch, organ, md["n_pairs"],
                md["mean_age_diff"], md["mean_pred_diff"],
                md["mean_abs_residual"],
            )

        log_sigmas = model.log_sigma_per_organ.detach().cpu().numpy()
        val_row: dict[str, Any] = {
            "epoch": epoch, "weighted_val_mae": w_mae,
            "ad_mean_std_years": ad_diag["healthy_val_inter_organ_mean_std_years"],
            "ad_mean_cosine_similarity": ad_diag["healthy_val_inter_organ_mean_cosine_similarity"],
        }
        for o_idx, organ in enumerate(schema.organ_names):
            for tp_name in schema.organ_timepoint_availability.get(organ, []):
                key = f"{organ}__{tp_name}"
                md = val_metrics.get(key, {})
                val_row[f"mae_{key}"] = md.get("mae", "")
                val_row[f"pearson_{key}"] = md.get("pearson_r", "")
                val_row[f"cal_{key}"] = md.get("calibration_spearman", "")
                val_row[f"n_{key}"] = md.get("n", "")
            val_row[f"log_sigma_{organ}"] = float(log_sigmas[o_idx])
        val_csv.log(val_row)

        history.append({
            "epoch": epoch, "duration_sec": epoch_dur,
            "weighted_val_mae": w_mae,
            "avg_train_loss_total": run_total,
            "val": val_metrics,
            "ad_diag": ad_diag,
            "long_diag": long_diag,
            "log_sigma": {
                o: float(log_sigmas[i])
                for i, o in enumerate(schema.organ_names)
            },
        })

        improved = math.isfinite(w_mae) and (w_mae < best_w_mae - 1e-4)
        if improved:
            best_w_mae = w_mae
            best_epoch = epoch
            epochs_since_best = 0
            best_val_metrics = val_metrics
            age_bias = fit_age_bias_correction(val_raw, schema)
            save_ft_checkpoint(
                output_dir / train_cfg.best_ckpt_name,
                model_state_dict=model.state_dict(),
                schema=schema.raw,
                normalization=norm_dict,
                config={
                    "embed_dim": model_cfg.embed_dim,
                    "n_layers": report.n_layers,
                    "n_organs": schema.n_organs(),
                    "n_timepoints": schema.n_timepoints,
                    "feature_columns": list(schema.feature_columns),
                    "organ_offsets": {
                        o: list(v) for o, v in schema.organ_offsets.items()
                    },
                    "features_per_group": model_cfg.features_per_group,
                    "num_thinking_rows": model_cfg.num_thinking_rows,
                    "k_ensemble": model_cfg.k_ensemble,
                    "head_hidden_dim": model_cfg.head_hidden_dim,
                    "head_dropout": model_cfg.head_dropout,
                    "rank1_init_std": model_cfg.rank1_init_std,
                    "orthogonal_projected_dim": model_cfg.orthogonal_projected_dim,
                },
                best_epoch=best_epoch,
                best_val_mae={
                    f"{o}__{tp}": val_metrics.get(f"{o}__{tp}", {}).get("mae", float("nan"))
                    for o in schema.organ_names
                    for tp in schema.organ_timepoint_availability.get(o, [])
                },
                weighted_val_mae=best_w_mae,
                log_sigma_per_organ={
                    o: float(log_sigmas[i])
                    for i, o in enumerate(schema.organ_names)
                },
                age_bias_correction=age_bias,
                organ_presence_threshold=threshold,
                train_history=history,
            )
            with (output_dir / "age_bias_correction_params.json").open(
                "w", encoding="utf-8",
            ) as f:
                json.dump(age_bias, f, indent=2)
            with (output_dir / "degeneracy_diagnostic.json").open(
                "w", encoding="utf-8",
            ) as f:
                json.dump({"ad_diag": ad_diag, "long_diag": long_diag}, f, indent=2)
            write_per_organ_val_predictions(output_dir, val_raw, schema)
            # ★ New in V5: single-organ predictions (each organ predicted
            # with all OTHER organ_presence flags forced to 0).  10 extra
            # CSVs side-by-side with the "all-organ context" CSVs above.
            single_raw = run_single_organ_validation(
                model, val_loader, schema, device=device,
            )
            write_per_organ_val_predictions_single(
                output_dir, single_raw, schema,
            )
            logger.info(
                "New best at epoch %d (weighted_val_mae=%.3f). Saved "
                "best_model.pt + age-bias + per-organ val predictions "
                "(full + single-organ).",
                epoch, best_w_mae,
            )
        else:
            epochs_since_best += 1
            if epochs_since_best >= train_cfg.early_stop_patience:
                logger.info(
                    "Early stop at epoch %d (no improvement for %d epochs).",
                    epoch, train_cfg.early_stop_patience,
                )
                break

        if smoke_test:
            break

    # Save final model.
    save_ft_checkpoint(
        output_dir / train_cfg.final_ckpt_name,
        model_state_dict=model.state_dict(),
        schema=schema.raw,
        normalization=norm_dict,
        config={
            "embed_dim": model_cfg.embed_dim,
            "n_layers": report.n_layers,
            "n_organs": schema.n_organs(),
            "n_timepoints": schema.n_timepoints,
            "feature_columns": list(schema.feature_columns),
            "organ_offsets": {o: list(v) for o, v in schema.organ_offsets.items()},
            "features_per_group": model_cfg.features_per_group,
            "num_thinking_rows": model_cfg.num_thinking_rows,
            "k_ensemble": model_cfg.k_ensemble,
            "head_hidden_dim": model_cfg.head_hidden_dim,
            "head_dropout": model_cfg.head_dropout,
            "rank1_init_std": model_cfg.rank1_init_std,
            "orthogonal_projected_dim": model_cfg.orthogonal_projected_dim,
        },
        best_epoch=best_epoch,
        best_val_mae={
            f"{o}__{tp}": best_val_metrics.get(f"{o}__{tp}", {}).get("mae", float("nan"))
            for o in schema.organ_names
            for tp in schema.organ_timepoint_availability.get(o, [])
        },
        weighted_val_mae=best_w_mae,
        log_sigma_per_organ={
            o: float(model.log_sigma_per_organ.detach().cpu().numpy()[i])
            for i, o in enumerate(schema.organ_names)
        },
        age_bias_correction=None,
        organ_presence_threshold=threshold,
        train_history=history,
    )
    logger.info("Training complete. Best epoch=%d, weighted_val_mae=%.3f.",
                best_epoch, best_w_mae)
    return {
        "organ_presence_threshold": threshold,
        "best_epoch": best_epoch,
        "best_weighted_val_mae": best_w_mae,
        "best_val_metrics": best_val_metrics,
        "n_per_organ_effective": per_organ_effective,
    }


# ---------------------------------------------------------------------------
# Top-level entry point (single-threshold run)
# ---------------------------------------------------------------------------

def main(
    *,
    smoke_test: bool = False,
    nrows: int | None = None,
    output_dir: Path | None = None,
    num_workers: int = 0,
    gpu_id: int = 0,
    organ_presence_threshold: float | None = None,
    override_n_epochs: int | None = None,
) -> None:
    output_dir = output_dir or _cfg.OUTPUT_DIR
    train_run(
        output_dir=output_dir,
        organ_presence_threshold=organ_presence_threshold,
        smoke_test=smoke_test,
        nrows=nrows,
        num_workers=num_workers,
        gpu_id=gpu_id,
        override_n_epochs=override_n_epochs,
    )

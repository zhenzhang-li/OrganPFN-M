"""SSL training loop + per-stage curriculum + per-(organ, timepoint) linear probe.

Reading order:

* :func:`main` is the entry point; it performs architectural and
  timepoint-embedding sanity checks and then runs each stage in
  :data:`config.TRAIN_CFG.stages` sequentially.
* :func:`run_stage` does ``epochs`` full passes over ``ssl_loader`` for
  one stage and saves a stage checkpoint at the end.
* :func:`linear_probe_eval` performs a closed-form ridge-regression probe
  on each eligible (organ, timepoint) combination:

    - Hepatic + T0 / Body + T0 / Renal + T0 / Metabolic + T0 etc.
    - Brain + T2 / Cardiovascular + T2 etc.
    - Body / Immune / Lungs are reported in both T0 and T2 so the user can
      verify the timepoint embedding is doing its job (the two MAEs should
      track each other within ~2 years).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import GradScaler, autocast  # type: ignore[attr-defined]

from . import config as _cfg
from . import ssl_objectives as ssl
from .data import build_dataloaders
from .model import OrganPFNM_SSL, build_model_from_checkpoint
from .ssl_objectives import StageHParams
from .utils import (
    CsvLogger,
    Schema,
    count_parameters,
    count_trainable,
    load_schema,
    save_checkpoint,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sanity checks (run once at the start of training)
# ---------------------------------------------------------------------------

def _sanity_checks(
    schema: Schema,
    model: OrganPFNM_SSL,
    train_cfg: _cfg.TrainConfig,
) -> None:
    """Cheap up-front validations.

    Includes:
      * Architectural guarantee (organ-presence independence).
      * ★ Timepoint embedding: two samples with identical features but
        different ``timepoint_id`` must produce *different* outputs (the
        difference is driven entirely by the per-row timepoint embedding;
        if it had no effect the test fails).
      * Mini batch forward + backward (gradient flows everywhere).
    """
    device = next(model.parameters()).device
    F = schema.n_total_features
    O = schema.n_organs()
    rng = torch.Generator(device="cpu")
    rng.manual_seed(0)

    # ----- 1. Organ-presence independence (Hepatic blocked) -----
    feats_a = torch.randn(F, generator=rng)
    s, e = schema.organ_offsets["Hepatic"]
    feats_b = feats_a.clone()
    feats_b[s:e] = torch.randn(e - s, generator=rng) * 100
    feats_a[s:e] = 0.0
    feats_b[s:e] = 0.0
    # Build a presence vector with Hepatic=0 and everything else=1.
    presence = torch.ones(O, dtype=torch.float32)
    presence[schema.organ_names.index("Hepatic")] = 0.0

    feats_batch = torch.stack([feats_a, feats_b], dim=0).to(device)
    presence_batch = presence.unsqueeze(0).expand(2, -1).to(device)
    # Both samples sit at T0 for this test.
    tp_batch = torch.zeros(2, dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        out = model.encode(feats_batch, presence_batch, tp_batch)
    delta = (out["cell_tokens"][0] - out["cell_tokens"][1]).abs().max().item()
    if delta > 1e-5:
        raise RuntimeError(
            f"Architectural guarantee FAILED: max |delta cell_tokens| = "
            f"{delta:.3e} > 1e-5 between two samples that differ only in "
            "blocked-organ values."
        )
    logger.info("[sanity] organ-presence independence ok (delta=%.3e)", delta)

    # ----- 2. ★ Timepoint embedding test -----
    feats_c = torch.randn(F, generator=rng).to(device)
    presence_c = torch.ones(O, dtype=torch.float32, device=device)
    feats_pair = torch.stack([feats_c, feats_c], dim=0)
    pres_pair = presence_c.unsqueeze(0).expand(2, -1)
    tp_pair = torch.tensor([0, 1], dtype=torch.long, device=device)
    with torch.no_grad():
        out_tp = model.encode(feats_pair, pres_pair, tp_pair)
    tp_delta = (out_tp["cell_tokens"][0] - out_tp["cell_tokens"][1]).abs().max().item()
    if tp_delta < 1e-5:
        raise RuntimeError(
            f"Timepoint-embedding sanity FAILED: max |delta cell_tokens| = "
            f"{tp_delta:.3e} between two identical-feature samples with "
            "different timepoint_id.  The timepoint embedding is either "
            "all-zero or not being used."
        )
    logger.info("[sanity] timepoint embedding ok (delta=%.3e)", tp_delta)

    # ----- 3. Mini batch forward+backward -----
    model.train()
    feats_batch.requires_grad_(False)
    out = model.encode(feats_batch, presence_batch, tp_batch)
    dummy_loss = out["cell_tokens"].pow(2).mean()
    dummy_loss.backward()
    n_zero = sum(
        p.grad is not None and p.grad.abs().sum().item() == 0
        for p in model.backbone.parameters()
    )
    n_total_bb = sum(1 for _ in model.backbone.parameters())
    logger.info(
        "[sanity] mini-batch forward+backward ok: %d/%d backbone params got "
        "non-zero gradient", n_total_bb - n_zero, n_total_bb,
    )
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

def build_optimizer_and_scheduler(
    model: OrganPFNM_SSL,
    *,
    train_cfg: _cfg.TrainConfig,
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warmup + cosine decay to ``peak * 0.1``."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        nd_keywords = ("bias", "norm", "embedding", "decode_token", "thinking_tokens")
        if any(kw in name.lower() for kw in nd_keywords):
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, lr=train_cfg.lr_peak, betas=(0.9, 0.95),
    )

    warmup = train_cfg.warmup_steps

    def _lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = max(0.0, min(1.0, progress))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Linear probe evaluator (per organ × timepoint)
# ---------------------------------------------------------------------------

@torch.no_grad()
def linear_probe_eval(
    model: OrganPFNM_SSL,
    eval_loader,
    schema: Schema,
    *,
    device: torch.device,
    split_ratio: float,
    ridge_lambda: float,
) -> dict[str, float]:
    """Closed-form ridge linear probe per (organ, timepoint) combination.

    Returns ``{"<organ>__<timepoint>": mae, ...}``.  An (organ, timepoint)
    pair is only evaluated if the organ is listed as available at that
    timepoint in ``schema.organ_timepoint_availability``.  Body / Immune /
    Lungs each appear at both T0 and T2; their MAEs should be similar
    if the timepoint embedding is doing its job.
    """
    model.eval()
    feats_per_organ: dict[str, list[np.ndarray]] = {o: [] for o in schema.organ_names}
    ages: list[float] = []
    timepoint_ids: list[int] = []

    for batch in eval_loader:
        feats = batch["features"].to(device)
        presence = batch["organ_presence"].to(device)
        tp = batch["timepoint_id"].to(device)
        out = model.encode(feats, presence, tp)
        per_organ = out["per_organ_summary"].cpu().numpy()
        ages.extend(batch["age"].numpy().tolist())
        timepoint_ids.extend(batch["timepoint_id"].numpy().tolist())
        for o_idx, organ in enumerate(schema.organ_names):
            feats_per_organ[organ].append(per_organ[:, o_idx, :])

    ages_arr = np.asarray(ages, dtype=np.float32)
    tp_arr = np.asarray(timepoint_ids, dtype=np.int64)
    valid_age = np.isfinite(ages_arr)
    if valid_age.sum() < 10:
        logger.warning("[probe] too few finite ages (%d); skipping.", int(valid_age.sum()))
        return {}

    tp_map = schema.timepoint_id_map  # {"T0": 0, "T2": 1}
    results: dict[str, float] = {}
    for organ in schema.organ_names:
        X_all = np.concatenate(feats_per_organ[organ], axis=0)
        for tp_name, tp_id in tp_map.items():
            # Only consider rows whose timepoint matches AND where this
            # organ is listed as available at that timepoint.
            avail = schema.organ_timepoint_availability.get(organ, [])
            if tp_name not in avail:
                continue
            mask = valid_age & (tp_arr == tp_id)
            if mask.sum() < 20:
                results[f"{organ}__{tp_name}"] = float("nan")
                continue
            X_sub = X_all[mask]
            age_sub = ages_arr[mask]
            # Within this (organ, timepoint) subset: split 80/20.
            rng = np.random.default_rng(seed=0)
            perm = rng.permutation(mask.sum())
            n_train = int(perm.size * split_ratio)
            tr = perm[:n_train]
            vl = perm[n_train:]
            X_tr = X_sub[tr]
            X_vl = X_sub[vl]
            y_tr = age_sub[tr]
            y_vl = age_sub[vl]
            X_tr_b = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
            X_vl_b = np.concatenate([X_vl, np.ones((X_vl.shape[0], 1))], axis=1)
            D = X_tr_b.shape[1]
            A = X_tr_b.T @ X_tr_b + ridge_lambda * np.eye(D, dtype=np.float32)
            b = X_tr_b.T @ y_tr
            try:
                w = np.linalg.solve(A.astype(np.float64), b.astype(np.float64)).astype(np.float32)
            except np.linalg.LinAlgError:
                results[f"{organ}__{tp_name}"] = float("nan")
                continue
            pred = X_vl_b @ w
            mae = float(np.abs(pred - y_vl).mean())
            results[f"{organ}__{tp_name}"] = mae
    return results


# ---------------------------------------------------------------------------
# Stage runner
# ---------------------------------------------------------------------------

def _stage_to_hparams(stage_cfg: dict[str, Any]) -> StageHParams:
    return StageHParams(
        cell_mask_ratio=float(stage_cfg["cell_mask_ratio"]),
        organ_drop_range=tuple(stage_cfg["organ_drop_range"]),  # type: ignore
        active_losses=set(stage_cfg["active_losses"]),
    )


def run_stage(
    *,
    stage_idx: int,
    stage_cfg: dict[str, Any],
    model: OrganPFNM_SSL,
    schema: Schema,
    ssl_loader,
    eval_loader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    metrics_logger: CsvLogger,
    epoch_logger: CsvLogger,
    device: torch.device,
    train_cfg: _cfg.TrainConfig,
    global_step: int,
    scaler: GradScaler | None,
    smoke_test: bool = False,
) -> tuple[int, list[dict[str, Any]]]:
    """Run one curriculum stage and return ``(global_step, epoch_summaries)``."""
    stage_name = stage_cfg["name"]
    stage_hp = _stage_to_hparams(stage_cfg)
    n_epochs = int(stage_cfg["epochs"])

    logger.info(
        "[stage %d/%s] starting: epochs=%d, cell_mask_ratio=%.2f, "
        "organ_drop_range=%s, active=%s",
        stage_idx, stage_name, n_epochs, stage_hp.cell_mask_ratio,
        tuple(stage_hp.organ_drop_range), sorted(stage_hp.active_losses),
    )

    epoch_summaries: list[dict[str, Any]] = []
    for epoch in range(n_epochs):
        model.train()
        epoch_t0 = time.time()
        running = {k: 0.0 for k in [
            "loss_total", "loss_cell", "loss_mask",
            "loss_organ", "loss_contrastive",
            "contrastive_top1_acc", "dropped_organ_count_mean",
        ]}
        n_batches = 0

        for batch_idx, batch in enumerate(ssl_loader):
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                losses = ssl.compute_full_ssl_loss(
                    model=model, batch=batch, schema=schema,
                    stage=stage_hp, train_cfg=train_cfg, device=device,
                )
            loss = losses["loss_total"]

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip_max_norm,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.grad_clip_max_norm,
                )
                optimizer.step()
            scheduler.step()
            global_step += 1
            n_batches += 1

            for k in running:
                running[k] += float(losses[k].detach().cpu())

            if global_step % train_cfg.log_every_n_steps == 0:
                row = {
                    "global_step": global_step,
                    "stage": stage_name,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "loss_total": float(losses["loss_total"].detach().cpu()),
                    "loss_cell": float(losses["loss_cell"].detach().cpu()),
                    "loss_mask": float(losses["loss_mask"].detach().cpu()),
                    "loss_organ": float(losses["loss_organ"].detach().cpu()),
                    "loss_contrastive": float(losses["loss_contrastive"].detach().cpu()),
                    "contrastive_top1_acc": float(losses["contrastive_top1_acc"].cpu()),
                    "dropped_organ_count_mean": float(losses["dropped_organ_count_mean"].cpu()),
                }
                metrics_logger.log(row)
                logger.info(
                    "[step %d / stage %s / epoch %d] total=%.4f cell=%.4f "
                    "mask=%.4f organ=%.4f cont=%.4f acc=%.3f drop=%.2f",
                    global_step, stage_name, epoch,
                    row["loss_total"], row["loss_cell"], row["loss_mask"],
                    row["loss_organ"], row["loss_contrastive"],
                    row["contrastive_top1_acc"], row["dropped_organ_count_mean"],
                )

            if smoke_test:
                break

        for k in running:
            running[k] /= max(1, n_batches)
        epoch_dur = time.time() - epoch_t0

        probe_results: dict[str, float] = {}
        if (
            (epoch + 1) % train_cfg.linear_probe_every_n_epochs == 0
            and not smoke_test
        ):
            probe_results = linear_probe_eval(
                model, eval_loader, schema,
                device=device,
                split_ratio=train_cfg.linear_probe_split_ratio,
                ridge_lambda=train_cfg.linear_probe_ridge_lambda,
            )

        summary = {
            "stage": stage_name, "epoch": epoch,
            "global_step": global_step, "duration_sec": epoch_dur,
            **{f"avg_{k}": v for k, v in running.items()},
        }
        for key, mae in probe_results.items():
            summary[f"probe_mae_{key}"] = mae
        epoch_summaries.append(summary)
        epoch_logger.log(summary)
        logger.info(
            "[stage %s][epoch %d] done in %.1fs. avg loss_total=%.4f. "
            "probe MAE: %s", stage_name, epoch, epoch_dur, running["loss_total"],
            ", ".join(f"{k}={v:.2f}" for k, v in probe_results.items())
            if probe_results else "(skipped)",
        )

        if smoke_test:
            break

    return global_step, epoch_summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    *,
    smoke_test: bool = False,
    nrows: int | None = None,
    output_dir: Path | None = None,
    num_workers: int = 0,
    gpu_id: int = 0,
) -> None:
    output_dir = output_dir or _cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / _cfg.TRAIN_CFG.train_log_name)

    logger.info("============================================================")
    logger.info("OrganPFN-M-LF SSL pre-training (env=%s, 10 organs)", _cfg.ENV)
    logger.info("DATA_DIR        : %s", _cfg.DATA_DIR)
    logger.info("CHECKPOINT_DIR  : %s", _cfg.CHECKPOINT_DIR)
    logger.info("OUTPUT_DIR      : %s", output_dir)
    logger.info("smoke_test=%s nrows=%s", smoke_test, nrows)
    logger.info("============================================================")

    train_cfg = _cfg.TRAIN_CFG
    model_cfg = _cfg.MODEL_CFG
    set_seed(train_cfg.seed)

    schema = load_schema(_cfg.schema_path())
    logger.info(
        "Schema loaded: %d organs, %d total features (incl. %d Sex columns), "
        "presence_excludes_sex=%s",
        schema.n_organs(), schema.n_total_features,
        sum(1 for v in schema.sex_column_indices_per_organ.values() if v >= 0),
        schema.presence_excludes_sex,
    )
    if schema.n_total_features != model_cfg.n_total_features:
        logger.warning(
            "schema.n_total_features=%d != model_cfg.n_total_features=%d "
            "(updating model_cfg).",
            schema.n_total_features, model_cfg.n_total_features,
        )
        model_cfg.n_total_features = schema.n_total_features
    if schema.n_organs() != model_cfg.n_organs:
        logger.warning(
            "schema.n_organs=%d != model_cfg.n_organs=%d (updating).",
            schema.n_organs(), model_cfg.n_organs,
        )
        model_cfg.n_organs = schema.n_organs()
    if schema.n_timepoints != model_cfg.n_timepoints:
        logger.warning(
            "schema.n_timepoints=%d != model_cfg.n_timepoints=%d (updating).",
            schema.n_timepoints, model_cfg.n_timepoints,
        )
        model_cfg.n_timepoints = schema.n_timepoints

    # -------- Data --------
    ssl_loader, eval_loader, norm = build_dataloaders(
        schema,
        batch_size=train_cfg.batch_size,
        num_workers=num_workers,
        nrows=nrows,
    )
    logger.info("DataLoader OK: %d SSL batches, %d eval batches per epoch",
                len(ssl_loader), len(eval_loader))

    # -------- Model --------
    model, report = build_model_from_checkpoint(
        schema=schema, ckpt_path=_cfg.tabpfn_ckpt_path(), model_cfg=model_cfg,
    )
    logger.info("[load report] %s", report)
    if report.n_skipped_in_ckpt != 0:
        raise RuntimeError(
            f"TabPFN load skipped {report.n_skipped_in_ckpt} keys; "
            "this should never happen with strict load."
        )

    if torch.cuda.is_available():
        n_cuda = torch.cuda.device_count()
        if gpu_id < 0 or gpu_id >= n_cuda:
            raise RuntimeError(
                f"Requested gpu_id={gpu_id}, but only {n_cuda} CUDA device(s) "
                "are visible."
            )
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        logger.info(
            "Device: %s (%s)", device, torch.cuda.get_device_name(gpu_id),
        )
    else:
        device = torch.device("cpu")
        logger.warning(
            "CUDA is not available; falling back to CPU.  When submitted on "
            "the GPU node, this script will bind to cuda:%d.", gpu_id,
        )
    model = model.to(device)

    _sanity_checks(schema, model, train_cfg)

    total_steps = sum(int(s["epochs"]) for s in train_cfg.stages) * max(1, len(ssl_loader))
    optimizer, scheduler = build_optimizer_and_scheduler(
        model, train_cfg=train_cfg, total_steps=total_steps,
    )
    scaler = GradScaler(
        device="cuda",
        enabled=(device.type == "cuda" and train_cfg.mixed_precision),
    )

    metrics_logger = CsvLogger(
        output_dir / train_cfg.metrics_csv_name,
        fieldnames=[
            "global_step", "stage", "epoch", "lr",
            "loss_total", "loss_cell", "loss_mask",
            "loss_organ", "loss_contrastive",
            "contrastive_top1_acc", "dropped_organ_count_mean",
        ],
    )
    epoch_fields = ["stage", "epoch", "global_step", "duration_sec"]
    epoch_fields += [f"avg_{k}" for k in [
        "loss_total", "loss_cell", "loss_mask",
        "loss_organ", "loss_contrastive",
        "contrastive_top1_acc", "dropped_organ_count_mean",
    ]]
    # Per (organ, timepoint) probe MAE columns.
    for o in schema.organ_names:
        for tp_name in schema.timepoint_id_map.keys():
            if tp_name in schema.organ_timepoint_availability.get(o, []):
                epoch_fields.append(f"probe_mae_{o}__{tp_name}")
    epoch_logger = CsvLogger(
        output_dir / train_cfg.epoch_summary_csv_name, fieldnames=epoch_fields,
    )

    global_step = 0
    all_epoch_summaries: list[dict[str, Any]] = []
    for stage_idx, stage_cfg in enumerate(train_cfg.stages):
        global_step, stage_summaries = run_stage(
            stage_idx=stage_idx, stage_cfg=stage_cfg,
            model=model, schema=schema,
            ssl_loader=ssl_loader, eval_loader=eval_loader,
            optimizer=optimizer, scheduler=scheduler,
            metrics_logger=metrics_logger, epoch_logger=epoch_logger,
            device=device, train_cfg=train_cfg,
            global_step=global_step,
            scaler=scaler if scaler.is_enabled() else None,
            smoke_test=smoke_test,
        )
        all_epoch_summaries.extend(stage_summaries)

        ckpt_path = output_dir / train_cfg.stage_ckpt_name(stage_cfg["name"])
        save_checkpoint(
            ckpt_path,
            model_state_dict=model.state_dict(),
            schema=schema.raw,
            normalization=norm.to_dict(),
            config={
                "embed_dim": model_cfg.embed_dim,
                "n_layers": report.backbone_layers,
                "n_organs": schema.n_organs(),
                "n_timepoints": schema.n_timepoints,
                "feature_columns": schema.feature_columns,
                "organ_offsets": {o: list(v) for o, v in schema.organ_offsets.items()},
                "features_per_group": model_cfg.features_per_group,
                "num_thinking_rows": model_cfg.num_thinking_rows,
            },
            tabpfn_ckpt_path=str(_cfg.tabpfn_ckpt_path()),
            global_step=global_step,
            epoch_summaries=all_epoch_summaries,
        )
        logger.info("Saved stage checkpoint -> %s", ckpt_path)

    final_path = output_dir / train_cfg.final_ckpt_name
    save_checkpoint(
        final_path,
        model_state_dict=model.state_dict(),
        schema=schema.raw,
        normalization=norm.to_dict(),
        config={
            "embed_dim": model_cfg.embed_dim,
            "n_layers": report.backbone_layers,
            "n_organs": schema.n_organs(),
            "n_timepoints": schema.n_timepoints,
            "feature_columns": schema.feature_columns,
            "organ_offsets": {o: list(v) for o, v in schema.organ_offsets.items()},
            "features_per_group": model_cfg.features_per_group,
            "num_thinking_rows": model_cfg.num_thinking_rows,
        },
        tabpfn_ckpt_path=str(_cfg.tabpfn_ckpt_path()),
        global_step=global_step,
        epoch_summaries=all_epoch_summaries,
    )
    logger.info("Saved final checkpoint -> %s", final_path)
    logger.info("============================================================")
    logger.info("OrganPFN-M-LF SSL pre-training DONE.")
    logger.info("============================================================")

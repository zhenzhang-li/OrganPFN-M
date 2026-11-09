"""SSL training objectives (long-format, 10-organ).

The four objectives used in the study are:

    Task 1  cell-level reconstruction      (Huber)
    Task 2  mask-position prediction       (BCE-with-logits)
    Task 3  organ-level reconstruction     (Huber, only on present organs)
    Task 4  SCARF contrastive               (InfoNCE on two corrupted views)

The loss wrapper reads the per-row
``timepoint_id`` from the batch and forwards it through ``model.encode``;
task-level functions don't need to know about timepoints because the
backbone handles the embedding upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import config as _cfg
from .model import OrganPFNM_SSL
from .utils import Schema


# ---------------------------------------------------------------------------
# Cell-level masking (shared by Task 1 and Task 2)
# ---------------------------------------------------------------------------

@dataclass
class CellMaskedView:
    masked_features: torch.Tensor
    artificial_mask: torch.Tensor
    valid_for_loss: torch.Tensor


def make_cell_mask(
    features_BF: torch.Tensor,
    missing_mask_BF: torch.Tensor,
    *,
    cell_mask_ratio: float,
    organ_drop_block_BF: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> CellMaskedView:
    """Mask only positions that are non-NaN and not inside an organ-drop block."""
    eligible = ~missing_mask_BF
    if organ_drop_block_BF is not None:
        eligible = eligible & (~organ_drop_block_BF)

    rand = torch.empty_like(features_BF).uniform_(0, 1, generator=generator)
    artificial_mask = (rand < cell_mask_ratio) & eligible

    masked = torch.where(artificial_mask, torch.zeros_like(features_BF), features_BF)
    return CellMaskedView(
        masked_features=masked,
        artificial_mask=artificial_mask,
        valid_for_loss=eligible,
    )


# ---------------------------------------------------------------------------
# Organ-level dropping (Task 3)
# ---------------------------------------------------------------------------

@dataclass
class OrganDropView:
    effective_presence: torch.Tensor
    drop_mask_BO: torch.Tensor
    drop_block_BF: torch.Tensor


def make_organ_drop(
    organ_presence_BO: torch.Tensor,
    schema: Schema,
    *,
    drop_range: tuple[int, int],
    generator: torch.Generator | None = None,
) -> OrganDropView:
    """Randomly drop a uniform-in-range number of present organs per sample."""
    B, O = organ_presence_BO.shape
    device = organ_presence_BO.device

    drop_mask = torch.zeros((B, O), dtype=torch.bool, device=device)
    if drop_range == (0, 0):
        return OrganDropView(
            effective_presence=organ_presence_BO.clone(),
            drop_mask_BO=drop_mask,
            drop_block_BF=torch.zeros(
                (B, schema.n_total_features), dtype=torch.bool, device=device,
            ),
        )

    lo, hi = drop_range
    if hi < lo:
        raise ValueError(f"organ_drop_range high < low: {drop_range}")

    for b in range(B):
        present_idx = torch.nonzero(organ_presence_BO[b] > 0).flatten()
        n_present = present_idx.numel()
        if n_present <= 1:
            continue
        max_drop = min(hi, n_present - 1)
        min_drop = min(lo, max_drop)
        if max_drop < min_drop:
            continue
        if generator is None:
            n_drop = torch.randint(min_drop, max_drop + 1, (1,), device=device).item()
        else:
            n_drop = torch.randint(
                min_drop, max_drop + 1, (1,), device=device, generator=generator,
            ).item()
        if n_drop <= 0:
            continue
        if generator is None:
            perm = torch.randperm(n_present, device=device)
        else:
            perm = torch.randperm(n_present, device=device, generator=generator)
        chosen = present_idx[perm[:n_drop]]
        drop_mask[b, chosen] = True

    effective = organ_presence_BO * (~drop_mask).float()

    drop_block_BF = torch.zeros(
        (B, schema.n_total_features), dtype=torch.bool, device=device,
    )
    for o_idx, organ in enumerate(schema.organ_names):
        s, e = schema.organ_offsets[organ]
        col_drop = drop_mask[:, o_idx].view(B, 1).expand(-1, e - s)
        drop_block_BF[:, s:e] = drop_block_BF[:, s:e] | col_drop

    return OrganDropView(
        effective_presence=effective,
        drop_mask_BO=drop_mask,
        drop_block_BF=drop_block_BF,
    )


# ---------------------------------------------------------------------------
# Task 1: Cell-level reconstruction
# ---------------------------------------------------------------------------

def cell_reconstruction_loss(
    cell_tokens_BGE: torch.Tensor,
    target_features_BF: torch.Tensor,
    artificial_mask_BF: torch.Tensor,
    *,
    head: nn.Module,
    features_per_group: int,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    B, G, E = cell_tokens_BGE.shape
    F_real = target_features_BF.shape[1]
    F_padded = G * features_per_group

    pred_BGfpg = head(cell_tokens_BGE)
    pred_BFpad = pred_BGfpg.reshape(B, F_padded)
    pred_BF = pred_BFpad[:, :F_real]

    if artificial_mask_BF.sum() == 0:
        return pred_BF.sum() * 0.0

    pred_sel = pred_BF[artificial_mask_BF]
    target_sel = target_features_BF[artificial_mask_BF]
    return F.huber_loss(pred_sel, target_sel, delta=huber_delta, reduction="mean")


# ---------------------------------------------------------------------------
# Task 2: Mask-position prediction
# ---------------------------------------------------------------------------

def mask_prediction_loss(
    cell_tokens_BGE: torch.Tensor,
    artificial_mask_BF: torch.Tensor,
    valid_for_loss_BF: torch.Tensor,
    *,
    head: nn.Module,
    features_per_group: int,
) -> torch.Tensor:
    B, G, E = cell_tokens_BGE.shape
    F_real = artificial_mask_BF.shape[1]
    F_padded = G * features_per_group

    logit_BGfpg = head(cell_tokens_BGE)
    logit_BFpad = logit_BGfpg.reshape(B, F_padded)
    logit_BF = logit_BFpad[:, :F_real]

    if valid_for_loss_BF.sum() == 0:
        return logit_BF.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        logit_BF[valid_for_loss_BF],
        artificial_mask_BF[valid_for_loss_BF].float(),
        reduction="mean",
    )


# ---------------------------------------------------------------------------
# Task 3: Organ-level reconstruction
# ---------------------------------------------------------------------------

def organ_reconstruction_loss(
    per_organ_summary_BOE: torch.Tensor,
    drop_mask_BO: torch.Tensor,
    target_features_BF: torch.Tensor,
    missing_mask_BF: torch.Tensor,
    *,
    organ_decoders: nn.ModuleDict,
    schema: Schema,
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if drop_mask_BO.sum() == 0:
        return per_organ_summary_BOE.sum() * 0.0

    B, O, E = per_organ_summary_BOE.shape

    losses: list[torch.Tensor] = []
    for o_idx, organ in enumerate(schema.organ_names):
        sample_idx = torch.nonzero(drop_mask_BO[:, o_idx]).flatten()
        if sample_idx.numel() == 0:
            continue
        s, e = schema.organ_offsets[organ]
        decoder = organ_decoders[organ]
        ctx_sum = per_organ_summary_BOE[sample_idx]
        keep = torch.ones(O, dtype=torch.bool, device=ctx_sum.device)
        keep[o_idx] = False
        ctx_sum_masked = ctx_sum[:, keep, :]
        ctx_BD = ctx_sum_masked.mean(dim=1)
        pred = decoder(ctx_BD)
        target = target_features_BF[sample_idx, s:e]
        valid = ~missing_mask_BF[sample_idx, s:e]
        if valid.sum() == 0:
            continue
        loss_o = F.huber_loss(
            pred[valid], target[valid], delta=huber_delta, reduction="mean",
        )
        losses.append(loss_o)

    if not losses:
        return per_organ_summary_BOE.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Task 4: SCARF contrastive
# ---------------------------------------------------------------------------

def scarf_corrupt(
    features_BF: torch.Tensor,
    missing_mask_BF: torch.Tensor,
    *,
    corrupt_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    B, F = features_BF.shape
    eligible = ~missing_mask_BF
    rand = torch.empty_like(features_BF).uniform_(0, 1, generator=generator)
    corrupt = (rand < corrupt_ratio) & eligible

    if generator is None:
        perm_BF = torch.argsort(torch.rand_like(features_BF), dim=0)
    else:
        rand_perm = torch.empty_like(features_BF)
        rand_perm.uniform_(0, 1, generator=generator)
        perm_BF = torch.argsort(rand_perm, dim=0)
    marginal_BF = torch.gather(features_BF, dim=0, index=perm_BF)
    return torch.where(corrupt, marginal_BF, features_BF)


def contrastive_loss(
    z1_BD: torch.Tensor, z2_BD: torch.Tensor, *, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z1 = F.normalize(z1_BD, dim=-1)
    z2 = F.normalize(z2_BD, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    loss = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean()
    return loss, acc


# ---------------------------------------------------------------------------
# Full SSL loss for a single batch
# ---------------------------------------------------------------------------

@dataclass
class StageHParams:
    cell_mask_ratio: float
    organ_drop_range: tuple[int, int]
    active_losses: set[str]


def compute_full_ssl_loss(
    *,
    model: OrganPFNM_SSL,
    batch: dict[str, torch.Tensor],
    schema: Schema,
    stage: StageHParams,
    train_cfg: _cfg.TrainConfig,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    feats = batch["features"].to(device)
    missing = batch["missing_mask"].to(device)
    presence = batch["organ_presence"].to(device)
    timepoint_id = batch["timepoint_id"].to(device)

    B, F_in = feats.shape
    O = presence.shape[1]

    organ_view = make_organ_drop(
        presence, schema, drop_range=stage.organ_drop_range, generator=generator,
    )
    cell_view = make_cell_mask(
        feats, missing,
        cell_mask_ratio=stage.cell_mask_ratio,
        organ_drop_block_BF=organ_view.drop_block_BF,
        generator=generator,
    )

    masked_feats = cell_view.masked_features.clone()
    masked_feats = torch.where(
        organ_view.drop_block_BF, torch.zeros_like(masked_feats), masked_feats,
    )

    out_main = model.encode(masked_feats, organ_view.effective_presence, timepoint_id)
    cell_tokens = out_main["cell_tokens"]
    per_organ = out_main["per_organ_summary"]
    global_summary = out_main["global_summary"]

    losses: dict[str, torch.Tensor] = {}

    if "cell" in stage.active_losses:
        losses["loss_cell"] = cell_reconstruction_loss(
            cell_tokens, feats, cell_view.artificial_mask,
            head=model.cell_recon_head,
            features_per_group=model.model_cfg.features_per_group,
        )
    else:
        losses["loss_cell"] = feats.sum() * 0.0

    if "mask" in stage.active_losses:
        losses["loss_mask"] = mask_prediction_loss(
            cell_tokens, cell_view.artificial_mask, cell_view.valid_for_loss,
            head=model.mask_pred_head,
            features_per_group=model.model_cfg.features_per_group,
        )
    else:
        losses["loss_mask"] = feats.sum() * 0.0

    if "organ" in stage.active_losses:
        losses["loss_organ"] = organ_reconstruction_loss(
            per_organ, organ_view.drop_mask_BO, feats, missing,
            organ_decoders=model.organ_decoders, schema=schema,
        )
    else:
        losses["loss_organ"] = feats.sum() * 0.0

    if "contrastive" in stage.active_losses:
        v1 = scarf_corrupt(
            feats, missing,
            corrupt_ratio=train_cfg.scarf_corrupt_ratio, generator=generator,
        )
        v2 = scarf_corrupt(
            feats, missing,
            corrupt_ratio=train_cfg.scarf_corrupt_ratio, generator=generator,
        )
        out_v1 = model.encode(v1, presence, timepoint_id)
        out_v2 = model.encode(v2, presence, timepoint_id)
        z1 = model.contrastive_head(out_v1["global_summary"])
        z2 = model.contrastive_head(out_v2["global_summary"])
        loss_c, acc_c = contrastive_loss(
            z1, z2, temperature=model.model_cfg.contrastive_temperature,
        )
        losses["loss_contrastive"] = loss_c
        losses["contrastive_top1_acc"] = acc_c.detach()
    else:
        losses["loss_contrastive"] = feats.sum() * 0.0
        losses["contrastive_top1_acc"] = torch.zeros((), device=device)

    losses["loss_total"] = (
        train_cfg.lambda_cell        * losses["loss_cell"]
        + train_cfg.lambda_mask      * losses["loss_mask"]
        + train_cfg.lambda_organ     * losses["loss_organ"]
        + train_cfg.lambda_contrastive * losses["loss_contrastive"]
    )

    losses["dropped_organ_count_mean"] = (
        organ_view.drop_mask_BO.sum(dim=1).float().mean().detach()
    )
    return losses

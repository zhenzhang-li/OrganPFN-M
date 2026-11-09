"""OrganPFN-M-LF SSL model (long-format, 10-organ).

Modules:
* :class:`OrganPFNMBackbone` -- wraps the TabPFN-2.5 PerFeatureTransformer
  and adds an ``organ_id_embedding`` + ★ ``timepoint_embedding``.  The
  latent zero-gate enforces organ-presence independence.
* :class:`OrganPFNM_SSL` -- backbone + 4 SSL heads.
* :func:`build_model_from_checkpoint` -- TabPFN-2.5 loader + sanity checks.

Long-format design:
* per-row timepoint embedding added BEFORE the first transformer block;
* ``encode`` takes an explicit ``timepoint_id_B`` (B,) long tensor.

Organ-presence independence proof:
Two mechanisms together guarantee that absent-organ raw values cannot
influence present-organ outputs (delta < 1e-5):

  1. Input-side zero-fill in the dataloader.
  2. Latent zero-gate after every transformer block.

The training entry point performs an empirical check before optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einops
import numpy as np
import torch
from torch import nn

from . import config as _cfg
from .data import build_group_organ_membership
from .utils import Schema, count_parameters, count_trainable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class OrganPFNMBackbone(nn.Module):
    """Wraps the TabPFN-2.5 PerFeatureTransformer.

    Long-format additions:
      * ``timepoint_embedding``: (n_timepoints, E) added per-row before the
        first transformer block (broadcast over the G dimension).
      * ``forward`` accepts ``timepoint_id_BN`` (B, N) integer tensor.
    """

    def __init__(
        self,
        *,
        tabpfn_model: nn.Module,
        schema: Schema,
        embed_dim: int,
        features_per_group: int,
        num_thinking_rows: int,
        n_timepoints: int,
    ) -> None:
        super().__init__()
        self.tabpfn = tabpfn_model
        self.embed_dim = int(embed_dim)
        self.features_per_group = int(features_per_group)
        self.num_thinking_rows = int(num_thinking_rows)
        self.n_total_features = schema.n_total_features
        self.n_organs = schema.n_organs()
        self.n_timepoints = int(n_timepoints)

        majority, organ_in_group = build_group_organ_membership(
            schema, self.features_per_group,
        )
        self.G = int(majority.shape[0])
        self.register_buffer(
            "_group_majority_organ",
            torch.as_tensor(majority, dtype=torch.long),
        )
        self.register_buffer(
            "_organ_in_group",
            torch.as_tensor(organ_in_group, dtype=torch.bool),
        )

        # Organ-id embedding (per group).
        self.organ_id_embedding = nn.Embedding(self.n_organs, self.embed_dim)
        nn.init.normal_(
            self.organ_id_embedding.weight, mean=0.0,
            std=_cfg.MODEL_CFG.organ_id_emb_init_std,
        )

        # ★ Timepoint embedding (per row).  Stored on the backbone so FT can
        # re-load and continue using it without touching the SSL heads.
        self.timepoint_embedding = nn.Embedding(self.n_timepoints, self.embed_dim)
        nn.init.normal_(
            self.timepoint_embedding.weight, mean=0.0,
            std=_cfg.MODEL_CFG.timepoint_emb_init_std,
        )

    # ------------------------------------------------------------------
    # Latent zero-gate
    # ------------------------------------------------------------------
    def build_group_presence_mask(
        self, organ_presence_BNO: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N, O) -> (B, N, G), 1 if any of group g's organs is present."""
        og = self._organ_in_group.float()
        any_org = torch.einsum("bno,og->bng", organ_presence_BNO, og)
        return (any_org > 0).float()

    def _gate_blocked_tokens(
        self,
        state_BSCE: torch.Tensor,
        group_presence_BNG: torch.Tensor,
        single_eval_pos: int,
    ) -> torch.Tensor:
        B, S, C, E = state_BSCE.shape
        T = single_eval_pos - group_presence_BNG.shape[1]
        N = group_presence_BNG.shape[1]
        G = group_presence_BNG.shape[2]
        assert T == self.num_thinking_rows
        assert C == G + 1
        mask = state_BSCE.new_ones((B, S, C))
        mask[:, T:T + N, :G] = group_presence_BNG
        return state_BSCE * mask.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        features_BNF: torch.Tensor,
        organ_presence_BNO: torch.Tensor,
        timepoint_id_BN: torch.Tensor,
        *,
        return_full_state: bool = False,
    ) -> dict[str, torch.Tensor]:
        if features_BNF.dim() != 3:
            raise ValueError(
                f"features_BNF must be 3D (B, N, F), got shape {features_BNF.shape}"
            )
        B, N, F_in = features_BNF.shape
        if F_in != self.n_total_features:
            raise ValueError(
                f"Expected F={self.n_total_features} input features, got {F_in}"
            )
        if timepoint_id_BN.shape != (B, N):
            raise ValueError(
                f"timepoint_id_BN must be (B={B}, N={N}); "
                f"got {tuple(timepoint_id_BN.shape)}"
            )
        device = features_BNF.device
        dtype = features_BNF.dtype

        # 1. Reshape into TabPFN's expected layout: (N, B, F).
        x_NBF = features_BNF.permute(1, 0, 2).contiguous()

        # Pad to multiple of features_per_group.
        missing_to_next = (
            self.features_per_group - (F_in % self.features_per_group)
        ) % self.features_per_group
        if missing_to_next > 0:
            x_NBF = torch.cat(
                [
                    x_NBF,
                    torch.zeros(N, B, missing_to_next, device=device, dtype=dtype),
                ],
                dim=-1,
            )
        F_padded = x_NBF.shape[-1]
        assert F_padded == self.G * self.features_per_group

        # 2. Run TabPFN's official encoders.
        single_eval_pos = N
        y_NB = torch.full((N, B), float("nan"), device=device, dtype=dtype)

        x_dict = {"main": x_NBF}
        x_dict["main"] = einops.rearrange(
            x_dict["main"], "s b (f n) -> b s f n", n=self.features_per_group,
        )
        x_dict["main"] = einops.rearrange(x_dict["main"], "b s f n -> s (b f) n")
        y_dict = {"main": y_NB.unsqueeze(-1)}

        embedded_y_BS_E = self.tabpfn.y_encoder(
            y_dict, single_eval_pos=single_eval_pos,
            cache_trainset_representation=False,
        ).transpose(0, 1)

        embedded_x_BSGE = einops.rearrange(
            self.tabpfn.encoder(
                x_dict, single_eval_pos=single_eval_pos,
                cache_trainset_representation=False,
            ),
            "s (b f) e -> b s f e", b=B,
        )

        embedded_x_BSGE, embedded_y_BS_E = self.tabpfn.add_embeddings(
            embedded_x_BSGE, embedded_y_BS_E,
            data_dags=None, num_features=F_in,
            seq_len=N, cache_embeddings=False, use_cached_embeddings=False,
        )

        # 3. Inject organ-id embedding (per group).
        organ_emb_GE = self.organ_id_embedding(self._group_majority_organ)
        embedded_x_BSGE = embedded_x_BSGE + organ_emb_GE.view(
            1, 1, self.G, self.embed_dim,
        )

        # 3b. ★ Inject timepoint embedding (per row, broadcast over G).
        tp_emb_BNE = self.timepoint_embedding(timepoint_id_BN.long())   # (B, N, E)
        embedded_x_BSGE = embedded_x_BSGE + tp_emb_BNE.unsqueeze(2)
        # Also add to the y column so the row's timepoint identity is
        # consistent across the whole token sequence.
        embedded_y_BS_E = embedded_y_BS_E + tp_emb_BNE

        embedded_input = torch.cat(
            [embedded_x_BSGE, embedded_y_BS_E.unsqueeze(2)], dim=2,
        )

        if torch.isnan(embedded_input).any():
            raise RuntimeError(
                "NaNs detected in embedded input - this should never happen "
                "because we feed already-standardized features and rely on "
                "the NanHandlingEncoderStep for both x and y."
            )

        # 4. Add thinking rows.
        if self.tabpfn.add_thinking_tokens is None:
            raise RuntimeError(
                "Loaded TabPFN model has no thinking-token module; v2.5 "
                "default checkpoint is expected to have it."
            )
        embedded_input, single_eval_pos_after = self.tabpfn.add_thinking_tokens(
            embedded_input, single_eval_pos,
        )
        T = single_eval_pos_after - single_eval_pos
        assert T == self.num_thinking_rows

        # 5. Apply latent zero-gate before & after each transformer layer.
        group_presence = self.build_group_presence_mask(organ_presence_BNO)
        state = self._gate_blocked_tokens(
            embedded_input, group_presence, single_eval_pos_after,
        )
        for layer in self.tabpfn.transformer_encoder.layers:
            state = layer(
                state, single_eval_pos=single_eval_pos_after,
                cache_trainset_representation=False, save_peak_mem_factor=None,
            )
            state = self._gate_blocked_tokens(
                state, group_presence, single_eval_pos_after,
            )

        state = state[:, single_eval_pos_after - N:, :, :]
        cell_tokens = state[:, :, :self.G, :]

        # 6. Per-organ summary.
        org_grp_mask = self._organ_in_group.float()
        weights_BNOG = (
            organ_presence_BNO.unsqueeze(-1) * org_grp_mask.view(1, 1, *org_grp_mask.shape)
        )
        denom = weights_BNOG.sum(dim=-1, keepdim=True).clamp(min=1.0)
        per_organ_summary = (
            torch.einsum("bnog,bnge->bnoe", weights_BNOG, cell_tokens) / denom
        )

        # 7. Global summary.
        gp_b_n_g_1 = group_presence.unsqueeze(-1)
        global_denom = group_presence.sum(dim=-1, keepdim=True).clamp(min=1.0)
        global_summary = (cell_tokens * gp_b_n_g_1).sum(dim=2) / global_denom

        out = {
            "cell_tokens": cell_tokens,
            "per_organ_summary": per_organ_summary,
            "global_summary": global_summary,
            "group_presence": group_presence,
        }
        if return_full_state:
            out["full_state"] = state
        return out


# ---------------------------------------------------------------------------
# SSL heads
# ---------------------------------------------------------------------------

class CellReconstructionHead(nn.Module):
    def __init__(self, embed_dim: int, features_per_group: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, features_per_group),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MaskPredictionHead(nn.Module):
    def __init__(self, embed_dim: int, features_per_group: int, hidden: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, features_per_group),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class OrganLevelDecoder(nn.Module):
    def __init__(
        self, *,
        organ_n_features: int,
        backbone_emb_dim: int, decoder_emb_dim: int,
        n_layers: int, nhead: int, ffn_factor: int = 4,
    ) -> None:
        super().__init__()
        self.organ_n_features = int(organ_n_features)
        self.decoder_emb_dim = int(decoder_emb_dim)
        self.input_proj = nn.Linear(backbone_emb_dim, decoder_emb_dim)
        self.decode_token = nn.Parameter(torch.zeros(1, 1, decoder_emb_dim))
        nn.init.normal_(self.decode_token, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_emb_dim, nhead=nhead,
            dim_feedforward=decoder_emb_dim * ffn_factor,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # norm_first=True 与 nested-tensor 快路径不兼容，显式关闭以消除无害的
        # "enable_nested_tensor is True ..." UserWarning（不改变任何数值/结果）。
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.output_proj = nn.Linear(decoder_emb_dim, self.organ_n_features)

    def forward(self, organ_summary_BD: torch.Tensor) -> torch.Tensor:
        proj_B1D = self.input_proj(organ_summary_BD).unsqueeze(1)
        dec_B1D = self.decode_token.expand(proj_B1D.shape[0], 1, -1)
        seq_B2D = torch.cat([dec_B1D, proj_B1D], dim=1)
        out = self.transformer(seq_B2D)
        return self.output_proj(out[:, 0, :])


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, proj_dim: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# ---------------------------------------------------------------------------
# Top-level SSL model
# ---------------------------------------------------------------------------

class OrganPFNM_SSL(nn.Module):
    """Backbone + 4 SSL heads."""

    def __init__(
        self,
        *,
        backbone: OrganPFNMBackbone,
        schema: Schema,
        model_cfg: _cfg.ModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.schema = schema
        self.model_cfg = model_cfg

        E = model_cfg.embed_dim
        fpg = model_cfg.features_per_group

        self.cell_recon_head = CellReconstructionHead(
            embed_dim=E, features_per_group=fpg, hidden=model_cfg.cell_recon_hidden,
        )
        self.mask_pred_head = MaskPredictionHead(
            embed_dim=E, features_per_group=fpg, hidden=model_cfg.mask_pred_hidden,
        )

        self.organ_decoders = nn.ModuleDict()
        for organ in schema.organ_names:
            self.organ_decoders[organ] = OrganLevelDecoder(
                organ_n_features=schema.n_features_per_organ[organ],
                backbone_emb_dim=E,
                decoder_emb_dim=model_cfg.organ_decoder_emb_dim,
                n_layers=model_cfg.organ_decoder_n_layers,
                nhead=model_cfg.organ_decoder_nhead,
            )

        self.contrastive_head = ContrastiveProjectionHead(
            embed_dim=E, hidden=model_cfg.contrastive_proj_hidden,
            proj_dim=model_cfg.contrastive_proj_dim,
        )

    def encode(
        self,
        features_BF: torch.Tensor,
        organ_presence_BO: torch.Tensor,
        timepoint_id_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch through the backbone.

        The SSL backbone treats the whole batch as a single in-context
        learning task (batch=1, sequence=B), so we add a leading dim and
        then squeeze it out before returning to the SSL heads.
        """
        feats_1NF = features_BF.unsqueeze(0)
        pres_1NO = organ_presence_BO.unsqueeze(0)
        tp_1N = timepoint_id_B.unsqueeze(0)
        out = self.backbone(feats_1NF, pres_1NO, tp_1N)
        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class _LoadReport:
    total_params: int
    trainable_params: int
    backbone_params: int
    backbone_layers: int
    n_loaded: int
    n_skipped_in_ckpt: int
    n_extra_in_model: int
    embed_dim: int
    features_per_group: int


def _import_load_model():
    try:
        import tabpfn  # noqa: PLC0415
        from tabpfn.model_loading import load_model  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "Could not import tabpfn.model_loading.load_model.\n"
            "Install with: pip install tabpfn"
        ) from e

    version = tuple(int(part) for part in tabpfn.__version__.split(".")[:3])
    if version < (6, 3, 2):
        raise RuntimeError(
            "Installed tabpfn version "
            f"{tabpfn.__version__!r} is too old for the "
            f"{_cfg.TABPFN_CKPT_NAME!r} checkpoint.  Use tabpfn>=6.3.2."
        )
    return load_model


def build_model_from_checkpoint(
    schema: Schema,
    *,
    ckpt_path: Path,
    model_cfg: _cfg.ModelConfig,
) -> tuple[OrganPFNM_SSL, _LoadReport]:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"TabPFN-2.5 checkpoint not found at {ckpt_path}.\n"
            f"Make sure the file '{_cfg.TABPFN_CKPT_NAME}' is present in "
            f"the checkpoint directory."
        )

    load_model = _import_load_model()
    logger.info("Loading TabPFN-2.5 backbone from %s", ckpt_path)
    tabpfn_model, _criterion, arch_config, _inference_config = load_model(
        path=ckpt_path, cache_trainset_representation=False,
    )

    actual_emsize = int(arch_config.emsize)
    actual_fpg = int(arch_config.features_per_group)
    if actual_emsize != model_cfg.embed_dim:
        raise RuntimeError(
            f"Checkpoint emsize {actual_emsize} != expected {model_cfg.embed_dim}"
        )
    if actual_fpg != model_cfg.features_per_group:
        raise RuntimeError(
            f"Checkpoint features_per_group {actual_fpg} != expected "
            f"{model_cfg.features_per_group}."
        )

    backbone_layers = len(tabpfn_model.transformer_encoder.layers)
    if backbone_layers != model_cfg.n_layers_expected:
        logger.warning(
            "Checkpoint has %d transformer layers; expected %d (non-fatal).",
            backbone_layers, model_cfg.n_layers_expected,
        )

    for p in tabpfn_model.parameters():
        p.requires_grad_(True)

    backbone_param_count = count_parameters(tabpfn_model)
    n_loaded = len(tabpfn_model.state_dict())
    logger.info(
        "TabPFN-2.5 backbone loaded: %d params (%d state_dict keys), "
        "%d transformer layers, emsize=%d, fpg=%d, num_thinking_rows=%d",
        backbone_param_count, n_loaded, backbone_layers,
        actual_emsize, actual_fpg, int(arch_config.num_thinking_rows),
    )

    backbone = OrganPFNMBackbone(
        tabpfn_model=tabpfn_model, schema=schema,
        embed_dim=actual_emsize, features_per_group=actual_fpg,
        num_thinking_rows=int(arch_config.num_thinking_rows),
        n_timepoints=model_cfg.n_timepoints,
    )

    model = OrganPFNM_SSL(
        backbone=backbone, schema=schema, model_cfg=model_cfg,
    )

    report = _LoadReport(
        total_params=count_parameters(model),
        trainable_params=count_trainable(model),
        backbone_params=backbone_param_count,
        backbone_layers=backbone_layers,
        n_loaded=n_loaded,
        n_skipped_in_ckpt=0,
        n_extra_in_model=count_parameters(model) - backbone_param_count
                          - count_parameters(backbone.organ_id_embedding)
                          - count_parameters(backbone.timepoint_embedding),
        embed_dim=actual_emsize,
        features_per_group=actual_fpg,
    )
    return model, report

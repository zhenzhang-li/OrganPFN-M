"""OrganPFN-M-LF Stage 2 model (long-format, 10 organs).

Five pieces live here:

* :class:`BatchEnsembleLinear` -- TabM-style rank-1 BatchEnsemble linear
  layer with k=32 members (Kobyzev et al. 2024, "TabM").
* :class:`OrganRegressionHead` -- per-organ regression head built from
  three ``BatchEnsembleLinear`` blocks with GELU/Dropout.
* :class:`OrthogonalSummaryProjection` -- anti-degeneracy #5
  (per-organ projector + QR decomposition).
* :class:`OPFNMLF_FT` -- top-level model: SSL backbone + N per-organ
  ``OrganRegressionHead``s (N = ``schema.n_organs()``; here N = 10) +
  ``OrthogonalSummaryProjection`` + learnable ``log_sigma`` per organ.
* :func:`build_ft_model_from_ssl_checkpoint` -- builder that handles
  SSL ckpt loading + backbone re-instantiation + strict state-dict load.

Representation-separation mechanisms
------------------------------------
1. **Parameter isolation**: 10 fully independent ``OrganRegressionHead``s.
2. **Architectural attention masking**: the SSL backbone's latent
   zero-gate (see ``OrganPFNMBackbone._gate_blocked_tokens``) is reused
   verbatim - blocked organ tokens stay at zero through every layer.
3. **Modality dropout**: applied to the presence vector before forward
   (see ``data.apply_modality_dropout``).
4. **Cross-organ contrastive**: computed in ``losses.py`` from the
   per-organ summaries produced here.
5. **Orthogonal projection**: implemented as the
   ``OrthogonalSummaryProjection`` module below; its output feeds the
   regression heads, decoupling the heads' inputs at the representation
   level.

The orthogonal projection always *runs* but its effect is gated by the
``contrastive`` / ``divergence`` losses: even when those losses are zero,
the QR-style projection still mathematically guarantees inter-organ row
orthogonality at the projector's output, which prevents the heads from
trivially sharing a common input direction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import config as _cfg
from .utils import Schema, count_parameters, count_trainable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verify that the self-supervised package from this distribution is available.
# ---------------------------------------------------------------------------

def _ensure_ssl_pkg_importable() -> None:
    try:
        import organpfnm_ssl  # noqa: F401
        return
    except ImportError as exc:
        raise ImportError(
            "Cannot import organpfnm_ssl. Install this repository with "
            "`pip install -e .[training]`."
        ) from exc


# ---------------------------------------------------------------------------
# BatchEnsembleLinear
# ---------------------------------------------------------------------------

class BatchEnsembleLinear(nn.Module):
    r"""TabM-style rank-1 BatchEnsemble linear layer with ``k`` members.

    Each member maintains its own rank-1 multiplicative adapter on the
    input and output of a *shared* dense matrix ``W``:

    .. math::

        y_{i} = ((x \odot r_{i}) W^{T}) \odot s_{i} + b_{i}

    where ``r_i`` (in_dim,) is the per-member input scaling, ``s_i``
    (out_dim,) is the per-member output scaling, and ``b_i`` (out_dim,)
    is the per-member bias.

    Forward path:

        x: (B, k, in_dim)
        x_scaled    = x * r          # broadcast (k, in_dim)
        h           = x_scaled @ W.T  # (B, k, out_dim)
        out         = h * s + b
    """

    def __init__(
        self, in_dim: int, out_dim: int, *,
        k: int = 32, rank1_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.k = int(k)

        self.W = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.W)

        # r, s ~ N(1, std):  centered at 1 so the shared path dominates at
        # init and members start close to one another.
        self.r = nn.Parameter(1.0 + torch.randn(self.k, in_dim) * rank1_init_std)
        self.s = nn.Parameter(1.0 + torch.randn(self.k, out_dim) * rank1_init_std)
        self.b = nn.Parameter(torch.zeros(self.k, out_dim))

    def forward(self, x_BkI: torch.Tensor) -> torch.Tensor:
        if x_BkI.dim() != 3 or x_BkI.shape[1] != self.k or x_BkI.shape[2] != self.in_dim:
            raise ValueError(
                f"BatchEnsembleLinear expected (B, k={self.k}, in_dim={self.in_dim}); "
                f"got shape {tuple(x_BkI.shape)}"
            )
        x_scaled = x_BkI * self.r.unsqueeze(0)
        h = x_scaled @ self.W.T
        return h * self.s.unsqueeze(0) + self.b.unsqueeze(0)


# ---------------------------------------------------------------------------
# Per-organ regression head (anti-degeneracy #1)
# ---------------------------------------------------------------------------

class OrganRegressionHead(nn.Module):
    """Per-organ regression head built from BatchEnsembleLinear blocks.

    Architecture::

        LayerNorm
          -> BatchEnsembleLinear(in_dim -> hidden_dim)  + GELU + Dropout
          -> BatchEnsembleLinear(hidden_dim -> hidden_dim) + GELU + Dropout
          -> BatchEnsembleLinear(hidden_dim -> 1)

    Input shape  : (B, in_dim)  -- usually orthogonally projected summary
    Output shape : (B, k)       -- k predicted ages
    """

    def __init__(
        self, *,
        in_dim: int, hidden_dim: int, k: int,
        dropout: float, rank1_init_std: float,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.k = int(k)

        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = BatchEnsembleLinear(
            in_dim, hidden_dim, k=k, rank1_init_std=rank1_init_std,
        )
        self.fc2 = BatchEnsembleLinear(
            hidden_dim, hidden_dim, k=k, rank1_init_std=rank1_init_std,
        )
        self.fc3 = BatchEnsembleLinear(
            hidden_dim, 1, k=k, rank1_init_std=rank1_init_std,
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, summary_BD: torch.Tensor) -> torch.Tensor:
        x = self.norm(summary_BD)
        x_BkD = x.unsqueeze(1).expand(-1, self.k, -1)
        h = self.dropout(self.act(self.fc1(x_BkD)))
        h = self.dropout(self.act(self.fc2(h)))
        y = self.fc3(h)             # (B, k, 1)
        return y.squeeze(-1)


# ---------------------------------------------------------------------------
# Orthogonal projection (anti-degeneracy #5)
# ---------------------------------------------------------------------------

class OrthogonalSummaryProjection(nn.Module):
    r"""Per-organ projector + QR decomposition.

    For a batch of stacked per-organ summaries
    ``summaries`` of shape ``(B, n_organs, embed_dim)``, we
    independently project each organ's summary through its own
    ``nn.Linear(embed_dim, projected_dim)``, then run a *batched*
    QR decomposition on the transposed ``(B, projected_dim, n_organs)``
    view of that tensor.  Transposing the resulting ``Q`` back to
    ``(B, n_organs, projected_dim)`` gives orthonormal rows along the
    n_organs axis, which is exactly the anti-degeneracy property we want.

    Implementation note: ``torch.linalg.qr`` returns column-orthonormal
    ``Q``.  To obtain row-orthonormal organ embeddings while preserving
    the final dimensionality, we decompose the transposed matrix and then
    transpose ``Q`` back.
    """

    def __init__(
        self, *, n_organs: int, embed_dim: int, projected_dim: int,
    ) -> None:
        super().__init__()
        if projected_dim < n_organs:
            raise ValueError(
                f"projected_dim ({projected_dim}) must be >= n_organs "
                f"({n_organs}) for the QR decomposition to be well-defined."
            )
        self.n_organs = int(n_organs)
        self.projected_dim = int(projected_dim)
        self.projectors = nn.ModuleList([
            nn.Linear(embed_dim, projected_dim) for _ in range(n_organs)
        ])

    def forward(self, summaries_BOE: torch.Tensor) -> torch.Tensor:
        # CUDA batched QR does not support fp16. Keep the rest of training
        # under AMP, but force this projector + QR block to run in fp32.
        original_dtype = summaries_BOE.dtype
        with torch.autocast(device_type=summaries_BOE.device.type, enabled=False):
            summaries_fp32 = summaries_BOE.float()
            proj = torch.stack(
                [
                    self.projectors[o](summaries_fp32[:, o, :])
                    for o in range(self.n_organs)
                ],
                dim=1,
            )  # (B, n_organs, projected_dim)
            # QR returns column-orthonormal bases, so decompose the transpose
            # and transpose back to keep row orthogonality across organs.
            Q_t, _R = torch.linalg.qr(proj.transpose(1, 2), mode="reduced")
            Q = Q_t.transpose(1, 2).contiguous()
        return Q.to(dtype=original_dtype)  # (B, n_organs, projected_dim)


# ---------------------------------------------------------------------------
# Top-level fine-tune model
# ---------------------------------------------------------------------------

class OPFNMLF_FT(nn.Module):
    """SSL backbone + orthogonal projection + 10 BE heads + log_sigma per organ.

    Forward signature::

        out = model(features_BF, organ_presence_BO, timepoint_id_B)

    where ``timepoint_id_B`` is required (long format).
    """

    def __init__(
        self, *,
        backbone_module: nn.Module,
        schema: Schema,
        model_cfg: _cfg.ModelConfig,
    ) -> None:
        super().__init__()
        self.backbone = backbone_module
        self.schema = schema
        self.model_cfg = model_cfg
        self.organ_names: list[str] = list(schema.organ_names)
        self.k = int(model_cfg.k_ensemble)

        # ---- Orthogonal projector ----
        self.orthogonal_projector = OrthogonalSummaryProjection(
            n_organs=schema.n_organs(),
            embed_dim=int(model_cfg.embed_dim),
            projected_dim=int(model_cfg.orthogonal_projected_dim),
        )

        # ---- Per-organ heads (input = projected_dim) ----
        self.heads = nn.ModuleDict()
        for organ in self.organ_names:
            self.heads[organ] = OrganRegressionHead(
                in_dim=int(model_cfg.orthogonal_projected_dim),
                hidden_dim=int(model_cfg.head_hidden_dim),
                k=self.k,
                dropout=float(model_cfg.head_dropout),
                rank1_init_std=float(model_cfg.rank1_init_std),
            )

        # Per-organ learnable log_sigma for uncertainty weighting.
        self.log_sigma_per_organ = nn.Parameter(
            torch.full((schema.n_organs(),), float(0.0))
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        features_BF: torch.Tensor,
        organ_presence_BO: torch.Tensor,
        timepoint_id_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return per-organ k-predictions + raw + projected summaries."""
        out = self.backbone_encode(features_BF, organ_presence_BO, timepoint_id_B)
        per_organ_summary_BOE = out["per_organ_summary"]   # (B, O, E)

        # Orthogonal projection: (B, O, projected_dim) with row-orthogonal Q.
        projected_BOD = self.orthogonal_projector(per_organ_summary_BOE)

        preds: list[torch.Tensor] = []
        for o_idx, organ in enumerate(self.organ_names):
            head_input = projected_BOD[:, o_idx, :]        # (B, projected_dim)
            preds.append(self.heads[organ](head_input))    # (B, k)
        per_organ_pred = torch.stack(preds, dim=1)         # (B, O, k)

        return {
            "per_organ_pred": per_organ_pred,
            "per_organ_summary": per_organ_summary_BOE,
            "projected_summary": projected_BOD,
            "global_summary": out["global_summary"],
            "group_presence": out["group_presence"],
        }

    def backbone_encode(
        self,
        features_BF: torch.Tensor,
        organ_presence_BO: torch.Tensor,
        timepoint_id_B: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the SSL backbone (TabPFN-2.5 wrapped) with timepoint emb.

        The backbone treats the batch as one in-context-learning task, so
        its internal layout is (1, B, F) -> per-cell tokens.
        """
        feats_1NF = features_BF.unsqueeze(0)
        pres_1NO = organ_presence_BO.unsqueeze(0)
        tp_1N = timepoint_id_B.unsqueeze(0)
        out = self.backbone(feats_1NF, pres_1NO, tp_1N)
        return {k: v.squeeze(0) if v.dim() > 1 else v for k, v in out.items()}


# ---------------------------------------------------------------------------
# Builder: load SSL ckpt and assemble OPFNMLF_FT
# ---------------------------------------------------------------------------

@dataclass
class _FTLoadReport:
    total_params: int
    trainable_params: int
    backbone_params: int
    head_params: int
    proj_params: int
    n_state_dict_loaded: int
    embed_dim: int
    n_organs: int
    n_timepoints: int
    n_layers: int
    k_ensemble: int


def build_ft_model_from_ssl_checkpoint(
    *,
    ssl_ckpt_path: Path,
    schema: Schema,
    model_cfg: _cfg.ModelConfig,
    tabpfn_ckpt_path: Path | None = None,
) -> tuple[OPFNMLF_FT, dict[str, Any], _FTLoadReport]:
    """Construct OPFNMLF_FT and initialize backbone from the SSL checkpoint."""
    _ensure_ssl_pkg_importable()
    from organpfnm_ssl import config as _ssl_cfg              # noqa: PLC0415
    from organpfnm_ssl.model import build_model_from_checkpoint  # noqa: PLC0415

    if not ssl_ckpt_path.exists():
        raise FileNotFoundError(
            f"SSL checkpoint not found at {ssl_ckpt_path}.\n"
            "Stage 2 cannot proceed without a Stage 1 checkpoint -- run "
            "scripts/train_ssl_thresholds.py first, then point config.SSL_CKPT_PATH "
            "at its final_ssl_checkpoint.pt."
        )
    logger.info("Loading SSL checkpoint from %s", ssl_ckpt_path)
    ssl_ckpt = torch.load(
        ssl_ckpt_path, map_location="cpu", weights_only=False,
    )

    required_keys = ("model_state_dict", "schema", "normalization", "config")
    for k in required_keys:
        if k not in ssl_ckpt:
            raise KeyError(f"SSL checkpoint missing key {k!r}; got {sorted(ssl_ckpt.keys())}")

    # 1. Sanity: schema feature columns must agree.
    ssl_schema = ssl_ckpt["schema"]
    ssl_feat_cols = list(ssl_schema["feature_columns"])
    if list(schema.feature_columns) != ssl_feat_cols:
        raise ValueError(
            "schema.feature_columns mismatch between current schema.json and "
            f"SSL ckpt ({len(schema.feature_columns)} vs "
            f"{len(ssl_feat_cols)} columns).  Refusing to fine-tune across "
            "schema changes."
        )

    # 2. Sanity: model architecture config.
    ssl_cfg_d = ssl_ckpt["config"]
    if int(ssl_cfg_d["embed_dim"]) != int(model_cfg.embed_dim):
        raise ValueError(
            f"embed_dim mismatch: SSL ckpt has {ssl_cfg_d['embed_dim']}, "
            f"FT model_cfg expects {model_cfg.embed_dim}"
        )
    ckpt_n_layers = int(ssl_cfg_d["n_layers"])
    if ckpt_n_layers != model_cfg.n_layers_expected:
        logger.warning(
            "SSL ckpt has %d backbone layers; FT expected %d (non-fatal).",
            ckpt_n_layers, model_cfg.n_layers_expected,
        )
    if int(ssl_cfg_d.get("features_per_group", model_cfg.features_per_group)) != \
            model_cfg.features_per_group:
        raise ValueError("features_per_group mismatch between SSL ckpt and FT cfg.")
    if int(ssl_cfg_d.get("n_organs", model_cfg.n_organs)) != model_cfg.n_organs:
        raise ValueError(
            f"n_organs mismatch: SSL ckpt has {ssl_cfg_d['n_organs']}, "
            f"FT cfg expects {model_cfg.n_organs}"
        )
    if int(ssl_cfg_d.get("n_timepoints", model_cfg.n_timepoints)) != model_cfg.n_timepoints:
        raise ValueError(
            f"n_timepoints mismatch: SSL ckpt has {ssl_cfg_d['n_timepoints']}, "
            f"FT cfg expects {model_cfg.n_timepoints}"
        )

    # 3. Re-build the SSL model and load SSL weights into it (strict).
    if tabpfn_ckpt_path is None:
        baked = ssl_ckpt.get("tabpfn_ckpt_path")
        if baked and Path(baked).exists():
            tabpfn_ckpt_path = Path(baked)
        else:
            tabpfn_ckpt_path = _cfg.TABPFN_CHECKPOINT_DIR / _ssl_cfg.TABPFN_CKPT_NAME
    logger.info("Re-building TabPFN-2.5 backbone from %s", tabpfn_ckpt_path)

    from organpfnm_ssl.utils import Schema as SslSchema  # noqa: PLC0415
    ssl_schema_obj = SslSchema(
        organ_names=list(schema.organ_names),
        feature_columns=list(schema.feature_columns),
        organ_offsets=dict(schema.organ_offsets),
        n_features_per_organ=dict(schema.n_features_per_organ),
        n_total_features=schema.n_total_features,
        metadata_columns=list(schema.metadata_columns),
        organ_presence_threshold_ssl=float(schema.organ_presence_threshold_ssl),
        organ_presence_threshold_finetune=float(schema.organ_presence_threshold_finetune),
        timepoint_id_map=dict(schema.timepoint_id_map),
        n_timepoints=int(schema.n_timepoints),
        organ_timepoint_availability=dict(schema.organ_timepoint_availability),
        raw=schema.raw,
        sex_column_per_organ=dict(schema.sex_column_per_organ),
        sex_column_indices_per_organ=dict(schema.sex_column_indices_per_organ),
        organ_feature_map_excluding_sex={
            o: list(v) for o, v in schema.organ_feature_map_excluding_sex.items()
        },
        n_features_per_organ_excluding_sex=dict(schema.n_features_per_organ_excluding_sex),
        presence_excludes_sex=bool(schema.presence_excludes_sex),
    )

    from organpfnm_ssl.config import ModelConfig as SslModelCfg  # noqa: PLC0415
    ssl_model_cfg = SslModelCfg(
        embed_dim=model_cfg.embed_dim,
        n_layers_expected=ckpt_n_layers,
        features_per_group=model_cfg.features_per_group,
        num_thinking_rows=model_cfg.num_thinking_rows,
        n_organs=model_cfg.n_organs,
        n_timepoints=model_cfg.n_timepoints,
        n_total_features=model_cfg.n_total_features,
    )

    ssl_model, ssl_report = build_model_from_checkpoint(
        schema=ssl_schema_obj,
        ckpt_path=tabpfn_ckpt_path,
        model_cfg=ssl_model_cfg,
    )

    missing, unexpected = ssl_model.load_state_dict(
        ssl_ckpt["model_state_dict"], strict=True,
    )
    if len(missing) or len(unexpected):
        raise RuntimeError(
            f"SSL state_dict load returned missing={missing}, "
            f"unexpected={unexpected}"
        )
    logger.info(
        "SSL weights loaded into rebuilt model: backbone params=%d "
        "(loaded keys=%d, ckpt step=%s)",
        ssl_report.backbone_params, ssl_report.n_loaded,
        ssl_ckpt.get("global_step", "?"),
    )

    # 4. Strip SSL heads, keep backbone, attach FT pieces.
    backbone_module = ssl_model.backbone
    for p in backbone_module.parameters():
        p.requires_grad_(True)

    ft_model = OPFNMLF_FT(
        backbone_module=backbone_module,
        schema=schema, model_cfg=model_cfg,
    )

    backbone_param_count = count_parameters(backbone_module)
    head_param_count = count_parameters(ft_model.heads)
    proj_param_count = count_parameters(ft_model.orthogonal_projector)
    report = _FTLoadReport(
        total_params=count_parameters(ft_model),
        trainable_params=count_trainable(ft_model),
        backbone_params=backbone_param_count,
        head_params=head_param_count,
        proj_params=proj_param_count,
        n_state_dict_loaded=len(ssl_ckpt["model_state_dict"]),
        embed_dim=int(model_cfg.embed_dim),
        n_organs=int(schema.n_organs()),
        n_timepoints=int(schema.n_timepoints),
        n_layers=ckpt_n_layers,
        k_ensemble=int(model_cfg.k_ensemble),
    )
    return ft_model, ssl_ckpt["normalization"], report

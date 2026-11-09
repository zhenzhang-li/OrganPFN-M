"""Path-free, reproducible configuration for OrganPFN-M self-supervision."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV = "public"
DATA_DIR = Path(os.environ.get("ORGANPFNM_SSL_DATA_DIR", REPO_ROOT / "data" / "processed" / "thr_90" / "ssl"))
CHECKPOINT_DIR = Path(os.environ.get("ORGANPFNM_TABPFN_CHECKPOINT_DIR", REPO_ROOT / "checkpoints" / "tabpfn"))
OUTPUT_DIR = Path(os.environ.get("ORGANPFNM_SSL_OUTPUT_DIR", REPO_ROOT / "outputs" / "ssl"))
LINEAR_PROBE_DATA_DIR = Path(os.environ.get("ORGANPFNM_LINEAR_PROBE_DATA_DIR", REPO_ROOT / "data" / "processed" / "thr_90" / "finetune"))
TABPFN_CKPT_NAME = "tabpfn-v2.5-regressor-v2.5_default.ckpt"


def tabpfn_ckpt_path() -> Path:
    return CHECKPOINT_DIR / TABPFN_CKPT_NAME


def schema_path() -> Path:
    return DATA_DIR / "schema.json"


def ssl_csv_path() -> Path:
    return DATA_DIR / "ssl_subset.csv"


def finetune_csv_path() -> Path:
    return LINEAR_PROBE_DATA_DIR / "finetune_subset.csv"


@dataclass
class ModelConfig:
    embed_dim: int = 192
    n_layers_expected: int = 18
    nhead_expected: int = 3
    features_per_group: int = 3
    num_thinking_rows: int = 64
    n_organs: int = 10
    n_timepoints: int = 2
    n_total_features: int = 786
    organ_id_emb_init_std: float = 0.02
    timepoint_emb_init_std: float = 0.02
    cell_recon_hidden: int = 192
    mask_pred_hidden: int = 192
    organ_decoder_emb_dim: int = 256
    organ_decoder_n_layers: int = 2
    organ_decoder_nhead: int = 4
    contrastive_proj_hidden: int = 192
    contrastive_proj_dim: int = 128
    contrastive_temperature: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 20260512
    batch_size: int = 64
    lr_peak: float = 3e-4
    weight_decay: float = 0.05
    grad_clip_max_norm: float = 1.0
    warmup_steps: int = 500
    mixed_precision: bool = True
    lambda_cell: float = 1.0
    lambda_mask: float = 0.3
    lambda_organ: float = 1.0
    lambda_contrastive: float = 0.1
    cell_mask_ratio_default: float = 0.15
    organ_drop_range_default: tuple[int, int] = (1, 4)
    scarf_corrupt_ratio: float = 0.30
    stages: list[dict] = field(default_factory=lambda: [
        {"name": "warmup", "epochs": 1, "cell_mask_ratio": 0.05,
         "organ_drop_range": (0, 0), "active_losses": ["cell", "mask"]},
        {"name": "main", "epochs": 15, "cell_mask_ratio": 0.15,
         "organ_drop_range": (1, 4), "active_losses": ["cell", "mask", "organ", "contrastive"]},
        {"name": "hard", "epochs": 5, "cell_mask_ratio": 0.30,
         "organ_drop_range": (1, 6), "active_losses": ["cell", "mask", "organ", "contrastive"]},
    ])
    linear_probe_every_n_epochs: int = 2
    linear_probe_split_ratio: float = 0.8
    linear_probe_ridge_lambda: float = 1.0
    log_every_n_steps: int = 50
    metrics_csv_name: str = "metrics.csv"
    epoch_summary_csv_name: str = "epoch_summary.csv"
    train_log_name: str = "train.log"
    final_ckpt_name: str = "final_ssl_checkpoint.pt"

    def stage_ckpt_name(self, stage_name: str) -> str:
        return f"checkpoint_stage_{stage_name}.pt"


MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()
DEBUG_NROWS: int | None = None
SMOKE_TEST = False

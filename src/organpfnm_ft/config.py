"""Path-free configuration for supervised OrganPFN-M fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV = "public"
DATA_ROOT = Path(os.environ.get("ORGANPFNM_DATA_ROOT", REPO_ROOT / "data" / "processed"))
DATA_DIR = Path(os.environ.get("ORGANPFNM_FT_DATA_DIR", DATA_ROOT / "thr_30" / "finetune"))
SSL_CKPT_PATH = Path(os.environ.get("ORGANPFNM_SSL_CHECKPOINT", REPO_ROOT / "checkpoints" / "ssl" / "thr_90" / "final_ssl_checkpoint.pt"))
TABPFN_CHECKPOINT_DIR = Path(os.environ.get("ORGANPFNM_TABPFN_CHECKPOINT_DIR", REPO_ROOT / "checkpoints" / "tabpfn"))
OUTPUT_DIR = Path(os.environ.get("ORGANPFNM_FT_OUTPUT_DIR", REPO_ROOT / "outputs" / "finetune"))


def set_data_dir(new_dir: Path) -> None:
    global DATA_DIR
    DATA_DIR = Path(new_dir)


def schema_path() -> Path:
    return DATA_DIR / "schema.json"


def finetune_csv_path() -> Path:
    return DATA_DIR / "finetune_subset.csv"


def participant_pairs_csv_path() -> Path:
    return DATA_DIR / "participant_pairs.csv"


def ssl_csv_path() -> Path:
    return DATA_ROOT / "thr_90" / "ssl" / "ssl_subset.csv"


def ssl_schema_path() -> Path:
    return DATA_ROOT / "thr_90" / "ssl" / "schema.json"


@dataclass
class ModelConfig:
    embed_dim: int = 192
    n_layers_expected: int = 18
    features_per_group: int = 3
    num_thinking_rows: int = 64
    n_organs: int = 10
    n_timepoints: int = 2
    n_total_features: int = 786
    k_ensemble: int = 32
    head_hidden_dim: int = 256
    head_dropout: float = 0.3
    rank1_init_std: float = 0.1
    orthogonal_projected_dim: int = 128
    age_bias_reg_weight: float = 0.1


@dataclass
class TrainConfig:
    seed: int = 20260512
    val_split_ratio: float = 0.20
    batch_size: int = 32
    paired_samples_per_batch: int = 8
    n_epochs: int = 50
    early_stop_patience: int = 10
    grad_clip_max_norm: float = 1.0
    mixed_precision: bool = True
    lr_backbone_early: float = 1e-5
    lr_backbone_late: float = 5e-5
    lr_backbone_top: float = 2e-4
    lr_heads: float = 1e-3
    weight_decay_backbone: float = 0.01
    weight_decay_heads: float = 0.01
    warmup_steps: int = 200
    huber_delta: float = 2.0
    log_sigma_init: float = 0.0
    inverse_freq_use_sqrt: bool = True
    modality_dropout_prob: float = 0.30
    modality_dropout_min_drop: int = 1
    modality_dropout_max_drop: int = 2
    gamma_contrastive: float = 0.5
    gamma_divergence: float = 0.1
    contrastive_margin: float = 0.3
    prediction_min_std: float = 1.5
    prediction_max_std: float = 5.0
    gamma_long: float = 0.1
    long_loss_organs: tuple[str, ...] = ("Body", "Immune", "Lungs")
    log_every_n_steps: int = 50
    train_metrics_csv: str = "train_metrics.csv"
    val_metrics_csv: str = "val_metrics.csv"
    train_log_name: str = "train.log"
    best_ckpt_name: str = "best_model.pt"
    final_ckpt_name: str = "final_model.pt"


MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()

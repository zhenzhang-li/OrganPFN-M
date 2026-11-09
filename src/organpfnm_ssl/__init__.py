"""OrganPFN-M-LF (Long-Format, 10-organ) SSL pre-training package.

Stage 1 of the OrganPFN-M-LF training pipeline:
self-supervised pre-training on healthy + diseased samples, in long
format (one row per (Participant, Timepoint) observation).

Architecture
------------
  - Backbone: TabPFN-2.5 PerFeatureTransformer (loaded 100 % from
    ``tabpfn-v2.5-regressor-v2.5_default.ckpt``).
  - Cell embedder: TabPFN's feature_group_embedder + nan-indicator
    pipeline + organ-id embedding + ★ timepoint embedding (per-row,
    T0=0 / T2=1).
  - 4 SSL heads attached to backbone output:
      Task 1  cell-level reconstruction (VIME-style).
      Task 2  mask-position prediction (binary classification).
      Task 3  organ-level reconstruction (per-organ light decoders).
      Task 4  SCARF contrastive learning.

Study configuration
-------------------
  * 10 organs (Body / Hepatic / Immune / Lungs / Metabolic / Renal
    + Brain / Cardiovascular / Cognitive / Mental).
  * Long-format CSV: ``timepoint`` and ``timepoint_id`` metadata
    columns drive a learnable ``timepoint_embedding`` added per-row.
  * Organ-presence threshold (SSL stage): 0.30 (read from schema.json).
"""

__all__ = [
    "config",
    "data",
    "model",
    "ssl_objectives",
    "train",
    "utils",
]

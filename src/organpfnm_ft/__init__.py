"""OrganPFN-M-LF Stage 2: supervised fine-tuning (long-format, 10 organs).

Loads the SSL checkpoint from Stage 1, attaches per-organ
TabM-style BatchEnsemble regression heads (k=32), and trains with
chronological-age supervision on the long-format healthy subset.

The package exposes:

* `config`     -- centralized hyperparameters and env-driven paths
* `data`       -- Dataset, PairedBatchSampler, modality dropout,
                  by-Participant_ID train/val split
* `model`      -- BatchEnsembleLinear, OrganRegressionHead,
                  OrthogonalSummaryProjection, OPFNMLF_FT (= SSL
                  backbone + 10 BE heads + orthogonal projector)
* `losses`     -- per-organ Huber + uncertainty weighting + inverse
                  frequency weighting + contrastive + divergence +
                  age-bias + longitudinal consistency
* `train`      -- training loop + early stopping + ablation hook
                  (organ-presence threshold sweep)
* `inference`  -- predict() API + age-bias linear correction
* `evaluation` -- healthy-vs-diseased age-gap analysis
"""

__all__ = [
    "config", "data", "model", "losses", "train",
    "inference", "evaluation", "utils",
]

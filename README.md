# OrganPFN-M

Reproducible code for multiorgan biological-age prediction, Panel-Invariant
Organ-ageing Asynchrony (PIOAA), and the Organ Aging-Disease Propagation
Cascade. The repository is organized for peer review and archival release with
the associated Nature Medicine submission.

This repository contains analysis code only. It does not contain UK Biobank
participant-level data, derived individual predictions, trained weights, or the
third-party TabPFN checkpoint. Access to UK Biobank data must be obtained from
UK Biobank and used under its data-access terms.

## Study workflow

1. Harmonize T0 physiological and T2 imaging follow-up measurements as
   independent participant-visit rows. A participant observed at both visits
   therefore contributes two samples, not one horizontally joined sample.
2. Preserve ten organ blocks, 776 measurement features, and one sample-level
   Sex field. Unmeasured organs and unmeasured features remain missing.
3. Create datasets at maximum within-organ missingness thresholds of 0%, 30%,
   50%, 70%, and 90%, retaining samples with at least two eligible organs.
4. Self-supervise OrganPFN-M on all retained samples at each threshold. Input
   normalization is estimated from healthy-reference observations across T0
   and T2 and then applied unchanged to all self-supervised samples.
5. Fine-tune on healthy participants only using participant-level,
   organ-balanced five-fold cross-validation. The saved self-supervised means
   and standard deviations are reused without re-estimation, and age remains
   in years.
6. Derive organ age gaps, PIOAA scores, and observational organ-disease
   propagation patterns.

The primary downstream analysis used the self-supervised threshold-90
checkpoint, threshold-30 healthy fine-tuning data, and the fold-4 checkpoint
from the asymmetric-loss five-fold analysis. All five thresholds and all five
folds remain exposed for sensitivity analyses.

## Repository map

```text
configs/                         non-sensitive configuration templates
docs/                            data, methods, and reproducibility notes
examples/                        synthetic data generator
scripts/prepare_data.py          five-threshold cohort preparation
scripts/train_ssl_thresholds.py  self-supervision at 0/30/50/70/90
scripts/train_finetune_thresholds.py  healthy-only five-fold fine-tuning
scripts/compute_pioaa.py         panel-invariant asynchrony scores
scripts/run_cascade.py           four edge families, routes, loops, comorbidity
src/organpfnm_ssl/               OrganPFN-M self-supervised model
src/organpfnm_ft/                supervised organ-age model
src/organpfnm/                   preprocessing and downstream methods
tests/                           synthetic unit tests
```

## Installation

Python 3.11 is recommended. Create a clean environment and install the package:

```bash
conda env create -f environment.yml
conda activate organpfnm
```

The OrganPFN-M backbone requires the TabPFN 2.5 regressor checkpoint named
`tabpfn-v2.5-regressor-v2.5_default.ckpt`. Obtain it under the applicable
TabPFN terms and place it in `checkpoints/tabpfn/`. It is intentionally ignored
by Git.

## Minimal reproducible run

Generate a non-identifying synthetic dataset and test preprocessing and
downstream analysis:

```bash
python examples/make_synthetic_data.py --output-dir examples/generated
python scripts/prepare_data.py \
  --input examples/generated/harmonized_visits.csv \
  --feature-map examples/generated/feature_map.json \
  --output-root examples/generated/processed
python -m pytest
```

Full training requires authorized data and a CUDA-capable environment:

```bash
python scripts/train_ssl_thresholds.py --data-root data/processed
python scripts/train_finetune_thresholds.py \
  --data-root data/processed \
  --ssl-checkpoint outputs/ssl/thr_90/final_ssl_checkpoint.pt
```

Use `--smoke-test` for a short integration check. Existing final checkpoints
are skipped unless `--overwrite` is supplied.

## Reproducibility and reporting

The seed is 20260512 for model training and cross-validation. Self-supervision
uses a 1-epoch warm-up, 15 main epochs, and 5 hard-mask epochs. Fine-tuning uses
50 epochs with early stopping, an organ-balanced participant split, and the
organ-specific asymmetric Huber loss described in `docs/METHODS_TO_CODE.md`.
The cascade uses FDR control and stated minimum-count criteria; it describes
temporal and prospective associations and does not establish causality.

See `docs/DATA_FORMAT.md`, `docs/REPRODUCIBILITY.md`, and
`docs/METHODS_TO_CODE.md` before running restricted analyses. Refactoring
provenance is recorded in `docs/SOURCE_PROVENANCE.md`.

## Public-release checklist

- Select an institutional software license; see `LICENSE_TO_SELECT.md`.
- Replace the repository URL and complete authors/DOI in `CITATION.cff`.
- Create a tagged release and archive the exact manuscript commit.
- Confirm with UK Biobank that no restricted data or individual outputs are in
  Git history.
- Record hardware, package lock file, and checkpoint hashes for the final run.

## Responsible use

Organ-age estimates are research outputs and are not validated for individual
clinical diagnosis or treatment decisions.

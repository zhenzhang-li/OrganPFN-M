# Reproducibility record

## Fixed analysis choices

- Training and cross-validation seed: 20260512.
- O2O bootstrap seed: 20260701.
- Data thresholds: 0, 30, 50, 70, and 90% maximum within-organ missingness.
- Minimum eligible organs per sample: 2.
- Self-supervised data: all retained samples.
- Normalization reference: healthy observations across T0 and T2.
- Fine-tuning data: healthy samples only.
- Cross-validation: five folds grouped by participant and balanced by organ
  availability.
- Primary SSL checkpoint: threshold 90.
- Primary fine-tuning dataset: threshold 30.
- Primary downstream checkpoint: asymmetric-loss fold 4.

## What each checkpoint must contain

The self-supervised checkpoint records model configuration, schema,
normalization means and standard deviations, and the TabPFN checkpoint path.
The fine-tuned checkpoint records the inherited self-supervised configuration,
the organ-presence threshold, fold-specific weights, and asymmetric-loss
parameters. Do not strip these metadata when archiving trained weights.

## Hardware and software record

For the final archived run, record:

- operating system and Python version;
- GPU model, CUDA version, and number of GPUs;
- `pip freeze` or an exported Conda lock file;
- SHA-256 hashes of the TabPFN, SSL, and selected FT checkpoints;
- exact Git commit and release tag;
- command line and environment variables used for each stage;
- fold assignment JSON and all threshold summary JSON files.

Floating-point results can vary slightly across GPU architectures. Report
performance using the archived out-of-fold predictions and retain tolerances in
automated integration checks.

## Validation order

1. Run `python examples/make_synthetic_data.py --output-dir examples/generated`.
2. Run the preprocessing command in the main README.
3. Run `python -m pytest`.
4. Run self-supervised and fine-tuning scripts with `--smoke-test` on an
   authorized, disclosure-safe slice.
5. Confirm fold assignments are participant-disjoint and that all visits for a
   participant remain in one fold.
6. Confirm fine-tuning checkpoint normalization arrays are byte-identical to
   those loaded from the selected SSL checkpoint.

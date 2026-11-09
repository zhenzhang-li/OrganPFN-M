# Source provenance

This release was assembled from the study's final working code and then
refactored to remove machine-specific paths and restricted data assumptions.

- Self-supervised model code was taken from the final `prog_10_v10`
  `organpfnm_ssl` package. The corresponding package in `prog_10_v10_2` was
  byte-identical at review time.
- Fine-tuning model code was taken from `prog_10_v10_2`, which contains the
  final asymmetric-loss implementation used for the selected model.
- PIOAA was refactored from the final participant-level T0/T2 computation into
  a fitted, reusable class with the same empirical marginal transform,
  Spearman-to-latent-normal correlation mapping, positive-definite correction,
  fixed 0.10 shrinkage, decomposition, and panel-size calibration.
- The cascade module was refactored from the final reconstructed edge and route
  analyses. Machine-specific paths, figure-only scripts, cached participant
  tables, and result files were not copied.

The public preprocessing interface intentionally stores one sample-level Sex
field. The loader broadcasts it into internal organ-block slots only at runtime
to retain checkpoint compatibility. The public table therefore follows the
study description of 776 measurements plus one Sex field rather than presenting
the repeated implementation slots as separate variables.

No scientific result was regenerated during packaging. Before manuscript
release, compare hashes and selected outputs from this package against the
archived analysis environment using authorized data.

# Data contract

## Participant-visit table

The preprocessing entry point consumes one harmonized table, separate T0 and
T2 tables, or a manifest of organ-specific files. Each output row is one participant at one visit. T0 and T2
rows are stacked; they are not horizontally joined. Repeated participant IDs
are expected and all rows from one participant are kept in the same
cross-validation fold.

Required metadata columns:

| Column | Meaning |
|---|---|
| `Participant_ID` | Study-local participant key; never release populated values |
| `Sex` | One sample-level value, encoded as M/F, Male/Female, or 0/1 |
| `timepoint` | `T0` or `T2` |
| `timepoint_id` | 0 for T0 and 1 for T2 |
| `Age` | Chronological age in years at the represented visit |
| `Disease` | Study disease label or summary field |
| `is_healthy` | 1 for the predefined healthy reference, otherwise 0 |

The measurement columns are assigned to the ten organs in a JSON feature map.
The study table contains 776 measurement features plus one sample-level Sex
field. An unmeasured organ is represented by missing values across its block;
partly observed organs retain their original feature-level missingness.

For restricted organ-specific source files, use
`configs/data_manifest.example.json`. Only explicitly listed columns are
joined to the visit metadata spine, with one-to-one participant validation.
This keeps raw field selection reviewable without publishing file paths or
participant data.

## Five preprocessing thresholds

For each organ and sample, the missing fraction is calculated using measurement
features only. Sex is excluded. An organ is eligible at threshold q when its
missing fraction is no greater than q%, for q in 0, 30, 50, 70, and 90.
Non-eligible organ blocks are set to missing, and a sample is retained only if
at least two organs remain eligible.

The model's original loaders apply a second, fixed observed-feature check of
30% during self-supervision and 50% during fine-tuning. These values are stored
separately from the five data-construction thresholds in `schema.json`.

## Sex representation

The saved CSV has one Sex column. To preserve compatibility with the original
organ-block architecture and checkpoints, the loader broadcasts this value to
the first implementation slot of each organ block. All such slots contain the
same source variable. They are not ten distinct biological predictors and are
excluded from organ availability and missingness calculations.

## Generated layout

```text
data/processed/
  thr_0/
    ssl/ssl_subset.csv
    ssl/schema.json
    finetune/finetune_subset.csv
    finetune/participant_pairs.csv
    finetune/schema.json
    summary.json
  thr_30/ ...
  thr_50/ ...
  thr_70/ ...
  thr_90/ ...
```

The self-supervised subset contains all retained samples. The fine-tuning
subset contains healthy-reference samples only.

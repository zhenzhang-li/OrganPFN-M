# Methods-to-code map

## 1. Cohort construction and missingness sensitivity

`src/organpfnm/preprocessing.py` and `scripts/prepare_data.py` implement the
independent T0/T2 sample representation, ten organ blocks, five maximum
within-organ missingness thresholds, and the requirement for at least two
eligible organs. Thresholding never uses Sex, age, disease status, or model
outputs.

## 2. OrganPFN-M self-supervision

`src/organpfnm_ssl` contains the TabPFN-2.5-based OrganPFN-M backbone, organ and
timepoint embeddings, four self-supervised objectives, and the 21-epoch
curriculum. The losses are cell reconstruction, missingness prediction,
organ-block reconstruction, and contrastive representation learning. The
default curriculum is 1 warm-up epoch, 15 main epochs, and 5 harder masking
epochs. Optimization uses AdamW, peak learning rate 3e-4, weight decay 0.05,
batch size 64, 500 warm-up steps, gradient norm clipping at 1.0, and seed
20260512.

All retained samples, healthy and diseased, contribute to self-supervised
training. Per-feature means and standard deviations are estimated only from the
predefined healthy-reference observations across T0 and T2. Missing values do
not contribute to these estimates. Standardized missing values are represented
by zero together with an explicit missingness mask. The fitted normalization
parameters are saved in the self-supervised checkpoint.

`scripts/train_ssl_thresholds.py` runs all five preprocessing thresholds.

## 3. Healthy-only supervised fine-tuning

`src/organpfnm_ft` loads the self-supervised backbone and its fixed
normalization parameters. Fine-tuning does not estimate means or standard
deviations from training or validation folds. Chronological age remains in
years. Participants, including both visits when available, are assigned to one
of five folds by the organ-balanced algorithm in
`src/organpfnm/cross_validation.py`.

The default schedule uses 50 epochs, batch size 32, early-stopping patience 10,
and layer-specific learning rates of 1e-5, 5e-5, 2e-4, and 1e-3 from early
backbone layers to prediction heads. The final asymmetric Huber loss increases
the weight of underprediction progressively from each organ's healthy median
age to its healthy 95th-percentile age. With lambda=3, the multiplier increases
from 1 to a maximum of 4. `scripts/train_finetune_thresholds.py` runs five folds
for each of the five datasets. By default, all fine-tuning sensitivity datasets
use the primary SSL threshold-90 checkpoint. Pass `--match-ssl-threshold` for
the paired-threshold sensitivity design in which each dataset is initialized
from its corresponding SSL checkpoint.

The principal downstream prediction configuration is:

- self-supervised checkpoint: threshold 90;
- supervised data: threshold 30, healthy samples;
- cross-validation: five participant-level folds;
- downstream applied model: asymmetric-loss fold 4.

## 4. Panel-Invariant Organ-ageing Asynchrony

`src/organpfnm/pioaa.py` combines available visit-specific organ age gaps into
one participant profile for this downstream analysis only. Repeated Immune or
Lungs observations are averaged when both visits are available. Each organ's
healthy-reference marginal distribution is converted to a normal score. A
pairwise complete Spearman matrix is mapped to latent normal correlation,
projected to a valid correlation matrix, and stabilized with fixed shrinkage
0.10.

For each observed organ panel, PIOAA decomposes total multiorgan deviation into
a shared magnitude component and a residual asynchrony component. Both are
then empirically calibrated within the number of observed organs, yielding
`s_magnitude` and `s_asynchrony`. At least two organs are required. The command
entry point is `scripts/compute_pioaa.py`.

## 5. Organ Aging-Disease Propagation Cascade

`src/organpfnm/cascade.py` and `scripts/run_cascade.py` implement the analysis
as eight linked steps:

1. Define ten organ nodes and disease first-onset nodes in the eligible cohort.
2. Estimate organ-to-disease edges with prospective Cox models after excluding
   baseline-prevalent disease, adjusting for age and sex and censoring at death
   or the administrative end of follow-up.
3. Estimate disease-to-organ edges by comparing prevalent cases with the
   healthy reference in age- and sex-adjusted regression.
4. Estimate organ-to-organ edges from paired visits using cross-lagged models
   of target-organ change, controlling for its baseline value, follow-up time,
   age, sex, and the common organ-aging component.
5. Estimate disease-to-disease direction from clearly ordered first onsets,
   requiring at least a 0.5-year separation and testing the directional
   fraction against 0.5.
6. Control the false-discovery rate, require prespecified minimum counts, and
   retain only effects exceeding their negative-control floor. O2D and D2O
   floors may differ by disease or organ; supply them in the
   `negative_control_floor` column before `run_cascade.py admit`.
7. Convert effects to within-edge-type percentile strengths. Identify the
   maximum-product route from any organ to each disease and report mixed
   organ-disease directed cycles separately as feedback patterns.
8. Summarize baseline comorbidity, PIOAA amplification, and adjusted all-cause
   mortality associations. Mortality is treated as a terminal outcome, not as
   evidence that the preceding observational routes are causal.

Primary count rules are 300 incident events for O2D, 200 prevalent cases for
D2O, 1,000 paired observations and a positive bootstrap interval for O2O, and
100 clearly ordered disease pairs for D2D. All retained edge families use
FDR<0.05. Cascade routes and loops represent linked associations and must not be
described as causal transmission.

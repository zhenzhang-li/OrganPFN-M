# Data access and non-redistribution

The study uses UK Biobank data. This repository does not redistribute raw or
processed participant-level records. Researchers must obtain their own
authorization from UK Biobank and reconstruct the analysis table from approved
fields within their secure environment.

Files under `data/raw`, `data/processed`, `checkpoints`, `outputs`, and
`results` are ignored by Git. This is a safeguard, not a substitute for review.
Before every public push, inspect staged files and the complete Git history for
participant identifiers, dates, disease histories, predictions, access tokens,
and checkpoint artifacts.

The feature map may be shared only if field names and derived definitions are
permitted by the applicable data-use terms. Aggregate, disclosure-controlled
cohort counts should be supplied in the article's supplementary tables rather
than copied from individual-level result files.

Recommended Code availability wording:

> Analysis code used for cohort preparation, OrganPFN-M training, PIOAA and the
> organ aging-disease propagation cascade is available at the archived release
> cited in the manuscript. UK Biobank participant-level data are not included
> and may be accessed through the UK Biobank application process.

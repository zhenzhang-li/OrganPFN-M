# GitHub and archival release checklist

1. Complete `CITATION.cff`, choose a license, and replace all placeholder URLs.
2. Keep the repository private during institutional and UK Biobank disclosure
   review.
3. Run `python -m pytest` and the path/privacy scan documented below.
4. Initialize Git, commit only the reviewed directory, and inspect every staged
   file with `git diff --cached --stat` and `git diff --cached`.
5. Push to a new empty repository. Do not import the historical research
   folders because their history may contain data and absolute paths.
6. Tag the manuscript version, for example `v1.0.0`, and create a GitHub
   release.
7. Archive the release in Zenodo or an institutional repository and add its DOI
   to the manuscript and `CITATION.cff`.

Suggested local checks:

```bash
python -m pytest
python -m compileall -q src scripts examples
rg -n '[A-Za-z]:\\|/home/|Participant_ID,' .
git status --short
```

The `Participant_ID` schema name is expected; populated values are not. Review
all matches manually.

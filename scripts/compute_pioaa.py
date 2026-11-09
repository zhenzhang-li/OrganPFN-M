#!/usr/bin/env python
"""Compute PIOAA scores from organ age-gap columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from organpfnm.pioaa import PIOAA, aggregate_age_gaps_by_participant


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--organs", nargs="+", required=True,
                        help="Organ age-gap columns in the intended panel order.")
    parser.add_argument("--healthy-column", default="is_healthy")
    parser.add_argument("--participant-column", default="Participant_ID")
    parser.add_argument("--aggregate-visits", action="store_true")
    args = parser.parse_args()

    data = read_table(args.input)
    if args.aggregate_visits:
        reference = data.groupby(args.participant_column, sort=False)[args.healthy_column].all()
        panel = aggregate_age_gaps_by_participant(
            data, args.organs, participant_column=args.participant_column
        )
        healthy = panel[args.participant_column].map(reference).fillna(False).to_numpy()
    else:
        panel = data[[args.participant_column, *args.organs]].copy()
        healthy = data[args.healthy_column].fillna(False).astype(bool).to_numpy()
    scores = PIOAA(args.organs).fit_transform(panel[args.organs], healthy)
    output = pd.concat([panel[[args.participant_column]].reset_index(drop=True), scores], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

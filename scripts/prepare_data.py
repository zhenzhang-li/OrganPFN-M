#!/usr/bin/env python
"""Prepare five missingness-threshold datasets from non-public source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from organpfnm.preprocessing import (
    DEFAULT_THRESHOLDS,
    harmonize_visit_rows,
    load_feature_map,
    prepare_threshold_datasets,
    assemble_visits_from_manifest,
)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Harmonized participant-visit CSV/Parquet.")
    source.add_argument("--t0", type=Path, help="T0 CSV/Parquet; requires --t2.")
    source.add_argument("--manifest", type=Path,
                        help="JSON manifest for organ-specific T0/T2 source tables.")
    parser.add_argument("--t2", type=Path)
    parser.add_argument("--feature-map", type=Path, required=True,
                        help="JSON mapping each organ to its measurement columns.")
    parser.add_argument("--output-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--thresholds", type=int, nargs="+", default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--min-organs", type=int, default=2)
    args = parser.parse_args()

    if args.input:
        table = read_table(args.input)
    elif args.manifest:
        table = assemble_visits_from_manifest(args.manifest)
    else:
        if args.t2 is None:
            parser.error("--t0 requires --t2")
        table = harmonize_visit_rows(read_table(args.t0), read_table(args.t2))
    summaries = prepare_threshold_datasets(
        table,
        load_feature_map(args.feature_map),
        args.output_root,
        thresholds=args.thresholds,
        min_organs=args.min_organs,
    )
    print(json.dumps([summary.__dict__ for summary in summaries], indent=2))


if __name__ == "__main__":
    main()

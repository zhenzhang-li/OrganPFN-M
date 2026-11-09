#!/usr/bin/env python
"""Command-line entry points for the organ aging-disease cascade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from organpfnm.cascade import (
    AdmissionRule,
    admit_edges,
    comorbidity_summary,
    cross_lagged_organ_to_organ,
    enumerate_feedback_loops,
    maximum_strength_routes,
    mortality_associations,
    prevalent_disease_to_organ,
    prospective_organ_to_disease,
    temporal_disease_to_disease,
)


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"wrote {len(table):,} rows to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    o2d = sub.add_parser("o2d", help="Prospective organ-to-disease Cox edges.")
    o2d.add_argument("--input", type=Path, required=True)
    o2d.add_argument("--organ-columns", nargs="+", required=True)
    o2d.add_argument("--onset-map", type=Path, required=True)
    o2d.add_argument("--baseline-column", default="baseline_year")
    o2d.add_argument("--death-column", default="death_year")
    o2d.add_argument("--age-column", default="age")
    o2d.add_argument("--sex-column", default="sex")
    o2d.add_argument("--administrative-censoring", type=float, required=True)
    o2d.add_argument("--min-events", type=int, default=300)
    o2d.add_argument("--output", type=Path, required=True)

    d2o = sub.add_parser("d2o", help="Prevalent disease-to-organ contrasts.")
    d2o.add_argument("--input", type=Path, required=True)
    d2o.add_argument("--organ-columns", nargs="+", required=True)
    d2o.add_argument("--onset-map", type=Path, required=True)
    d2o.add_argument("--healthy-column", default="is_healthy")
    d2o.add_argument("--baseline-column", default="baseline_year")
    d2o.add_argument("--age-column", default="age")
    d2o.add_argument("--sex-column", default="sex")
    d2o.add_argument("--min-cases", type=int, default=200)
    d2o.add_argument("--output", type=Path, required=True)

    d2d = sub.add_parser("d2d", help="Temporally ordered disease-to-disease edges.")
    d2d.add_argument("--input", type=Path, required=True)
    d2d.add_argument("--onset-map", type=Path, required=True)
    d2d.add_argument("--minimum-lag-years", type=float, default=0.5)
    d2d.add_argument("--min-pairs", type=int, default=100)
    d2d.add_argument("--minimum-fraction", type=float, default=0.55)
    d2d.add_argument("--output", type=Path, required=True)

    o2o = sub.add_parser("o2o", help="Cross-lagged organ-to-organ edges.")
    o2o.add_argument("--input", type=Path, required=True)
    o2o.add_argument("--organs", nargs="+", required=True)
    o2o.add_argument("--baseline-suffix", default="_baseline")
    o2o.add_argument("--followup-suffix", default="_followup")
    o2o.add_argument("--age-column", default="age_baseline")
    o2o.add_argument("--sex-column", default="sex")
    o2o.add_argument("--interval-column", default="followup_years")
    o2o.add_argument("--min-pairs", type=int, default=1000)
    o2o.add_argument("--n-bootstrap", type=int, default=200)
    o2o.add_argument("--seed", type=int, default=20260701)
    o2o.add_argument("--output", type=Path, required=True)

    admit = sub.add_parser("admit", help="Apply negative-control floors.")
    admit.add_argument("--edges", type=Path, nargs="+", required=True)
    admit.add_argument("--rules", type=Path, required=True,
                       help="JSON: edge type -> minimum_count/effect_floor/direction.")
    admit.add_argument("--output", type=Path, required=True)

    routes = sub.add_parser("routes", help="Build maximum-product routes.")
    routes.add_argument("--admitted-edges", type=Path, required=True)
    routes.add_argument("--organs", nargs="+", required=True)
    routes.add_argument("--diseases", nargs="+", required=True)
    routes.add_argument("--output", type=Path, required=True)

    loops = sub.add_parser("loops", help="Enumerate mixed feedback loops.")
    loops.add_argument("--admitted-edges", type=Path, required=True)
    loops.add_argument("--organs", nargs="+", required=True)
    loops.add_argument("--diseases", nargs="+", required=True)
    loops.add_argument("--max-length", type=int, default=8)
    loops.add_argument("--output", type=Path, required=True)

    comorbid = sub.add_parser("comorbidity", help="Summarize baseline comorbidity.")
    comorbid.add_argument("--input", type=Path, required=True)
    comorbid.add_argument("--disease-columns", nargs="+", required=True)
    comorbid.add_argument("--burden-output", type=Path, required=True)
    comorbid.add_argument("--pairs-output", type=Path, required=True)

    mortality = sub.add_parser("mortality", help="Adjusted all-cause mortality Cox models.")
    mortality.add_argument("--input", type=Path, required=True)
    mortality.add_argument("--exposures", nargs="+", required=True)
    mortality.add_argument("--duration-column", default="followup_years")
    mortality.add_argument("--event-column", default="death")
    mortality.add_argument("--covariates", nargs="+", default=["age", "sex"])
    mortality.add_argument("--no-standardize", action="store_true")
    mortality.add_argument("--min-events", type=int, default=100)
    mortality.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "o2d":
        result = prospective_organ_to_disease(
            read_table(args.input), args.organ_columns, read_json(args.onset_map),
            baseline_column=args.baseline_column, death_column=args.death_column,
            age_column=args.age_column, sex_column=args.sex_column,
            administrative_censoring=args.administrative_censoring,
            min_events=args.min_events,
        )
        write_table(result, args.output)
    elif args.command == "d2o":
        data = read_table(args.input)
        result = prevalent_disease_to_organ(
            data, args.organ_columns, read_json(args.onset_map),
            data[args.healthy_column].fillna(False).astype(bool),
            baseline_column=args.baseline_column, age_column=args.age_column,
            sex_column=args.sex_column, min_cases=args.min_cases,
        )
        write_table(result, args.output)
    elif args.command == "d2d":
        result = temporal_disease_to_disease(
            read_table(args.input), read_json(args.onset_map),
            minimum_lag_years=args.minimum_lag_years,
            min_directional_pairs=args.min_pairs,
            minimum_fraction=args.minimum_fraction,
        )
        write_table(result, args.output)
    elif args.command == "o2o":
        result = cross_lagged_organ_to_organ(
            read_table(args.input), args.organs,
            baseline_suffix=args.baseline_suffix, followup_suffix=args.followup_suffix,
            age_column=args.age_column, sex_column=args.sex_column,
            interval_column=args.interval_column, min_pairs=args.min_pairs,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        write_table(result, args.output)
    elif args.command == "admit":
        edges = pd.concat([read_table(path) for path in args.edges], ignore_index=True)
        rules = {
            edge_type: AdmissionRule(edge_type=edge_type, **parameters)
            for edge_type, parameters in read_json(args.rules).items()
        }
        write_table(admit_edges(edges, rules), args.output)
    elif args.command == "routes":
        write_table(maximum_strength_routes(
            read_table(args.admitted_edges), args.organs, args.diseases
        ), args.output)
    elif args.command == "loops":
        write_table(enumerate_feedback_loops(
            read_table(args.admitted_edges), args.organs, args.diseases,
            max_length=args.max_length,
        ), args.output)
    elif args.command == "comorbidity":
        data = read_table(args.input)
        burden, pairs = comorbidity_summary(data[args.disease_columns])
        write_table(burden.reset_index(drop=True), args.burden_output)
        write_table(pairs, args.pairs_output)
    elif args.command == "mortality":
        write_table(mortality_associations(
            read_table(args.input), args.exposures,
            duration_column=args.duration_column, event_column=args.event_column,
            covariate_columns=args.covariates,
            standardize_exposures=not args.no_standardize,
            min_events=args.min_events,
        ), args.output)


if __name__ == "__main__":
    main()

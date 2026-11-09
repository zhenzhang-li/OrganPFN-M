#!/usr/bin/env python
"""Run healthy-only five-fold fine-tuning for all five data thresholds.

The asymmetric Huber multiplier increases from 1 at each organ's healthy
median age to 4 at its healthy 95th-percentile age (lambda=3), only when the
prediction is below chronological age.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from organpfnm.cross_validation import organ_balanced_kfold, per_organ_age_ranges
from organpfnm_ft import config
from organpfnm_ft.train import train_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--ssl-checkpoint", type=Path,
                        default=Path("outputs/ssl/thr_90/final_ssl_checkpoint.pt"))
    parser.add_argument("--ssl-output-root", type=Path, default=Path("outputs/ssl"),
                        help="Root used with --match-ssl-threshold.")
    parser.add_argument("--match-ssl-threshold", action="store_true",
                        help="Initialize each FT threshold from the SSL checkpoint at the same threshold.")
    parser.add_argument("--tabpfn-checkpoint-dir", type=Path, default=Path("checkpoints/tabpfn"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/finetune"))
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 30, 50, 70, 90])
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--lambda-asym", type=float, default=3.0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config.TABPFN_CHECKPOINT_DIR = args.tabpfn_checkpoint_dir.resolve()
    for threshold in args.thresholds:
        config.SSL_CKPT_PATH = (
            (args.ssl_output_root / f"thr_{threshold}" / "final_ssl_checkpoint.pt").resolve()
            if args.match_ssl_threshold
            else args.ssl_checkpoint.resolve()
        )
        data_dir = (args.data_root / f"thr_{threshold}" / "finetune").resolve()
        config.set_data_dir(data_dir)
        data = pd.read_csv(data_dir / "finetune_subset.csv")
        with (data_dir / "schema.json").open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        feature_map = schema["organ_feature_map_excluding_sex"]
        presence_threshold = float(schema["filter_thresholds"]["finetune"]["organ_presence_threshold"])
        folds, report = organ_balanced_kfold(
            data, feature_map, n_splits=5, seed=args.seed,
            presence_threshold=presence_threshold,
        )
        age_low, age_high = per_organ_age_ranges(
            data, feature_map, presence_threshold=presence_threshold,
            lower_percentile=50, upper_percentile=95,
        )
        threshold_output = (args.output_root / f"thr_{threshold}").resolve()
        threshold_output.mkdir(parents=True, exist_ok=True)
        with (threshold_output / "fold_assignment.json").open("w", encoding="utf-8") as handle:
            json.dump({"folds": [sorted(fold) for fold in folds], "report": report}, handle, indent=2)
        with (threshold_output / "asymmetric_loss.json").open("w", encoding="utf-8") as handle:
            json.dump({"lambda": args.lambda_asym, "organs": list(feature_map),
                       "age_median": age_low, "age_95th_percentile": age_high}, handle, indent=2)

        for fold_index in args.folds:
            output = threshold_output / f"fold_{fold_index}"
            final = output / config.TRAIN_CFG.final_ckpt_name
            if final.exists() and not args.overwrite:
                print(f"[skip] threshold={threshold} fold={fold_index}: {final} exists")
                continue
            print(f"[FT] threshold={threshold}% fold={fold_index}/4 output={output}")
            train_run(
                output_dir=output,
                gpu_id=args.gpu_id,
                num_workers=args.num_workers,
                smoke_test=args.smoke_test,
                override_n_epochs=(1 if args.smoke_test else args.epochs),
                val_pids_override=folds[fold_index],
                asym=(float(args.lambda_asym), age_low, age_high),
            )


if __name__ == "__main__":
    main()

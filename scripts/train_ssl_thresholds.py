#!/usr/bin/env python
"""Run OrganPFN-M self-supervision for each missingness threshold."""

from __future__ import annotations

import argparse
from pathlib import Path

from organpfnm_ssl import config
from organpfnm_ssl.train import main as train_ssl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/ssl"))
    parser.add_argument("--tabpfn-checkpoint-dir", type=Path, default=Path("checkpoints/tabpfn"))
    parser.add_argument("--thresholds", type=int, nargs="+", default=[0, 30, 50, 70, 90])
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--nrows", type=int)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config.CHECKPOINT_DIR = args.tabpfn_checkpoint_dir.resolve()
    for threshold in args.thresholds:
        output = (args.output_root / f"thr_{threshold}").resolve()
        final = output / config.TRAIN_CFG.final_ckpt_name
        if final.exists() and not args.overwrite:
            print(f"[skip] threshold {threshold}: {final} exists")
            continue
        config.DATA_DIR = (args.data_root / f"thr_{threshold}" / "ssl").resolve()
        config.LINEAR_PROBE_DATA_DIR = (args.data_root / f"thr_{threshold}" / "finetune").resolve()
        print(f"[SSL] threshold={threshold}% data={config.DATA_DIR} output={output}")
        train_ssl(
            smoke_test=args.smoke_test,
            nrows=args.nrows,
            output_dir=output,
            num_workers=args.num_workers,
            gpu_id=args.gpu_id,
        )


if __name__ == "__main__":
    main()

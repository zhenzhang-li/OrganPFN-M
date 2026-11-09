#!/usr/bin/env python
"""Generate non-identifying data for interface and unit-testing demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORGANS = [
    "Body", "Hepatic", "Immune", "Lungs", "Metabolic", "Renal",
    "Brain", "Cardiovascular", "Cognitive", "Mental",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participants", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260512)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    feature_map = {organ: [f"{organ}_measurement_{i:03d}" for i in range(1, 4)] for organ in ORGANS}
    for participant in range(args.participants):
        sex = "F" if rng.random() < 0.52 else "M"
        healthy = int(rng.random() < 0.35)
        baseline_age = rng.uniform(40, 75)
        visits = ["T0", "T2"] if rng.random() < 0.65 else [rng.choice(["T0", "T2"])]
        latent = rng.normal()
        for visit in visits:
            age = baseline_age + (rng.uniform(2, 8) if visit == "T2" else 0)
            row = {
                "Participant_ID": f"SYN{participant:06d}",
                "Sex": sex,
                "timepoint": visit,
                "timepoint_id": 0 if visit == "T0" else 1,
                "Age": age,
                "Disease": "healthy" if healthy else "synthetic_condition",
                "is_healthy": healthy,
            }
            for organ_index, organ in enumerate(ORGANS):
                organ_available = rng.random() > (0.08 + 0.04 * (organ_index % 3))
                for column in feature_map[organ]:
                    value = age / 10 + latent + organ_index / 10 + rng.normal(scale=0.7)
                    row[column] = value if organ_available and rng.random() > 0.12 else np.nan
            rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "harmonized_visits.csv", index=False)
    with (args.output_dir / "feature_map.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_map, handle, indent=2)
    print(f"wrote {len(rows):,} synthetic participant-visit rows to {args.output_dir}")


if __name__ == "__main__":
    main()

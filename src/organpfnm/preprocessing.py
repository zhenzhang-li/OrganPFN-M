"""Create threshold-specific OrganPFN-M inputs without exposing UKB data.

The public contract is a harmonized participant-visit table. T0 and T2 are
kept as separate rows even when they share a participant identifier. Each row
contains one sample-level ``Sex`` value and any available measurements from ten
organ systems. Missing organs and missing measurements remain ``NaN``.

The model internally broadcasts the single Sex value to its organ blocks. The
broadcast columns are recorded in ``schema.json`` for checkpoint compatibility;
they are not additional biological variables and are not written to the public
CSV files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_THRESHOLDS: tuple[int, ...] = (0, 30, 50, 70, 90)
REQUIRED_METADATA: tuple[str, ...] = (
    "Participant_ID",
    "Sex",
    "timepoint",
    "timepoint_id",
    "Age",
    "Disease",
    "is_healthy",
)


@dataclass(frozen=True)
class PreparationSummary:
    threshold: int
    n_input_rows: int
    n_retained_rows: int
    n_ssl_rows: int
    n_finetune_rows: int
    per_organ_present: dict[str, int]


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".csv.gz"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}. Use CSV or Parquet.")


def load_feature_map(path: Path) -> dict[str, list[str]]:
    """Load ``{organ: [measurement columns]}`` from a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if "organ_feat_cols" in raw:  # accepts the study merge schema directly
        raw = raw["organ_feat_cols"]
    result = {str(organ): [str(col) for col in cols] for organ, cols in raw.items()}
    if not result or any(not cols for cols in result.values()):
        raise ValueError("Every organ must have at least one measurement column.")
    flat = [col for cols in result.values() for col in cols]
    if len(flat) != len(set(flat)):
        raise ValueError("A measurement column may belong to only one organ.")
    return result


def harmonize_visit_rows(
    t0: pd.DataFrame,
    t2: pd.DataFrame,
    *,
    metadata_columns: Sequence[str] = REQUIRED_METADATA,
) -> pd.DataFrame:
    """Stack T0 and T2 as independent participant-visit observations.

    This is row-wise concatenation, not a participant-level join. Repeated
    participant identifiers therefore remain as distinct visit records.
    """
    frames: list[pd.DataFrame] = []
    for frame, label, tp_id in ((t0, "T0", 0), (t2, "T2", 1)):
        current = frame.copy()
        current["timepoint"] = label
        current["timepoint_id"] = tp_id
        frames.append(current)
    merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    for col in metadata_columns:
        if col not in merged:
            merged[col] = np.nan
    if merged["Participant_ID"].isna().any():
        raise ValueError("Participant_ID contains missing values.")
    if merged["Sex"].isna().any():
        raise ValueError("Sex contains missing values; resolve these before release training.")
    return merged


def _resolve_table(path: str | Path, base_dir: Path) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = base_dir / resolved
    return _read_table(resolved)


def assemble_visits_from_manifest(manifest_path: Path) -> pd.DataFrame:
    """Assemble organ-specific T0/T2 files using a non-sensitive manifest.

    Each visit must define a metadata table and may define any number of organ
    tables. Source-to-public column renames are explicit, preventing accidental
    inclusion of unlisted UKB fields. The metadata table is the participant
    spine and organ tables are joined one-to-one by participant ID.
    """
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    base_dir = manifest_path.parent
    participant = manifest.get("participant_column", "Participant_ID")
    frames: list[pd.DataFrame] = []
    for visit_name, visit in manifest["visits"].items():
        metadata_spec = visit["metadata"]
        metadata = _resolve_table(metadata_spec["path"], base_dir)
        metadata = metadata.rename(columns=metadata_spec.get("columns", {}))
        if participant not in metadata:
            raise ValueError(f"{visit_name} metadata lacks {participant!r}.")
        if metadata[participant].duplicated().any():
            raise ValueError(f"{visit_name} metadata has duplicate participant IDs.")
        assembled = metadata.copy()
        for organ, specification in visit.get("organ_tables", {}).items():
            organ_table = _resolve_table(specification["path"], base_dir)
            organ_table = organ_table.rename(columns=specification.get("columns", {}))
            if participant not in organ_table:
                raise ValueError(f"{visit_name}/{organ} lacks {participant!r}.")
            if organ_table[participant].duplicated().any():
                raise ValueError(f"{visit_name}/{organ} has duplicate participant IDs.")
            keep = [participant, *specification.get("keep", [])]
            if len(keep) == 1:
                keep = list(organ_table.columns)
            missing = [column for column in keep if column not in organ_table]
            if missing:
                raise ValueError(f"{visit_name}/{organ} lacks configured columns: {missing}")
            overlap = (set(assembled.columns) & set(keep)) - {participant}
            if overlap:
                raise ValueError(
                    f"{visit_name}/{organ} would overwrite columns: {sorted(overlap)}"
                )
            assembled = assembled.merge(
                organ_table[keep], on=participant, how="left", validate="one_to_one"
            )
        assembled["timepoint"] = visit_name
        assembled["timepoint_id"] = int(visit["timepoint_id"])
        frames.append(assembled)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    rename = manifest.get("final_column_renames", {})
    combined = combined.rename(columns=rename)
    missing = [column for column in REQUIRED_METADATA if column not in combined]
    if missing:
        raise ValueError(f"Manifest output lacks required metadata: {missing}")
    return combined


def organ_presence(
    table: pd.DataFrame,
    feature_map: Mapping[str, Sequence[str]],
    threshold: int,
) -> pd.DataFrame:
    """Return organ availability under a maximum missingness threshold.

    For threshold ``q``, an organ is available when its within-organ fraction
    of missing measurement features is at most ``q / 100``. Sex is excluded
    from this calculation.
    """
    if threshold < 0 or threshold > 100:
        raise ValueError("threshold must be between 0 and 100")
    allowed = threshold / 100.0
    result: dict[str, np.ndarray] = {}
    for organ, columns in feature_map.items():
        missing = table[list(columns)].isna().mean(axis=1).to_numpy()
        result[organ] = missing <= allowed + 1e-12
    return pd.DataFrame(result, index=table.index)


def apply_missingness_threshold(
    table: pd.DataFrame,
    feature_map: Mapping[str, Sequence[str]],
    threshold: int,
    *,
    min_organs: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank non-qualifying organ blocks and retain rows with >= min_organs."""
    presence = organ_presence(table, feature_map, threshold)
    retained = presence.sum(axis=1) >= int(min_organs)
    out = table.loc[retained].copy().reset_index(drop=True)
    kept_presence = presence.loc[retained].reset_index(drop=True)
    for organ, columns in feature_map.items():
        absent = ~kept_presence[organ].to_numpy()
        if absent.any():
            out.loc[absent, list(columns)] = np.nan
    return out, kept_presence


def build_training_schema(
    feature_map: Mapping[str, Sequence[str]],
    *,
    threshold: int,
    min_organs: int = 2,
    metadata_columns: Sequence[str] = REQUIRED_METADATA,
    ssl_presence_threshold: float = 0.30,
    finetune_presence_threshold: float = 0.50,
) -> dict:
    """Build the schema used by both self-supervision and fine-tuning."""
    organ_names = list(feature_map)
    feature_columns: list[str] = []
    organ_feature_map: dict[str, list[str]] = {}
    offsets: dict[str, list[int]] = {}
    sex_columns: dict[str, str] = {}
    sex_indices: dict[str, int] = {}
    n_per_organ: dict[str, int] = {}
    n_per_organ_excluding_sex: dict[str, int] = {}
    cursor = 0
    for organ in organ_names:
        sex_col = f"__model_{organ}_Sex"
        block = [sex_col, *list(feature_map[organ])]
        sex_columns[organ] = sex_col
        sex_indices[organ] = cursor
        organ_feature_map[organ] = block
        offsets[organ] = [cursor, cursor + len(block)]
        n_per_organ[organ] = len(block)
        n_per_organ_excluding_sex[organ] = len(block) - 1
        feature_columns.extend(block)
        cursor += len(block)
    return {
        "schema_version": "1.0",
        "organ_names": organ_names,
        "measurement_feature_columns": [
            col for organ in organ_names for col in feature_map[organ]
        ],
        "feature_columns": feature_columns,
        "organ_feature_map": organ_feature_map,
        "organ_feature_map_excluding_sex": {
            organ: list(feature_map[organ]) for organ in organ_names
        },
        "organ_offsets": offsets,
        "n_features_per_organ": n_per_organ,
        "n_features_per_organ_excluding_sex": n_per_organ_excluding_sex,
        "n_measurement_features": sum(n_per_organ_excluding_sex.values()),
        "n_total_features": len(feature_columns),
        "sex_source_column": "Sex",
        "sex_column_per_organ": sex_columns,
        "sex_column_indices_per_organ": sex_indices,
        "metadata_columns": list(metadata_columns),
        "timepoint_id_map": {"T0": 0, "T2": 1},
        "n_timepoints": 2,
        "organ_timepoint_availability": {
            organ: ["T0", "T2"] for organ in organ_names
        },
        "filter_thresholds": {
            "ssl": {
                "organ_presence_threshold": float(ssl_presence_threshold),
                "max_missing_fraction": threshold / 100.0,
                "min_organs": int(min_organs),
                "presence_excludes_sex": True,
            },
            "finetune": {
                "organ_presence_threshold": float(finetune_presence_threshold),
                "max_missing_fraction": threshold / 100.0,
                "min_organs": int(min_organs),
                "presence_excludes_sex": True,
                "healthy_only": True,
            },
        },
        "per_organ_effective_samples": {
            "ssl": {organ: 0 for organ in organ_names},
            "finetune": {organ: 0 for organ in organ_names},
        },
        "notes": {
            "visit_representation": "T0 and T2 are independent sample rows.",
            "sex_representation": (
                "One sample-level Sex field is broadcast internally to organ blocks; "
                "broadcast columns are implementation slots, not independent features."
            ),
            "threshold_distinction": (
                "max_missing_fraction defines the five preprocessing datasets; "
                "organ_presence_threshold is the fixed observed-feature ratio used "
                "inside the original SSL or fine-tuning loader."
            ),
        },
    }


def make_participant_pairs(table: pd.DataFrame) -> pd.DataFrame:
    """List participants with both T0 and T2 for longitudinal regularization."""
    required = {"Participant_ID", "timepoint", "Age"}
    if not required.issubset(table.columns):
        return pd.DataFrame(columns=["Participant_ID", "Age_T0", "Age_T2", "Age_diff"])
    work = table[["Participant_ID", "timepoint", "Age"]].dropna(subset=["Participant_ID"])
    wide = work.pivot_table(
        index="Participant_ID", columns="timepoint", values="Age", aggfunc="first"
    )
    if not {"T0", "T2"}.issubset(wide.columns):
        return pd.DataFrame(columns=["Participant_ID", "Age_T0", "Age_T2", "Age_diff"])
    pairs = wide.dropna(subset=["T0", "T2"]).reset_index()
    pairs = pairs.rename(columns={"T0": "Age_T0", "T2": "Age_T2"})
    pairs["Age_diff"] = pairs["Age_T2"] - pairs["Age_T0"]
    return pairs[["Participant_ID", "Age_T0", "Age_T2", "Age_diff"]]


def _write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def prepare_threshold_datasets(
    table: pd.DataFrame,
    feature_map: Mapping[str, Sequence[str]],
    output_root: Path,
    *,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
    min_organs: int = 2,
) -> list[PreparationSummary]:
    """Write all-sample SSL and healthy-only fine-tuning sets for each threshold."""
    missing_meta = [col for col in REQUIRED_METADATA if col not in table.columns]
    if missing_meta:
        raise ValueError(f"Missing required metadata columns: {missing_meta}")
    missing_features = [
        col for cols in feature_map.values() for col in cols if col not in table.columns
    ]
    if missing_features:
        raise ValueError(f"Missing measurement columns: {missing_features[:10]}")

    ordered_columns = [
        *REQUIRED_METADATA,
        *[col for cols in feature_map.values() for col in cols],
    ]
    base = table.loc[:, ordered_columns].copy()
    summaries: list[PreparationSummary] = []
    for threshold in thresholds:
        threshold = int(threshold)
        filtered, presence = apply_missingness_threshold(
            base, feature_map, threshold, min_organs=min_organs
        )
        schema = build_training_schema(
            feature_map, threshold=threshold, min_organs=min_organs
        )
        target = Path(output_root) / f"thr_{threshold}"
        ssl_dir = target / "ssl"
        ft_dir = target / "finetune"
        _write_csv(filtered, ssl_dir / "ssl_subset.csv")
        healthy = filtered.loc[
            filtered["is_healthy"].fillna(0).astype(int) == 1
        ].reset_index(drop=True)
        schema["per_organ_effective_samples"] = {
            "ssl": {
                organ: int((filtered[list(columns)].notna().mean(axis=1) >= 0.30).sum())
                for organ, columns in feature_map.items()
            },
            "finetune": {
                organ: int((healthy[list(columns)].notna().mean(axis=1) >= 0.50).sum())
                for organ, columns in feature_map.items()
            },
        }
        _write_csv(healthy, ft_dir / "finetune_subset.csv")
        make_participant_pairs(healthy).to_csv(
            ft_dir / "participant_pairs.csv", index=False
        )
        for directory in (ssl_dir, ft_dir):
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "schema.json").open("w", encoding="utf-8") as handle:
                json.dump(schema, handle, ensure_ascii=False, indent=2)
        summary = PreparationSummary(
            threshold=threshold,
            n_input_rows=len(base),
            n_retained_rows=len(filtered),
            n_ssl_rows=len(filtered),
            n_finetune_rows=len(healthy),
            per_organ_present={organ: int(presence[organ].sum()) for organ in presence},
        )
        with (target / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary.__dict__, handle, ensure_ascii=False, indent=2)
        summaries.append(summary)
    return summaries


def prepare_from_files(
    input_path: Path,
    feature_map_path: Path,
    output_root: Path,
    *,
    thresholds: Iterable[int] = DEFAULT_THRESHOLDS,
    min_organs: int = 2,
) -> list[PreparationSummary]:
    return prepare_threshold_datasets(
        _read_table(input_path),
        load_feature_map(feature_map_path),
        output_root,
        thresholds=thresholds,
        min_organs=min_organs,
    )

"""Participant-level, organ-balanced cross-validation helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def participant_organ_presence(
    data: pd.DataFrame,
    organ_feature_map: Mapping[str, Sequence[str]],
    *,
    presence_threshold: float = 0.5,
) -> pd.DataFrame:
    presence: dict[str, np.ndarray] = {}
    for organ, columns in organ_feature_map.items():
        available = [column for column in columns if column in data]
        if not available:
            presence[organ] = np.zeros(len(data), dtype=np.int8)
            continue
        ratio = data[available].notna().mean(axis=1).to_numpy()
        presence[organ] = (ratio >= presence_threshold).astype(np.int8)
    frame = pd.DataFrame(presence)
    frame["Participant_ID"] = data["Participant_ID"].astype(str).to_numpy()
    return frame.groupby("Participant_ID", sort=False).max()


def organ_balanced_kfold(
    data: pd.DataFrame,
    organ_feature_map: Mapping[str, Sequence[str]],
    *,
    n_splits: int = 5,
    seed: int = 20260512,
    presence_threshold: float = 0.5,
) -> tuple[list[set[str]], dict]:
    """Assign participants while preferentially balancing rarer organs."""
    presence = participant_organ_presence(
        data, organ_feature_map, presence_threshold=presence_threshold
    )
    organs = list(organ_feature_map)
    totals = presence[organs].sum().sort_values()
    organ_index = {organ: index for index, organ in enumerate(organs)}
    rng = np.random.default_rng(seed)
    participants = presence.index.to_list()
    rng.shuffle(participants)

    rarity_order = list(totals.index)

    def rarity(participant: str) -> int:
        observed = [organ for organ in rarity_order if presence.loc[participant, organ] > 0]
        return rarity_order.index(observed[0]) if observed else len(rarity_order)

    participants.sort(key=rarity)
    fold_counts = np.zeros((n_splits, len(organs)), dtype=np.int64)
    fold_sizes = np.zeros(n_splits, dtype=np.int64)
    assignments: dict[str, int] = {}
    for participant in participants:
        vector = presence.loc[participant, organs].to_numpy(dtype=np.int64)
        observed = [organ for organ in rarity_order if presence.loc[participant, organ] > 0]
        if observed:
            rare_column = organ_index[observed[0]]
            fold = int(np.lexsort((fold_sizes, fold_counts[:, rare_column]))[0])
        else:
            fold = int(np.argmin(fold_sizes))
        assignments[participant] = fold
        fold_counts[fold] += vector
        fold_sizes[fold] += 1
    folds = [
        {participant for participant, fold in assignments.items() if fold == index}
        for index in range(n_splits)
    ]
    report = {
        "n_splits": n_splits,
        "seed": seed,
        "presence_threshold": presence_threshold,
        "participants_per_fold": fold_sizes.tolist(),
        "organ_counts_per_fold": {
            organ: fold_counts[:, index].tolist()
            for index, organ in enumerate(organs)
        },
    }
    return folds, report


def per_organ_age_ranges(
    data: pd.DataFrame,
    organ_feature_map: Mapping[str, Sequence[str]],
    *,
    presence_threshold: float = 0.5,
    lower_percentile: float = 50,
    upper_percentile: float = 95,
) -> tuple[list[float], list[float]]:
    """Age anchors for the organ-specific asymmetric regression loss."""
    age = pd.to_numeric(data["Age"], errors="coerce").to_numpy()
    lower, upper = [], []
    for organ, columns in organ_feature_map.items():
        ratio = data[list(columns)].notna().mean(axis=1).to_numpy()
        values = age[(ratio >= presence_threshold) & np.isfinite(age)]
        if len(values) < 20:
            raise ValueError(f"Too few age observations for {organ!r}: {len(values)}")
        lower.append(float(np.percentile(values, lower_percentile)))
        upper.append(float(np.percentile(values, upper_percentile)))
    return lower, upper

"""Panel-Invariant Organ-ageing Asynchrony (PIOAA).

PIOAA separates an individual's multiorgan age-gap profile into a shared
directional magnitude and a residual asynchrony component. It supports partial
organ panels and empirically calibrates both scores within the number of
observed organs, reducing dependence on panel size.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cho_factor, cho_solve


def _nearest_correlation(matrix: np.ndarray, eigen_floor: float = 1e-4) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    positive = vectors @ np.diag(np.clip(values, eigen_floor, None)) @ vectors.T
    scale = np.sqrt(np.clip(np.diag(positive), eigen_floor, None))
    result = positive / np.outer(scale, scale)
    np.fill_diagonal(result, 1.0)
    return result


def _empirical_normal(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(reference[np.isfinite(reference)])
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if len(reference) == 0:
        return output
    probability = (
        np.searchsorted(reference, values[valid], side="right") + 0.5
    ) / (len(reference) + 1.0)
    lower = 1.0 / (len(reference) + 1.0)
    output[valid] = stats.norm.ppf(np.clip(probability, lower, 1.0 - lower))
    return output


@dataclass
class PIOAA:
    """Estimate panel-invariant magnitude and asynchrony scores.

    Parameters mirror the analysis used in the study. ``min_pairwise_n`` is
    the minimum healthy-reference overlap for estimating an organ pair;
    ``shrinkage`` stabilizes the latent correlation matrix; and
    ``min_calibration_n`` is the minimum healthy-reference size at each panel
    size for empirical score calibration.
    """

    organ_names: Sequence[str]
    shrinkage: float = 0.10
    min_pairwise_n: int = 30
    min_calibration_n: int = 50
    eigen_floor: float = 1e-4

    def __post_init__(self) -> None:
        self.organ_names = tuple(self.organ_names)
        if len(self.organ_names) < 2:
            raise ValueError("PIOAA requires at least two organs.")
        if not 0 <= self.shrinkage < 1:
            raise ValueError("shrinkage must be in [0, 1).")
        self.reference_marginals_: list[np.ndarray] | None = None
        self.correlation_: np.ndarray | None = None
        self.magnitude_null_: dict[int, np.ndarray] | None = None
        self.asynchrony_null_: dict[int, np.ndarray] | None = None

    def _matrix(self, age_gaps: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(age_gaps, pd.DataFrame):
            missing = [o for o in self.organ_names if o not in age_gaps]
            if missing:
                raise ValueError(f"Missing organ age-gap columns: {missing}")
            return age_gaps.loc[:, self.organ_names].to_numpy(dtype=float)
        matrix = np.asarray(age_gaps, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.organ_names):
            raise ValueError("age_gaps has an incompatible shape.")
        return matrix

    def _fit_marginals(self, matrix: np.ndarray, reference: np.ndarray) -> None:
        marginals: list[np.ndarray] = []
        for column in range(matrix.shape[1]):
            values = matrix[reference & np.isfinite(matrix[:, column]), column]
            values = np.sort(values)
            if len(values) < self.min_pairwise_n:
                raise ValueError(
                    f"Organ {self.organ_names[column]!r} has only {len(values)} "
                    "healthy-reference observations."
                )
            marginals.append(values)
        self.reference_marginals_ = marginals

    def _copula_transform(self, matrix: np.ndarray) -> np.ndarray:
        if self.reference_marginals_ is None:
            raise RuntimeError("Call fit before transform.")
        transformed = np.full(matrix.shape, np.nan, dtype=float)
        for column, reference in enumerate(self.reference_marginals_):
            transformed[:, column] = _empirical_normal(matrix[:, column], reference)
        return transformed

    def _fit_correlation(self, matrix: np.ndarray, reference: np.ndarray) -> None:
        n_organs = matrix.shape[1]
        rank_correlation = np.eye(n_organs)
        for left in range(n_organs):
            for right in range(left + 1, n_organs):
                valid = (
                    reference
                    & np.isfinite(matrix[:, left])
                    & np.isfinite(matrix[:, right])
                )
                if valid.sum() < self.min_pairwise_n:
                    continue
                rho = stats.spearmanr(
                    matrix[valid, left], matrix[valid, right]
                ).correlation
                if np.isfinite(rho):
                    latent = 2.0 * np.sin(np.pi * float(rho) / 6.0)
                    rank_correlation[left, right] = latent
                    rank_correlation[right, left] = latent
        correlation = _nearest_correlation(rank_correlation, self.eigen_floor)
        correlation = (
            (1.0 - self.shrinkage) * correlation
            + self.shrinkage * np.eye(n_organs)
        )
        np.fill_diagonal(correlation, 1.0)
        self.correlation_ = correlation

    def _decompose(self, z_scores: np.ndarray) -> pd.DataFrame:
        if self.correlation_ is None:
            raise RuntimeError("Call fit before transform.")
        n_rows = len(z_scores)
        k = np.isfinite(z_scores).sum(axis=1).astype(int)
        total = np.full(n_rows, np.nan)
        magnitude = np.full(n_rows, np.nan)
        common_level = np.full(n_rows, np.nan)
        asynchrony = np.full(n_rows, np.nan)
        patterns: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for row in range(n_rows):
            patterns[tuple(np.flatnonzero(np.isfinite(z_scores[row])))] .append(row)
        for columns, rows in patterns.items():
            if len(columns) < 2:
                continue
            selected = list(columns)
            covariance = self.correlation_[np.ix_(selected, selected)]
            factor = cho_factor(covariance, lower=True, check_finite=False)
            z = z_scores[np.ix_(rows, selected)]
            solved = cho_solve(factor, z.T, check_finite=False)
            total_values = np.einsum("ij,ji->i", z, solved)
            ones = np.ones(len(selected))
            precision_ones = cho_solve(factor, ones, check_finite=False)
            scale = float(ones @ precision_ones)
            level_values = (z @ precision_ones) / scale
            magnitude_values = scale * level_values**2
            total[rows] = total_values
            common_level[rows] = level_values
            magnitude[rows] = magnitude_values
            asynchrony[rows] = np.maximum(total_values - magnitude_values, 0.0)
        return pd.DataFrame({
            "observed_organs": k,
            "total_deviation": total,
            "magnitude": magnitude,
            "common_level": common_level,
            "asynchrony": asynchrony,
        })

    def fit(
        self,
        age_gaps: pd.DataFrame | np.ndarray,
        healthy_reference: Sequence[bool] | np.ndarray,
    ) -> "PIOAA":
        matrix = self._matrix(age_gaps)
        reference = np.asarray(healthy_reference, dtype=bool)
        if reference.shape != (len(matrix),):
            raise ValueError("healthy_reference must have one value per row.")
        self._fit_marginals(matrix, reference)
        self._fit_correlation(matrix, reference)
        components = self._decompose(self._copula_transform(matrix))
        self.magnitude_null_ = {}
        self.asynchrony_null_ = {}
        for panel_size in sorted(components["observed_organs"].unique()):
            selected = reference & (components["observed_organs"].to_numpy() == panel_size)
            magnitude = components.loc[selected, "magnitude"].dropna().to_numpy()
            asynchrony = components.loc[selected, "asynchrony"].dropna().to_numpy()
            if len(magnitude) >= self.min_calibration_n:
                self.magnitude_null_[int(panel_size)] = np.sort(magnitude)
            if len(asynchrony) >= self.min_calibration_n:
                self.asynchrony_null_[int(panel_size)] = np.sort(asynchrony)
        return self

    def transform(self, age_gaps: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if self.magnitude_null_ is None or self.asynchrony_null_ is None:
            raise RuntimeError("Call fit before transform.")
        matrix = self._matrix(age_gaps)
        result = self._decompose(self._copula_transform(matrix))
        s_magnitude = np.full(len(result), np.nan)
        s_asynchrony = np.full(len(result), np.nan)
        k = result["observed_organs"].to_numpy(dtype=int)
        for panel_size in np.unique(k):
            rows = k == panel_size
            if int(panel_size) in self.magnitude_null_:
                s_magnitude[rows] = _empirical_normal(
                    result.loc[rows, "magnitude"].to_numpy(),
                    self.magnitude_null_[int(panel_size)],
                )
            if int(panel_size) in self.asynchrony_null_:
                s_asynchrony[rows] = _empirical_normal(
                    result.loc[rows, "asynchrony"].to_numpy(),
                    self.asynchrony_null_[int(panel_size)],
                )
        result["s_magnitude"] = s_magnitude
        result["s_asynchrony"] = s_asynchrony
        return result

    def fit_transform(
        self,
        age_gaps: pd.DataFrame | np.ndarray,
        healthy_reference: Sequence[bool] | np.ndarray,
    ) -> pd.DataFrame:
        return self.fit(age_gaps, healthy_reference).transform(age_gaps)


def aggregate_age_gaps_by_participant(
    table: pd.DataFrame,
    organ_names: Sequence[str],
    *,
    participant_column: str = "Participant_ID",
) -> pd.DataFrame:
    """Combine available visit-specific organ age gaps into one profile.

    Repeated measurements of the same organ across T0 and T2 are averaged;
    visit-specific organs contribute their available value. This operation is
    for the downstream PIOAA analysis only and does not alter training samples.
    """
    required = {participant_column, *organ_names}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return (
        table[[participant_column, *organ_names]]
        .groupby(participant_column, sort=False, as_index=False)
        .mean(numeric_only=True)
    )

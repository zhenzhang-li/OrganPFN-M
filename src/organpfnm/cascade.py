"""Organ Aging-Disease Propagation Cascade analysis.

This module implements the four evidence-defined directed edge families used
in the study, their admission rules, route construction, feedback-loop
enumeration, and basic comorbidity summaries. It deliberately labels routes as
observational propagation patterns rather than causal pathways.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment with NaN support."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return q
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * len(order) / np.arange(1, len(order) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def _zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    scale = np.nanstd(x, ddof=1)
    return (x - np.nanmean(x)) / scale if scale > 0 else np.full_like(x, np.nan)


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    yv, xv = y[valid], x[valid]
    if len(yv) <= xv.shape[1]:
        raise ValueError("Insufficient complete observations for regression.")
    beta, _, _, _ = np.linalg.lstsq(xv, yv, rcond=None)
    residual = yv - xv @ beta
    dof = len(yv) - xv.shape[1]
    variance = float(residual @ residual) / dof
    covariance = variance * np.linalg.pinv(xv.T @ xv)
    return beta, np.sqrt(np.clip(np.diag(covariance), 0, None)), dof


def prospective_organ_to_disease(
    data: pd.DataFrame,
    organ_columns: Sequence[str],
    onset_columns: Mapping[str, str],
    *,
    baseline_column: str,
    death_column: str,
    age_column: str,
    sex_column: str,
    administrative_censoring: float,
    min_events: int = 300,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Estimate organ-to-future-disease edges using age/sex-adjusted Cox models."""
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover - dependency message
        raise ImportError("Install the 'cascade' extra to run Cox models.") from exc
    rows: list[dict] = []
    for disease, onset_column in onset_columns.items():
        onset = pd.to_numeric(data[onset_column], errors="coerce")
        baseline = pd.to_numeric(data[baseline_column], errors="coerce")
        death = pd.to_numeric(data[death_column], errors="coerce")
        eligible = ~(onset.notna() & baseline.notna() & (onset < baseline))
        subset = data.loc[eligible].copy()
        onset = onset.loc[eligible]
        baseline = baseline.loc[eligible]
        death = death.loc[eligible]
        disease_time = np.where(onset.notna() & (onset >= baseline), onset, np.inf)
        censor_time = np.minimum(death.fillna(np.inf), administrative_censoring)
        subset["__event"] = (disease_time <= censor_time).astype(int)
        subset["__duration"] = np.minimum(disease_time, censor_time) - baseline
        subset = subset.loc[subset["__duration"] > 0]
        for organ in organ_columns:
            model_data = subset[["__duration", "__event", age_column, sex_column, organ]].dropna()
            n_events = int(model_data["__event"].sum())
            if n_events < min_events or model_data[organ].std() == 0:
                rows.append({"source": organ, "target": disease, "effect": np.nan,
                             "p": np.nan, "count": n_events, "edge_type": "O2D"})
                continue
            model_data = model_data.rename(columns={age_column: "age", sex_column: "sex", organ: "exposure"})
            model_data["exposure"] = _zscore(model_data["exposure"])
            try:
                model = CoxPHFitter().fit(model_data, "__duration", "__event")
                summary = model.summary.loc["exposure"]
                effect, p_value = float(np.exp(summary["coef"])), float(summary["p"])
            except Exception:
                effect, p_value = np.nan, np.nan
            rows.append({"source": organ, "target": disease, "effect": effect,
                         "p": p_value, "count": n_events, "edge_type": "O2D"})
    result = pd.DataFrame(rows)
    result["q"] = benjamini_hochberg(result["p"])
    result["eligible"] = (
        (result["effect"] > 1) & (result["q"] < fdr) & (result["count"] >= min_events)
    )
    return result


def prevalent_disease_to_organ(
    data: pd.DataFrame,
    organ_columns: Sequence[str],
    onset_columns: Mapping[str, str],
    healthy_reference: Sequence[bool],
    *,
    baseline_column: str,
    age_column: str,
    sex_column: str,
    min_cases: int = 200,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Estimate disease-to-organ edges with adjusted standardized contrasts."""
    healthy = np.asarray(healthy_reference, dtype=bool)
    rows: list[dict] = []
    baseline = pd.to_numeric(data[baseline_column], errors="coerce").to_numpy()
    for disease, onset_column in onset_columns.items():
        onset = pd.to_numeric(data[onset_column], errors="coerce").to_numpy()
        prevalent = np.isfinite(onset) & np.isfinite(baseline) & (onset < baseline)
        for organ in organ_columns:
            selected = prevalent | healthy
            case = prevalent[selected].astype(float)
            y = pd.to_numeric(data.loc[selected, organ], errors="coerce").to_numpy()
            age = pd.to_numeric(data.loc[selected, age_column], errors="coerce").to_numpy()
            sex = pd.to_numeric(data.loc[selected, sex_column], errors="coerce").to_numpy()
            complete = np.isfinite(y) & np.isfinite(age) & np.isfinite(sex)
            n_cases = int(case[complete].sum())
            try:
                x = np.column_stack([np.ones(selected.sum()), case, age, sex])
                beta, se, dof = _ols(y, x)
                residual_sd = np.nanstd(y[complete] - x[complete] @ beta, ddof=x.shape[1])
                effect = float(beta[1] / residual_sd)
                p_value = float(2 * stats.t.sf(abs(beta[1] / se[1]), dof))
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
                effect, p_value = np.nan, np.nan
            rows.append({"source": disease, "target": organ, "effect": effect,
                         "p": p_value, "count": n_cases, "edge_type": "D2O"})
    result = pd.DataFrame(rows)
    result["q"] = benjamini_hochberg(result["p"])
    result["eligible"] = (
        (result["effect"] > 0) & (result["q"] < fdr) & (result["count"] >= min_cases)
    )
    return result


def temporal_disease_to_disease(
    data: pd.DataFrame,
    onset_columns: Mapping[str, str],
    *,
    minimum_lag_years: float = 0.5,
    min_directional_pairs: int = 100,
    minimum_fraction: float = 0.55,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Estimate disease ordering using clearly separated first-onset dates."""
    rows: list[dict] = []
    for source, target in combinations(onset_columns, 2):
        a = pd.to_numeric(data[onset_columns[source]], errors="coerce").to_numpy()
        b = pd.to_numeric(data[onset_columns[target]], errors="coerce").to_numpy()
        valid = np.isfinite(a) & np.isfinite(b)
        difference = b[valid] - a[valid]
        a_before = int((difference >= minimum_lag_years).sum())
        b_before = int((difference <= -minimum_lag_years).sum())
        directional = a_before + b_before
        if directional < min_directional_pairs:
            continue
        for left, right, count in ((source, target, a_before), (target, source, b_before)):
            fraction = count / directional
            p_value = stats.binomtest(
                count, directional, p=0.5, alternative="greater"
            ).pvalue
            rows.append({"source": left, "target": right, "effect": fraction,
                         "p": p_value, "count": directional, "edge_type": "D2D"})
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["source", "target", "effect", "p", "count", "edge_type", "q", "eligible"])
    result["q"] = benjamini_hochberg(result["p"])
    result["eligible"] = (
        (result["effect"] > minimum_fraction)
        & (result["q"] < fdr)
        & (result["count"] >= min_directional_pairs)
    )
    return result


def cross_lagged_organ_to_organ(
    paired: pd.DataFrame,
    organ_names: Sequence[str],
    *,
    baseline_suffix: str = "_baseline",
    followup_suffix: str = "_followup",
    age_column: str = "age_baseline",
    sex_column: str = "sex",
    interval_column: str = "followup_years",
    min_pairs: int = 1000,
    n_bootstrap: int = 200,
    seed: int = 20260701,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Estimate cross-lagged organ-to-organ associations on paired visits."""
    baseline_cols = [f"{o}{baseline_suffix}" for o in organ_names]
    followup_cols = [f"{o}{followup_suffix}" for o in organ_names]
    required = [*baseline_cols, *followup_cols, age_column, sex_column, interval_column]
    work = paired.dropna(subset=required).copy()
    work = work.loc[work[interval_column] > 0.5].reset_index(drop=True)
    if len(work) < min_pairs:
        return pd.DataFrame(columns=["source", "target", "effect", "p", "count", "ci_low", "ci_high", "edge_type", "q", "eligible"])
    age = work[age_column].to_numpy(float)
    sex = work[sex_column].to_numpy(float)
    base = work[baseline_cols].to_numpy(float)
    followup = work[followup_cols].to_numpy(float)

    def residualize(panel: np.ndarray) -> np.ndarray:
        common = np.nanmean(panel, axis=1)
        result = np.full_like(panel, np.nan)
        for column in range(panel.shape[1]):
            x = np.column_stack([np.ones(len(panel)), age, sex, common])
            beta, _, _ = _ols(panel[:, column], x)
            result[:, column] = panel[:, column] - x @ beta
        return result

    base_residual = residualize(base)
    followup_residual = residualize(followup)
    interval = work[interval_column].to_numpy(float)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for target_index, target in enumerate(organ_names):
        change = _zscore((followup_residual[:, target_index] - base_residual[:, target_index]) / interval)
        target_baseline = _zscore(base_residual[:, target_index])
        for source_index, source in enumerate(organ_names):
            if source == target:
                continue
            source_baseline = _zscore(base_residual[:, source_index])
            x = np.column_stack([np.ones(len(work)), source_baseline, target_baseline, age, sex])
            try:
                beta, se, dof = _ols(change, x)
                effect = float(beta[1])
                p_value = float(2 * stats.t.sf(abs(effect / se[1]), dof))
                boot = np.empty(n_bootstrap)
                for iteration in range(n_bootstrap):
                    sample = rng.integers(0, len(work), len(work))
                    boot[iteration] = np.linalg.lstsq(x[sample], change[sample], rcond=None)[0][1]
                lower, upper = np.percentile(boot, [2.5, 97.5])
            except (ValueError, np.linalg.LinAlgError):
                effect = p_value = lower = upper = np.nan
            rows.append({"source": source, "target": target, "effect": effect,
                         "p": p_value, "count": len(work), "ci_low": lower,
                         "ci_high": upper, "edge_type": "O2O"})
    result = pd.DataFrame(rows)
    result["q"] = benjamini_hochberg(result["p"])
    result["eligible"] = (
        (result["effect"] > 0) & (result["q"] < fdr)
        & (result["count"] >= min_pairs) & (result["ci_low"] > 0)
    )
    return result


@dataclass(frozen=True)
class AdmissionRule:
    edge_type: str
    minimum_count: int
    effect_floor: float
    direction: str = "greater"


def admit_edges(
    edges: pd.DataFrame,
    rules: Mapping[str, AdmissionRule],
    *,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """Apply type-specific negative-control floors and assign edge strengths."""
    admitted_parts: list[pd.DataFrame] = []
    for edge_type, group in edges.groupby("edge_type", sort=False):
        if edge_type not in rules:
            raise KeyError(f"No admission rule for {edge_type}.")
        rule = rules[edge_type]
        floors = (
            pd.to_numeric(group["negative_control_floor"], errors="coerce")
            if "negative_control_floor" in group
            else pd.Series(rule.effect_floor, index=group.index, dtype=float)
        )
        direction = group["effect"] > floors if rule.direction == "greater" else group["effect"] < floors
        selected = group.loc[direction & (group["q"] < fdr) & (group["count"] >= rule.minimum_count)].copy()
        selected["floor"] = floors.loc[selected.index].to_numpy()
        selected["strength"] = selected["effect"].rank(method="average", pct=True)
        admitted_parts.append(selected)
    if not admitted_parts:
        return edges.iloc[:0].assign(floor=np.nan, strength=np.nan)
    return pd.concat(admitted_parts, ignore_index=True)


def maximum_strength_routes(
    admitted_edges: pd.DataFrame,
    start_organs: Sequence[str],
    target_diseases: Sequence[str],
) -> pd.DataFrame:
    """Find the maximum-product admissible route from any organ to each disease."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'cascade' extra to construct routes.") from exc
    graph = nx.DiGraph()
    for row in admitted_edges.itertuples(index=False):
        strength = float(row.strength)
        if not 0 < strength <= 1:
            continue
        graph.add_edge(str(row.source), str(row.target), weight=-np.log(strength), strength=strength, edge_type=str(row.edge_type))
    rows: list[dict] = []
    for target in target_diseases:
        best: tuple[float, list[str]] | None = None
        for source in start_organs:
            if source not in graph or target not in graph:
                continue
            try:
                distance, path = nx.single_source_dijkstra(graph, source, target)
            except nx.NetworkXNoPath:
                continue
            joint = float(np.exp(-distance))
            if best is None or joint > best[0]:
                best = (joint, path)
        rows.append({
            "disease": target,
            "reachable": best is not None,
            "joint_strength": np.nan if best is None else best[0],
            "n_edges": 0 if best is None else len(best[1]) - 1,
            "path": "" if best is None else " -> ".join(best[1]),
        })
    return pd.DataFrame(rows)


def enumerate_feedback_loops(
    admitted_edges: pd.DataFrame,
    organ_nodes: Sequence[str],
    disease_nodes: Sequence[str],
    *,
    max_length: int = 8,
) -> pd.DataFrame:
    """Enumerate mixed organ-disease directed cycles up to a stated length."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'cascade' extra to enumerate loops.") from exc
    graph = nx.DiGraph()
    for row in admitted_edges.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), strength=float(row.strength))
    organs, diseases = set(organ_nodes), set(disease_nodes)
    records: list[dict] = []
    for cycle in nx.simple_cycles(graph, length_bound=max_length):
        if len(cycle) < 3 or not (set(cycle) & organs) or not (set(cycle) & diseases):
            continue
        strengths = [
            graph[cycle[i]][cycle[(i + 1) % len(cycle)]]["strength"]
            for i in range(len(cycle))
        ]
        records.append({"nodes": " -> ".join([*cycle, cycle[0]]),
                        "length": len(cycle), "loop_strength": float(np.prod(strengths))})
    return pd.DataFrame(records).sort_values("loop_strength", ascending=False, ignore_index=True) if records else pd.DataFrame(columns=["nodes", "length", "loop_strength"])


def comorbidity_summary(prevalence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return participant burden and pairwise baseline co-prevalence counts."""
    binary = prevalence.fillna(False).astype(bool)
    burden = pd.DataFrame({"n_comorbid": binary.sum(axis=1).astype(int)}, index=binary.index)
    pairs = []
    for left, right in combinations(binary.columns, 2):
        pairs.append({"disease_1": left, "disease_2": right,
                      "n_coprevalent": int((binary[left] & binary[right]).sum())})
    return burden, pd.DataFrame(pairs)


def mortality_associations(
    data: pd.DataFrame,
    exposure_columns: Sequence[str],
    *,
    duration_column: str,
    event_column: str,
    covariate_columns: Sequence[str] = ("age", "sex"),
    standardize_exposures: bool = True,
    min_events: int = 100,
) -> pd.DataFrame:
    """Fit separate covariate-adjusted all-cause mortality Cox models.

    Use continuous exposures for organ age, PIOAA, or comorbidity burden and
    binary exposures for prevalent diseases. Set ``standardize_exposures`` to
    false when effects should be reported per disease or per comorbidity count.
    """
    try:
        from lifelines import CoxPHFitter
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install the 'cascade' extra to run Cox models.") from exc
    rows: list[dict] = []
    for exposure in exposure_columns:
        columns = [duration_column, event_column, exposure, *covariate_columns]
        model_data = data[columns].dropna().copy()
        n_events = int(model_data[event_column].sum())
        if n_events < min_events or model_data[exposure].std() == 0:
            rows.append({"exposure": exposure, "hazard_ratio": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                         "events": n_events})
            continue
        if standardize_exposures:
            model_data[exposure] = _zscore(model_data[exposure])
        try:
            model = CoxPHFitter().fit(model_data, duration_column, event_column)
            row = model.summary.loc[exposure]
            rows.append({
                "exposure": exposure,
                "hazard_ratio": float(np.exp(row["coef"])),
                "ci_low": float(np.exp(row["coef lower 95%"])),
                "ci_high": float(np.exp(row["coef upper 95%"])),
                "p": float(row["p"]),
                "events": n_events,
            })
        except Exception:
            rows.append({"exposure": exposure, "hazard_ratio": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "p": np.nan,
                         "events": n_events})
    result = pd.DataFrame(rows)
    result["q"] = benjamini_hochberg(result["p"])
    return result

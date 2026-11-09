"""Loss functions for OrganPFN-M-LF Stage 2 (long-format, 10 organs).

The total loss is::

    L_FT =  sum_o present_o * inv_freq_o * (
                Huber(pred_mean_o, age) / (2 * exp(2 * log_sigma_o)) + log_sigma_o
            )                                  # A. main task (Huber + dual weighting)
          + gamma_c * L_contrastive            # B. cross-organ contrastive
          + gamma_d * L_divergence             # C. age-prediction divergence
          + beta   * L_bias                    # D. age-bias regularization
          + gamma_l * L_long                   # E. ★ longitudinal consistency

Each anti-degeneracy / consistency term has a dedicated function below;
:func:`compute_ft_loss` combines them and returns a diagnostic dict.

Inverse-frequency weights
-------------------------
Weights are calculated as::

    raw_weights[o] = 1 / sqrt(n_o)       # n_o from per_organ_effective_samples.finetune
    w[o] = raw_weights[o] / mean(raw_weights)

This pushes T2-only organs (Brain etc.) above 1.0 and pulls
both-timepoint organs (Body, Immune, Lungs) below 1.0, exactly because
those latter organs accumulate samples from BOTH timepoints in long
format.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Inverse-frequency weights
# ---------------------------------------------------------------------------

def compute_inverse_frequency_weights(
    per_organ_samples: Mapping[str, int],
    organ_names: Sequence[str],
    *,
    use_sqrt: bool = True,
) -> torch.Tensor:
    """Per-organ loss weights normalized to mean 1."""
    raw = np.array([
        (1.0 / math.sqrt(max(1, per_organ_samples[o]))
         if use_sqrt
         else 1.0 / max(1, per_organ_samples[o]))
        for o in organ_names
    ], dtype=np.float64)
    mean_w = float(raw.mean())
    if mean_w <= 0:
        return torch.ones(len(organ_names), dtype=torch.float32)
    return torch.as_tensor(raw / mean_w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# A. Per-organ Huber on the ensemble mean
# ---------------------------------------------------------------------------

def per_organ_huber_loss(
    per_organ_pred_BOk: torch.Tensor,
    target_age_B: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    *,
    delta: float,
    sample_weight_B: torch.Tensor | None = None,
    asym: tuple | None = None,
) -> torch.Tensor:
    """Returns (n_organs,) per-organ mean Huber over present samples.

    Organs with no present samples in this batch contribute 0.

    ``sample_weight_B`` (NEW): optional per-sample weights (B,) for age-aware
    re-weighting (Method-1 / LDS).

    ``asym`` (NEW): optional ``(lam, age_lo, age_hi)`` for the age-conditional
    asymmetric loss (Method-2).  Under-predictions (pred < true) are penalised
    by an extra factor ``1 + lam * age_norm`` where
    ``age_norm = clip((age-age_lo)/(age_hi-age_lo), 0, 1)`` — i.e. only at high
    age and only in the under-prediction direction.  ``None`` -> symmetric.
    """
    B, O, k = per_organ_pred_BOk.shape
    pred_mean_BO = per_organ_pred_BOk.mean(dim=-1)
    target_BO = target_age_B.unsqueeze(1).expand(-1, O)
    elementwise = F.huber_loss(
        pred_mean_BO, target_BO, delta=delta, reduction="none",
    )
    if asym is not None:
        lam, age_lo, age_hi = asym
        # age_lo/age_hi 可为标量(全局) 或 长度O的序列(逐器官)；自动广播。
        lo = torch.as_tensor(age_lo, dtype=elementwise.dtype, device=elementwise.device)
        hi = torch.as_tensor(age_hi, dtype=elementwise.dtype, device=elementwise.device)
        rng = (hi - lo).clamp(min=1e-6)
        age_norm = ((target_age_B.unsqueeze(1) - lo) / rng).clamp(0.0, 1.0)  # (B,1) 或 (B,O)
        under = (pred_mean_BO < target_BO).to(elementwise.dtype)             # (B,O) 低估方向
        elementwise = elementwise * (1.0 + float(lam) * age_norm * under)
    if sample_weight_B is not None:
        w_B1 = sample_weight_B.to(elementwise.dtype).unsqueeze(1)        # (B,1)
        masked = elementwise * organ_presence_BO * w_B1
        denom = (organ_presence_BO * w_B1).sum(dim=0).clamp(min=1e-6)
        return masked.sum(dim=0) / denom
    masked = elementwise * organ_presence_BO
    n_present_per_organ = organ_presence_BO.sum(dim=0).clamp(min=1)
    return masked.sum(dim=0) / n_present_per_organ


def per_organ_ccc_loss(
    per_organ_pred_BOk: torch.Tensor,
    target_age_B: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    *,
    min_n: int = 6,
) -> torch.Tensor:
    """Method-3: per-organ (1 - CCC) over present samples in the batch.

    CCC = 2*cov(pred,age) / (var(pred)+var(age)+(mean_pred-mean_age)^2).
    Minimising (1-CCC) simultaneously maximises correlation, MATCHES the
    prediction variance to the age variance (undoes regression-to-the-mean
    shrinkage -> fixes tail bias & lowers MAE) and aligns the means -- all
    without using chronological age at inference.

    Organs with < ``min_n`` present samples this batch contribute 0 (no grad),
    avoiding unstable statistics on rare organs.
    """
    B, O, k = per_organ_pred_BOk.shape
    pred_BO = per_organ_pred_BOk.mean(dim=-1)
    out = pred_BO.new_zeros(O)
    for o in range(O):
        m = organ_presence_BO[:, o] > 0
        n = int(m.sum())
        if n < min_n:
            continue
        x = pred_BO[m, o]
        y = target_age_B[m].to(x.dtype)
        mx, my = x.mean(), y.mean()
        vx = ((x - mx) ** 2).mean()
        vy = ((y - my) ** 2).mean()
        cov = ((x - mx) * (y - my)).mean()
        ccc = 2.0 * cov / (vx + vy + (mx - my) ** 2 + 1e-6)
        out[o] = 1.0 - ccc
    return out


# ---------------------------------------------------------------------------
# B. Cross-organ contrastive (anti-degeneracy #4)
# ---------------------------------------------------------------------------

def cross_organ_contrastive_loss(
    per_organ_summary_BOE: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    r"""Hinge penalty on pairwise cosine similarity between present organs.

    .. math::

        L_c = \frac{1}{|\mathcal{P}|} \sum_{(i, j) \in \mathcal{P}}
              \max(0,\, \cos(s_i, s_j) - m)

    where :math:`\mathcal{P}` is the set of organ-organ pairs that are
    BOTH present in at least one sample of the batch.
    """
    B, O, E = per_organ_summary_BOE.shape
    # Mean cosine similarity between organ pairs, masked by per-row presence.
    # We compute per-batch-row cosine similarity, mask by per-row pair
    # presence, then average over present rows.
    losses: list[torch.Tensor] = []
    for i in range(O):
        for j in range(i + 1, O):
            pair_mask = (organ_presence_BO[:, i] * organ_presence_BO[:, j]) > 0
            if pair_mask.sum() < 1:
                continue
            si = per_organ_summary_BOE[pair_mask, i]    # (k, E)
            sj = per_organ_summary_BOE[pair_mask, j]
            sim = F.cosine_similarity(si, sj, dim=-1)   # (k,)
            penalty = torch.clamp(sim - margin, min=0).mean()
            losses.append(penalty)
    if not losses:
        return per_organ_summary_BOE.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# C. Age-prediction divergence (anti-degeneracy supplementary)
# ---------------------------------------------------------------------------

def age_prediction_divergence_loss(
    per_organ_pred_BOk: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    is_healthy_B: torch.Tensor,
    *,
    min_std: float,
    max_std: float,
) -> torch.Tensor:
    """Keep healthy-sample per-row prediction std in [min_std, max_std]."""
    B, O, k = per_organ_pred_BOk.shape
    pred_mean_BO = per_organ_pred_BOk.mean(dim=-1)   # (B, O)
    healthy_mask = is_healthy_B.bool()
    if healthy_mask.sum() < 1:
        return per_organ_pred_BOk.sum() * 0.0

    losses: list[torch.Tensor] = []
    for b in torch.nonzero(healthy_mask).flatten().tolist():
        present_idx = torch.nonzero(organ_presence_BO[b] > 0).flatten()
        if present_idx.numel() < 2:
            continue
        preds_present = pred_mean_BO[b, present_idx]   # (n_present,)
        std = preds_present.std(unbiased=False)
        below = torch.clamp(min_std - std, min=0)
        above = torch.clamp(std - max_std, min=0)
        losses.append(below + above)
    if not losses:
        return per_organ_pred_BOk.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# D. Age-bias correlation regularizer
# ---------------------------------------------------------------------------

def age_bias_correlation_penalty(
    per_organ_pred_BOk: torch.Tensor,
    target_age_B: torch.Tensor,
    organ_presence_BO: torch.Tensor,
) -> torch.Tensor:
    """Sum over organs of squared Pearson(residual, chronological age)."""
    B, O, k = per_organ_pred_BOk.shape
    pred_mean_BO = per_organ_pred_BOk.mean(dim=-1)
    residual_BO = pred_mean_BO - target_age_B.unsqueeze(1)

    age_centered_B = target_age_B - target_age_B.mean()

    penalties: list[torch.Tensor] = []
    for o in range(O):
        mask = organ_presence_BO[:, o] > 0
        if mask.sum() < 2:
            continue
        r = residual_BO[mask, o]
        a = age_centered_B[mask]
        r_centered = r - r.mean()
        cov = (r_centered * a).mean()
        r_var = r_centered.pow(2).mean().clamp(min=1e-8)
        a_var_local = a.pow(2).mean().clamp(min=1e-8)
        corr = cov / torch.sqrt(r_var * a_var_local)
        penalties.append(corr.pow(2))

    if not penalties:
        return per_organ_pred_BOk.sum() * 0.0
    return torch.stack(penalties).sum()


# ---------------------------------------------------------------------------
# E. ★ Longitudinal consistency loss (long-format unique)
# ---------------------------------------------------------------------------

def longitudinal_consistency_loss(
    per_organ_pred_BOk: torch.Tensor,
    age_B: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    timepoint_id_B: torch.Tensor,
    pid_B: torch.Tensor,
    organ_names: Sequence[str],
    long_loss_organs: Sequence[str],
    *,
    huber_delta: float = 2.0,
) -> torch.Tensor:
    """Penalize ``|pred_diff - age_diff|`` over batch-internal paired rows.

    A "pair" is two rows in the batch that share the same Participant_ID
    and have different timepoint_id.  Only organs in ``long_loss_organs``
    (= Body / Immune / Lungs) contribute, because only those organs are
    measured at BOTH timepoints in the long-format schema.

    Returns 0 if the batch happens to contain no eligible pairs.
    """
    B, O, k = per_organ_pred_BOk.shape
    organ_to_idx = {o: i for i, o in enumerate(organ_names)}
    pred_mean_BO = per_organ_pred_BOk.mean(dim=-1)

    # Group rows by Participant_ID.
    pid_np = pid_B.detach().cpu().numpy()
    tp_np = timepoint_id_B.detach().cpu().numpy()
    rows_by_pid: dict[int, dict[int, int]] = {}
    for i in range(B):
        rows_by_pid.setdefault(int(pid_np[i]), {})[int(tp_np[i])] = i

    losses: list[torch.Tensor] = []
    for pid, d in rows_by_pid.items():
        if 0 not in d or 1 not in d:
            continue
        idx_T1, idx_T2 = d[0], d[1]
        age_diff = age_B[idx_T2] - age_B[idx_T1]
        for organ in long_loss_organs:
            if organ not in organ_to_idx:
                continue
            o_idx = organ_to_idx[organ]
            if not (organ_presence_BO[idx_T1, o_idx] > 0 and
                    organ_presence_BO[idx_T2, o_idx] > 0):
                continue
            pred_diff = pred_mean_BO[idx_T2, o_idx] - pred_mean_BO[idx_T1, o_idx]
            losses.append(
                F.huber_loss(
                    pred_diff.unsqueeze(0),
                    age_diff.unsqueeze(0),
                    delta=huber_delta, reduction="mean",
                )
            )
    if not losses:
        return per_organ_pred_BOk.sum() * 0.0
    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# Uncertainty-weighted total loss
# ---------------------------------------------------------------------------

@dataclass
class FTLossOutput:
    total: torch.Tensor                    # scalar
    per_organ_huber: torch.Tensor          # (n_organs,)
    per_organ_weighted: torch.Tensor       # (n_organs,)
    contrastive: torch.Tensor              # scalar
    divergence: torch.Tensor               # scalar
    age_bias_penalty: torch.Tensor         # scalar
    longitudinal: torch.Tensor             # scalar


def compute_ft_loss(
    *,
    per_organ_pred_BOk: torch.Tensor,
    per_organ_summary_BOE: torch.Tensor,
    target_age_B: torch.Tensor,
    organ_presence_BO: torch.Tensor,
    log_sigma_O: torch.Tensor,
    inv_freq_weights_O: torch.Tensor,
    is_healthy_B: torch.Tensor,
    timepoint_id_B: torch.Tensor,
    pid_B: torch.Tensor,
    organ_names: Sequence[str],
    huber_delta: float,
    age_bias_reg_weight: float,
    gamma_contrastive: float,
    gamma_divergence: float,
    gamma_long: float,
    contrastive_margin: float,
    prediction_min_std: float,
    prediction_max_std: float,
    long_loss_organs: Sequence[str],
    sample_weight_B: torch.Tensor | None = None,
    asym: tuple | None = None,
    ccc_weight: float = 0.0,
    ccc_min_n: int = 6,
) -> FTLossOutput:
    """Compute the full FT loss (A + B + C + D + E).

    ``sample_weight_B`` (NEW): per-sample weights for age-aware re-weighting
    (Method-1 / LDS).  ``asym`` (NEW): ``(lam, age_lo, age_hi)`` for the
    age-conditional asymmetric Huber (Method-2).  ``ccc_weight`` (NEW>0):
    add ``ccc_weight * Σ inv_freq * (1-CCC)`` per organ (Method-3, accuracy:
    matches pred variance to age variance -> ↑r, ↓MAE, ↓tail bias).
    All defaults reproduce the original loss.
    """
    O = per_organ_pred_BOk.shape[1]
    if log_sigma_O.shape != (O,):
        raise ValueError(
            f"log_sigma_O shape {tuple(log_sigma_O.shape)} != ({O},)"
        )

    huber_O = per_organ_huber_loss(
        per_organ_pred_BOk, target_age_B, organ_presence_BO, delta=huber_delta,
        sample_weight_B=sample_weight_B, asym=asym,
    )
    inv_2sigma2 = 1.0 / (2.0 * torch.exp(2.0 * log_sigma_O))
    weighted_O = inv_freq_weights_O.to(huber_O.device) * (
        inv_2sigma2 * huber_O + log_sigma_O
    )
    any_present_O = (organ_presence_BO.sum(dim=0) > 0).float()
    weighted_total = (weighted_O * any_present_O).sum()

    contrastive = cross_organ_contrastive_loss(
        per_organ_summary_BOE, organ_presence_BO, margin=contrastive_margin,
    )
    divergence = age_prediction_divergence_loss(
        per_organ_pred_BOk, organ_presence_BO, is_healthy_B,
        min_std=prediction_min_std, max_std=prediction_max_std,
    )
    bias_penalty = age_bias_correlation_penalty(
        per_organ_pred_BOk, target_age_B, organ_presence_BO,
    )
    longitudinal = longitudinal_consistency_loss(
        per_organ_pred_BOk, target_age_B, organ_presence_BO,
        timepoint_id_B, pid_B, organ_names, long_loss_organs,
        huber_delta=huber_delta,
    )

    total = (
        weighted_total
        + gamma_contrastive * contrastive
        + gamma_divergence  * divergence
        + age_bias_reg_weight * bias_penalty
        + gamma_long        * longitudinal
    )
    if ccc_weight > 0:                              # Method-3: CCC accuracy term
        ccc_O = per_organ_ccc_loss(
            per_organ_pred_BOk, target_age_B, organ_presence_BO, min_n=ccc_min_n)
        total = total + ccc_weight * (
            inv_freq_weights_O.to(ccc_O.device) * ccc_O).sum()
    return FTLossOutput(
        total=total,
        per_organ_huber=huber_O.detach(),
        per_organ_weighted=weighted_O.detach(),
        contrastive=contrastive.detach(),
        divergence=divergence.detach(),
        age_bias_penalty=bias_penalty.detach(),
        longitudinal=longitudinal.detach(),
    )


# ---------------------------------------------------------------------------
# Validation metrics (no-grad)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_val_metrics_per_organ_timepoint(
    pred_BOk: torch.Tensor,
    age_B: torch.Tensor,
    presence_BO: torch.Tensor,
    timepoint_id_B: torch.Tensor,
    *,
    organ_names: Sequence[str],
    organ_timepoint_availability: Mapping[str, Sequence[str]],
    timepoint_id_map: Mapping[str, int],
) -> dict[str, dict[str, float]]:
    """Compute per-(organ, timepoint) MAE / Pearson r / calibration Spearman.

    Returns a dict keyed by ``"<organ>__<tp_name>"`` (e.g. ``"Body__T0"``);
    only the (organ, timepoint) combinations listed as available in
    ``organ_timepoint_availability`` are evaluated.
    """
    pred_BOk = pred_BOk.float().cpu()
    age_B = age_B.float().cpu()
    presence_BO = presence_BO.float().cpu()
    tp_B = timepoint_id_B.long().cpu()

    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    pred_std_BO = pred_BOk.std(dim=-1).numpy()
    age = age_B.numpy()
    present = presence_BO.numpy() > 0
    tp_np = tp_B.numpy()

    out: dict[str, dict[str, float]] = {}
    for o_idx, organ in enumerate(organ_names):
        avail = list(organ_timepoint_availability.get(organ, []))
        for tp_name in avail:
            tp_id = int(timepoint_id_map[tp_name])
            mask = (
                present[:, o_idx]
                & (tp_np == tp_id)
                & np.isfinite(age)
                & np.isfinite(pred_mean_BO[:, o_idx])
            )
            key = f"{organ}__{tp_name}"
            if mask.sum() < 2:
                out[key] = {
                    "mae": float("nan"), "pearson_r": float("nan"),
                    "calibration_spearman": float("nan"), "n": int(mask.sum()),
                }
                continue
            err = pred_mean_BO[mask, o_idx] - age[mask]
            mae = float(np.mean(np.abs(err)))
            x, y = age[mask], pred_mean_BO[mask, o_idx]
            x_c, y_c = x - x.mean(), y - y.mean()
            denom = float(np.sqrt(np.sum(x_c ** 2) * np.sum(y_c ** 2)))
            pearson = float((x_c * y_c).sum() / denom) if denom > 0 else float("nan")
            try:
                from scipy.stats import spearmanr  # noqa: PLC0415
                rho, _ = spearmanr(np.abs(err), pred_std_BO[mask, o_idx])
                cal = float(rho) if np.isfinite(rho) else float("nan")
            except Exception:
                cal = float("nan")
            out[key] = {
                "mae": mae, "pearson_r": pearson,
                "calibration_spearman": cal, "n": int(mask.sum()),
            }
    return out

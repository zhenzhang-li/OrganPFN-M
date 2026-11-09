"""Inference utility: load a fine-tune checkpoint and predict per-organ age.

Two entry points:

* :func:`load_predictor` -- re-instantiate the FT model from a saved
  ``best_model.pt`` / ``final_model.pt``.  It re-builds the SSL backbone
  from the SSL checkpoint, attaches the FT modules, then strict-loads
  the FT state_dict.
* :func:`predict` -- given a (cached) ``Predictor`` and a long-format
  DataFrame, return a tidy DataFrame with one row per input sample and
  per-organ prediction columns plus age-gap columns (raw + per-(organ,
  timepoint) bias-corrected).

The two are wrapped by :func:`predict_from_checkpoint` so a user can do
the entire round-trip in one call:

.. code-block:: python

    import pandas as pd
    from organpfnm_ft.inference import predict_from_checkpoint

    out = predict_from_checkpoint(
        ckpt_path="ft_outputs/best_model.pt",
        df=pd.read_csv("E:/Organ_age/data_10org/ssl_subset.csv"),
        out_csv="inference_demo.csv",
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from . import config as _cfg
from .data import FineTuneDataset, NormalizationStats, ft_collate
from .model import OPFNMLF_FT, build_ft_model_from_ssl_checkpoint
from .utils import Schema, load_ft_checkpoint, load_schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Predictor (cached model + ancillary state)
# ---------------------------------------------------------------------------

@dataclass
class Predictor:
    model: OPFNMLF_FT
    schema: Schema
    norm: NormalizationStats
    age_bias_correction: dict[str, dict[str, float]] | None
    organ_presence_threshold: float
    device: torch.device


def load_predictor(
    *,
    ckpt_path: Path,
    schema_path: Path | None = None,
    ssl_ckpt_path: Path | None = None,
    device: torch.device | None = None,
) -> Predictor:
    """Re-instantiate the model from a FT checkpoint.

    The FT checkpoint stores: model_state_dict, schema, normalization,
    config, organ_presence_threshold, age_bias_correction (best only).

    Because the on-disk state_dict is just the FT model state, we still
    need access to the SSL backbone architecture to *construct* the
    matching model -- so we re-run ``build_ft_model_from_ssl_checkpoint``
    with the same SSL ckpt, then overwrite its weights with the FT
    state_dict.
    """
    ck = load_ft_checkpoint(ckpt_path)
    if "schema" not in ck or "normalization" not in ck:
        raise KeyError(
            f"FT ckpt missing schema/normalization; got keys {sorted(ck.keys())}"
        )

    if schema_path is not None:
        schema = load_schema(schema_path)
    else:
        raw = ck["schema"]
        from .utils import Schema as SchemaCls   # noqa: PLC0415
        _organs = list(raw["organ_names"])
        _feat = list(raw["feature_columns"])
        # V5 Sex-as-organ-feature 字段 (raw 里有就用, 没有则按 load_schema 的方式派生默认)
        _sexcol_raw = raw.get("sex_column_per_organ", {})
        _sexidx_raw = raw.get("sex_column_indices_per_organ", {})
        _fmap_excl_raw = raw.get("organ_feature_map_excluding_sex", {})
        _npo_excl_raw = raw.get("n_features_per_organ_excluding_sex", {})
        _sex_col = {}
        _sex_idx = {}
        _fmap_excl = {}
        _npo_excl = {}
        for _o in _organs:
            _sc = _sexcol_raw.get(_o, f"{_o}_Sex")
            _sex_col[_o] = _sc
            if _o in _sexidx_raw:
                _sex_idx[_o] = int(_sexidx_raw[_o])
            else:
                _sex_idx[_o] = _feat.index(_sc) if _sc in _feat else -1
            if _o in _fmap_excl_raw:
                _fmap_excl[_o] = list(_fmap_excl_raw[_o])
            else:
                _blk = list(raw["organ_feature_map"][_o])
                _fmap_excl[_o] = [c for c in _blk if c != _sc]
            _npo_excl[_o] = int(_npo_excl_raw.get(_o, len(_fmap_excl[_o])))
        _presence_excl_sex = bool(
            raw["filter_thresholds"]["finetune"].get("presence_excludes_sex", False)
            or raw["filter_thresholds"]["ssl"].get("presence_excludes_sex", False)
        )
        schema = SchemaCls(
            organ_names=_organs,
            feature_columns=_feat,
            organ_offsets={
                o: tuple(raw["organ_offsets"][o]) for o in _organs
            },
            n_features_per_organ={
                o: int(raw["n_features_per_organ"][o]) for o in _organs
            },
            n_total_features=int(raw["n_total_features"]),
            metadata_columns=list(raw["metadata_columns"]),
            organ_presence_threshold_ssl=float(
                raw["filter_thresholds"]["ssl"]["organ_presence_threshold"]
            ),
            organ_presence_threshold_finetune=float(
                raw["filter_thresholds"]["finetune"]["organ_presence_threshold"]
            ),
            per_organ_effective_samples_finetune={
                o: int(raw["per_organ_effective_samples"]["finetune"][o])
                for o in _organs
            },
            timepoint_id_map={
                str(k): int(v) for k, v in raw["timepoint_id_map"].items()
            },
            n_timepoints=int(raw["n_timepoints"]),
            organ_timepoint_availability={
                o: list(raw["organ_timepoint_availability"][o])
                for o in _organs
            },
            raw=raw,
            sex_column_per_organ=_sex_col,
            sex_column_indices_per_organ=_sex_idx,
            organ_feature_map_excluding_sex=_fmap_excl,
            n_features_per_organ_excluding_sex=_npo_excl,
            presence_excludes_sex=_presence_excl_sex,
        )

    norm = NormalizationStats.from_ssl_checkpoint(ck["normalization"])

    cfg = ck["config"]
    from .config import ModelConfig                  # noqa: PLC0415
    model_cfg = ModelConfig(
        embed_dim=int(cfg["embed_dim"]),
        n_layers_expected=int(cfg["n_layers"]),
        features_per_group=int(cfg["features_per_group"]),
        num_thinking_rows=int(cfg["num_thinking_rows"]),
        n_organs=int(cfg["n_organs"]),
        n_timepoints=int(cfg.get("n_timepoints", 2)),
        n_total_features=len(cfg["feature_columns"]),
        k_ensemble=int(cfg.get("k_ensemble", 32)),
        head_hidden_dim=int(cfg.get("head_hidden_dim", 256)),
        head_dropout=float(cfg.get("head_dropout", 0.3)),
        rank1_init_std=float(cfg.get("rank1_init_std", 0.1)),
        orthogonal_projected_dim=int(cfg.get("orthogonal_projected_dim", 128)),
    )
    ssl_ckpt = ssl_ckpt_path or _cfg.SSL_CKPT_PATH
    model, _norm_unused, _report = build_ft_model_from_ssl_checkpoint(
        ssl_ckpt_path=ssl_ckpt, schema=schema, model_cfg=model_cfg,
    )
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"FT state_dict load missing={missing}, unexpected={unexpected}"
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    threshold = float(
        ck.get("organ_presence_threshold")
        or schema.organ_presence_threshold_finetune
    )

    return Predictor(
        model=model,
        schema=schema,
        norm=norm,
        age_bias_correction=ck.get("age_bias_correction"),
        organ_presence_threshold=threshold,
        device=device,
    )


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_single_organ(
    predictor: Predictor,
    df: pd.DataFrame,
    organ: str,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Predict ages with ONLY ``organ`` visible (all other organ_presence
    flags forced to 0).

    Returns a DataFrame with one row per input sample and columns
    ``Participant_ID, Sex, timepoint, chronological_age, is_healthy,
    {organ}_predicted_age_singleorgan, {organ}_std_singleorgan,
    {organ}_age_gap_singleorgan``.  Rows where ``organ`` is NOT present in
    the input sample (under the active FT presence threshold) get NaN
    for the prediction columns - which is the right thing to do since the
    architectural latent zero-gate would force a zero summary for such rows.
    """
    schema = predictor.schema
    norm = predictor.norm
    model = predictor.model
    device = predictor.device
    if organ not in schema.organ_names:
        raise ValueError(f"Unknown organ {organ!r}; "
                         f"valid: {schema.organ_names}")
    target_idx = schema.organ_names.index(organ)

    ds = FineTuneDataset(
        df, schema, norm,
        organ_presence_threshold=predictor.organ_presence_threshold,
        min_organs=0,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=ft_collate,
    )

    all_pred: list[torch.Tensor] = []
    all_age: list[torch.Tensor] = []
    all_pres: list[torch.Tensor] = []
    all_pid: list[torch.Tensor] = []
    all_healthy: list[torch.Tensor] = []
    all_sex: list[torch.Tensor] = []
    all_tp: list[torch.Tensor] = []
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        full_pres = batch["organ_presence"].to(device, non_blocking=True)
        tp = batch["timepoint_id"].to(device, non_blocking=True)
        # Single-organ presence: keep only the target organ on rows where
        # it is actually present.
        single = torch.zeros_like(full_pres)
        single[:, target_idx] = full_pres[:, target_idx]
        out = model(feats, single, tp)
        all_pred.append(out["per_organ_pred"].cpu())
        all_age.append(batch["age"])
        all_pres.append(single.cpu())
        all_pid.append(batch["pid"])
        all_healthy.append(batch["is_healthy"])
        all_sex.append(batch["sex"])
        all_tp.append(batch["timepoint_id"])

    pred_BOk = torch.cat(all_pred, dim=0)
    age_B = torch.cat(all_age, dim=0)
    pres_BO = torch.cat(all_pres, dim=0)
    pid_B = torch.cat(all_pid, dim=0)
    healthy_B = torch.cat(all_healthy, dim=0)
    sex_B = torch.cat(all_sex, dim=0)
    tp_B = torch.cat(all_tp, dim=0)

    pred_mean = pred_BOk[:, target_idx, :].mean(dim=-1).numpy()
    pred_std = pred_BOk[:, target_idx, :].std(dim=-1).numpy()
    age_np = age_B.numpy()
    presence_np = pres_BO.numpy()
    present_mask = presence_np[:, target_idx] > 0

    tp_map_inv = {v: k for k, v in schema.timepoint_id_map.items()}
    tp_letters = [tp_map_inv.get(int(t), str(t)) for t in tp_B.numpy().tolist()]
    sex_letters = ["M" if int(s) == 0 else ("F" if int(s) == 1 else "U")
                   for s in sex_B.tolist()]

    out_df = pd.DataFrame({
        "Participant_ID": pid_B.numpy(),
        "Sex": sex_letters,
        "timepoint": tp_letters,
        "chronological_age": age_np,
        "is_healthy": healthy_B.numpy(),
    })
    if "Disease" in df.columns:
        out_df["Disease"] = df["Disease"].astype(str).fillna("").reset_index(drop=True)

    pred = np.where(present_mask, pred_mean, np.nan)
    std = np.where(present_mask, pred_std, np.nan)
    gap = np.where(present_mask, pred - age_np, np.nan)

    out_df[f"{organ}_predicted_age_singleorgan"] = pred
    out_df[f"{organ}_std_singleorgan"] = std
    out_df[f"{organ}_age_gap_singleorgan"] = gap
    return out_df


@torch.no_grad()
def predict(
    predictor: Predictor,
    df: pd.DataFrame,
    *,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Predict per-organ ages for every row in ``df``.

    Returns DataFrame columns:
        Participant_ID, Sex, timepoint, chronological_age, is_healthy,
        <Disease if present>,
        <organ>_predicted_age   (point estimate, ensemble mean)
        <organ>_std             (epistemic uncertainty)
        <organ>_age_gap         (predicted - chronological)
        <organ>_age_gap_corrected   (linear age-bias corrected; uses
                                      per-(organ, timepoint) coefficients)
    """
    schema = predictor.schema
    norm = predictor.norm
    model = predictor.model
    device = predictor.device

    ds = FineTuneDataset(
        df, schema, norm,
        organ_presence_threshold=predictor.organ_presence_threshold,
        min_organs=0,  # at inference we don't drop rows
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=ft_collate,
    )

    all_pred: list[torch.Tensor] = []
    all_age: list[torch.Tensor] = []
    all_pres: list[torch.Tensor] = []
    all_pid: list[torch.Tensor] = []
    all_healthy: list[torch.Tensor] = []
    all_sex: list[torch.Tensor] = []
    all_tp: list[torch.Tensor] = []
    for batch in loader:
        feats = batch["features"].to(device, non_blocking=True)
        presence = batch["organ_presence"].to(device, non_blocking=True)
        tp = batch["timepoint_id"].to(device, non_blocking=True)
        out = model(feats, presence, tp)
        all_pred.append(out["per_organ_pred"].cpu())
        all_age.append(batch["age"])
        all_pres.append(batch["organ_presence"])
        all_pid.append(batch["pid"])
        all_healthy.append(batch["is_healthy"])
        all_sex.append(batch["sex"])
        all_tp.append(batch["timepoint_id"])

    pred_BOk = torch.cat(all_pred, dim=0)
    age_B = torch.cat(all_age, dim=0)
    pres_BO = torch.cat(all_pres, dim=0)
    pid_B = torch.cat(all_pid, dim=0)
    healthy_B = torch.cat(all_healthy, dim=0)
    sex_B = torch.cat(all_sex, dim=0)
    tp_B = torch.cat(all_tp, dim=0)

    pred_mean_BO = pred_BOk.mean(dim=-1).numpy()
    pred_std_BO = pred_BOk.std(dim=-1).numpy()
    age_np = age_B.numpy()
    presence_np = pres_BO.numpy()
    tp_np = tp_B.numpy()

    tp_map_inv = {v: k for k, v in schema.timepoint_id_map.items()}
    tp_letters = [tp_map_inv.get(int(t), str(t)) for t in tp_np.tolist()]

    sex_letters = ["M" if int(s) == 0 else ("F" if int(s) == 1 else "U")
                   for s in sex_B.tolist()]

    out_df = pd.DataFrame({
        "Participant_ID": pid_B.numpy(),
        "Sex": sex_letters,
        "timepoint": tp_letters,
        "chronological_age": age_np,
        "is_healthy": healthy_B.numpy(),
    })
    if "Disease" in df.columns:
        # NOTE: we filtered with min_organs=0 so row count is preserved.
        out_df["Disease"] = df["Disease"].astype(str).fillna("").reset_index(drop=True)

    abc = predictor.age_bias_correction or {}
    for o_idx, organ in enumerate(schema.organ_names):
        present_mask = presence_np[:, o_idx] > 0
        pred = pred_mean_BO[:, o_idx].copy()
        std = pred_std_BO[:, o_idx].copy()
        gap = pred - age_np
        pred = np.where(present_mask, pred, np.nan)
        std = np.where(present_mask, std, np.nan)
        gap = np.where(present_mask, gap, np.nan)

        out_df[f"{organ}_predicted_age"] = pred
        out_df[f"{organ}_std"] = std
        out_df[f"{organ}_age_gap"] = gap

        # Apply per-(organ, timepoint) age-bias correction.
        if abc:
            corrected = np.full_like(age_np, np.nan, dtype=np.float64)
            for tp_name, tp_id in schema.timepoint_id_map.items():
                key = f"{organ}__{tp_name}"
                if key not in abc:
                    continue
                m = present_mask & (tp_np == tp_id)
                if not m.any():
                    continue
                a = float(abc[key].get("a", 0.0))
                b = float(abc[key].get("b", 0.0))
                corrected[m] = gap[m] - (a * age_np[m] + b)
            out_df[f"{organ}_age_gap_corrected"] = corrected
        else:
            out_df[f"{organ}_age_gap_corrected"] = np.nan

    return out_df


def predict_from_checkpoint(
    *,
    ckpt_path: Path,
    df: pd.DataFrame,
    out_csv: Path | None = None,
    schema_path: Path | None = None,
    ssl_ckpt_path: Path | None = None,
    batch_size: int = 64,
    device: torch.device | None = None,
    also_single_organ: bool = False,
    single_organ_out_dir: Path | None = None,
) -> pd.DataFrame:
    """Round-trip: load predictor + run predict + optionally save CSV.

    When ``also_single_organ`` is True, additionally run the single-organ
    prediction mode for every organ in the schema and write one CSV per
    organ under ``single_organ_out_dir`` (defaults to ``out_csv``'s parent
    directory + ``single_organ/``).
    """
    predictor = load_predictor(
        ckpt_path=ckpt_path,
        schema_path=schema_path,
        ssl_ckpt_path=ssl_ckpt_path,
        device=device,
    )
    out_df = predict(predictor, df, batch_size=batch_size)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_csv, index=False)
        logger.info("Saved predictions -> %s", out_csv)

    if also_single_organ:
        target_dir = (
            Path(single_organ_out_dir) if single_organ_out_dir
            else (out_csv.parent / "single_organ" if out_csv else Path("single_organ"))
        )
        target_dir.mkdir(parents=True, exist_ok=True)
        for organ in predictor.schema.organ_names:
            single_df = predict_single_organ(
                predictor, df, organ, batch_size=batch_size,
            )
            single_path = target_dir / f"prediction_singleorgan_{organ}.csv"
            single_df.to_csv(single_path, index=False)
            logger.info("Saved single-organ predictions (%s) -> %s",
                        organ, single_path)
    return out_df

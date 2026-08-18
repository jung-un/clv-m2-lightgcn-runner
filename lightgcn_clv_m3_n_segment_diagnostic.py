"""No-training diagnostic for the existing Dunnhumby M3-N checkpoint.

The full-CLV compositional M3 model should be trained only when its N relation
already helps users whose historical CLV is activity-dominant.  This module
loads the existing seed-42 M1 and M3-N checkpoints, evaluates the validation
split, and fails closed if either checkpoint is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_m3_transfer as transfer
import lightgcn_clv_v3 as v3


CODE_VERSION = "m3-n-segment-diagnostic-v1"
SEED = 42
MODEL_ID = "m3_n_transfer"


def _checkpoint_path(arch: str, cfg: dict) -> Path:
    filename = (
        f"ckpt_{arch}_{cfg['DATASET']}_s{SEED}_"
        f"{v3.cfg_hash(cfg, v3.DCFG, arch, SEED)}.pt"
    )
    return Path(cfg["OUT_DIR"]) / filename


def _require_existing_checkpoint(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} checkpoint not found: {path}. "
            "This diagnostic never trains a replacement model."
        )


def segment_decision(frame: pd.DataFrame) -> dict:
    indexed = frame.set_index("segment")
    required = "N-oriented (pi_N>0.5)"
    if required not in indexed.index:
        raise ValueError(f"missing required segment: {required}")
    row = indexed.loc[required]
    revenue_positive = bool(row["revenue@10_delta"] > 0.0)
    recall_positive = bool(row["recall@20_delta"] > 0.0)
    proceed = revenue_positive and recall_positive
    return {
        "proceed_to_compositional_m3": proceed,
        "n_oriented_revenue_positive": revenue_positive,
        "n_oriented_recall20_positive": recall_positive,
        "reason": (
            "N-transfer is positive for activity-dominant CLV users"
            if proceed
            else "N-transfer is not jointly positive for activity-dominant CLV users"
        ),
        "next_step": (
            "run M1 vs full-CLV compositional M3"
            if proceed
            else "stop the current compositional M3; redesign the N edge relation"
        ),
    }


def _activate(cfg: dict) -> None:
    transfer.validate_screening_config(cfg)
    settings = {
        key: value
        for key, value in cfg.items()
        if key not in {"DATASET", "OUT_DIR", "GRAPH_MODES"}
    }
    settings["GRAPH_MODE"] = "n_transfer"
    v3.configure_run(cfg["DATASET"], out_dir=cfg["OUT_DIR"], **settings)


def _load_existing_model(
    arch: str,
    cfg: dict,
    data: dict,
    gate_t: torch.Tensor,
    x_item: np.ndarray,
    item_cat: np.ndarray,
    meta: dict,
    val_cache: v3.EvalCache,
):
    checkpoint = _checkpoint_path(arch, cfg)
    _require_existing_checkpoint(checkpoint, arch)
    return v3.get_or_train(
        arch,
        SEED,
        data,
        gate_t,
        data["x_val_u"],
        x_item,
        item_cat,
        meta,
        val_cache,
        cfg,
    )[0]


def _evaluate_two_k(
    model,
    data: dict,
    gate_t: torch.Tensor,
    cache: v3.EvalCache,
    meta: dict,
    cfg: dict,
) -> dict:
    at10 = v3.evaluate(
        model,
        0.0,
        gate_t,
        cache,
        meta,
        [10],
        data["csr_ptr"],
        data["csr_items"],
        cfg,
        per_user=True,
    )
    at20 = v3.evaluate(
        model,
        0.0,
        gate_t,
        cache,
        meta,
        [20],
        data["csr_ptr"],
        data["csr_items"],
        cfg,
        per_user=True,
    )
    return {
        "revenue@10": at10["per_user"]["revenue"],
        "recall@20": at20["per_user"]["recall"],
    }


def run_diagnostic(cfg: dict | None = None) -> pd.DataFrame:
    cfg = transfer.configure_m3_transfer_dunnhumby_run() if cfg is None else dict(cfg)
    _activate(cfg)
    print("M3-N segment diagnostic: existing checkpoints only; no training")

    data = v3.prepare_data(v3.CFG, v3.DCFG)
    if data["m3_transfer_graph"] is None:
        raise RuntimeError("n_transfer graph diagnostics were not constructed")

    gt, rev = data["splits"]["val"]
    seg_th = v3.segment_thresholds(data["clv"], v3.CFG["SEG_EDGES"])
    cache = v3.EvalCache(gt, rev, data["clv"], seg_th, data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    gate_t = torch.ones(data["n_users"], dtype=torch.float32, device=v3.DEVICE)

    base_data, base_cfg = v3.binary_baseline(data, v3.CFG)
    model_checkpoint = _checkpoint_path("pref_only", v3.CFG)
    baseline_checkpoint = _checkpoint_path("pref_only", base_cfg)
    _require_existing_checkpoint(model_checkpoint, "M3-N")
    _require_existing_checkpoint(baseline_checkpoint, "M1")

    model = _load_existing_model(
        "pref_only", v3.CFG, data, gate_t, x_item, item_cat, meta, cache
    )
    baseline = _load_existing_model(
        "pref_only", base_cfg, base_data, gate_t, x_item, item_cat, meta, cache
    )
    model_metrics = _evaluate_two_k(model, data, gate_t, cache, meta, v3.CFG)
    base_metrics = _evaluate_two_k(
        baseline, base_data, gate_t, cache, meta, base_cfg
    )

    pi_n = data["m3_transfer_graph"].pi_n[cache.users]
    segment_masks = {
        "N-oriented (pi_N>0.5)": pi_n > 0.5,
        "balanced (pi_N=0.5)": np.isclose(pi_n, 0.5),
        "V-oriented (pi_N<0.5)": pi_n < 0.5,
    }
    rows = []
    for name, mask in segment_masks.items():
        if not mask.any():
            continue
        revenue_diff = (
            model_metrics["revenue@10"][mask] - base_metrics["revenue@10"][mask]
        )
        recall_diff = (
            model_metrics["recall@20"][mask] - base_metrics["recall@20"][mask]
        )
        revenue_ci = v3.paired_bootstrap([revenue_diff], v3.CFG["N_BOOT"])
        recall_ci = v3.paired_bootstrap([recall_diff], v3.CFG["N_BOOT"])
        rows.append(
            {
                "segment": name,
                "n_users": int(mask.sum()),
                "pi_n_mean": float(pi_n[mask].mean()),
                "revenue@10_delta": revenue_ci["mean"],
                "revenue@10_lo": revenue_ci["lo"],
                "revenue@10_hi": revenue_ci["hi"],
                "recall@20_delta": recall_ci["mean"],
                "recall@20_lo": recall_ci["lo"],
                "recall@20_hi": recall_ci["hi"],
            }
        )

    frame = pd.DataFrame(rows)
    decision = segment_decision(frame)
    out = Path(cfg["OUT_DIR"]) / "checkpoint_diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "m3_n_clv_composition_segment_diagnostic.csv"
    json_path = out / "m3_n_clv_composition_segment_diagnostic.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "code_version": CODE_VERSION,
                "formula": {
                    "log_clv": "log(1+N_u) + log(1+V_u)",
                    "q_clv": "percentile(log_clv)",
                    "pi_n": "log(1+N_u) / log_clv",
                    "pi_v": "1 - pi_n",
                },
                "checkpoints": {
                    "m3_n": str(model_checkpoint),
                    "m1": str(baseline_checkpoint),
                },
                "decision": decision,
                "rows": frame.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    frame.attrs["decision"] = decision
    frame.attrs["paths"] = {"csv": str(csv_path), "json": str(json_path)}
    return frame


if __name__ == "__main__":
    print(json.dumps(transfer.preflight_summary(
        transfer.configure_m3_transfer_dunnhumby_run()
    ), ensure_ascii=False, indent=2))
    print("Import run_diagnostic() explicitly; __main__ never starts evaluation.")

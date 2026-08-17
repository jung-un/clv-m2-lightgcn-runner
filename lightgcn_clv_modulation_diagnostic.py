"""No-retraining checkpoint diagnostics for CLV-conditioned modulation M2."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_modulation as modulation
import lightgcn_clv_v3 as v3


CODE_VERSION = "m2-clv-modulation-checkpoint-diagnostic-v1"
VIEW_MODES = ("none", "n_only", "v_only", "both", "shuffled_user")


def find_modulation_checkpoint(root: str | Path) -> Path:
    candidates = sorted(
        Path(root).glob("m2_clv_modulation_dunnhumby_s42_*.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"modulation checkpoint not found under {root}")
    return candidates[0]


def load_modulation_checkpoint(model, checkpoint: str | Path, cfg, input_hash: str):
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    recorded = payload.get("config", {})
    expected = asdict(cfg)
    identity_keys = (
        "dataset",
        "seed",
        "input_days",
        "id_dim",
        "modulation_rank",
        "tau",
        "n_layers",
    )
    mismatch = {
        key: {"expected": expected[key], "actual": recorded.get(key)}
        for key in identity_keys
        if recorded.get(key) != expected[key]
    }
    if payload.get("input_hash") != input_hash:
        mismatch["input_hash"] = {
            "expected": input_hash,
            "actual": payload.get("input_hash"),
        }
    if mismatch:
        raise RuntimeError(f"modulation checkpoint identity mismatch: {mismatch}")
    model.load_state_dict(payload["state"], strict=True)
    model.eval()
    return payload


@torch.no_grad()
def modulation_structure(model) -> dict:
    user_n, user_v, item_n, item_v = model._axis_modulations()
    components = {
        "user_n": user_n,
        "user_v": user_v,
        "item_n": item_n,
        "item_v": item_v,
    }
    report = {}
    for name, values in components.items():
        transformed = torch.tanh(values)
        report[name] = {
            "raw_abs_mean": float(values.abs().mean()),
            "raw_std": float(values.std()),
            "tanh_abs_mean": float(transformed.abs().mean()),
            "saturation_share": float(transformed.abs().gt(0.95).float().mean()),
            "positive_share": float(transformed.gt(0).float().mean()),
        }
    for mode in ("none", "n_only", "v_only", "both"):
        model.set_eval_axes(mode)
        user_mod, item_mod = model._combined_modulations()
        user_scale = 1.0 + model.tau * torch.tanh(user_mod)
        item_scale = 1.0 + model.tau * torch.tanh(item_mod)
        report[f"scale_{mode}"] = {
            "user_mean": float(user_scale.mean()),
            "user_std": float(user_scale.std()),
            "user_min": float(user_scale.min()),
            "user_max": float(user_scale.max()),
            "item_mean": float(item_scale.mean()),
            "item_std": float(item_scale.std()),
            "item_min": float(item_scale.min()),
            "item_max": float(item_scale.max()),
        }
    model.set_eval_axes("both")
    return report


def _evaluate_view(model, prepared: dict, mode: str):
    model.set_eval_axes(mode)
    metrics, per_user = modulation._evaluate(model, prepared, per_user=True)
    return modulation._with_public_metric_names(metrics), per_user


def _evaluate_shuffled_user(model, prepared: dict, seed: int):
    names = ("user_activity", "user_value", "user_activity_valid", "user_value_valid")
    originals = {name: getattr(model, name).detach().clone() for name in names}
    permutation = np.random.default_rng(seed).permutation(model.n_users)
    index = torch.as_tensor(permutation, device=model.user_activity.device)
    try:
        for name in names:
            target = getattr(model, name)
            target.copy_(originals[name].index_select(0, index))
        return _evaluate_view(model, prepared, "both")
    finally:
        for name in names:
            getattr(model, name).copy_(originals[name])
        model.set_eval_axes("both")


def _metric_row(view: str, metrics: dict) -> dict:
    return {"view": view, **metrics}


def _paired_rows(per_user: dict[str, dict], n_boot: int) -> list[dict]:
    baseline = per_user["none"]
    rows = []
    for view in VIEW_MODES[1:]:
        for metric in ("recall", "ndcg", "revenue", "arp"):
            difference = per_user[view][metric] - baseline[metric]
            rows.append(
                {
                    "view": view,
                    "reference": "none",
                    "metric": metric,
                    **v3.paired_bootstrap([difference], n_boot),
                }
            )
    return rows


def _persist(report: dict, cfg) -> dict[str, str]:
    root = Path(cfg.out_dir) / "checkpoint_diagnostics"
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "views_csv": root / "m2_modulation_checkpoint_views.csv",
        "paired_csv": root / "m2_modulation_checkpoint_paired.csv",
        "json": root / "m2_modulation_checkpoint_diagnostic.json",
    }
    report["views"].to_csv(paths["views_csv"], index=False)
    report["paired"].to_csv(paths["paired_csv"], index=False)
    payload = {
        "code_version": CODE_VERSION,
        "scope": "existing checkpoint only; no model training",
        "checkpoint": report["checkpoint"],
        "config": asdict(cfg),
        "structure": report["structure"],
        "views": report["views"].to_dict("records"),
        "paired": report["paired"].to_dict("records"),
    }
    paths["json"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {name: str(path) for name, path in paths.items()}


def run_checkpoint_diagnostics(
    cfg=None, *, checkpoint_path: str | None = None
) -> pd.DataFrame:
    """Evaluate an existing M2 checkpoint under axis masks and user shuffling."""
    cfg = modulation.validate_modulation_config(
        cfg or modulation.configure_modulation_dunnhumby_run()
    )
    prepared = modulation._prepare(cfg)
    checkpoint = (
        Path(checkpoint_path)
        if checkpoint_path
        else find_modulation_checkpoint(prepared["out_dir"])
    )
    model = modulation._build_model(prepared, cfg)
    load_modulation_checkpoint(model, checkpoint, cfg, prepared["input_hash"])
    structure = modulation_structure(model)

    metrics_by_view = {}
    per_user = {}
    for view in ("none", "n_only", "v_only", "both"):
        metrics_by_view[view], per_user[view] = _evaluate_view(
            model, prepared, view
        )
    metrics_by_view["shuffled_user"], per_user["shuffled_user"] = (
        _evaluate_shuffled_user(model, prepared, cfg.seed)
    )
    views = pd.DataFrame(
        [_metric_row(view, metrics_by_view[view]) for view in VIEW_MODES]
    )
    paired = pd.DataFrame(
        _paired_rows(per_user, prepared["base_cfg"]["N_BOOT"])
    )
    report = {
        "checkpoint": str(checkpoint),
        "structure": structure,
        "views": views,
        "paired": paired,
    }
    paths = _persist(report, cfg)
    views.attrs["paired"] = paired
    views.attrs["structure"] = structure
    views.attrs["paths"] = paths
    views.attrs["checkpoint"] = str(checkpoint)
    model.set_eval_axes("both")
    print("M2 modulation checkpoint views:")
    print(views.to_string(index=False))
    print("\nPaired delta vs none:")
    print(paired.to_string(index=False))
    print("\nSaved files:", paths)
    return views


if __name__ == "__main__":
    print(
        "No training is started automatically. "
        "Call run_checkpoint_diagnostics() explicitly."
    )

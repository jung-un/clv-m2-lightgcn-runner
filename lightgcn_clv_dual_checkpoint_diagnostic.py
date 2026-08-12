"""Re-evaluate trained dual-axis CLV adapters without any model training."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import lightgcn_clv_dual as dual
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
from clv_dual_axis_model import CLVDualAxisEmbeddingModel, DualItemProfile
from clv_moe_features import UserProfileArtifact


CODE_VERSION = "clv-dual-checkpoint-diagnostic-v1.0"
AXIS_MODES = ("n_only", "v_only", "n_plus_v")
QUADRANTS = ("low_low", "activity", "value", "core")


def diagnostic_spec(dataset: str, window_days: int | None) -> dict:
    """Return the pre-specified local grid around each existing operating point."""
    if dataset == "dunnhumby" and window_days is None:
        return {
            "gate_shape": "equal",
            "lambdas": (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0),
        }
    if dataset == "hm" and window_days == 60:
        return {
            "gate_shape": "high",
            "lambdas": (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
        }
    raise ValueError("진단 범위는 Dunnhumby 전체 또는 H&M 60일만 지원합니다")


def _quadrant_labels(q_n, q_v, valid, eval_users):
    q_n, q_v = np.asarray(q_n), np.asarray(q_v)
    valid, users = np.asarray(valid, bool), np.asarray(eval_users, np.int64)
    labels = np.full(len(users), "invalid", dtype=object)
    usable = valid[users]
    n_high, v_high = q_n[users] >= 0.5, q_v[users] >= 0.5
    labels[usable & ~n_high & ~v_high] = "low_low"
    labels[usable & n_high & ~v_high] = "activity"
    labels[usable & ~n_high & v_high] = "value"
    labels[usable & n_high & v_high] = "core"
    return labels


def quadrant_metrics(
    *,
    q_n,
    q_v,
    valid,
    eval_users,
    model_per_user,
    baseline_per_user,
    model_id,
    lam,
    n_boot,
    gate_shape="unknown",
):
    """Summarize paired M1 gains in four train-only CLV-axis quadrants."""
    labels = _quadrant_labels(q_n, q_v, valid, eval_users)
    rows = []
    for quadrant in QUADRANTS:
        mask = labels == quadrant
        for metric in ("recall", "ndcg", "revenue", "arp"):
            baseline = np.asarray(baseline_per_user[metric], float)[mask]
            model = np.asarray(model_per_user[metric], float)[mask]
            if len(model):
                diff = model - baseline
                ci = v3.paired_bootstrap([diff], n_boot)
                row = {
                    "baseline_mean": float(baseline.mean()),
                    "model_mean": float(model.mean()),
                    "mean_delta": float(diff.mean()),
                    "median_delta": float(np.median(diff)),
                    "improved_user_share": float(np.mean(diff > 0)),
                    "lo": ci["lo"],
                    "hi": ci["hi"],
                }
            else:
                row = {
                    key: float("nan")
                    for key in (
                        "baseline_mean",
                        "model_mean",
                        "mean_delta",
                        "median_delta",
                        "improved_user_share",
                        "lo",
                        "hi",
                    )
                }
            rows.append(
                {
                    "model_id": model_id,
                    "gate_shape": gate_shape,
                    "lambda": float(lam),
                    "quadrant": quadrant,
                    "metric": metric,
                    "user_count": int(mask.sum()),
                    **row,
                }
            )
    return pd.DataFrame(rows)


def _validate_payload(payload: dict) -> None:
    config = payload.get("config", {})
    if config.get("eval_test") or config.get("eval_holdout"):
        raise ValueError("checkpoint 진단은 validation-only입니다")
    if tuple(config.get("seed_list", ())) != (42,):
        raise ValueError("checkpoint 진단은 seed 42 validation-only입니다")
    paths, hashes = payload.get("checkpoint_paths", {}), payload.get(
        "checkpoint_sha256", {}
    )
    required = ("m1_s42", "encoder_s42", "dual_clv_fixed_s42")
    missing = [name for name in required if name not in paths or name not in hashes]
    if missing:
        raise ValueError(f"필수 checkpoint 정보 누락: {missing}")


def _verify_checkpoints(payload: dict) -> dict[str, Path]:
    verified = {}
    for name, expected in payload["checkpoint_sha256"].items():
        if name not in payload["checkpoint_paths"]:
            raise RuntimeError(f"checkpoint path 누락: {name}")
        path = Path(payload["checkpoint_paths"][name])
        if not path.is_file():
            raise RuntimeError(f"checkpoint 파일 없음: {path}")
        actual = moe.file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"checkpoint hash 불일치: {name}")
        verified[name] = path
    return verified


def _configure_base(payload: dict, m1_path: Path) -> dict:
    dataset = payload["config"]["dataset"]
    overrides = {
        key: value
        for key, value in payload["base_config"].items()
        if key not in {"DATASET", "OUT_DIR"}
    }
    overrides.update(EVAL_TEST=False, EVAL_HOLDOUT=False, SEED_LIST=[42])
    base_cfg = dict(
        v3.configure_run(
            dataset=dataset, out_dir=str(m1_path.parent), **overrides
        )
    )
    for key, expected in (
        ("ARCH", "pref_only"),
        ("GRAPH_MODE", "binary"),
        ("LOSS_MODE", "plain"),
        ("NEG_MODE", "uniform"),
    ):
        if base_cfg[key] != expected:
            raise RuntimeError(f"M1 기준설정 오염: {key}={base_cfg[key]!r}")
    return base_cfg


def _validate_inputs(payload: dict) -> dict:
    current = moe.build_input_manifest(v3.DCFG)
    saved = payload.get("input_manifest", {})
    if not saved or moe.manifest_hash(current) != moe.manifest_hash(saved):
        raise RuntimeError("원본 데이터 manifest 불일치")
    return current


def _load_models(payload, paths, data, base_cfg, x_item, item_cat):
    encoder_blob = torch.load(
        paths["encoder_s42"], map_location="cpu", weights_only=False
    )
    if encoder_blob.get("source_revision") != payload["source_revision"]:
        raise RuntimeError("encoder/source revision 불일치")
    clv_proxy = np.asarray(encoder_blob["clv_proxy_all"], np.float32)

    base = v3.build_model(data, data["x_val_u"], x_item, item_cat, base_cfg)
    base_blob = torch.load(paths["m1_s42"], map_location=v3.DEVICE)
    base.load_state_dict(base_blob["state"])
    base.eval()

    dual_blob = torch.load(
        paths["dual_clv_fixed_s42"], map_location=v3.DEVICE, weights_only=False
    )
    if dual_blob.get("source_revision") != payload["source_revision"]:
        raise RuntimeError("dual/source revision 불일치")
    baseline_hash = moe.state_hash(base)
    if dual_blob.get("baseline_state_hash") != baseline_hash:
        raise RuntimeError("dual checkpoint의 M1 state hash 불일치")

    user_names = tuple(dual_blob["user_feature_names"])
    activity_names = tuple(dual_blob["item_activity_names"])
    value_names = tuple(dual_blob["item_value_names"])
    dummy_user = UserProfileArtifact(
        np.zeros((data["n_users"], len(user_names)), np.float32),
        np.ones(data["n_users"], bool),
        user_names,
    )
    dummy_item = DualItemProfile(
        np.zeros((data["n_items"], len(activity_names)), np.float32),
        np.zeros((data["n_items"], len(value_names)), np.float32),
        np.ones(data["n_items"], bool),
        activity_names,
        value_names,
    )
    model = CLVDualAxisEmbeddingModel(
        base,
        dummy_user,
        dummy_item,
        np.zeros(data["n_users"], np.float32),
        np.zeros(data["n_users"], np.float32),
        control=dual.PRIMARY_MODEL,
        seed=42,
        hidden_dim=int(payload["config"]["expert_hidden_dim"]),
        expert_dim=int(payload["config"]["expert_dim"]),
    ).to(v3.DEVICE)
    model.load_state_dict(dual_blob["state"])
    model.eval()
    return base, model, clv_proxy, baseline_hash


def _axis_ratio(axis_mode: str, diagnostics: dict) -> float:
    key = {
        "n_only": "effective_n_ratio",
        "v_only": "effective_v_ratio",
        "n_plus_v": "effective_total_ratio",
    }[axis_mode]
    return float(diagnostics[key])


def _overall_delta(axis_mode, gate_shape, lam, per_user, baseline, n_boot):
    rows = []
    for metric in ("recall", "ndcg", "revenue", "arp"):
        diff = np.asarray(per_user[metric]) - np.asarray(baseline[metric])
        rows.append(
            {
                "model_id": axis_mode,
                "gate_shape": gate_shape,
                "lambda": float(lam),
                "metric": metric,
                **v3.paired_bootstrap([diff], n_boot),
            }
        )
    return rows


def _matched_strength_table(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis_mode in AXIS_MODES:
        source = curves[curves.model_id.eq(axis_mode)].sort_values(
            "effective_strength"
        )
        strength = source.effective_strength.to_numpy(float)
        revenue = source["revenue@10"].to_numpy(float)
        for _, target in curves[curves.model_id.eq("n_plus_v")].iterrows():
            nearest = source.iloc[
                np.abs(
                    source.effective_strength.to_numpy(float)
                    - float(target.effective_strength)
                ).argmin()
            ]
            rows.append(
                {
                    "reference_model": "n_plus_v",
                    "reference_lambda": float(target["lambda"]),
                    "reference_effective_strength": float(
                        target.effective_strength
                    ),
                    "compared_model": axis_mode,
                    "matched_lambda": float(nearest["lambda"]),
                    "matched_effective_strength": float(
                        nearest.effective_strength
                    ),
                    "strength_gap": float(
                        nearest.effective_strength - target.effective_strength
                    ),
                    "reference_revenue@10": float(target["revenue@10"]),
                    "compared_revenue@10": float(nearest["revenue@10"]),
                    "nearest_revenue_delta": float(
                        target["revenue@10"] - nearest["revenue@10"]
                    ),
                    "interpolated_compared_revenue@10": (
                        float(
                            np.interp(
                                float(target.effective_strength), strength, revenue
                            )
                        )
                        if strength[0]
                        <= float(target.effective_strength)
                        <= strength[-1]
                        else float("nan")
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table["interpolated_revenue_delta"] = (
        table["reference_revenue@10"]
        - table["interpolated_compared_revenue@10"]
    )
    return table


def _plot_curves(curves: pd.DataFrame, output: Path, dataset: str) -> list[str]:
    paths = []
    for x, suffix in (("lambda", "lambda"), ("effective_strength", "strength")):
        figure, axis = plt.subplots(figsize=(9, 5))
        for model_id in AXIS_MODES:
            part = curves[curves.model_id.eq(model_id)].sort_values(x)
            axis.plot(part[x], part["revenue@10"], marker="o", label=model_id)
        axis.set(xlabel=x, ylabel="weighted hit@10", title=f"{dataset}: axis diagnostic")
        axis.legend()
        axis.grid(alpha=0.25)
        path = output / f"{dataset}_{suffix}_curve.png"
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        paths.append(str(path))
    return paths


def run_checkpoint_diagnostic(
    result_json: str | Path, output_dir: str | Path | None = None
) -> pd.DataFrame:
    """Load one finished screening result and run validation-only diagnostics."""
    result_json = Path(result_json)
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    _validate_payload(payload)
    paths = _verify_checkpoints(payload)
    if "base_config" not in payload:
        raise ValueError("원본 결과 JSON에 base_config가 없습니다")
    base_cfg = _configure_base(payload, paths["m1_s42"])
    manifest = _validate_inputs(payload)
    data = v3.prepare_data(base_cfg, v3.DCFG)
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    meta = v3.item_meta(data["train"], data["n_items"])
    base, model, clv_proxy, baseline_hash = _load_models(
        payload, paths, data, base_cfg, x_item, item_cat
    )
    thresholds = v3.segment_thresholds(clv_proxy, base_cfg["SEG_EDGES"])
    cache = v3.EvalCache(
        *data["splits"]["val"], clv_proxy, thresholds, data["n_items"]
    )
    baseline_flat, baseline_per_user = moe._flat_evaluation(
        base, 0.0, cache, meta, data, base_cfg, per_user=True
    )

    spec = diagnostic_spec(payload["config"]["dataset"], payload["config"]["window_days"])
    model.set_gate_shape(spec["gate_shape"])
    diagnostics = model.axis_diagnostics(spec["gate_shape"])
    n_boot = int(base_cfg["N_BOOT"])
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "lambda": 0.0,
            "effective_strength": 0.0,
            **baseline_flat,
        }
    ]
    delta_rows, quadrant_frames = [], []
    for axis_mode in AXIS_MODES:
        model.set_eval_axes(axis_mode)
        ratio = _axis_ratio(axis_mode, diagnostics)
        for lam in spec["lambdas"]:
            flat, per_user = moe._flat_evaluation(
                model, float(lam), cache, meta, data, base_cfg, per_user=True
            )
            rows.append(
                {
                    "seed": 42,
                    "model_id": axis_mode,
                    "split": "val",
                    "gate_shape": spec["gate_shape"],
                    "lambda": float(lam),
                    "effective_strength": float(lam) * ratio,
                    **flat,
                }
            )
            delta_rows.extend(
                _overall_delta(
                    axis_mode,
                    spec["gate_shape"],
                    lam,
                    per_user,
                    baseline_per_user,
                    n_boot,
                )
            )
            quadrant_frames.append(
                quadrant_metrics(
                    q_n=model.q_n.detach().cpu().numpy(),
                    q_v=model.q_v.detach().cpu().numpy(),
                    valid=model.has_profile.detach().cpu().numpy(),
                    eval_users=cache.users,
                    model_per_user=per_user,
                    baseline_per_user=baseline_per_user,
                    model_id=axis_mode,
                    gate_shape=spec["gate_shape"],
                    lam=lam,
                    n_boot=n_boot,
                )
            )

    curves = pd.DataFrame(rows)
    nonbaseline = curves[~curves.model_id.eq("m1")].copy()
    matched = _matched_strength_table(nonbaseline)
    quadrants = pd.concat(quadrant_frames, ignore_index=True)
    output = Path(output_dir or result_json.parent / "checkpoint_diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    stem = f"dual_axis_checkpoint_{payload['config']['dataset']}_{payload['result_fingerprint']}"
    csv_path = output / f"{stem}.csv"
    delta_path = output / f"{stem}_delta.csv"
    quadrant_path = output / f"{stem}_quadrants.csv"
    strength_path = output / f"{stem}_matched_strength.csv"
    json_path = output / f"{stem}.json"
    curves.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    quadrants.to_csv(quadrant_path, index=False)
    matched.to_csv(strength_path, index=False)
    plot_paths = _plot_curves(
        nonbaseline, output, payload["config"]["dataset"]
    )
    report = {
        "code_version": CODE_VERSION,
        "source_revision": moe.source_revision(),
        "original_result_json": str(result_json),
        "original_result_fingerprint": payload["result_fingerprint"],
        "original_source_revision": payload["source_revision"],
        "input_manifest": manifest,
        "baseline_state_hash": baseline_hash,
        "checkpoint_paths": {name: str(path) for name, path in paths.items()},
        "checkpoint_sha256": payload["checkpoint_sha256"],
        "diagnostic_spec": {**spec, "axis_modes": list(AXIS_MODES)},
        "axis_diagnostics": diagnostics,
        "absolute_rows": curves.to_dict("records"),
        "delta_rows": delta_rows,
        "quadrant_rows": quadrants.to_dict("records"),
        "matched_strength_rows": matched.to_dict("records"),
        "interpretation": (
            "single-seed validation exploratory mechanism diagnostic; "
            "not a new screening or confirmation result"
        ),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    curves.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "quadrant_csv": str(quadrant_path),
        "matched_strength_csv": str(strength_path),
        "json": str(json_path),
        "plots": plot_paths,
    }
    curves.attrs["diagnostic_spec"] = spec
    return curves


__all__ = ["diagnostic_spec", "quadrant_metrics", "run_checkpoint_diagnostic"]

"""One development-screen arm: preserve binary LightGCN propagation and add N/V graph deviation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_m3_centered_value_graph as m3
import lightgcn_clv_v3 as v3
import lightgcn_clv_axis_specific_test10 as io


CODE_VERSION = "clv-m3-binary-residual-nv-dev-v1"
MODEL_ID = "m3_binary_residual_nv_bpr_k1"
ALPHA = 0.5
CHECKPOINT_EPOCHS = (100, 300)


def configure(**overrides):
    defaults = {
        "seeds": (43,),
        "epochs": 300,
        "eval_every": 25,
        "out_dir": f"{v3.default_out_dir('dunnhumby')}_clv_m3_binary_residual_nv_s43_v1",
    }
    cfg = m3.configure_centered_graph(**(defaults | overrides))
    if cfg.seeds != (43,) or cfg.epochs != 300 or cfg.allow_baseline_training:
        raise ValueError("이번 스크린은 seed 43·300 epoch·M1 재사용만 허용합니다")
    return cfg


def _original_curves(cfg, beta: float) -> tuple[pd.DataFrame, dict]:
    root = Path(v3.default_out_dir("dunnhumby") + "_clv_m3_centered_value_graph_v1")
    matches = sorted(root.glob("clv_m3_centered_value_graph_*.json"))
    if len(matches) != 1:
        raise RuntimeError(f"기존 N/V M3 전체 결과 JSON을 한 개 찾지 못했습니다: {root}")
    raw = matches[0].read_bytes()
    old = json.loads(raw)
    fixed = ("epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim",
             "pref_reg", "negative_count", "target_cv")
    if any(old["config"][key] != getattr(cfg, key) for key in fixed):
        raise RuntimeError("기존 M3와 새 실험의 학습 설정이 달라 비교할 수 없습니다")
    if 43 not in old["config"]["seeds"]:
        raise RuntimeError("기존 M1·M3의 seed 43 결과가 없습니다")
    old_beta = old["betas"][m3.ARM_VALUE_ACTIVITY]
    if not np.isclose(beta, old_beta, rtol=0, atol=1e-6):
        raise RuntimeError(f"N/V 엣지 입력이 기존 M3와 다릅니다: beta {beta} vs {old_beta}")
    curve = pd.DataFrame(old["curve"])
    curve = curve[curve.model_id.isin((m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY))
                  & curve.seed.eq(43)].copy()
    for model_id in (m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY):
        epochs = set(curve.loc[curve.model_id.eq(model_id), "epoch"])
        if set(CHECKPOINT_EPOCHS) - epochs:
            raise RuntimeError(f"기존 {model_id}의 100·300 epoch 결과가 없습니다")
    return curve, {"path": str(matches[0]), "sha256": hashlib.sha256(raw).hexdigest(),
                   "code_version": old["code_version"],
                   "source_revision": old["source_revision"]}


def _mixed_adjacency(prepared: dict, weights: np.ndarray) -> tuple[torch.Tensor, dict]:
    signals, data = prepared["signals"], prepared["data"]
    users, items = signals["edge_users"], signals["edge_items"]
    n_users, n_items = data["n_users"], data["n_items"]
    binary = v3.build_adj(users, items, np.ones(len(users), np.float32), n_users, n_items)
    weighted = v3.build_adj(users, items, weights.astype(np.float32), n_users, n_items)
    if not torch.equal(binary.indices(), weighted.indices()):
        raise RuntimeError("M1과 N/V M3의 엣지 목록이 다릅니다")
    values = (1.0 - ALPHA) * binary.values() + ALPHA * weighted.values()
    mixed = torch.sparse_coo_tensor(binary.indices(), values, binary.shape,
                                    device=values.device).coalesce()
    if not bool(torch.isfinite(mixed.values()).all()) or bool((mixed.values() <= 0).any()):
        raise RuntimeError("혼합 전파행렬에 비유한값 또는 비양수 엣지가 있습니다")
    if not torch.equal(mixed.indices(), binary.indices()):
        raise RuntimeError("혼합 과정에서 구매 그래프의 엣지가 바뀌었습니다")
    delta = weighted.values() - binary.values()
    audit = {
        "alpha": ALPHA,
        "edges_unchanged": True,
        "relative_operator_change_vs_m1": float(
            torch.linalg.vector_norm(ALPHA * delta) / torch.linalg.vector_norm(binary.values())
        ),
        "max_absolute_operator_change_vs_m1": float((ALPHA * delta).abs().max()),
    }
    return mixed, audit


def self_test() -> None:
    """Small runnable check before starting the expensive development fit."""
    prepared = {"signals": {"edge_users": np.array([0, 0, 1]),
                            "edge_items": np.array([0, 1, 1])},
                "data": {"n_users": 2, "n_items": 2}}
    weights = np.array([0.8, 1.2, 0.9], np.float32)
    mixed, audit = _mixed_adjacency(prepared, weights)
    base = v3.build_adj(prepared["signals"]["edge_users"],
                        prepared["signals"]["edge_items"],
                        np.ones(3, np.float32), 2, 2)
    weighted = v3.build_adj(prepared["signals"]["edge_users"],
                            prepared["signals"]["edge_items"], weights, 2, 2)
    assert torch.equal(mixed.indices(), base.indices())
    assert torch.allclose(mixed.values(), (base.values() + weighted.values()) / 2)
    assert audit["edges_unchanged"] and audit["relative_operator_change_vs_m1"] > 0


def _comparison(curve: pd.DataFrame) -> pd.DataFrame:
    index = curve.set_index(["model_id", "seed", "epoch"])
    metrics = [column for column in curve.columns if "@" in column or column ==
               "user_value_tendency_recommended_price_alignment"]
    rows = []
    for epoch in CHECKPOINT_EPOCHS:
        candidate = index.loc[(MODEL_ID, 43, epoch)]
        for reference in (m3.M1_MODEL_ID, m3.ARM_VALUE_ACTIVITY):
            baseline = index.loc[(reference, 43, epoch)]
            for metric in metrics:
                base = float(baseline[metric])
                value = float(candidate[metric])
                rows.append({"seed": 43, "epoch": epoch, "reference": reference,
                             "metric": metric, "reference_value": base,
                             "candidate_value": value, "delta": value - base,
                             "ratio": value / base if base else float("nan")})
    return pd.DataFrame(rows)


def run(cfg=None) -> dict:
    cfg = configure() if cfg is None else configure(**asdict(cfg))
    prepared = m3._prepare(cfg)
    prepared["config_hash"] = hashlib.sha256(
        f"{CODE_VERSION}:{ALPHA}:{prepared['config_hash']}".encode()
    ).hexdigest()[:12]
    spec = {"model_id": MODEL_ID, "arm": "binary_plus_nv_deviation", "gamma": 1.0,
            "question": "Does preserving M1 propagation plus N/V deviation improve overall results?",
            "mix_alpha": ALPHA, "code_version": CODE_VERSION}
    graph = m3.build_arm_graph(prepared, cfg, spec)
    old_curve, old_reference = _original_curves(
        cfg, graph["beta"]
    )  # Fail before training if reference mismatches.
    graph["adjacency"], operator_audit = _mixed_adjacency(prepared, graph["weights"])
    arm_path = m3._arm_paths(prepared, MODEL_ID, 43)["result"]
    if arm_path.exists():
        cached = json.loads(arm_path.read_text(encoding="utf-8"))
        if (cached.get("source_revision") != prepared["revision"]
                or cached.get("code_version") != CODE_VERSION
                or cached.get("mix_alpha") != ALPHA):
            raise RuntimeError("기존 arm 결과의 코드·강도가 다릅니다. 별도 결과 경로를 사용하세요")
    old_stages = m3.clear_stale_progress(prepared)
    arm = m3._run_arm(prepared, cfg, spec, graph, 43)
    curve = pd.concat([m3.curve_table([arm], {}), old_curve], ignore_index=True)
    if curve.duplicated(["model_id", "seed", "epoch"]).any():
        raise RuntimeError("비교 곡선에 중복 모형·시드·epoch가 있습니다")
    comparison = _comparison(curve)
    at_300 = comparison[comparison.epoch.eq(300)]
    accuracy = [f"{name}@{k}" for name in ("recall", "ndcg") for k in (10, 20, 50)]
    m1 = at_300[at_300.reference.eq(m3.M1_MODEL_ID)].set_index("metric")
    old = at_300[at_300.reference.eq(m3.ARM_VALUE_ACTIVITY)].set_index("metric")
    economic = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
    reading = {
        "development_screen_only": True,
        "fixed_epoch": 300,
        "accuracy_guard_vs_m1": bool((m1.loc[accuracy, "ratio"] >= 0.99).all()),
        "both_economic_at10_above_m1_and_old_m3": bool(
            (m1.loc[list(economic), "delta"] > 0).all()
            and (old.loc[list(economic), "delta"] > 0).all()
        ),
        "significance_claim": False,
    }
    out = Path(cfg.out_dir)
    stem = f"{CODE_VERSION}_{prepared['config_hash']}"
    paths = {"absolute_csv": out / f"{stem}_absolute.csv",
             "comparison_csv": out / f"{stem}_comparison.csv",
             "json": out / f"{stem}.json", "arm_result": arm_path}
    io._atomic_csv(paths["absolute_csv"], curve)
    io._atomic_csv(paths["comparison_csv"], comparison)
    io._atomic_json(paths["json"], {
        "code_version": CODE_VERSION, "config": asdict(cfg), "alpha": ALPHA,
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "graph_audit": graph["audit"], "operator_audit": operator_audit,
        "old_m1_m3_reference": old_reference,
        "dropped_stale_progress": old_stages, "reading": reading,
        "result_paths": {name: str(path) for name, path in paths.items()},
    })
    print(json.dumps({"reading": reading, "operator_audit": operator_audit,
                      "paths": {name: str(path) for name, path in paths.items()}},
                     ensure_ascii=False, indent=2))
    return {"absolute": curve, "comparison": comparison, "reading": reading,
            "paths": paths}

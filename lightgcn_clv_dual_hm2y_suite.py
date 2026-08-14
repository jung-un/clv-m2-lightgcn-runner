"""Fast, resumable H&M full-period M2 suite (seed 42, validation only)."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_dual as dual
import lightgcn_clv_moe as moe
import lightgcn_clv_v3 as v3
from clv_run_state import ProgressStore, RunIdentity, file_sha256


CODE_VERSION = "clv-dual-hm2y-suite-v1.0"
MODELS = ("m1", dual.PRIMARY_MODEL, *dual.CONTROLS)
GATE_SHAPE = "high"
TARGET_RHO = 0.2
BATCH_CANDIDATES = (131072, 65536, 32768)
ACCURACY_METRICS = tuple(
    f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50)
)
STAGES = (
    "prepare_data",
    "batch_preflight",
    "m1",
    "encoder_select",
    "encoder_final",
    dual.PRIMARY_MODEL,
    *dual.CONTROLS,
    "validation_evaluation",
    "comparison_and_decision",
)


def configure_hm2y_suite(**overrides) -> moe.MoEConfig:
    defaults = {
        "seed_list": (42,),
        "eval_test": False,
        "eval_holdout": False,
        "out_dir": f"{v3.default_out_dir('hm')}_clv_dual_hm2y_suite",
        "m1_checkpoint_dir": v3.default_out_dir("hm"),
    }
    return validate_suite_config(
        dual.configure_dual_run("hm", short_hm=False, **(defaults | overrides))
    )


def validate_suite_config(cfg: moe.MoEConfig) -> moe.MoEConfig:
    if cfg.dataset != "hm" or cfg.window_days is not None:
        raise ValueError("이 suite는 H&M 전체기간만 허용합니다")
    if tuple(cfg.seed_list) != (42,):
        raise ValueError("H&M 2년 M2 suite는 seed 42만 허용합니다")
    if cfg.eval_test or cfg.eval_holdout:
        raise ValueError("H&M 2년 M2 suite는 validation-only입니다")
    return dual.validate_dual_config(cfg)


def suite_root(out_dir: str | Path) -> Path:
    return Path(out_dir) / "hm2y_m2_suite"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def suite_identity(config_hash: str, source_revision: str, input_hash: str) -> dict:
    return {
        "code_version": CODE_VERSION,
        "config_hash": str(config_hash),
        "source_revision": str(source_revision),
        "input_hash": str(input_hash),
        "dataset": "hm",
        "period": "full",
        "seed": 42,
    }


class SuiteManifest:
    def __init__(self, root: Path, payload: dict):
        self.root = root
        self.path = root / "run_manifest.json"
        self.payload = payload

    @classmethod
    def open(cls, root: str | Path, identity: dict) -> "SuiteManifest":
        root = Path(root)
        path = root / "run_manifest.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("identity") != identity:
                raise RuntimeError(
                    "suite manifest identity mismatch: "
                    f"expected={identity}, actual={payload.get('identity')}"
                )
        else:
            payload = {
                "identity": identity,
                "selected_batch_size": None,
                "stages": {stage: {"status": "pending"} for stage in STAGES},
                "artifacts": {},
            }
            _atomic_json(path, payload)
        return cls(root, payload)

    def save(self) -> None:
        _atomic_json(self.path, self.payload)

    def stage(self, name: str) -> dict:
        return dict(self.payload["stages"].get(name, {"status": "pending"}))

    def is_completed(self, name: str) -> bool:
        return self.stage(name).get("status") == "completed"

    def start(self, name: str, **fields) -> None:
        self.payload["stages"][name] = {"status": "running", **fields}
        self.save()

    def complete(self, name: str, **fields) -> None:
        self.payload["stages"][name] = {"status": "completed", **fields}
        self.save()

    def fail(self, name: str, error: Exception | str) -> None:
        self.payload["stages"][name] = {
            "status": "failed",
            "error": str(error),
        }
        self.save()

    def set_batch_size(self, batch_size: int) -> None:
        selected = self.payload.get("selected_batch_size")
        if selected is not None and int(selected) != int(batch_size):
            raise RuntimeError("재개 중 batch size를 바꿀 수 없습니다")
        self.payload["selected_batch_size"] = int(batch_size)
        self.save()

    def set_artifact(self, name: str, path: str | Path) -> None:
        path = Path(path)
        self.payload["artifacts"][name] = {
            "path": str(path),
            "sha256": file_sha256(path),
        }
        self.save()

    def artifact(self, name: str) -> Path | None:
        entry = self.payload.get("artifacts", {}).get(name)
        if entry is None:
            return None
        path = Path(entry["path"])
        if not path.exists() or file_sha256(path) != entry["sha256"]:
            raise RuntimeError(f"완료 산출물 hash 불일치: {name}")
        return path


def _config_hash(cfg: moe.MoEConfig) -> str:
    payload = {
        "config": asdict(cfg),
        "batch_candidates": BATCH_CANDIDATES,
        "gate_shape": GATE_SHAPE,
        "target_rho": TARGET_RHO,
        "models": MODELS,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def choose_batch_size(
    candidates: tuple[int, ...], probe: Callable[[int], bool]
) -> int:
    for batch_size in candidates:
        try:
            if probe(int(batch_size)):
                return int(batch_size)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"모든 batch 후보가 CUDA 메모리 점검에 실패했습니다: {candidates}")


def _m1_batch_probe(data: dict, base_cfg: dict, batch_size: int) -> bool:
    """Run one throwaway M1 step; never persist its parameters."""
    probe_cfg = dict(base_cfg, BATCH_SIZE=int(batch_size))
    v3.set_seed(42)
    x_item, item_cat = v3.item_value_features(data["train"], data["n_items"])
    model = v3.build_model(
        data, data["x_val_u"], x_item, item_cat, probe_cfg
    )
    rng = np.random.default_rng(42)
    count = min(int(batch_size), len(data["tr_u"]))
    indices = np.arange(count)
    users = data["tr_u"][indices]
    positives = data["tr_i"][indices]
    negatives = v3.sample_negatives(
        users,
        positives,
        data["n_items"],
        data["pos_key"],
        rng,
        probe_cfg["NEG_MODE"],
        data["item_cat"],
        data["cat_items"],
    )
    tensors = [
        torch.as_tensor(values, dtype=torch.long, device=v3.DEVICE)
        for values in (users, positives, negatives)
    ]
    gate = torch.ones(data["n_users"], device=v3.DEVICE)
    loss, _ = model.bpr_loss(*tensors, gate, 0.0, None)
    loss.backward()
    del loss, tensors, gate, model
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return True


def _progress_store(root: Path, identity: dict, stage: str) -> ProgressStore:
    return ProgressStore(
        root,
        RunIdentity(
            stage=stage,
            model_id=stage,
            seed=42,
            config_hash=identity["config_hash"],
            source_revision=identity["source_revision"],
            input_hash=identity["input_hash"],
        ),
    )


def operating_point(raw_effective_ratio: float) -> dict:
    ratio = float(raw_effective_ratio)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("raw_effective_ratio는 양의 유한값이어야 합니다")
    return {
        "gate_shape": GATE_SHAPE,
        "rho": TARGET_RHO,
        "raw_effective_ratio": ratio,
        "lambda": TARGET_RHO / ratio,
        "effective_strength": TARGET_RHO,
    }


def suite_decision(baseline: dict, fixed: dict, controls: dict[str, dict]) -> dict:
    accuracy_ratios = {
        metric: float(fixed[metric]) / float(baseline[metric])
        for metric in ACCURACY_METRICS
    }
    failed_controls = [
        model_id
        for model_id, row in controls.items()
        if float(fixed["revenue@10"]) <= float(row["revenue@10"])
    ]
    conditions = {
        "six_accuracy_ratios_at_least_0.99": all(
            ratio >= 0.99 for ratio in accuracy_ratios.values()
        ),
        "revenue@10_above_m1": (
            float(fixed["revenue@10"]) > float(baseline["revenue@10"])
        ),
        "revenue@10_above_required_controls": not failed_controls,
    }
    return {
        "success": all(conditions.values()),
        "conditions": conditions,
        "failed_conditions": [
            name for name, passed in conditions.items() if not passed
        ],
        "failed_controls": failed_controls,
        "accuracy_ratios": accuracy_ratios,
        "revenue@10_delta_vs_m1": (
            float(fixed["revenue@10"]) - float(baseline["revenue@10"])
        ),
        "control_revenue@10": {
            name: float(row["revenue@10"]) for name, row in controls.items()
        },
    }


def _load_variant(model_id, prepared, cfg, checkpoint: Path):
    base_model = dual._fresh_base(prepared, seed=42)
    model = dual.CLVDualAxisEmbeddingModel(
        base_model,
        prepared["user_profile"],
        prepared["item_profile"],
        prepared["q_n"],
        prepared["q_v"],
        control=model_id,
        seed=42,
        hidden_dim=cfg.expert_hidden_dim,
        expert_dim=cfg.expert_dim,
    ).to(v3.DEVICE)
    blob = torch.load(checkpoint, map_location=v3.DEVICE, weights_only=False)
    model.load_state_dict(blob["state"])
    moe._set_base_trainable(model.base_model, False)
    model.eval()
    return {
        "model": model,
        "training": blob["training"],
        "diagnostics": blob["diagnostics"],
        "checkpoint": str(checkpoint),
    }


def _evaluate_variant(run: dict, prepared: dict) -> tuple[dict, dict]:
    model = run["model"]
    model.set_eval_axes("n_plus_v")
    model.set_gate_shape(GATE_SHAPE)
    ratio = run["diagnostics"]["gate_shape_diagnostics"][GATE_SHAPE][
        "effective_total_ratio"
    ]
    point = operating_point(ratio)
    flat, per_user = moe._flat_evaluation(
        model,
        point["lambda"],
        prepared["cache"],
        prepared["meta"],
        prepared["data"],
        prepared["base_cfg"],
        per_user=True,
    )
    return {**point, **flat}, per_user


def _persist(cfg, prepared, root, runs, evaluated, decision, manifest):
    rows = [
        {
            "seed": 42,
            "model_id": "m1",
            "split": "val",
            "gate_shape": "none",
            "rho": 0.0,
            "raw_effective_ratio": 0.0,
            "lambda": 0.0,
            "effective_strength": 0.0,
            **prepared["baseline_flat"],
        }
    ]
    delta_rows = []
    for model_id in (dual.PRIMARY_MODEL, *dual.CONTROLS):
        flat, per_user = evaluated[model_id]
        rows.append(
            {
                "seed": 42,
                "model_id": model_id,
                "split": "val",
                **flat,
            }
        )
        for metric in ("recall", "ndcg", "revenue", "arp"):
            diff = per_user[metric] - prepared["baseline_per_user"][metric]
            delta_rows.append(
                {
                    "model_id": model_id,
                    "split": "val",
                    "gate_shape": GATE_SHAPE,
                    "rho": TARGET_RHO,
                    "lambda": flat["lambda"],
                    "metric": metric,
                    **v3.paired_bootstrap(
                        [diff], prepared["base_cfg"]["N_BOOT"]
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    stem = f"clv_dual_hm2y_suite_{prepared['fingerprint']}"
    csv_path = root / f"{stem}.csv"
    delta_path = root / f"{stem}_delta.csv"
    json_path = root / f"{stem}.json"
    frame.to_csv(csv_path, index=False, float_format="%.8f")
    pd.DataFrame(delta_rows).to_csv(delta_path, index=False)
    checkpoints = {
        "m1": prepared["m1_checkpoint"],
        "encoder": prepared["encoder_checkpoint"],
        **{name: run["checkpoint"] for name, run in runs.items()},
    }
    payload = {
        "code_version": CODE_VERSION,
        "source_revision": prepared["revision"],
        "result_fingerprint": prepared["fingerprint"],
        "input_manifest": prepared["manifest"],
        "config": asdict(cfg),
        "actual_batch_size": int(prepared["base_cfg"]["BATCH_SIZE"]),
        "models": list(MODELS),
        "gate_shape": GATE_SHAPE,
        "target_rho": TARGET_RHO,
        "decision": decision,
        "training": {name: run["training"] for name, run in runs.items()},
        "diagnostics": {name: run["diagnostics"] for name, run in runs.items()},
        "checkpoint_paths": checkpoints,
        "checkpoint_sha256": {
            name: file_sha256(path) for name, path in checkpoints.items()
        },
        "absolute_rows": rows,
        "delta": delta_rows,
        "run_manifest": manifest.payload,
        "interpretation": {
            "clv": "historical CLV-related behavior proxy, not future realized CLV",
            "revenue": "price/purchase-amount weighted hit, not incremental revenue",
        },
    }
    _atomic_json(json_path, payload)
    frame.attrs["decision"] = decision
    frame.attrs["result_paths"] = {
        "csv": str(csv_path),
        "delta_csv": str(delta_path),
        "json": str(json_path),
        "manifest": str(manifest.path),
        "progress": str(root / "progress.json"),
    }
    return frame


def run_hm2y_suite(cfg: moe.MoEConfig | None = None) -> pd.DataFrame:
    cfg = validate_suite_config(cfg or configure_hm2y_suite())
    root = suite_root(cfg.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_manifest = moe.build_input_manifest(v3.SCHEMA["hm"])
    input_hash = moe.manifest_hash(input_manifest)
    revision = moe.source_revision()
    identity = suite_identity(_config_hash(cfg), revision, input_hash)
    manifest = SuiteManifest.open(root, identity)
    stores = {stage: _progress_store(root, identity, stage) for stage in STAGES}

    def select_batch(data, base_cfg):
        selected = manifest.payload.get("selected_batch_size")
        if selected is not None:
            return int(selected)
        manifest.start("batch_preflight")
        stores["batch_preflight"].mark_stage(
            "running", candidates=list(BATCH_CANDIDATES)
        )
        chosen = choose_batch_size(
            BATCH_CANDIDATES,
            lambda batch: _m1_batch_probe(data, base_cfg, batch),
        )
        manifest.set_batch_size(chosen)
        manifest.complete("batch_preflight", batch_size=chosen)
        stores["batch_preflight"].mark_complete(batch_size=chosen)
        return chosen

    manifest.start("prepare_data")
    stores["prepare_data"].mark_stage("running")
    encoder_checkpoint = manifest.artifact("encoder")
    try:
        prepared = dual._prepare(
            cfg,
            seed=42,
            encoder_checkpoint=encoder_checkpoint,
            progress_stores=stores,
            batch_selector=select_batch,
        )
    except Exception as error:
        stores["prepare_data"].mark_failed(str(error))
        manifest.fail("prepare_data", error)
        raise
    manifest.complete(
        "prepare_data",
        users=int(prepared["data"]["n_users"]),
        items=int(prepared["data"]["n_items"]),
    )
    stores["prepare_data"].mark_complete(
        users=int(prepared["data"]["n_users"]),
        items=int(prepared["data"]["n_items"]),
    )
    for name, path in (
        ("m1", prepared["m1_checkpoint"]),
        ("encoder", prepared["encoder_checkpoint"]),
    ):
        manifest.set_artifact(name, path)
    m1_sha = file_sha256(prepared["m1_checkpoint"])
    stores["m1"].mark_complete(checkpoint_sha256=m1_sha)
    manifest.complete("m1", checkpoint=prepared["m1_checkpoint"], sha256=m1_sha)
    encoder_sha = file_sha256(prepared["encoder_checkpoint"])
    for stage in ("encoder_select", "encoder_final"):
        stores[stage].mark_complete(checkpoint_sha256=encoder_sha)
        manifest.complete(
            stage,
            checkpoint=prepared["encoder_checkpoint"],
            sha256=encoder_sha,
        )

    runs = {}
    for model_id in (dual.PRIMARY_MODEL, *dual.CONTROLS):
        checkpoint = manifest.artifact(model_id)
        if manifest.is_completed(model_id) and checkpoint is not None:
            runs[model_id] = _load_variant(model_id, prepared, cfg, checkpoint)
            continue
        manifest.start(model_id)
        try:
            run = dual._train_variant(
                model_id,
                dual._fresh_base(prepared, seed=42),
                prepared,
                cfg,
                seed=42,
                gate_shapes=(GATE_SHAPE,),
                lambda_eval=(),
                progress_store=stores[model_id],
            )
        except Exception as error:
            stores[model_id].mark_failed(str(error))
            manifest.fail(model_id, error)
            raise
        runs[model_id] = run
        manifest.set_artifact(model_id, run["checkpoint"])
        checkpoint_sha = file_sha256(run["checkpoint"])
        stores[model_id].mark_complete(checkpoint_sha256=checkpoint_sha)
        manifest.complete(
            model_id, checkpoint=run["checkpoint"], sha256=checkpoint_sha
        )

    manifest.start("validation_evaluation")
    stores["validation_evaluation"].mark_stage("running")
    evaluated = {
        model_id: _evaluate_variant(run, prepared)
        for model_id, run in runs.items()
    }
    manifest.complete("validation_evaluation")
    stores["validation_evaluation"].mark_complete()
    manifest.start("comparison_and_decision")
    stores["comparison_and_decision"].mark_stage("running")
    fixed = evaluated[dual.PRIMARY_MODEL][0]
    controls = {name: evaluated[name][0] for name in dual.CONTROLS}
    decision = suite_decision(prepared["baseline_flat"], fixed, controls)
    manifest.complete(
        "comparison_and_decision", success=bool(decision["success"])
    )
    frame = _persist(cfg, prepared, root, runs, evaluated, decision, manifest)
    manifest.complete(
        "comparison_and_decision",
        success=bool(decision["success"]),
        result_json=frame.attrs["result_paths"]["json"],
    )
    stores["comparison_and_decision"].mark_complete(
        success=bool(decision["success"]),
        result_json=frame.attrs["result_paths"]["json"],
    )
    print("H&M 2년 M2 4모형 판정:", decision)
    print("결과 파일:", frame.attrs["result_paths"])
    return frame


def read_progress(out_dir: str | Path) -> dict:
    path = suite_root(out_dir) / "progress.json"
    if not path.exists():
        return {"status": "not_started", "progress_path": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def preflight_summary(cfg: moe.MoEConfig) -> dict:
    cfg = validate_suite_config(cfg)
    return {
        "code_version": CODE_VERSION,
        "dataset": "hm",
        "period": "full",
        "seed_list": [42],
        "models": list(MODELS),
        "batch_candidates": list(BATCH_CANDIDATES),
        "gate_shape": GATE_SHAPE,
        "target_rho": TARGET_RHO,
        "eval_test": False,
        "eval_holdout": False,
        "out_dir": str(cfg.out_dir),
        "suite_root": str(suite_root(cfg.out_dir)),
    }


if __name__ == "__main__":
    print(json.dumps(preflight_summary(configure_hm2y_suite()), ensure_ascii=False, indent=2))
    print("학습은 Colab의 실행 셀에서만 시작하세요.")

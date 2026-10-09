"""One exploratory M5 fit; reuse the verified seed48 M1, never train controls."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

import clv_m2_nv_conditional_basis_screen as baseline
import clv_m5_m3_m4_user_centered_screen as m4
from clv_m5_shared_attributes_model import JointAttributeLightGCN, build_signals
from clv_run_state import (
    ProgressStore,
    RunIdentity,
    _atomic_json,
    _atomic_torch,
    file_sha256,
)
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_v3 as v3

CODE_VERSION = "clv-m5-shared-attributes-seed48-dev-v1-edge-backward-drive-io-fix"
PREVIOUS_CODE_VERSION = "clv-m5-shared-attributes-seed48-dev-v1-edge-backward-fix"
MODEL_ID = "m5_shared_attributes_nv_learned_graph_user_centered_bpr_k1"
SPLIT = baseline.SPLIT
DURABLE_CHECKPOINT_EVERY = 25


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (48,)
    epochs: int = 300
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    id_dim: int = 64
    pref_reg: float = .001
    negative_count: int = 1
    eta: float = .1
    epsilon: float = .1
    kappa: float = .2
    out_dir: str = baseline.ROOT + "_m5_shared_attributes_seed48_dev_v1_edge_backward_drive_io_fix"
    baseline_json: str = baseline.Config().baseline_json


def configure(**paths):
    if set(paths) - {"out_dir", "baseline_json"}:
        raise ValueError("실행 전 고정값은 변경하지 않습니다. 경로만 지정하세요")
    return validate_config(Config(**paths))


def validate_config(cfg):
    for key, value in asdict(Config()).items():
        if key not in ("out_dir", "baseline_json") and getattr(cfg, key) != value:
            raise ValueError(f"사전 고정값 불일치: {key}")
    if not cfg.out_dir or not cfg.baseline_json:
        raise ValueError("결과·기준 경로 누락")
    return cfg


def validate_prepared(prepared):
    data = prepared["data"]
    if set(data["splits"]) != {"test"} or data["train"].t.max() > 683:
        raise RuntimeError("DAY684~690 개발분할 외 평가자료")
    required = {"TIME_CUTOFF": 690, "EVAL_HOLDOUT": False, "HOLDOUT_DAYS": 0,
                "GRAPH_MODE": "binary", "LOSS_MODE": "plain", "NEG_MODE": "uniform",
                "MIN_ITEM_INTER": 1, "TRAIN_ON_VAL": True, "TEST_DAYS": 7}
    if any(prepared["base_cfg"].get(k) != v for k, v in required.items()):
        raise RuntimeError("분할·카탈로그·sampling 설정 불일치")
    if data.get("loss_w") is not None:
        raise RuntimeError("외부 기본 가중치와 M4-C가 중복됩니다")
    train_keys = np.asarray(data["pos_key"], dtype=np.int64)
    truths, _ = data["splits"]["test"]
    keys = np.concatenate([int(u) * data["n_items"] + np.asarray(items, np.int64)
                           for u, items in truths.items()]) if truths else np.empty(0, np.int64)
    if np.isin(keys, train_keys).any():
        raise RuntimeError("학습쌍이 신규상품 정답에 포함됩니다")


def prepare(cfg=None):
    cfg = validate_config(cfg or configure())
    if not Path(cfg.baseline_json).is_file():
        raise FileNotFoundError(f"기존 seed48 M1 JSON이 필요합니다. 새 M1 학습 없음: {cfg.baseline_json}")
    prepared = recheck._prepare(recheck.configure_component_recheck(out_dir=cfg.out_dir))
    validate_prepared(prepared)
    old, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    products = pd.read_csv(v3.DCFG["item_meta_path"],
                           usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC"])
    data = prepared["data"]
    signals = build_signals(data["train"], products, data["n_users"], data["n_items"])
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("학습형 M3와 binary graph의 엣지집합이 다릅니다")
    weights, weight_audit = m4.row_weights(prepared)
    attributes_hash = hashlib.sha256(products.to_csv(index=False).encode()).hexdigest()
    fingerprint = {"version": CODE_VERSION, "config": asdict(cfg), "source": prepared["revision"],
                   "input": prepared["input_hash"], "baseline_sha": provenance["sha256"],
                   "metadata_sha": attributes_hash}
    preflight = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "split": SPLIT,
        "new_fit_count": 1, "new_model": MODEL_ID, "m1": "existing seed48 reuse only",
        "seed_selection": "existing compatible M1 artifact; not selected by metric; previously exposed seed",
        "train_days": [1, 683], "development_days": [684, 690],
        "final_test": False, "holdout": False, "selection": "none; fixed epoch300",
        "same_forward_one_optimizer": True, "pretraining_or_freeze": False,
        "propagation_backward": "exact first-order edge gradients; chunk65536; no dense node-square dA",
        "checkpoint_io": "epoch-local on Colab VM; Drive recovery checkpoint every25 epochs; no Drive heartbeat",
        "external_score_addition_or_reranking": False, "min_item_interactions": 1,
        "new_item_task": True, "negative": "uniform unseen K=1",
        "m2": "shared subtype8+price8; N/V-conditioned per-feature history modulation; positive item LOO",
        "m3": "learned differentiable bounded edge weights; same train pairs; degree normalization",
        "m4": "existing customer-mass-preserving M4-C lambda=.5",
        "objective": "M4-weighted BPR + unchanged sampled ID L2; no extra shared-parameter L2",
        "clv": "original last365day observed N,V,C=N*V midranks; not future CLV",
        "price_caveat": "basket value V is not unit-price preference; price interaction is a hypothesis",
        "attribute_caveat": "shared attributes and capacity are additional explanations; no CLV attribution from M1-only comparison",
        "screen_rule": "six recall/ndcg ratios>=.99 and both economic@10 strictly>M1 at300",
        "limits": "one exposed development seed; no significance or final success claim",
        "signals": signals["audit"], "metadata_sha256": attributes_hash, "m4_audit": weight_audit}
    prepared["base_cfg"].update(EPOCHS=300, SEED_LIST=[48])
    prepared.update(out_dir=Path(cfg.out_dir), baseline=old, baseline_provenance=provenance,
                    signals=signals, row_weights=weights, preflight=preflight,
                    config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    return cfg, prepared


def build_model(cfg, prepared):
    v3.set_seed(48)
    data = prepared["data"]
    return JointAttributeLightGCN(n_users=data["n_users"], n_items=data["n_items"],
        signals=prepared["signals"], q_n=prepared["q_n"], q_v=prepared["q_v"],
        valid=prepared["clv_valid"], id_dim=cfg.id_dim, n_layers=cfg.n_layers,
        pref_reg=cfg.pref_reg, eta=cfg.eta, epsilon=cfg.epsilon, kappa=cfg.kappa).to(v3.DEVICE)


def _runtime_progress_root(prepared):
    """Keep frequent writes on the Colab VM, never on mounted Drive."""
    base = Path("/content") if Path("/content").is_dir() else Path(prepared["out_dir"])
    return base / ".clv_m5_shared_attributes_seed48_runtime" / prepared["config_hash"]


class LocalFirstProgressStore(ProgressStore):
    """Epoch-local resume with sparse durable snapshots.

    The parent store still records every epoch and minute heartbeat, but its
    root is the Colab VM.  Only evaluation-boundary checkpoints (25 epochs)
    and completion metadata touch Drive.
    """

    def __init__(self, local_root, durable_root, identity, *, max_epoch,
                 durable_every=DURABLE_CHECKPOINT_EVERY,
                 resume_source=None, resume_source_identity=None):
        super().__init__(local_root, identity)
        self.durable_root = Path(durable_root)
        self.max_epoch = int(max_epoch)
        self.durable_every = int(durable_every)
        safe = f"{identity.stage}_{identity.model_id}_s{identity.seed}"
        self.durable_checkpoint = self.durable_root / "resume" / f"{safe}_latest.pt"
        self.durable_state = self.durable_root / "stages" / f"{safe}.durable.json"
        self.durable_complete = self.durable_root / "stages" / f"{safe}.completed.json"
        if self.durable_every <= 0:
            raise ValueError("Drive checkpoint interval must be positive")
        self._bootstrap(resume_source, resume_source_identity)

    @staticmethod
    def _load_exact(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            raise RuntimeError(f"복구 checkpoint를 읽을 수 없습니다(원본 보존): {path}") from error

    def _bootstrap(self, resume_source, resume_source_identity):
        if self.latest_checkpoint.is_file():
            return
        source = self.durable_checkpoint if self.durable_checkpoint.is_file() else None
        expected = asdict(self.identity)
        migrated = False
        if source is None and resume_source and Path(resume_source).is_file():
            source = Path(resume_source)
            if resume_source_identity is None:
                raise RuntimeError("이전 checkpoint 신원이 없습니다")
            expected = dict(resume_source_identity)
            migrated = True
        if source is None:
            return
        payload = self._load_exact(source)
        if payload.get("identity") != expected:
            raise RuntimeError("복구 checkpoint identity mismatch")
        if migrated:
            payload = dict(payload)
            payload["identity"] = asdict(self.identity)
        _atomic_torch(self.latest_checkpoint, payload)
        action = "이전 저장본 이관" if migrated else "Drive 복구본 로드"
        print(f"  [{action}] epoch {int(payload['epoch'])} → Colab 로컬", flush=True)

    def _sync_durable(self, epoch):
        self.durable_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.durable_checkpoint.with_suffix(self.durable_checkpoint.suffix + ".tmp")
        shutil.copyfile(self.latest_checkpoint, temporary)
        os.replace(temporary, self.durable_checkpoint)
        _atomic_json(self.durable_state, {
            "identity": asdict(self.identity),
            "epoch": int(epoch),
            "max_epoch": self.max_epoch,
            "checkpoint_path": str(self.durable_checkpoint),
            "checkpoint_sha256": file_sha256(self.durable_checkpoint),
            "drive_write_policy": f"checkpoint every {self.durable_every} epochs; no heartbeat",
        })
        print(f"  [Drive 복구본] epoch {int(epoch)} 저장", flush=True)

    def save_epoch(self, model, optimizer, rng, **epoch_state):
        path = super().save_epoch(model, optimizer, rng, **epoch_state)
        epoch = int(epoch_state["epoch"])
        if epoch % self.durable_every == 0 or epoch == self.max_epoch:
            self._sync_durable(epoch)
        return path

    def mark_complete(self, **fields):
        payload = super().mark_complete(**fields)
        complete = {
            "identity": asdict(self.identity),
            **payload,
            "checkpoint_path": str(self.durable_checkpoint),
            "drive_write_policy": f"checkpoint every {self.durable_every} epochs; no heartbeat",
        }
        _atomic_json(self.durable_complete, complete)
        return complete


def reuse_prepared_after_memory_fix(cfg, prepared):
    """Bind already prepared immutable inputs to the fixed source, without
    preparing data again or silently abandoning completed old epochs."""
    validate_config(cfg)
    validate_prepared(prepared)
    old_checkpoint = (Path(prepared["out_dir"]) / "progress" / prepared["config_hash"])
    if list(old_checkpoint.rglob("*_latest.pt")):
        raise RuntimeError("이전 epoch checkpoint가 있습니다. 자동 초기화하지 말고 재개 경로를 확인하세요")
    old, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise RuntimeError("기존 prepared와 M1 원본이 달라졌습니다")
    preflight = dict(prepared["preflight"])
    if "metadata_sha256" not in preflight:
        raise RuntimeError("상품 속성 입력 신원이 없는 prepared입니다")
    revision = capacity.moe.source_revision()
    preflight.update(code_version=CODE_VERSION, config=asdict(cfg),
        propagation_backward="exact first-order edge gradients; chunk65536; no dense node-square dA",
        memory_fix_previous_source=prepared["revision"], data_reprepared=False)
    fingerprint = {"version": CODE_VERSION, "config": asdict(cfg), "source": revision,
                   "input": prepared["input_hash"], "baseline_sha": provenance["sha256"],
                   "metadata_sha": preflight["metadata_sha256"]}
    ready = dict(prepared)
    ready.update(out_dir=Path(cfg.out_dir), revision=revision, baseline=old,
                 preflight=preflight,
                 config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    print("[OOM 수정] 기존 데이터·CLV·M4 입력 재사용. 같은 seed48·구조·300epoch, 엣지 역전파만 수정.", flush=True)
    return ready


def reuse_prepared_after_drive_io_fix(cfg, prepared):
    """Reuse prepared arrays and, if present, migrate an exact old checkpoint."""
    validate_config(cfg)
    validate_prepared(prepared)
    previous_preflight = prepared["preflight"]
    if previous_preflight.get("code_version") != PREVIOUS_CODE_VERSION:
        raise RuntimeError("Drive I/O 수정 직전 실행이 아닌 prepared입니다")
    if previous_preflight.get("propagation_backward") != (
            "exact first-order edge gradients; chunk65536; no dense node-square dA"):
        raise RuntimeError("OOM 수정판의 prepared가 아닙니다")
    old, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise RuntimeError("기존 prepared와 M1 원본이 달라졌습니다")

    previous_identity = RunIdentity(
        PREVIOUS_CODE_VERSION, MODEL_ID, 48, prepared["config_hash"],
        prepared["revision"], prepared["input_hash"],
    )
    safe = f"{previous_identity.stage}_{previous_identity.model_id}_s{previous_identity.seed}"
    old_progress = Path(prepared["out_dir"]) / "progress" / prepared["config_hash"]
    old_checkpoint = old_progress / "resume" / f"{safe}_latest.pt"
    old_result = Path(prepared["out_dir"]) / "arms" / prepared["config_hash"] / "m5_s48.json"
    if old_result.is_file():
        raise RuntimeError("기존 실행이 이미 완료됐습니다. 이전 결과를 먼저 확인하세요")

    preflight = dict(previous_preflight)
    revision = capacity.moe.source_revision()
    preflight.update(
        code_version=CODE_VERSION,
        config=asdict(cfg),
        checkpoint_io=(
            "epoch-local on Colab VM; Drive recovery checkpoint every25 epochs; "
            "no Drive heartbeat"
        ),
        drive_io_fix_previous_source=prepared["revision"],
        data_reprepared=False,
    )
    fingerprint = {
        "version": CODE_VERSION,
        "config": asdict(cfg),
        "source": revision,
        "input": prepared["input_hash"],
        "baseline_sha": provenance["sha256"],
        "metadata_sha": preflight["metadata_sha256"],
    }
    ready = dict(prepared)
    ready.update(
        out_dir=Path(cfg.out_dir),
        revision=revision,
        baseline=old,
        preflight=preflight,
        config_hash=hashlib.sha256(
            json.dumps(fingerprint, sort_keys=True).encode()
        ).hexdigest()[:12],
        resume_source_checkpoint=str(old_checkpoint) if old_checkpoint.is_file() else None,
        resume_source_identity=asdict(previous_identity),
    )
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    status = "이전 epoch checkpoint 이관 예정" if old_checkpoint.is_file() else "완료 epoch 없음"
    print(
        f"[Drive I/O 수정] 데이터·CLV·M4 입력 재사용, {status}. "
        "빈번한 진행기록은 Colab 로컬, Drive 복구본은 25epoch마다 저장.",
        flush=True,
    )
    return ready


def run_arm(cfg, prepared):
    identity = RunIdentity(CODE_VERSION, MODEL_ID, 48, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / "m5_s48.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("identity") != asdict(identity) or payload["curve"][-1]["epoch"] != 300:
            raise RuntimeError("완료 결과 신원/epoch 불일치")
        return payload
    model = build_model(cfg, prepared)
    initial = model.representation_diagnostics()
    durable_root = prepared["out_dir"] / "progress" / prepared["config_hash"]
    store = LocalFirstProgressStore(
        _runtime_progress_root(prepared), durable_root, identity,
        max_epoch=cfg.epochs,
        resume_source=prepared.get("resume_source_checkpoint"),
        resume_source_identity=prepared.get("resume_source_identity"),
    )
    print("[학습] M5 seed48 1회만 실행. 학습형 그래프는 고정그래프보다 느릴 수 있습니다.", flush=True)
    curve = capacity._train_curve(model, prepared, cfg,
        {"model_id": MODEL_ID, "condition": "joint_learned_graph_attributes"}, 48,
        store, row_weights=prepared["row_weights"])
    for record in curve:
        if "score_split" in record:
            # Helper's legacy CLV labels refer to the entire attribute block,
            # which is NOT an isolated measurement of the N/V contribution.
            record["attribute_score_split"] = {
                k.replace("clv", "attribute_branch"): v for k, v in record.pop("score_split").items()}
            record["attribute_score_split"]["not_clv_attribution"] = True
    payload = {"identity": asdict(identity), "model_id": MODEL_ID, "seed": 48,
               "curve": curve, "initial_diagnostics": initial,
               "final_diagnostics": model.representation_diagnostics()}
    io._atomic_json(path, payload)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.durable_checkpoint), result_path=str(path))
    return payload


def report(cfg, prepared, arm):
    old = prepared["baseline"]
    metrics = list(old["curve"][-1]["metrics"])
    rows, diagnostics = [], []
    for current in (old, arm):
        for r in current["curve"]:
            if "metrics" not in r:
                continue
            if set(r["metrics"]) != set(metrics):
                raise RuntimeError("M1/M5 전체 metric key가 다릅니다")
            meta = {"model_id": current["model_id"], "seed": 48, "epoch": r["epoch"]}
            rows.append({**meta, **r["metrics"]})
            diagnostics.append({**meta, "loss": r["loss"], "p_correct": r["p_correct"],
                                **r.get("gradient_diagnostics", {}),
                                **r.get("attribute_score_split", {})})
    absolute = pd.DataFrame(rows)
    expected = {(m, ep) for m in (baseline.M1, MODEL_ID) for ep in capacity.evaluation_epochs(cfg)}
    if (absolute.duplicated(["model_id", "epoch"]).any()
            or set(zip(absolute.model_id, absolute.epoch)) != expected
            or not np.isfinite(absolute[metrics].to_numpy(float)).all()):
        raise RuntimeError("평가점 중복/누락/비유한 지표")
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for metric in metrics:
            value, reference = float(at.at[MODEL_ID, metric]), float(at.at[baseline.M1, metric])
            comparisons.append({"seed": 48, "epoch": int(epoch), "metric": metric,
                "m1": reference, "m5": value, "difference": value - reference,
                "relative_percent": 100 * (value / reference - 1) if reference else None})
    final = absolute[absolute.epoch.eq(300)].set_index("model_id")
    accuracy = {m: float(final.at[MODEL_ID, m] / final.at[baseline.M1, m]) for m in baseline.ACCURACY}
    economic = {m: float(final.at[MODEL_ID, m] / final.at[baseline.M1, m]) for m in baseline.ECONOMIC}
    reading = {"accuracy_ratios": accuracy, "economic_ratios": economic,
               "exploratory_screen_rule_passed": min(accuracy.values()) >= .99 and min(economic.values()) > 1,
               "significance_claim": False, "clv_attribution_claim": False, "final_success_claim": False,
               "requires_full_raw_results_and_diagnostics_review": True}
    tables = {"absolute": absolute, "comparison": pd.DataFrame(comparisons),
              "diagnostics": pd.DataFrame(diagnostics)}
    root = Path(cfg.out_dir) / "reports"
    paths = {k: str(root / f"{k}.csv") for k in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    paths["preflight"] = str(Path(cfg.out_dir) / "preflight.json")
    io._atomic_json(Path(paths["json"]), {**prepared["preflight"],
        "source_revision": prepared["revision"], "input_hash": prepared["input_hash"],
        "config_hash": prepared["config_hash"], "baseline_provenance": prepared["baseline_provenance"],
        "reading": reading, "arms": [old, arm], "paths": paths})
    return {**tables, "reading": reading, "paths": paths}


def run(cfg, prepared):
    validate_config(cfg)
    validate_prepared(prepared)
    if prepared["preflight"]["config"] != asdict(cfg):
        raise ValueError("prepare 이후 설정 변경")
    _, provenance = baseline.load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise ValueError("prepare 이후 M1 JSON 변경")
    return report(cfg, prepared, run_arm(cfg, prepared))

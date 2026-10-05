"""Development-only comparison of scalar-gated and N-conditioned RBF matching.

Reuse the verified seed48 M1; train two M2 arms sequentially, not M3/M4/M5.
Both keep ID-only graph propagation and append economic coordinates inside
the same forward/optimizer. No checkpoint or hyperparameter is selected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from clv_m5_n_conditioned_value_basis_model import M5NConditionedValueBasisLightGCN
from clv_run_state import ProgressStore, RunIdentity
import lightgcn_clv_axis_specific_test10 as io
import lightgcn_clv_component_recheck as recheck
import lightgcn_clv_m2_capacity_search as capacity
import lightgcn_clv_v3 as v3


CODE_VERSION = "clv-m2-nv-conditional-basis-dev-v1"
SPLIT = "historical_development_days_684_690"
M1 = "m1_bpr_k1"
OLD = "m2_clv_scaled_value_basis_bpr_k1"
NEW = "m2_nv_conditional_basis_bpr_k1"
ACCURACY = tuple(f"{metric}@{k}" for metric in ("recall", "ndcg") for k in (10, 20, 50))
ECONOMIC = ("price_purchase_amount_weighted_hit@10", "vndcg@10")
ROOT = "/content/drive/MyDrive/논문/data/results_v3_dunnhumby"


class ConditionalBasisLightGCN(M5NConditionedValueBasisLightGCN):
    """e_u = q_C * (T0 + (2*q_N-1)*TN) @ b(q_V), e_i = b(price).

T0 starts at identity and TN at zero: initial scores match the old gate=1
model. TN can receive BPR gradients immediately. Unconstrained matrices
replace the two scalar gate parameters; inherited L2 remains ID-only.
"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs, constant_gate=1.0, economic_propagation=False)
        self.T0 = nn.Parameter(torch.eye(3))
        self.TN = nn.Parameter(torch.zeros(3, 3))

    def economic_coordinates(self):
        basis = self.user_value_basis
        mapped = basis @ self.T0.T
        mapped = mapped + self.user_q_n_centered[:, None] * (basis @ self.TN.T)
        strength = self.user_clv_level * self.user_clv_valid
        return strength[:, None] * mapped, self.item_value_basis

    @torch.no_grad()
    def training_gradient_diagnostics(self):
        result = super().training_gradient_diagnostics()
        for name in ("T0", "TN"):
            param = getattr(self, name)
            result[f"{name}_gradient_norm"] = 0.0 if param.grad is None else float(param.grad.norm())
            result[f"{name}_norm"] = float(param.norm())
        n_part = self.user_q_n_centered[:, None] * (self.user_value_basis @ self.TN.T)
        n_part = n_part * (self.user_clv_valid * self.user_clv_level)[:, None]
        full, _ = self.economic_coordinates()
        result["n_conditional_coordinate_rms"] = float(n_part.square().mean().sqrt())
        result["economic_coordinate_rms"] = float(full.square().mean().sqrt())
        return result

    @torch.no_grad()
    def representation_diagnostics(self):
        return {
            "rho": self.rho, "basis_bandwidth": self.basis_bandwidth,
            "economic_graph_propagation": False, "joint_end_to_end_training": True,
            "external_reranking": False, "explicit_q_n_in_m2": True,
            "explicit_q_v_in_m2": True, "q_c_in_m2": True,
            "n_centering": "2*q_N-1 (percentile midpoint, not sample mean)",
            "n_role": "conditions the learned value-to-item-economic mapping",
            "v_role": "fixed normalized low/mid/high RBF input to learned mapping",
            "q_c_role": "scales the complete user economic block",
            "economic_trainable_parameters": 18, "matrix_l2": 0.0,
            "T0": self.T0.cpu().tolist(), "TN": self.TN.cpu().tolist(),
            **self.training_gradient_diagnostics(),
        }


@dataclass(frozen=True)
class Config:
    seeds: tuple[int, ...] = (48,)
    epochs: int = 300
    eval_every: int = 25
    batch_size: int = 8192
    lr: float = 5e-4
    n_layers: int = 2
    id_dim: int = 64
    pref_reg: float = 1e-3
    negative_count: int = 1
    rho: float = 0.25
    basis_bandwidth: float = 0.25
    out_dir: str = ROOT + "_m2_nv_conditional_basis_seed48_v1"
    baseline_json: str = ROOT + "_clv_reliability_orthogonal_nv_seed48_replication_v1/reports/result.json"


def configure(**paths):
    if set(paths) - {"out_dir", "baseline_json"}:
        raise ValueError("이번 개발 실험은 경로만 변경할 수 있습니다")
    return validate_config(Config(**paths))


def validate_config(cfg):
    defaults = asdict(Config())
    for key, expected in defaults.items():
        if key not in ("out_dir", "baseline_json") and getattr(cfg, key) != expected:
            raise ValueError(f"사전 고정 설정 불일치: {key}")
    if not cfg.out_dir or not cfg.baseline_json:
        raise ValueError("결과 경로와 M1 기준 JSON이 필요합니다")
    return cfg


def load_baseline(cfg, input_hash):
    path = Path(cfg.baseline_json)
    if not path.is_file():
        raise FileNotFoundError(f"기존 seed48 M1 결과가 필요합니다. M1을 새로 학습하지 않습니다: {path}")
    raw = path.read_bytes()
    old = json.loads(raw)
    required = {
        "code_version": "clv-reliability-orthogonal-nv-seed48-replication-v1",
        "source_revision": "b541ba7816143bb38da9461163ea470665eb5de8",
        "input_hash": input_hash, "seed": 48, "split": SPLIT,
        "final_test": False, "holdout": False,
    }
    for key, expected in required.items():
        if key not in old or old[key] != expected:
            raise ValueError(f"M1 재사용 신원 불일치: {key}")
    for key in ("epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim", "pref_reg", "negative_count"):
        if old["config"].get(key) != getattr(cfg, key):
            raise ValueError(f"M1 학습 설정 불일치: {key}")
    if old["config"].get("seeds") != [48]:
        raise ValueError("M1 seed 목록 불일치")
    arms = [a for a in old["arms"] if a.get("model_id") == M1]
    if len(arms) != 1 or any(arms[0].get(k) != required[k] for k in ("seed", "input_hash", "source_revision")):
        raise ValueError("M1 arm 신원 불일치")
    curve = arms[0]["curve"]
    if [r["epoch"] for r in curve] != list(range(1, 301)):
        raise ValueError("M1 300epoch 완료 곡선이 아닙니다")
    expected_epochs = capacity.evaluation_epochs(cfg)
    if [r["epoch"] for r in curve if "metrics" in r] != expected_epochs:
        raise ValueError("M1 평가 격자가 다릅니다")
    return arms[0], {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), **required}


def prepare(cfg=None):
    cfg = validate_config(cfg or configure())
    # Preserve the already verified data/CLV pipeline. Its auxiliary arrays
    # are unused: only data['adj'] and plain BPR enter these M2 arms.
    prepared = recheck._prepare(recheck.configure_component_recheck(out_dir=cfg.out_dir))
    if (set(prepared["data"]["splits"]) != {"test"}
            or prepared["data"]["train"].t.max() > 683):
        raise RuntimeError("개발684~690 분할 외 자료가 있습니다")
    required = {"TIME_CUTOFF": 690, "EVAL_HOLDOUT": False, "HOLDOUT_DAYS": 0,
                "GRAPH_MODE": "binary", "LOSS_MODE": "plain", "NEG_MODE": "uniform",
                "MIN_ITEM_INTER": 1, "TRAIN_ON_VAL": True, "TEST_DAYS": 7}
    if any(prepared["base_cfg"].get(key) != value for key, value in required.items()):
        raise RuntimeError("개발분할·M2 개입경계 설정 불일치")
    if prepared["data"].get("loss_w") is not None:
        raise RuntimeError("M2 단독에 행 가중치가 있습니다")
    prepared["base_cfg"].update(EPOCHS=300, SEED_LIST=[48])
    baseline, provenance = load_baseline(cfg, prepared["input_hash"])
    fingerprint = {"version": CODE_VERSION, "config": asdict(cfg),
                   "source": prepared["revision"], "input": prepared["input_hash"],
                   "baseline_sha": provenance["sha256"]}
    prepared.update(out_dir=Path(cfg.out_dir), baseline=baseline, baseline_provenance=provenance,
                    config_hash=hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()[:12])
    preflight = {
        "code_version": CODE_VERSION, "config": asdict(cfg), "split": SPLIT,
        "final_test": False, "holdout": False, "m1": "reuse only",
        "new_fits_in_order": [OLD, NEW], "same_forward_one_optimizer": True,
        "graph": "binary", "loss": "plain BPR + existing sampled ID L2",
        "negative": "uniform unseen K=1", "min_item_interactions": 1,
        "new_item_task": True, "economic_propagation": False,
        "internal_split_key": "test is historical dev684~690, NOT protected final test698~704",
        "selection": "none; final comparison at 300; 100/200 descriptive only",
        "initialization": "T0=identity, TN=zero; same initial ID and economic scores",
        "gate_vs_matrix_parameters": [2, 18], "matrix_l2": 0.0,
        "capacity_caveat": "more mapping parameters; improvement alone cannot establish N/V attribution",
        "candidate_rule": "six accuracy ratios>=0.99 and both economic@10 strictly>M1",
        "old_comparison": "report all deltas; both economic@10>old is a separate flag",
        "limits": "already exposed development seed48; no significance, CLV attribution or M5 claim",
    }
    io._atomic_json(Path(cfg.out_dir) / "preflight.json", preflight)
    prepared["preflight"] = preflight
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    return cfg, prepared


def build_model(prepared, cfg, model_id):
    if model_id not in (OLD, NEW):
        raise ValueError("M2 두 arm만 새로 학습합니다")
    v3.set_seed(48)
    data = prepared["data"]
    cls = ConditionalBasisLightGCN if model_id == NEW else M5NConditionedValueBasisLightGCN
    extra = {} if model_id == NEW else {"economic_propagation": False}
    return cls(n_users=data["n_users"], n_items=data["n_items"],
               user_q_n=prepared["q_n"], user_q_v=prepared["q_v"], user_q_c=prepared["q_c"],
               user_clv_valid=prepared["clv_valid"], item_price_percentile=prepared["item_amount_percentile"],
               item_price_valid=prepared["item_economic_valid"], adj=data["adj"],
               id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
               rho=cfg.rho, basis_bandwidth=cfg.basis_bandwidth, **extra).to(v3.DEVICE)


def run_arm(cfg, prepared, model_id):
    path = prepared["out_dir"] / "arms" / prepared["config_hash"] / f"{model_id}_s48.json"
    identity = RunIdentity(CODE_VERSION, model_id, 48, prepared["config_hash"],
                           prepared["revision"], prepared["input_hash"])
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("identity") != asdict(identity) or payload["curve"][-1]["epoch"] != 300:
            raise RuntimeError("완료 캐시 신원/epoch 불일치")
        print(f"[cached] {model_id}", flush=True)
        return payload
    model = build_model(prepared, cfg, model_id)
    initial = model.representation_diagnostics()
    store = ProgressStore(prepared["out_dir"] / "progress" / prepared["config_hash"], identity)
    spec = {"model_id": model_id, "condition": "binary_plain_bpr"}
    print(f"[학습] {model_id}: 최대300epoch, 매epoch 저장·자동재개", flush=True)
    curve = capacity._train_curve(model, prepared, cfg, spec, 48, store)
    payload = {"identity": asdict(identity), "model_id": model_id, "seed": 48,
               "curve": curve, "initial_diagnostics": initial,
               "final_diagnostics": model.representation_diagnostics()}
    io._atomic_json(path, payload)
    store.mark_complete(epoch=300, max_epoch=300, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def report(cfg, prepared, arms):
    rows, diagnostics = [], []
    for arm in arms:
        for record in arm["curve"]:
            if "metrics" not in record:
                continue
            meta = {"model_id": arm["model_id"], "seed": 48, "epoch": record["epoch"]}
            rows.append({**meta, **record["metrics"]})
            diagnostics.append({**meta, "loss": record["loss"], "p_correct": record["p_correct"],
                                **record.get("score_split", {}), **record.get("gradient_diagnostics", {})})
    absolute = pd.DataFrame(rows)
    if absolute.duplicated(["model_id", "epoch"]).any():
        raise RuntimeError("중복 평가 결과")
    metrics = list(arms[0]["curve"][-1]["metrics"])
    if not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("평가 지표에 결측/비유한값이 있습니다")
    comparisons = []
    for epoch, group in absolute.groupby("epoch"):
        at = group.set_index("model_id")
        for model_id, reference in ((OLD, M1), (NEW, M1), (NEW, OLD)):
            for metric in metrics:
                value, base = float(at.at[model_id, metric]), float(at.at[reference, metric])
                comparisons.append({"seed": 48, "epoch": int(epoch), "model_id": model_id,
                                    "reference": reference, "metric": metric, "value": value,
                                    "reference_value": base, "delta": value - base,
                                    "ratio": value / base if base else None})
    at = absolute[absolute.epoch.eq(300)].set_index("model_id")
    decisions = {}
    for model_id in (OLD, NEW):
        accuracy = {m: float(at.at[model_id, m] / at.at[M1, m]) for m in ACCURACY}
        economic = {m: float(at.at[model_id, m] / at.at[M1, m]) for m in ECONOMIC}
        decisions[model_id] = {"accuracy_ratios": accuracy, "economic_ratios": economic,
                               "candidate": min(accuracy.values()) >= .99 and min(economic.values()) > 1}
    decisions["new_both_economic_above_old"] = all(at.at[NEW, m] > at.at[OLD, m] for m in ECONOMIC)
    decisions.update(significance_claim=False, clv_attribution_claim=False, m5_claim=False)
    root = Path(cfg.out_dir) / "reports"
    tables = {"absolute": absolute, "comparison": pd.DataFrame(comparisons),
              "diagnostics": pd.DataFrame(diagnostics)}
    paths = {key: str(root / f"{key}.csv") for key in tables}
    for key, table in tables.items():
        io._atomic_csv(Path(paths[key]), table)
    paths["json"] = str(root / "result.json")
    io._atomic_json(Path(paths["json"]), {
        **prepared["preflight"], "source_revision": prepared["revision"],
        "input_hash": prepared["input_hash"], "config_hash": prepared["config_hash"],
        "baseline_provenance": prepared["baseline_provenance"], "reading": decisions,
        "arms": arms, "paths": paths,
    })
    return {**tables, "reading": decisions, "paths": paths}


def run(cfg, prepared):
    validate_config(cfg)
    if prepared["preflight"]["config"] != asdict(cfg):
        raise ValueError("prepare 이후 설정이 바뀌었습니다")
    _, provenance = load_baseline(cfg, prepared["input_hash"])
    if provenance != prepared["baseline_provenance"]:
        raise ValueError("prepare 이후 M1 기준파일이 바뀌었습니다")
    arms = [prepared["baseline"]]
    for model_id in (OLD, NEW):
        arms.append(run_arm(cfg, prepared, model_id))
    return report(cfg, prepared, arms)

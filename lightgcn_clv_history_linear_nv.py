"""Development-only affine-history M2/M5 and parameter-matched q controls.

Reuse-only anchors: never silently retrain M1, old M2, complementary M4/M5.
"""
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import lightgcn_clv_history_m5_strength as base
from clv_history_linear_nv_model import LinearNVHistoryLightGCN
from clv_run_state import ProgressStore, RunIdentity, file_sha256

CODE_VERSION = "history-linear-nv-m2-m5-development-v1"
PRIMARY = ("price_purchase_amount_weighted_hit@10", "vndcg@10")


def configure(**overrides):
    defaults = dict(m4_mode="complementary", dataset="dunnhumby", rhos=(0.05,),
                    out_dir=f"{base.v3.default_out_dir('dunnhumby')}_history_linear_nv_v1")
    cfg = base.configure_strength(**(defaults | overrides))
    validate(cfg)
    return cfg


def validate(cfg):
    base.validate_config(cfg)
    if cfg.dataset != "dunnhumby" or cfg.m4_mode != "complementary" or tuple(cfg.rhos) != (0.05,):
        raise ValueError("First affine comparison fixes Dunnhumby, complementary M4 and rho=0.05")
    return cfg


def new_specs():
    return [dict(model_id=f"{role.lower()}_linear_{condition}", role=role,
                 kind="history_fit", graph="binary", weighted=(role == "M5"),
                 rho=0.05, condition=condition)
            for condition in ("nv", "constant") for role in ("M2", "M5")]


def preflight(cfg):
    validate(cfg)
    return dict(code_version=CODE_VERSION, config=asdict(cfg), split=base.split_name(cfg),
        question="Does jointly learned W[p;q]+b improve overall M2 and M5 outcomes?",
        m2="N/V-component representation; no q_C; raw LOO profiles + q_N/q_V enter affine layers",
        m4="fixed complementary: 1+0.5*q_C*(1-RBF_value_fit), train-row mean normalized",
        control="same 48 extra parameters; q_N/q_V=0.5; N/V histories and masks retained",
        affine_initialization="seeded torch Linear default; paired observed/constant identical",
        affine_l2="existing pref_reg * sum of shared affine parameter squares, once per batch",
        masks="invalid or empty (including LOO singleton) profiles stay zero after affine",
        joint_optimizer=True, pretrained_frozen_components=False, external_reranking=False,
        graph="binary", negative="K=1 uniform unseen", min_item_interactions=1,
        train_pairs_excluded=True, final_test=False, holdout=False, epochs=100,
        primary=list(PRIMARY), primary_reference="M2 vs M1; M5 vs fixed complementary M4",
        guard="all six overall Recall/NDCG >= 99% of matched M1, per-seed AND mean reported",
        secondary="@20/@50, all CLV segments, exposure; old M2/M5 and constant controls",
        reading="exploratory directions, not significance, stability or CLV attribution; no best-epoch selection",
        anchor_policy="4 existing compatible results per seed required; missing/conflicting anchors stop before new training",
        new_fits=len(cfg.seeds)*4)


def load_anchors(prepared, cfg):
    anchors, missing = [], []
    roots = (*cfg.reuse_dirs,
             f"{base.v3.default_out_dir('dunnhumby')}_history_m5_strength_complementary_v1")
    for seed in cfg.seeds:
        for spec in base.arm_specifications(cfg):
            identity = base._identity(prepared, cfg, spec, seed)
            candidates = []
            for root in roots:
                path = Path(root) / "arms" / base._digest(identity) / "result.json"
                if path.is_file():
                    p = json.loads(path.read_text())
                    if p.get("identity") == identity:
                        candidates.append({**p, "origin": "reused_strength", "source_result": str(path)})
            if not candidates:
                p = base._reuse_component(prepared, cfg, spec, seed)
                if p is not None:
                    candidates.append(p)
            if not candidates:
                missing.append(f"seed={seed} {spec['model_id']}")
                continue
            if len({base._digest(p["metrics"]) for p in candidates}) != 1:
                raise RuntimeError(f"Conflicting anchors: {seed} {spec['model_id']}")
            p = candidates[0]
            required = (*base.ACCURACY, *PRIMARY)
            if not all(np.isfinite(p["metrics"].get(k, np.nan)) for k in required):
                raise RuntimeError(f"Incomplete anchor metrics: {seed} {spec['model_id']}")
            anchors.append(p)
    if missing:
        raise RuntimeError("No training started. Compatible anchors missing: " + ", ".join(missing))
    return anchors


def prepare(cfg):
    """Load data and verify anchors; this function never trains any model."""
    validate(cfg)
    prepared = base._prepare(cfg)
    anchors = load_anchors(prepared, cfg)
    prepared["linear_nv_config"] = asdict(cfg)
    prepared["linear_nv_anchors"] = anchors
    print(f"[verified] {len(anchors)} existing anchors; no training performed", flush=True)
    return prepared


def _build(prepared, cfg, spec, seed):
    base.v3.set_seed(seed)
    data = prepared["data"]
    valid = np.asarray(prepared["clv_valid"], bool)
    return LinearNVHistoryLightGCN(
        n_users=data["n_users"], n_items=data["n_items"], history=prepared["history"],
        q_n=np.where(valid, prepared["q_n"], 0), q_v=np.where(valid, prepared["q_v"], 0),
        activity_valid=valid, value_valid=valid, adj=data["adj"], id_dim=cfg.id_dim,
        axis_dim=cfg.history_axis_dim, n_layers=cfg.n_layers, rho=spec["rho"],
        pref_reg=cfg.pref_reg, constant_q=spec["condition"] == "constant").to(base.v3.DEVICE)


def _run_new(prepared, cfg, spec, seed):
    identity = base._identity(prepared, cfg, spec, seed)
    identity["version"] = CODE_VERSION
    for name in (Path(__file__).name, "clv_history_linear_nv_model.py"):
        identity["source_hashes"][name] = file_sha256(Path(__file__).with_name(name))
    key = base._digest(identity)
    root = Path(cfg.out_dir) / "arms" / key
    path = root / "result.json"
    if path.exists():
        p = json.loads(path.read_text())
        if p.get("identity") != identity:
            raise RuntimeError("Cached affine run identity mismatch")
        print(f"[cached] {seed} {spec['model_id']}", flush=True)
        return p
    model = _build(prepared, cfg, spec, seed)
    store = ProgressStore(root / "progress", RunIdentity(
        stage=CODE_VERSION, model_id=spec["model_id"], seed=seed, config_hash=key,
        source_revision="content:"+base._digest(identity["source_hashes"]), input_hash=prepared["input_hash"]))
    training = base.components._train_arm(model, prepared, cfg, spec, seed, store)
    metrics = base.capacity._evaluate(model, prepared)
    payload = dict(**spec, seed=seed, identity=identity, metrics=metrics, origin="trained_affine",
        training=training, checkpoint=str(store.latest_checkpoint),
        checkpoint_sha256=file_sha256(store.latest_checkpoint),
        diagnostics={**model.representation_diagnostics(), **model.training_gradient_diagnostics(),
                     **base.hm_budget._score_share(model, prepared),
                     "gradient_diagnostics_available": any(p.grad is not None for p in model.parameters()),
                     "gradient_scope": "last training batch total objective, including L2"},
        data_diagnostics=prepared.get("data_diagnostics", {}),
        m4_diagnostics=prepared["m4_diagnostics"] if spec["weighted"] else {})
    base.capacity.test10._atomic_json(path, payload)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        result_path=str(path), checkpoint_path=str(store.latest_checkpoint))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def tables(arms, seeds):
    absolute = pd.DataFrame([{**{k: a[k] for k in ("model_id", "role", "seed", "rho", "origin")},
                              **a["metrics"]} for a in arms])
    if absolute.duplicated(["seed", "model_id"]).any():
        raise ValueError("Duplicate seed/model result")
    rows = []
    metric_names = sorted(set().union(*(a["metrics"] for a in arms)))
    for seed, part in absolute.groupby("seed"):
        by_id = part.set_index("model_id")
        for mid, row in by_id.iterrows():
            refs = ["m1"] if mid != "m1" else []
            if row.role == "M5":
                refs += ["m4", mid.replace("m5_", "m2_", 1)]
            if "linear" in mid:
                refs += [f"{row.role.lower()}_rho0.05"]
                if mid.endswith("_nv"):
                    refs += [mid.removesuffix("_nv") + "_constant"]
            for ref in dict.fromkeys(refs):
                if ref not in by_id.index:
                    continue
                for metric in metric_names:
                    b, v = by_id.loc[ref, metric], row[metric]
                    if not isinstance(b, (float, int, np.number)) or not isinstance(v, (float, int, np.number)):
                        continue
                    rows.append(dict(seed=seed, model_id=mid, reference=ref, metric=metric,
                        reference_value=b, value=v, delta=v-b,
                        relative_change_pct=100*(v/b-1) if b else np.nan))
    comparison = pd.DataFrame(rows)
    summary = comparison.groupby(["model_id", "reference", "metric"], as_index=False).agg(
        seeds=("seed", "nunique"), mean_reference=("reference_value", "mean"),
        mean_value=("value", "mean"), mean_delta=("delta", "mean"),
        positive_seeds=("delta", lambda x: int((x > 0).sum())))
    summary["relative_change_pct"] = 100*(summary.mean_value/summary.mean_reference.replace(0, np.nan)-1)
    reading = []
    for spec in new_specs():
        mid = spec["model_id"]
        for label, required_seeds in [(s, {s}) for s in seeds] + [("mean", set(seeds))]:
            checks = []
            economic_ref = "m4" if spec["weighted"] else "m1"
            for ref, metric, ratio in [("m1", m, .99) for m in base.ACCURACY] + [(economic_ref, m, None) for m in PRIMARY]:
                p = comparison[(comparison.model_id == mid) & (comparison.reference == ref)
                               & (comparison.metric == metric) & comparison.seed.isin(required_seeds)]
                complete = set(p.seed) == required_seeds and np.isfinite(p[["value", "reference_value"]].to_numpy()).all()
                checks.append((complete, bool(complete and (p.value.mean() > p.reference_value.mean()
                    if ratio is None else p.value.mean() >= ratio*p.reference_value.mean()))))
            reading.append(dict(model_id=mid, seed=label, complete=all(c[0] for c in checks),
                accuracy_guard=all(c[1] for c in checks[:6]),
                overall_economic_gain=all(c[1] for c in checks[6:]),
                directional_pass=all(c[1] for c in checks), significance_claim=False))
    return absolute, comparison, summary, pd.DataFrame(reading)


def save(arms, cfg):
    result = tables(arms, cfg.seeds)
    root = Path(cfg.out_dir) / "reports" / base._digest(dict(config=asdict(cfg), version=CODE_VERSION))
    paths = {}
    for name, frame in zip(("absolute", "comparison", "summary", "reading"), result):
        path = root / f"{name}.csv"
        base.capacity.test10._atomic_csv(path, frame)
        paths[name] = str(path)
    paths["json"] = str(root / "result.json")
    base.capacity.test10._atomic_json(root / "result.json", dict(preflight=preflight(cfg), arms=arms,
        reading=result[-1].to_dict("records"), paths=paths))
    result[0].attrs["paths"] = paths
    return result[0]


def run(cfg, prepared=None):
    validate(cfg)
    if prepared is None:
        prepared = prepare(cfg)
    if prepared.get("linear_nv_config") != asdict(cfg):
        raise ValueError("Prepared protocol changed; rerun prepare before training")
    arms = list(prepared["linear_nv_anchors"])
    save(arms, cfg)
    for seed in cfg.seeds:
        for spec in new_specs():
            print(f"=== seed {seed} {spec['model_id']} ===", flush=True)
            arms.append(_run_new(prepared, cfg, spec, seed))
            save(arms, cfg)
    return save(arms, cfg)

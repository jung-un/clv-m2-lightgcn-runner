"""Fixed last-seven-day evaluation; ten seeds or explicitly scoped seed49 pilots.

Last weeks were exposed in early research: not an unseen confirmatory split.
No old development references/checkpoints, early stopping or test selection.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import clv_m2_nv_conditional_basis_m5_screen as previous
import lightgcn_clv_m3_dunnhumby_test1 as fixed
import lightgcn_clv_m3_centered_value_graph_hm2y as hm_graph
import lightgcn_clv_m5_nv_economic_positive_weight as economics
import clv_m5_m3_m4_split_nv_hm2y_screen as hm_m5
from clv_run_state import ProgressStore, RunIdentity

v3, io = fixed.v3, fixed.test10
CODE_VERSION = "clv-m2-m5-lastweek-test10-v1"
SEEDS = tuple(range(42, 52))
M1, M2, BASE, FULL = previous.M1, previous.M2, previous.M5_BASE, previous.M5_FULL
MODELS = (M1, BASE, M2, FULL)
M4_B = previous.m5.ARM_M4_B
M3 = previous.m5.m3.ARM_VALUE_ACTIVITY
M4_A = "m4_binary_graph_original_m4_bpr_k1"
M5_A = previous.m5.ARM_A
ACCURACY, ECONOMIC = previous.ACCURACY, previous.ECONOMIC
INTERVALS = {"dunnhumby": (1, 704, 711),
             "hm": ("2018-09-20", "2020-09-15", "2020-09-22")}


@dataclass(frozen=True)
class Config:
    experiment: str = "m2_m5"
    dataset: str = "dunnhumby"
    seeds: tuple = SEEDS
    epochs: int = 300
    id_dim: int = 64
    n_layers: int = 2
    batch_size: int = 8192
    lr: float = .0005
    pref_reg: float = .001
    negative_count: int = 1
    input_days: int = 365
    rho: float = .25
    basis_bandwidth: float = .25
    target_cv: float = .20
    max_degree_correlation: float = .05
    max_price_correlation: float = .20
    max_popularity_correlation: float = .20
    out_dir: str = ""


def configure(dataset="dunnhumby", *, out_dir=None, seeds=SEEDS, experiment="m2_m5"):
    if dataset not in INTERVALS:
        raise ValueError("dataset은 dunnhumby 또는 hm입니다")
    seeds = tuple(seeds)
    if experiment == "dh_m3_m4b_completion":
        if dataset != "dunnhumby" or seeds != (49,):
            raise ValueError("누락 비교는 Dunnhumby seed49만 허용합니다")
        return Config(experiment=experiment, seeds=seeds,
                      out_dir=out_dir or v3.default_out_dir(dataset)+"_m3_m4b_lastweek_seed49_v1")
    if experiment == "dh_m4a_m5a_completion":
        if dataset != "dunnhumby" or seeds != (49,):
            raise ValueError("A형 누락 비교는 Dunnhumby seed49만 허용합니다")
        return Config(experiment=experiment, seeds=seeds,
                      out_dir=out_dir or v3.default_out_dir(dataset)+"_m4a_m5a_lastweek_seed49_v1")
    if experiment == "hm_m4b_m5b":
        if dataset != "hm" or seeds != (49,):
            raise ValueError("H&M M4-B/M5-B 비교는 hm·seed49만 허용합니다")
        return Config(experiment=experiment, dataset="hm", seeds=seeds, batch_size=131072,
                      rho=0., out_dir=out_dir or v3.default_out_dir("hm")+"_m4b_m5b_lastweek_seed49_v1")
    if experiment != "m2_m5":
        raise ValueError("알 수 없는 실험입니다")
    if seeds != SEEDS and not (dataset == "dunnhumby" and seeds == (49,)):
        raise ValueError("고정 10시드 또는 Dunnhumby seed49 예비 실행만 가능합니다")
    suffix = "_m2_m5_lastweek_seed49_pilot_v1" if len(seeds) == 1 else "_m2_m5_lastweek_test10_v1"
    return Config(dataset=dataset, seeds=seeds, batch_size=8192 if dataset == "dunnhumby" else 131072,
                  out_dir=out_dir or v3.default_out_dir(dataset) + suffix)


def models_for(cfg):
    if cfg.experiment == "dh_m3_m4b_completion":
        return (M3, M4_B)
    if cfg.experiment == "dh_m4a_m5a_completion":
        return (M4_A, M5_A)
    return (M1, M4_B, BASE) if cfg.experiment == "hm_m4b_m5b" else MODELS


def base_config(cfg):
    return dict(v3.configure_run(
        cfg.dataset, out_dir=cfg.out_dir, ARCH="pref_only", SEED_LIST=list(cfg.seeds),
        WINDOW_DAYS=None, TIME_CUTOFF=None, VAL_DAYS=0, TEST_DAYS=7, HOLDOUT_DAYS=0,
        TRAIN_ON_VAL=True, EVAL_TEST=True, EVAL_HOLDOUT=False,
        GRAPH_MODE="binary", LOSS_MODE="plain", NEG_MODE="uniform",
        MIN_USER_INTER=1, MIN_ITEM_INTER=1, DIM=cfg.id_dim, N_LAYERS=cfg.n_layers,
        BATCH_SIZE=cfg.batch_size, LR=cfg.lr, PREF_REG=cfg.pref_reg,
        EPOCHS=cfg.epochs, EARLY_STOP=cfg.epochs, REPORT_LEGACY_VALUE_FEATURES=False))


def validate_split(data, cfg):
    stats = data["data_stats"]
    convert = pd.Timestamp if cfg.dataset == "hm" else float
    first, end, last = map(convert, INTERVALS[cfg.dataset])
    bounds = stats["split_boundaries"]
    if (set(data["splits"]) != {"test"} or stats["split_rows"]["val"] != 0
            or stats["split_rows"]["holdout"] != 0
            or convert(stats["source"]["time_min"]) != first
            or convert(stats["source"]["time_max"]) != last
            or convert(bounds["train"]["start_inclusive"]) != first
            or convert(bounds["train"]["end_inclusive"]) != end
            or convert(bounds["test"]["start_exclusive"]) != end
            or convert(bounds["test"]["end_inclusive"]) != last
            or convert(data["train"].t.max()) > end
            or data.get("loss_w") is not None):
        raise RuntimeError("마지막7일 test/앞기간 전체학습/검증·holdout 없음 조건 불일치")


def prepare(cfg):
    if cfg != configure(cfg.dataset, out_dir=cfg.out_dir, seeds=cfg.seeds, experiment=cfg.experiment):
        raise ValueError("사전 고정된 시드·300epoch·설정을 변경할 수 없습니다")
    manifest = fixed.moe.build_input_manifest(v3.SCHEMA[cfg.dataset])
    input_hash, revision = fixed.moe.manifest_hash(manifest), fixed.moe.source_revision()
    base = base_config(cfg)
    data = v3.prepare_data(base, v3.DCFG)
    validate_split(data, cfg)
    data["loss_w"] = None
    train = data["train"]
    snapshot = fixed.residual.build_final_snapshot(train, data["n_users"],
                                                    v3.DCFG["is_date"], cfg.input_days)
    axes = fixed.joint.build_user_axis_inputs(snapshot, data["n_users"])
    qn, qv, qc, valid = fixed.evaluation.build_clv_inputs(axes)
    qn, qv, qc = (np.where(valid, q, 0.).astype(np.float32) for q in (qn, qv, qc))
    econ = economics.build_nv_economic_inputs(
        train, n_users=data["n_users"], n_items=data["n_items"],
        q_n=qn, q_v=qv, q_c=qc, clv_valid=valid, n_bins=4,
        shrinkage_strength=10., degree_bins=10)
    thresholds = v3.segment_thresholds(axes["clv_proxy"], base["SEG_EDGES"])
    signals = fixed.graph.centered_edge_signals(
        hm_graph._purchase_keyed(train) if cfg.dataset == "hm" else train,
        data["n_users"], data["n_items"])
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("M3 엣지와 이진 그래프 TRAIN pair 순서 불일치")
    prep = dict(out_dir=Path(cfg.out_dir), input_hash=input_hash, revision=revision,
                base_cfg=base, data=data, axes=axes, q_n=qn, q_v=qv, q_c=qc,
                clv_valid=valid, signals=signals, q_value=qv.astype(float),
                q_activity=qn.astype(float), meta=v3.item_meta(train, data["n_items"]),
                cache=v3.EvalCache(*data["splits"]["test"], axes["clv_proxy"], thresholds, data["n_items"]))
    for key in ("item_amount_percentile", "item_economic_valid", "user_economic_valid", "user_bin_fit", "item_bin"):
        prep[key] = econ[key]
    graph = fixed.build_graph(prep, cfg)
    hm_comparison = cfg.experiment == "hm_m4b_m5b"
    a_comparison = cfg.experiment == "dh_m4a_m5a_completion"
    if hm_comparison:
        # Preserve the actual H&M B formula; only the training window changed.
        weights, audit = hm_m5.row_weights(prep, econ, check_development_reference=False)
        row_weight, weight_audit = weights["split_nv"], audit["split_nv"]
    else:
        weights, audit = previous.m5.row_weights(prep)
        key = M5_A if a_comparison else BASE
        row_weight, weight_audit = weights[key], audit[key]
    # New training windows must NOT be compared to the old dev row count/weight SHA.
    prep.update(m3_graph=graph, m4_weights=row_weight,
                m3_adj=v3.build_adj(signals["edge_users"], signals["edge_items"],
                                   graph["weights"].astype(np.float32), data["n_users"], data["n_items"]))
    protocol = dict(code_version=CODE_VERSION, config=asdict(cfg), models=list(models_for(cfg)),
                    total_fits=len(models_for(cfg))*len(cfg.seeds), pilot_only=len(cfg.seeds)==1,
                    split="last_seven_days_test", validation=False, holdout=False,
                    intervals=INTERVALS[cfg.dataset], source_revision=revision, input_hash=input_hash,
                    previously_exposed_test=True, unseen_confirmation_claim=False,
                    no_old_reference_reuse=True, no_early_stopping=True, test_evaluations_per_fit=1,
                    test_checkpoint="fixed epoch300 only", clv_input_days=cfg.input_days,
                    m4="split N/V B, invalid rows raw=1, normalized train-row mean",
                    m4_weight_audit=weight_audit, m3_beta=graph["beta"], m3_audit=graph["audit"],
                    m2="q_C*[T0+(2q_N-1)*TN]*b(q_V), item=b(amount percentile); joint optimizer",
                    primary=f"{len(cfg.seeds)}-seed mean economic@10: full > M3+M4 and M1; six accuracy means >= .99*M1",
                    m2_standalone_diagnostic_only=True, significance_claim=False,
                    data_stats=data["data_stats"])
    if hm_comparison:
        protocol.update(m4="(1+.5*q_N)*(1+.5*q_V*price_percentile*fit), train-row mean normalized; H&M invalid CLV axes=0, no extra economic-valid mask",
                        m2="not included", m2_standalone_diagnostic_only=False,
                        primary="seed49 M5-B economic@10 > M1 and identical M4-B; six accuracy metrics >= .99*M1",
                        comparison_scope="M3 addition conditional on M4-B; no factorial synergy or CLV attribution claim")
    if cfg.experiment == "dh_m3_m4b_completion":
        protocol.update(m2="not included", m2_standalone_diagnostic_only=False,
                        no_old_reference_reuse=False,
                        reference_policy="only SHA-pinned lastweek seed49 M1/M5-B; no retraining or reevaluation",
                        primary="M5-B economic@10 > M1/M3/M4-B; six accuracy metrics >= .99*M1 and .99*best standalone; all guards reported separately; new descriptive rule, no replacement of original decision",
                        comparison_scope="complete M1/M3/M4-B/M5-B factorial; descriptive only")
    if a_comparison:
        protocol.update(
            m4="original A: 1+.5*q_C*item_amount_percentile*user_economic_bin_fit; invalid rows raw=1; train-row mean normalized",
            m2="not included", m2_standalone_diagnostic_only=False,
            no_old_reference_reuse=False,
            reference_policy="only SHA-pinned lastweek seed49 M1/M3/M4-B/M5-B; no retraining or reevaluation",
            primary="M5-A economic@10 > M1/M3/M4-A; six accuracy metrics >= .99*M1 and .99*best standalone; A/B comparison descriptive only",
            comparison_scope="add M4-A and M5-A to the existing six-cell A/B factorial; descriptive pilot only")
    identity = dict(version=CODE_VERSION, config=asdict(cfg), input_hash=input_hash,
                    source_revision=revision, weights=weight_audit["sha256"], beta=graph["beta"])
    prep["config_hash"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    root = Path(cfg.out_dir) / prep["config_hash"]
    prep.update(run_dir=root, protocol=protocol)
    path = root / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("같은 실행 경로에 다른 사전설정이 있습니다")
    io._atomic_json(path, protocol)
    print(json.dumps({k:v for k,v in protocol.items() if k != "data_stats"}, ensure_ascii=False, indent=2))
    return prep


def build_model(prep, cfg, model_id, seed):
    if model_id not in models_for(cfg) or seed not in cfg.seeds:
        raise ValueError("사전등록 모형/시드가 아닙니다")
    v3.set_seed(seed)
    data = prep["data"]
    has_m2 = model_id in (M2, FULL)
    cls = previous.basis.ConditionalBasisLightGCN if has_m2 else fixed.M5NConditionedValueBasisLightGCN
    options = {} if has_m2 else dict(constant_gate=1., economic_propagation=False)
    return cls(n_users=data["n_users"], n_items=data["n_items"],
               user_q_n=prep["q_n"], user_q_v=prep["q_v"], user_q_c=prep["q_c"],
               user_clv_valid=prep["clv_valid"], item_price_percentile=prep["item_amount_percentile"],
               item_price_valid=prep["item_economic_valid"],
               adj=prep["m3_adj"] if model_id in (M3, BASE, FULL, M5_A) else data["adj"],
               id_dim=cfg.id_dim, n_layers=cfg.n_layers, pref_reg=cfg.pref_reg,
               rho=cfg.rho if has_m2 else 0., basis_bandwidth=cfg.basis_bandwidth, **options).to(v3.DEVICE)


def run_arm(prep, cfg, model_id, seed):
    identity = RunIdentity(CODE_VERSION, model_id, seed, prep["config_hash"], prep["revision"], prep["input_hash"])
    path = prep["run_dir"] / "arms" / f"{model_id}_s{seed}.json"
    if path.exists():
        row = json.loads(path.read_text())
        if row.get("identity") != asdict(identity) or row.get("epochs") != cfg.epochs:
            raise RuntimeError("완료 결과 신원 불일치")
        print(f"[재사용] 이번 분할 완료 결과: {model_id} seed{seed}")
        return row
    model = build_model(prep, cfg, model_id, seed)
    store = ProgressStore(prep["run_dir"] / "progress", identity)
    weights = prep["m4_weights"] if model_id in (M4_B, BASE, FULL, M4_A, M5_A) else None
    history = fixed._train(model, prep, cfg, model_id, seed, store, row_weights=weights)
    if len(history) != cfg.epochs or history[-1]["epoch"] != cfg.epochs:
        raise RuntimeError("고정 epoch 학습을 완료하지 않아 test를 평가하지 않습니다")
    metrics = fixed._evaluate(model, prep)  # Only evaluation call; no epoch loop test access.
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError("최종 평가 지표에 비유한값이 있습니다")
    diagnostic = model.representation_diagnostics() if model_id in (M2, FULL) else {"rho": 0.}
    diagnostic.update(graph="centered_nv" if model_id in (M3,BASE,FULL,M5_A) else "binary",
                      row_weighted=weights is not None,
                      weight_sha256=hashlib.sha256(weights.astype(np.float32).tobytes()).hexdigest() if weights is not None else None)
    row = dict(identity=asdict(identity), model_id=model_id, seed=seed, epochs=cfg.epochs,
               metrics=metrics, training_history=history, diagnostics=diagnostic)
    io._atomic_json(path, row)
    store.mark_complete(epoch=cfg.epochs, max_epoch=cfg.epochs, selection="none",
                        checkpoint_path=str(store.latest_checkpoint), result_path=str(path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def report(prep, cfg, rows):
    absolute = pd.DataFrame([dict(model_id=r["model_id"], seed=r["seed"], **r["metrics"]) for r in rows])
    expected = {(m,s) for m in models_for(cfg) for s in cfg.seeds}
    if (len(absolute) != len(expected) or absolute.duplicated(["model_id","seed"]).any()
            or set(zip(absolute.model_id,absolute.seed)) != expected):
        raise RuntimeError(f"예정된 {len(expected)}개 실행 모두 완료 전에는 판독을 생성하지 않습니다")
    metrics = list(rows[0]["metrics"])
    if not np.isfinite(absolute[metrics].to_numpy(float)).all():
        raise RuntimeError("집계 지표 누락/비유한값")
    means = absolute.groupby("model_id")[metrics].mean()
    summary = absolute.melt(id_vars=["model_id","seed"], var_name="metric", value_name="value")
    summary = summary.groupby(["model_id","metric"]).value.agg(["mean","std","count"]).reset_index()
    paired, changes = [], []
    hm_comparison = cfg.experiment == "hm_m4b_m5b"
    pairs = ((M4_B,M1),(BASE,M1),(BASE,M4_B)) if hm_comparison else ((M2,M1),(BASE,M1),(FULL,BASE),(FULL,M1))
    for model, ref in pairs:
        a = absolute[absolute.model_id.eq(model)].set_index("seed").loc[list(cfg.seeds)]
        b = absolute[absolute.model_id.eq(ref)].set_index("seed").loc[list(cfg.seeds)]
        for metric in metrics:
            delta = a[metric]-b[metric]
            for seed in cfg.seeds:
                paired.append(dict(model_id=model,reference=ref,metric=metric,seed=seed,
                                   value=float(a.at[seed,metric]),reference_value=float(b.at[seed,metric]),
                                   delta=float(delta.at[seed])))
            base_mean, value_mean = float(b[metric].mean()), float(a[metric].mean())
            changes.append(dict(model_id=model,reference=ref,metric=metric,
                                reference_mean=base_mean,mean=value_mean,
                                relative_change_pct=100*(value_mean/base_mean-1) if base_mean else None,
                                paired_delta_mean=float(delta.mean()),
                                paired_delta_std=float(delta.std(ddof=1)) if len(cfg.seeds)>1 else None,
                                seeds_improved=int((delta>0).sum()),seeds_total=len(cfg.seeds)))
    target, reference = (BASE,M4_B) if hm_comparison else (FULL,BASE)
    guard = all(means.at[target,m] >= .99*means.at[M1,m] for m in ACCURACY)
    economic = all(means.at[target,m] > max(means.at[M1,m],means.at[reference,m]) for m in ECONOMIC)
    decision = dict(complete=True,seed_count=len(cfg.seeds),pilot_only=len(cfg.seeds)==1,
                    final_ten_seed_report=len(cfg.seeds)==10,accuracy_mean_guard=bool(guard),
                    both_economic_means_above_m1_and_m3_m4=bool(economic),
                    combination_condition_met=bool(guard and economic),
                    significance_claim=False,clv_attribution_claim=False,previously_exposed_test=True)
    if hm_comparison:
        decision.pop("both_economic_means_above_m1_and_m3_m4")
        decision.update(both_economic_at10_above_m1_and_m4_b=bool(economic),
                        accuracy_guard_vs_m4_b=bool(all(means.at[BASE,m]>=.99*means.at[M4_B,m] for m in ACCURACY)),
                        evaluated_epoch=cfg.epochs,interaction_attribution_claim=False)
    tables = dict(absolute=absolute,summary=summary,comparison=pd.DataFrame(changes),
                  paired_comparison=pd.DataFrame(paired),
                  diagnostics=pd.DataFrame([dict(model_id=r["model_id"],seed=r["seed"],
                                                 **r["diagnostics"]) for r in rows]))
    root = prep["run_dir"] / "reports"
    paths = {k:str(root/f"{k}.csv") for k in tables}
    for key,table in tables.items():
        io._atomic_csv(Path(paths[key]),table)
    paths["json"] = str(root/"result.json")
    io._atomic_json(Path(paths["json"]), dict(protocol=prep["protocol"],reading=decision,
                                             rows=rows,paths=paths))
    return {**tables,"reading":decision,"paths":paths}


def run(cfg, prep):
    if cfg.experiment in ("dh_m3_m4b_completion", "dh_m4a_m5a_completion"):
        raise ValueError("기준 결과 검증이 필요한 실험입니다. clv_dh_m3_m4b_lastweek_completion.run을 사용하세요")
    if json.loads(json.dumps(asdict(cfg))) != json.loads(json.dumps(prep["protocol"]["config"])):
        raise ValueError("prepare 이후 설정이 변경됐습니다")
    rows = []
    for seed in cfg.seeds:
        for model in models_for(cfg):
            print(f"[{len(rows)+1}/{len(models_for(cfg))*len(cfg.seeds)}] {cfg.dataset} / seed{seed} / {model}",flush=True)
            rows.append(run_arm(prep,cfg,model,seed))
    return report(prep,cfg,rows)

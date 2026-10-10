"""Three read-only views of the existing seed48, epoch300 development M5.

No optimizer, fitting, checkpoint selection or final-week data. ID-only is
the same jointly trained M5, not M1; permutation is a functional diagnostic,
not a retrained control or a causal estimate of CLV's learning effect.
"""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

import clv_m5_shared_attributes_centered_seed48_screen as screen
from clv_run_state import file_sha256

VERSION = "clv-m5-shared-attributes-checkpoint-diagnostic-v1"
SOURCE_HASH = "804a67b9c06d"
SOURCE_REVISION = "883ffd084b9b2cec81a0aacc4ad3031fa183ad8b"
INPUT_HASH = "974f1986c80e1ba6541cc8de5e6b31da58f9eeb1317c4b36643b413d748aa0c6"
REPORT_SHA = "60ea04a9bf1c41a7e3b954d55399f07992ce1af23a0ac09fc4840a7a130ec09c"
ARM_SHA = "4aa49b3bbd46716cc731d9e050b1e890e7e72276ea52d6e895591bb5867ce8ea"
PERMUTATION_SEED = 1048
VIEWS = ("full", "joint_id_only", "m2_nv_permuted")
DEFAULT_SOURCE = screen.Config().out_dir
DEFAULT_OUT = "/content/clv_m5_shared_attributes_checkpoint_diagnostic_seed48"


def source_artifacts(source_dir):
    root = Path(source_dir).resolve()
    report_path = root / "reports" / "result.json"
    arm_path = root / "arms" / SOURCE_HASH / f"{screen.MODEL_ID}_s48.json"
    for path, sha in ((report_path, REPORT_SHA), (arm_path, ARM_SHA)):
        if not path.is_file() or file_sha256(path) != sha:
            raise RuntimeError(f"승인된 seed48 원본 결과 누락·변경: {path}. 재학습하지 않습니다")
    report, arm = json.loads(report_path.read_text()), json.loads(arm_path.read_text())
    identity = dict(stage=screen.CODE_VERSION, model_id=screen.MODEL_ID, seed=48,
                    config_hash=SOURCE_HASH, source_revision=SOURCE_REVISION, input_hash=INPUT_HASH)
    config = report["config"]
    screen.configure(**{k: config[k] for k in ("out_dir", "baseline_json", "c_reference_dir")})
    expected = screen.Config()
    for key in ("epochs", "eval_every", "batch_size", "lr", "n_layers", "id_dim",
                "pref_reg", "negative_count", "eta", "epsilon", "target_cv"):
        if config[key] != getattr(expected, key):
            raise RuntimeError(f"원본 설정 불일치: {key}")
    if (config["seeds"] != [48] or report.get("split") != screen.shared.SPLIT
            or report.get("final_test") is not False or report.get("holdout") is not False
            or arm.get("identity") != identity or arm["curve"][-1]["epoch"] != 300
            or len(arm["curve"][-1]["metrics"]) != 142):
        raise RuntimeError("원본 분할·신원·완료시점·전체지표 불일치")
    state, provenance = screen.ranks._checkpoint(root, screen.CODE_VERSION, screen.MODEL_ID,
        48, INPUT_HASH, SOURCE_HASH, SOURCE_REVISION)
    return report, arm, state, provenance


def prepare(report, out_dir):
    """Reuse train-only preparation, without the old C diagnostic or any fit."""
    cfg = screen.configure(out_dir=str(out_dir), baseline_json=report["config"]["baseline_json"],
                           c_reference_dir=report["config"]["c_reference_dir"])
    log = StringIO()
    with redirect_stdout(log):
        _, prepared = screen.shared.prepare(screen.shared.configure(
            out_dir=cfg.out_dir, baseline_json=cfg.baseline_json))
    # Inherited preflight describes the training runner, not this operation.
    print("[진단 준비] 학습자료·CLV·상품속성 읽기 완료. 신규 학습 0회", flush=True)
    screen.shared.validate_prepared(prepared)
    if prepared["input_hash"] != INPUT_HASH:
        raise RuntimeError("원본과 학습·평가 입력 fingerprint가 다릅니다")
    data = prepared["data"]
    signals = screen.m3.centered_edge_signals(data["train"], data["n_users"], data["n_items"])
    if not np.array_equal(signals["edge_users"] * data["n_items"] + signals["edge_items"], data["pos_key"]):
        raise RuntimeError("M3와 M2 엣지집합 불일치")
    valid = np.asarray(prepared["clv_valid"], bool)
    graph_prepared = dict(prepared, signals=signals,
        q_value=np.where(valid, prepared["q_v"], 0), q_activity=np.where(valid, prepared["q_n"], 0))
    graph_cfg = screen.m3.configure_centered_graph(seeds=(48,), out_dir=cfg.out_dir)
    spec = next(s for s in screen.m3.arm_specifications() if s["model_id"] == screen.m3.ARM_VALUE_ACTIVITY)
    graph = screen.m3.build_arm_graph(graph_prepared, graph_cfg, spec)
    graph["weight_sha256"] = hashlib.sha256(graph["weights"].astype(np.float32).tobytes()).hexdigest()
    if (graph["weight_sha256"] != report["m3_weight_sha256"]
            or prepared["preflight"]["m4_audit"]["sha256"] != report["m4_audit"]["sha256"]):
        raise RuntimeError("원본 M3·M4 가중치와 다릅니다")
    prepared.update(graph=graph, graph_prepared=graph_prepared, graph_cfg=graph_cfg)
    screen.io._atomic_json(Path(out_dir) / "preflight.json", dict(code_version=VERSION,
        split=screen.shared.SPLIT, seed=48, epoch=300, new_fit_count=0,
        final_test=False, holdout=False, views=list(VIEWS), permutation_seed=PERMUTATION_SEED,
        source_config_hash=SOURCE_HASH, input_hash=INPUT_HASH,
        m3_weight_sha256=graph["weight_sha256"], m4_audit=prepared["preflight"]["m4_audit"],
        interpretation="read-only inference views; not causal/retrained controls"))
    return cfg, prepared


def joint_permutation(degree, valid):
    """Joint N/V permutation in training-degree deciles, preserving validity.

    Invalid customers remain untouched; tied degrees stay in the same bin.
    A fixed RNG is used once, including possible fixed points (never redraw).
    """
    degree, valid = np.asarray(degree), np.asarray(valid, bool)
    if degree.ndim != 1 or valid.shape != degree.shape or not np.isfinite(degree).all():
        raise ValueError("학습 degree·valid 배열이 잘못됐습니다")
    edges = np.unique(np.quantile(degree[valid], np.arange(.1, 1, .1))) if valid.any() else np.array([])
    bins = np.searchsorted(edges, degree, side="left")
    donor = np.arange(len(degree))
    rng = np.random.default_rng(PERMUTATION_SEED)
    for group in np.unique(bins[valid]):
        members = np.flatnonzero(valid & (bins == group))
        donor[members] = rng.permutation(members)
    if not np.array_equal(np.sort(donor), np.arange(len(degree))) or not np.array_equal(valid[donor], valid):
        raise RuntimeError("N/V 순열의 고객·유효성 보존 실패")
    return donor, bins, edges


class EmbeddingView(torch.nn.Module):
    """Cache original expression dimensions to avoid a changed GEMM shape."""
    def __init__(self, users, items, id_dim):
        super().__init__()
        self.id_dim = id_dim
        self.register_buffer("users", users.detach())
        self.register_buffer("items", items.detach())

    def embeddings(self):
        return (self.users, self.items, self.users.new_zeros((len(self.users), 1)),
                self.items.new_zeros((len(self.items), 1)))


@torch.no_grad()
def embedding_views(model, donor):
    model.eval()
    users, items, _, _ = model.embeddings()
    full = EmbeddingView(users, items, model.id_dim)
    id_users, id_items = users.clone(), items.clone()
    id_users[:, model.id_dim:] = 0
    id_items[:, model.id_dim:] = 0
    original = model.context.clone()
    try:
        model.context.copy_(original[torch.as_tensor(donor, device=original.device)])
        swapped_users, swapped_items, _, _ = model.embeddings()
    finally:
        model.context.copy_(original)
    if not torch.equal(items, swapped_items) or not torch.equal(users[:, :model.id_dim], swapped_users[:, :model.id_dim]):
        raise RuntimeError("M2 N/V 교체가 아이템·ID·M3 경로를 변경했습니다")
    return dict(zip(VIEWS, (full, EmbeddingView(id_users, id_items, model.id_dim),
                            EmbeddingView(swapped_users, swapped_items, model.id_dim))))


def comparisons(metrics):
    rows = []
    for reference in VIEWS[1:]:
        if set(metrics[reference]) != set(metrics["full"]):
            raise RuntimeError("view 전체 지표 키가 다릅니다")
        for key, value in metrics["full"].items():
            base = metrics[reference][key]
            if not np.isfinite([value, base]).all():
                raise RuntimeError("비유한 전체 지표")
            rows.append(dict(seed=48, epoch=300, candidate="full", reference=reference,
                metric=key, reference_value=base, candidate_value=value, delta=value - base,
                relative_change_pct=100 * (value / base - 1) if base else None))
    return pd.DataFrame(rows)


@torch.no_grad()
def boundary_components(views, truth, recommendations, prepared):
    """Truth minus same-user full-ranking cutoff item, split into ID/attribute.

    Fixed full boundaries also compare actual/permuted N/V without the
    confounding movement of the cutoff item. These are direct score margins,
    not accuracy estimates or training attribution.
    """
    id_dim = views["full"].id_dim
    rows = []
    for k in (10, 20, 50):
        boundary = recommendations.loc[recommendations["rank"].eq(k), ["user", "item"]]
        part = truth.merge(boundary.rename(columns={"item": "boundary_item"}), on="user", validate="many_to_one")
        if len(part) != len(truth):
            raise RuntimeError("정답·동일고객 경계상품 결합 누락")
        u = torch.as_tensor(part.user.to_numpy(), device=views["full"].users.device)
        i = torch.as_tensor(part.item.to_numpy(), device=u.device)
        b = torch.as_tensor(part.boundary_item.to_numpy(), device=u.device)
        scores = {}
        for label in ("full", "m2_nv_permuted"):
            view = views[label]
            diff = view.items[i] - view.items[b]
            scores[label] = (view.users[u, id_dim:] * diff[:, id_dim:]).sum(1).cpu().numpy()
        view = views["full"]
        id_margin = (view.users[u, :id_dim] * (view.items[i, :id_dim] - view.items[b, :id_dim])).sum(1).cpu().numpy()
        part = part[["seed", "user", "item", "rank", "boundary_item"]].copy()
        part["k"] = k
        part["id_margin"] = id_margin
        part["actual_attribute_margin"] = scores["full"]
        part["permuted_attribute_margin"] = scores["m2_nv_permuted"]
        part["actual_minus_permuted_margin"] = scores["full"] - scores["m2_nv_permuted"]
        rows.append(part)
    result = pd.concat(rows, ignore_index=True)
    population = screen._population(prepared, 48)
    result = result.merge(population[["user", "segment", "degree_bin", "q_n", "q_v"]], on="user", validate="many_to_one")
    return result


@torch.no_grad()
def mechanism_audit(model, views, prepared):
    """Final scaled message changes; saturation is a fixed edge sample only."""
    users = torch.as_tensor(prepared["cache"].users, device=model.context.device)
    delta = views["full"].users[users, model.id_dim:] - views["m2_nv_permuted"].users[users, model.id_dim:]
    norm = delta.norm(dim=1).cpu().numpy()
    count = min(4096, len(model.edge_users))
    indices = np.linspace(0, len(model.edge_users) - 1, count, dtype=np.int64)
    edges = torch.as_tensor(indices, device=model.context.device)
    attrs = model.attributes()
    inputs = torch.cat([attrs[model.edge_items[edges]], model.context[model.edge_users[edges]],
                        model.relations[edges, 2:3]], dim=1)
    mod = torch.tanh(model.feature_net(inputs))
    return dict(final_scaled_user_attribute_change_norm_mean=float(norm.mean()),
        final_scaled_user_attribute_change_norm_max=float(norm.max()),
        final_scaled_user_attribute_change_norm_median=float(np.median(norm)),
        evaluated_users=len(users), effective_attribute_score_coefficient=model.eta ** 2,
        feature_tanh_sample_edges=count, feature_tanh_abs_ge_095_sample_fraction=float((mod.abs() >= .95).float().mean()),
        sample_note="deterministic evenly spaced train edges; not population saturation or gradient attribution")


def run(source_dir=DEFAULT_SOURCE, out_dir=DEFAULT_OUT):
    source, out = Path(source_dir).resolve(), Path(out_dir).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("진단 출력은 원본 결과와 별도 폴더여야 합니다")
    print("[읽기 진단] Dunnhumby seed48·300epoch·개발684~690일. 새 학습 0", flush=True)
    report, arm, state, provenance = source_artifacts(source)
    cfg, prepared = prepare(report, out)
    model = screen.build_model(cfg, prepared, screen.MODEL_ID)
    # Verify input-derived buffers too; loading must not silently replace data.
    for name, expected in model.named_buffers():
        actual = state[name].to(expected.device)
        if expected.is_sparse:
            good = (torch.equal(expected.coalesce().indices(), actual.coalesce().indices())
                    and torch.equal(expected.coalesce().values(), actual.coalesce().values()))
        else:
            good = torch.equal(expected, actual)
        if not good:
            raise RuntimeError(f"checkpoint 학습입력 buffer 불일치: {name}")
    model.load_state_dict(state, strict=True)
    del state
    donor, bins, edges = joint_permutation(np.diff(prepared["data"]["csr_ptr"]), prepared["clv_valid"])
    views = embedding_views(model, donor)
    audit = mechanism_audit(model, views, prepared)
    del model
    permutation = pd.DataFrame(dict(user=np.arange(len(donor)), donor_user=donor, degree_bin=bins,
        clv_valid=prepared["clv_valid"], changed_donor=donor != np.arange(len(donor))))
    screen.io._atomic_csv(out / "permutation.csv", permutation)
    metrics, truth, recs = {}, {}, {}
    for label in VIEWS:
        metrics[label] = screen.capacity._evaluate(views[label], prepared)
        if label == "full":
            readback = screen._readback(arm["curve"][-1]["metrics"], metrics[label], "full")
            screen.io._atomic_csv(out / "readback.csv", pd.DataFrame(readback))
            if not all(r["passed"] for r in readback):
                raise RuntimeError("원본 전체142지표 재현 불일치. readback.csv 확인; 진단 중단·새 학습 없음")
        records = []
        truth[label] = screen.ranks._score_truth(views[label], prepared, 48, records)
        recs[label] = records
        screen.io._atomic_csv(out / f"{label}_truth.csv", truth[label])
        screen.io._atomic_csv(out / f"{label}_recommendations.csv", pd.DataFrame(records))
        print(f"[진단 완료] {label}: 전체지표·정답순위·추천목록", flush=True)
    absolute = pd.DataFrame([dict(view=k, seed=48, epoch=300, **v) for k, v in metrics.items()])
    comparison = comparisons(metrics)
    screen.io._atomic_csv(out / "absolute.csv", absolute)
    screen.io._atomic_csv(out / "comparison.csv", comparison)
    pairs = {}
    for reference in VIEWS[1:]:
        pair_recs = [dict(r, model="m1") for r in recs[reference]] + [dict(r, model="m5") for r in recs["full"]]
        pairs[reference] = screen._save_pair(prepared, truth[reference], truth["full"], pair_recs,
            48, f"full_vs_{reference}", (reference, "full"))
        summary = pd.read_csv(pairs[reference]["summary"])
        for k in (10, 20, 50):
            row = summary.loc[summary.grouping.eq("all") & summary.k.eq(k)].iloc[0]
            for observed, key in ((row.mean_recall_delta, f"recall@{k}"),
                                  (row.mean_weighted_hit_delta, f"price_purchase_amount_weighted_hit@{k}")):
                if not np.isclose(observed, metrics["full"][key] - metrics[reference][key], atol=1e-7, rtol=1e-5):
                    raise RuntimeError("정답 이동과 전체 적중 지표 차이 불일치")
    components = boundary_components(views, truth["full"], pd.DataFrame(recs["full"]), prepared)
    screen.io._atomic_csv(out / "boundary_score_components.csv", components)
    parts = []
    for column in ("actual_attribute_margin", "actual_minus_permuted_margin"):
        part = components.assign(positive=components[column] > 1e-8, negative=components[column] < -1e-8)
        for all_users in (False, True):
            group = part.assign(segment="all") if all_users else part
            summary = group.groupby(["segment", "k"], observed=True).agg(
                truth_pairs=("item", "size"), mean_margin=(column, "mean"),
                median_margin=(column, "median"), positive_fraction=("positive", "mean"),
                negative_fraction=("negative", "mean")).reset_index()
            parts.append(summary.assign(component=column))
    screen.io._atomic_csv(out / "boundary_component_summary.csv", pd.concat(parts, ignore_index=True))
    screen.io._atomic_json(out / "result.json", dict(code_version=VERSION, seed=48, epoch=300,
        split=screen.shared.SPLIT, source_result_sha256=REPORT_SHA, source_arm_sha256=ARM_SHA,
        checkpoint=provenance, original_full_readback_passed=True, views=list(VIEWS),
        permutation_seed=PERMUTATION_SEED, degree_decile_edges=edges.tolist(),
        permutation_changed_users=int(permutation.changed_donor.sum()),
        permutation_changed_eval_users=int(permutation.set_index("user").loc[prepared["cache"].users, "changed_donor"].sum()),
        permutation_note="N/V jointly permuted in training-degree deciles; invalid untouched; no redraw",
        m3_m4_unchanged=True, new_training=0, checkpoint_selection=False, final_test=False, holdout=False,
        mechanism=audit, margin_direction_tolerance=1e-8,
        id_only_is_m1=False, causal_attribution_claim=False, significance_claim=False,
        next_training_authorized=False, pair_paths=pairs,
        caveat="one exposed development checkpoint, one permutation; does not remove indirect N/V learning effects",
        outputs=[p.name for p in sorted(out.glob("*.csv"))]))
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    return dict(out_dir=str(out), zip=archive, result=str(out / "result.json"))

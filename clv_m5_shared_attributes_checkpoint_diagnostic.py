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
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import clv_m5_shared_attributes_centered_seed48_screen as screen
from clv_run_state import file_sha256

VERSION = "clv-m5-shared-attributes-checkpoint-diagnostic-v2-cached-components"
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
    # Same operations as the original embeddings(), but cache the independent
    # ID/item paths. Repeated CUDA sparse propagation need not be bit-identical;
    # numerical equality after recomputation does not test path independence.
    user_id, item_id = model.id_vectors()
    attributes = model.attributes()
    profiles, messages = model.history_vectors(attributes)
    users = torch.cat([user_id, model.eta * profiles], 1)
    items = torch.cat([item_id, model.eta * attributes], 1)
    full = EmbeddingView(users, items, model.id_dim)
    id_users, id_items = users.clone(), items.clone()
    id_users[:, model.id_dim:] = 0
    id_items[:, model.id_dim:] = 0
    original = model.context.clone()
    try:
        model.context.copy_(original[torch.as_tensor(donor, device=original.device)])
        _, swapped_messages = model.history_vectors(attributes)
    finally:
        model.context.copy_(original)
    # Retain the actual full profile and aggregate only the N/V-induced message
    # difference. Identical tuples/invalid customers then have *exactly* zero
    # perturbation, rather than noise from a second reduction of the full base.
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        delta = messages.new_zeros(profiles.shape).index_add(
            0, model.edge_users, swapped_messages - messages)
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    swapped_profiles = profiles + delta / model.degree[:, None].clamp_min(1)
    swapped_users = torch.cat([user_id, model.eta * swapped_profiles], 1)
    if not torch.equal(original, model.context):
        raise RuntimeError("M2 N/V 진단 후 입력 복원 실패")
    if not torch.equal(users[:, :model.id_dim], swapped_users[:, :model.id_dim]):
        raise RuntimeError("M2 N/V 교체가 아이템·ID·M3 경로를 변경했습니다")
    return dict(zip(VIEWS, (full, EmbeddingView(id_users, id_items, model.id_dim),
                            EmbeddingView(swapped_users, items, model.id_dim))))


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


def _failed_context():
    """Recover only this failed runner's cached read-only inference context."""
    traceback = getattr(sys, "last_traceback", None)
    while traceback is not None:
        frame = traceback.tb_frame
        if (frame.f_code.co_name == "run" and Path(frame.f_code.co_filename).name == Path(__file__).name
                and {"prepared", "views", "arm", "provenance"}.issubset(frame.f_locals)):
            return {key: frame.f_locals[key] for key in ("prepared", "views", "arm", "provenance")}
        traceback = traceback.tb_next
    raise RuntimeError("실패 당시 캐시가 런타임에 없습니다. 재학습하지 말고 이 메시지를 알려주세요")


def _saved_tops(recommendations, users):
    part = recommendations.loc[recommendations.model.eq("m5")].copy()
    if (part.duplicated(["user", "rank"]).any() or part.duplicated(["user", "item"]).any()
            or set(part.user.unique()) != set(users)):
        raise RuntimeError("원본 full 추천목록의 고객·상품·순위 키 불일치")
    table = part.pivot(index="user", columns="rank", values="item").reindex(users)
    if set(table.columns) != set(range(1, 51)) or table.isna().any().any():
        raise RuntimeError("원본 고객별 Top50 누락")
    return table.reindex(columns=range(1, 51)).to_numpy(np.int64)


def _capture_evaluation(view, prepared, forced_tops=None):
    """Capture the SAME top50/score used by evaluation, not top100 re-ranking.

    Optional forced_tops replays the saved list to check whether that list
    itself reproduces the recorded metrics; no checkpoint or model is changed.
    """
    score_fn, metric_fn = screen.v3.combined_score_all, screen.v3.score_topk
    cache_users = np.asarray(prepared["cache"].users)
    lookup = {int(u): j for j, u in enumerate(cache_users)}
    pending, captured, order = {}, [], []
    def scores(*args, **kwargs):
        result = score_fn(*args, **kwargs)
        pending["scores"] = result
        return result
    def metrics(top, batch, *args, **kwargs):
        if forced_tops is not None:
            # evaluate also reads top after this callback for exposure metrics.
            # Replace its batch-local array, not the cached/saved lists.
            top[:] = forced_tops[[lookup[int(u)] for u in batch]]
        used = top
        matrix = pending.pop("scores")
        values = matrix.gather(1, torch.as_tensor(used, dtype=torch.long, device=matrix.device)).detach().cpu().numpy()
        captured.append((used.copy(), values))
        order.extend(batch)
        return metric_fn(used, batch, *args, **kwargs)
    with patch.object(screen.v3, "combined_score_all", side_effect=scores), \
         patch.object(screen.v3, "score_topk", side_effect=metrics):
        measured = screen.capacity._evaluate(view, prepared)
    if not np.array_equal(np.asarray(order), cache_users):
        raise RuntimeError("재현 점검 평가고객 순서 불일치")
    return measured, np.concatenate([v[0] for v in captured]), np.concatenate([v[1] for v in captured])


def probe_failed_readback(source_dir=DEFAULT_SOURCE, out_dir=DEFAULT_OUT, *, context=None):
    """Read cached full expressions from the failure; do not prepare/train.

    Preserve the failed readback and strict gate. This probe is not an accepted
    three-view result and cannot promote a checkpoint/model to success.
    """
    context = _failed_context() if context is None else context
    prepared, view = context["prepared"], context["views"]["full"]
    screen.shared.validate_prepared(prepared)
    source, root = Path(source_dir).resolve(), Path(out_dir).resolve() / "readback_ranking_probe"
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("원본과 진단 출력 폴더가 겹칩니다")
    recorded = context["arm"]["curve"][-1]["metrics"]
    users = np.asarray(prepared["cache"].users)
    source_path = source / "new_pair_diagnostic" / "recommendations.csv"
    recommendations = pd.read_csv(source_path)
    saved = _saved_tops(recommendations, users)
    for row, user in enumerate(users):
        a, b = prepared["data"]["csr_ptr"][user:user + 2]
        if np.isin(saved[row], prepared["data"]["csr_items"][a:b]).any():
            raise RuntimeError("저장 추천목록에 학습상품이 포함됐습니다")
    print("[경계 점검] 실패 당시 full 표현 재사용. 데이터 재준비·학습 0", flush=True)
    current_metrics, current, values = _capture_evaluation(view, prepared)
    replay_metrics, _, _ = _capture_evaluation(view, prepared, saved)
    audits, changes, boundaries = [], [], []
    for label, measured in (("current_probe", current_metrics), ("saved_list_replay", replay_metrics)):
        audits.extend(dict(row, probe=label) for row in screen._readback(recorded, measured, label))
    for j, user in enumerate(users):
        old_rank = {int(i): rank for rank, i in enumerate(saved[j], 1)}
        new_rank = {int(i): rank for rank, i in enumerate(current[j], 1)}
        truth = set(prepared["cache"].gt[user])
        for k in (10, 20, 50):
            removed = set(saved[j, :k]) - set(current[j, :k])
            added = set(current[j, :k]) - set(saved[j, :k])
            for item in sorted(removed | added):
                changes.append(dict(seed=48, user=int(user), segment=str(prepared["cache"].seg[j]), k=k,
                    item=int(item), movement="entered_current" if item in added else "exited_current",
                    saved_rank=old_rank.get(int(item)), current_rank=new_rank.get(int(item)), is_truth=item in truth))
            if removed or added:
                boundaries.append(dict(user=int(user), segment=str(prepared["cache"].seg[j]), k=k,
                    current_boundary_item=int(current[j, k - 1]), current_boundary_score=float(values[j, k - 1]),
                    next_item=int(current[j, k]) if k < 50 else None,
                    current_gap_to_next=float(values[j, k - 1] - values[j, k]) if k < 50 else None))
    change_columns = ["seed", "user", "segment", "k", "item", "movement", "saved_rank", "current_rank", "is_truth"]
    boundary_columns = ["user", "segment", "k", "current_boundary_item", "current_boundary_score", "next_item", "current_gap_to_next"]
    screen.io._atomic_csv(root / "metric_readback.csv", pd.DataFrame(audits))
    screen.io._atomic_csv(root / "topk_item_changes.csv", pd.DataFrame(changes, columns=change_columns))
    screen.io._atomic_csv(root / "changed_boundaries.csv", pd.DataFrame(boundaries, columns=boundary_columns))
    repeated = {}
    for k in (10, 20, 50):
        repeated[str(k)] = dict(changed_order_users=int(np.any(saved[:, :k] != current[:, :k], axis=1).sum()),
            changed_set_users=sum(set(a) != set(b) for a, b in zip(saved[:, :k], current[:, :k])))
    info = dict(code_version=VERSION + "-ranking-probe", source_recommendations_sha256=file_sha256(source_path),
        source_checkpoint=context["provenance"], split=screen.shared.SPLIT, seed=48, epoch=300,
        current_probe_readback_passed=all(r["passed"] for r in audits if r["probe"] == "current_probe"),
        saved_list_replay_matches_recorded=all(r["passed"] for r in audits if r["probe"] == "saved_list_replay"),
        recommendation_changes=repeated, strict_gate_bypassed=False, original_readback_modified=False,
        new_training=0, data_preparation=0, final_test=False, holdout=False,
        caveat="saved recommendations came from top100 diagnostic, not original top50 metric call; replay checks that distinction")
    screen.io._atomic_json(root / "result.json", info)
    archive = shutil.make_archive(str(root), "zip", root_dir=root)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)
    return dict(zip=archive, out_dir=str(root))


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
        view_construction="ID/item computed once; full unchanged; N/V message delta deterministically aggregated onto cached full profile",
        m3_m4_unchanged=True, new_training=0, checkpoint_selection=False, final_test=False, holdout=False,
        mechanism=audit, margin_direction_tolerance=1e-8,
        id_only_is_m1=False, causal_attribution_claim=False, significance_claim=False,
        next_training_authorized=False, pair_paths=pairs,
        caveat="one exposed development checkpoint, one permutation; does not remove indirect N/V learning effects",
        outputs=[p.name for p in sorted(out.glob("*.csv"))]))
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    return dict(out_dir=str(out), zip=archive, result=str(out / "result.json"))

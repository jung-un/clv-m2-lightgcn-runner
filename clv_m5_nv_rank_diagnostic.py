"""Read existing M3/M5-B development checkpoints; do not train or select a model.

The proposed N/V-conditioned user representation needs evidence that M5-B's
missed truth items lie near ranking boundaries and that losses vary with N/V.
These descriptive tables cannot establish that a new representation will help.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "clv-m5-nv-rank-diagnostic-v1"
KS = (10, 20, 50)
TOP = 100


def bucket(rank):
    if not np.isfinite(rank):
        return ">100"
    for k, label in ((10, "1-10"), (20, "11-20"), (50, "21-50"), (100, "51-100")):
        if rank <= k:
            return label
    raise ValueError("Unexpected rank")


def movement_tables(reference, candidate, users):
    keys = ["seed", "user", "item"]
    if reference.duplicated(keys).any() or candidate.duplicated(keys).any():
        raise ValueError("Duplicate truth key")
    truth = reference.merge(candidate, on=keys, suffixes=("_m3", "_m5"),
                            how="outer", validate="one_to_one", indicator=True)
    if not truth._merge.eq("both").all():
        raise ValueError("M3 and M5 truth populations differ")
    truth = truth.drop(columns="_merge")
    if not np.array_equal(truth.weight_m3, truth.weight_m5):
        raise ValueError("M3 and M5 truth weights differ")
    truth["weight"] = truth.weight_m3
    truth["m3_band"] = truth.rank_m3.map(bucket)
    truth["m5_band"] = truth.rank_m5.map(bucket)
    output = users.copy()
    if output.duplicated(["seed", "user"]).any():
        raise ValueError("Duplicate evaluation user")
    for k in KS:
        hit3, hit5 = truth.rank_m3.le(k), truth.rank_m5.le(k)
        truth[f"gained@{k}"] = (~hit3 & hit5).astype(int)
        truth[f"lost@{k}"] = (hit3 & ~hit5).astype(int)
        truth[f"net_weight@{k}"] = (hit5.astype(int) - hit3.astype(int)) * truth.weight
        part = truth.groupby(["seed", "user"])[
            [f"gained@{k}", f"lost@{k}", f"net_weight@{k}"]].sum().reset_index()
        output = output.merge(part, on=["seed", "user"], validate="one_to_one")
        output[f"net_recall@{k}"] = (
            output[f"gained@{k}"] - output[f"lost@{k}"]
        ) / output.truth_count
    return truth, output


def summaries(truth, users):
    rows, correlations = [], []
    for seed, part in users.groupby("seed"):
        groups = [("all", "all", part)]
        for column in ("segment", "n_bin", "v_bin", "degree_bin"):
            groups.extend((column, str(label), group) for label, group in part.groupby(column))
        for kind, label, group in groups:
            items = truth[truth.seed.eq(seed) & truth.user.isin(group.user)]
            for k in KS:
                lost = items[items[f"lost@{k}"].eq(1)]
                missed = items[~items.rank_m5.le(k)]
                rows.append(dict(seed=int(seed), grouping=kind, group=label, k=k,
                    users=len(group), truth_gained=int(group[f"gained@{k}"].sum()),
                    truth_lost=int(group[f"lost@{k}"].sum()),
                    users_with_loss=int(group[f"lost@{k}"].gt(0).sum()),
                    lost_to_next5=int(lost.rank_m5.le(k + 5).sum()),
                    m5_missed_truth=len(missed),
                    m5_missed_in_next5=int(missed.rank_m5.le(k + 5).sum()),
                    mean_recall_delta=float(group[f"net_recall@{k}"].mean()),
                    mean_weighted_hit_delta=float(group[f"net_weight@{k}"].mean())))
        valid = part.loc[part.clv_valid].copy()
        for axis in ("q_n", "q_v"):
            for k in KS:
                target = f"net_recall@{k}"
                ranks = valid[[axis, target]].rank(method="average")
                raw = ranks.corr().iloc[0, 1]
                # Descriptive within-degree adjustment; not a causal control.
                residual = ranks - ranks.groupby(valid.degree_bin).transform("mean")
                adjusted = residual.corr().iloc[0, 1]
                correlations.append(dict(seed=int(seed), axis=axis, metric=target,
                    valid_users=len(valid), spearman=raw,
                    within_degree_rank_correlation=adjusted))
    transitions = truth.groupby(["seed", "m3_band", "m5_band"], dropna=False).agg(
        truth_count=("item", "size"), truth_weight_sum=("weight", "sum")).reset_index()
    return pd.DataFrame(rows), pd.DataFrame(correlations), transitions


def train_bins(values, valid, evaluation_users):
    values, valid = np.asarray(values, float), np.asarray(valid, bool)
    valid = valid & np.isfinite(values)
    if not valid.any():
        raise ValueError("No valid training values for bins")
    edges = np.unique(np.quantile(values[valid], [.2, .4, .6, .8]))
    bins = np.searchsorted(edges, values[evaluation_users], side="left") + 1
    return np.where(valid[evaluation_users], bins, 0), edges.tolist()


def _checkpoint(root, stage, model_id, seed, input_hash, config_hash, revision):
    import torch
    from clv_run_state import file_sha256
    pattern = f"progress/{config_hash}/resume/{stage}_{model_id}_s{seed}_latest.pt"
    paths = list(Path(root).glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"기존 체크포인트를 한 개 찾지 못했습니다: {root}/{pattern}")
    path = paths[0]
    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    expected = dict(stage=stage, model_id=model_id, seed=seed, input_hash=input_hash,
                    config_hash=config_hash, source_revision=revision)
    if state.get("epoch") != 300 or any(identity.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"체크포인트의 epoch/입력/모형/출처가 다릅니다: {path}")
    return state["model_state"], dict(path=str(path), sha256=file_sha256(path),
                                      epoch=300, identity=identity)


def _score_truth(model, prepared, seed):
    import torch
    import lightgcn_clv_v3 as v3
    cache, data = prepared["cache"], prepared["data"]
    users = np.asarray(cache.users, np.int64)
    records = []
    model.eval()
    with torch.no_grad():
        up, ip, uv, iv = model.embeddings()
        ones = torch.ones(data["n_users"], device=v3.DEVICE)
        batch_size = int(prepared["base_cfg"]["EVAL_BATCH"])
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            scores = v3.combined_score_all(up, ip, uv, iv, ones, 0.0,
                torch.as_tensor(batch, dtype=torch.long, device=v3.DEVICE))
            for row, user in enumerate(batch):
                a, b = data["csr_ptr"][user:user + 2]
                seen = data["csr_items"][a:b]
                if np.isin(cache.gt[user], seen).any():
                    raise RuntimeError("개발 정답에 학습쌍이 포함됐습니다")
                scores[row, seen] = -1e9
            top_scores, top_items = scores.topk(TOP, dim=1)
            top_scores, top_items = top_scores.cpu().numpy(), top_items.cpu().numpy()
            for row, user in enumerate(batch):
                ranks = {int(item): r + 1 for r, item in enumerate(top_items[row])}
                actual = np.asarray(cache.gt[user], np.int64)
                weights = np.asarray(cache.rev[user], float)
                actual_scores = scores[row, torch.as_tensor(actual, device=v3.DEVICE)].cpu().numpy()
                if len(actual) != len(weights) or not np.isfinite(actual_scores).all():
                    raise RuntimeError("Invalid truth score/value")
                scale = max(float(top_scores[row].std()), 1e-12)
                for item, weight, value in zip(actual, weights, actual_scores):
                    record = dict(seed=seed, user=int(user), item=int(item), weight=float(weight),
                                  rank=ranks.get(int(item), np.nan), truth_score=float(value))
                    for k in KS:
                        margin = float(value - top_scores[row, k - 1])
                        record[f"margin@{k}"] = margin
                        record[f"scaled_margin@{k}"] = margin / scale
                    records.append(record)
    return pd.DataFrame(records)


def run(root="/content/drive/MyDrive/논문/data", seeds=(43, 44)):
    import torch
    import clv_m5_m3_m4_split_nv_screen as screen
    import lightgcn_clv_m3_centered_value_graph as m3
    import lightgcn_clv_m2_capacity_search as capacity
    import lightgcn_clv_axis_specific_test10 as io
    import lightgcn_clv_v3 as v3
    from clv_run_state import file_sha256

    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) - {43, 44}:
        raise ValueError("저장된 개발시드 43/44만 읽습니다")
    # CPU-only diagnostic; it can run while existing GPU training continues.
    v3.DEVICE = torch.device("cpu")
    root = Path(root)
    out = root / "results_v3_dunnhumby_m5_nv_rank_diagnostic_v1"
    cfg = screen.configure(seed=seeds[0], out_dir=str(out))
    print("[사전진단] 기존 M3/M5-B@300 읽기 · 새 학습 0 · 개발684~690일", flush=True)
    prepared = m3._prepare(cfg)
    data, cache = prepared["data"], prepared["cache"]
    if (set(data["splits"]) != {"test"} or data["train"].t.max() > 683
            or prepared["base_cfg"]["MIN_ITEM_INTER"] != 1
            or prepared["base_cfg"]["EVAL_HOLDOUT"]):
        raise RuntimeError("역사적 개발분할 계약이 다릅니다")
    spec = next(s for s in m3.arm_specifications() if s["model_id"] == m3.ARM_VALUE_ACTIVITY)
    graph = m3.build_arm_graph(prepared, cfg, spec)
    graph["adjacency"] = v3.build_adj(prepared["signals"]["edge_users"],
        prepared["signals"]["edge_items"], graph["weights"].astype(np.float32),
        data["n_users"], data["n_items"])
    m3root = root / "results_v3_dunnhumby_clv_m3_centered_value_graph_v1"
    m3paths = list(m3root.glob("clv_m3_centered_value_graph_*.json"))
    if len(m3paths) != 1:
        raise RuntimeError("기존 M3 JSON을 한 개 찾지 못했습니다")
    m3path = m3paths[0]
    m3report = json.loads(m3path.read_text())
    degree = np.diff(data["csr_ptr"])
    population = pd.DataFrame(dict(user=np.asarray(cache.users, np.int64), segment=cache.seg,
        truth_count=[len(cache.gt[u]) for u in cache.users],
        q_n=np.asarray(prepared["q_n"])[cache.users], q_v=np.asarray(prepared["q_v"])[cache.users],
        clv_valid=np.asarray(prepared["clv_valid"], bool)[cache.users], degree=degree[cache.users]))
    edges = {}
    for axis, column in (("q_n", "n_bin"), ("q_v", "v_bin")):
        population[column], edges[column] = train_bins(prepared[axis], prepared["clv_valid"], cache.users)
    population["degree_bin"], edges["degree_bin"] = train_bins(degree, degree > 0, cache.users)
    all_truth, all_users, audits, absolute, sources = [], [], [], [], []
    for seed in seeds:
        m5root = root / f"results_v3_dunnhumby_clv_m5_m3_m4_split_nv_s{seed}_v1"
        m5paths = list(m5root.glob(f"{screen.CODE_VERSION}_*.json"))
        if len(m5paths) != 1:
            raise RuntimeError(f"seed{seed} M5 원본 JSON을 한 개 찾지 못했습니다")
        report_path = m5paths[0]
        report = json.loads(report_path.read_text())
        if (report["input_hash"] != prepared["input_hash"] or report["final_test"]
                or report["holdout"] or report["split"] != "historical_development_days_684_690"
                or report["old_m1_m3_reference"]["sha256"] != file_sha256(m3path)
                or report["config"]["seeds"] != [seed]
                or not np.isclose(report["beta"], graph["beta"], atol=1e-6, rtol=0)):
            raise RuntimeError("M5 원본 입력/분할/그래프/M3 출처가 다릅니다")
        for key in ("epochs", "id_dim", "n_layers", "pref_reg", "negative_count", "batch_size", "lr"):
            if report["config"][key] != getattr(cfg, key):
                raise RuntimeError(f"M5 학습 설정 불일치: {key}")
        source_csv = Path(report["result_paths"]["absolute_csv"])
        sources.append(dict(report=str(report_path), report_sha256=file_sha256(report_path),
                            absolute=str(source_csv), absolute_sha256=file_sha256(source_csv)))
        curve = pd.read_csv(source_csv)
        scored = {}
        for label, model_id, folder, stage, config_hash, revision in (
            ("m3", m3.ARM_VALUE_ACTIVITY, m3root, "centered_graph_dev",
             m3path.stem.rsplit("_", 1)[-1], m3report["source_revision"]),
            ("m5", screen.ARM_B, m5root, "m5_m3_m4_dev",
             report_path.stem.rsplit("_", 1)[-1], report["source_revision"]),
        ):
            expected = curve[curve.model_id.eq(model_id) & curve.seed.eq(seed) & curve.epoch.eq(300)]
            if len(expected) != 1:
                raise RuntimeError("원본 300epoch 성과가 누락/중복됐습니다")
            state, source = _checkpoint(folder, stage, model_id, seed,
                prepared["input_hash"], config_hash, revision)
            sources.append(source)
            model = m3._build_model(prepared, cfg, graph, seed)
            model.load_state_dict(state, strict=True)
            measured = capacity._evaluate(model, prepared)
            for metric, value in measured.items():
                if metric not in expected.columns:
                    raise RuntimeError(f"원본에 지표가 없습니다: {metric}")
                reference = float(expected.iloc[0][metric])
                passed = bool(np.isclose(value, reference, rtol=1e-5, atol=1e-6, equal_nan=True))
                audits.append(dict(seed=seed, model_id=model_id, metric=metric,
                    recorded=reference, readback=float(value), passed=passed))
            io._atomic_csv(out / "readback.csv", pd.DataFrame(audits))
            if not all(row["passed"] for row in audits):
                raise RuntimeError("체크포인트 재현 차이: readback.csv를 확인하세요. 새 학습 없음")
            absolute.append(dict(seed=seed, model_id=model_id, epoch=300, **measured))
            scored[label] = _score_truth(model, prepared, seed)
            del model, state
            print(f"[진단 완료] seed{seed} {label}: 전체지표 재현·정답 순위 기록", flush=True)
        truth, users = movement_tables(scored["m3"], scored["m5"], population.assign(seed=seed))
        for k in KS:
            for column, metric in ((f"net_recall@{k}", f"recall@{k}"),
                                    (f"net_weight@{k}", f"price_purchase_amount_weighted_hit@{k}")):
                expected_delta = absolute[-1][metric] - absolute[-2][metric]
                if not np.isclose(users[column].mean(), expected_delta, atol=1e-6, rtol=1e-5):
                    raise RuntimeError(f"정답 이동량과 전체 성과 차이가 다릅니다: {column}")
        all_truth.append(truth)
        all_users.append(users)
    truth, users = pd.concat(all_truth, ignore_index=True), pd.concat(all_users, ignore_index=True)
    summary, correlations, transitions = summaries(truth, users)
    frames = dict(truth_movements=truth, user_movements=users, summary=summary,
                  axis_correlations=correlations, rank_transitions=transitions,
                  absolute=pd.DataFrame(absolute), readback=pd.DataFrame(audits))
    paths = {key: str(out / f"{key}.csv") for key in frames}
    for key, frame in frames.items():
        io._atomic_csv(Path(paths[key]), frame)
    paths["json"] = str(out / "result.json")
    io._atomic_json(Path(paths["json"]), dict(code_version=VERSION, seeds=list(seeds),
        epoch=300, training=False, final_test=False, holdout=False,
        split="historical_development_days_684_690", input_hash=prepared["input_hash"],
        sources=sources, train_bin_edges=edges, graph_audit=graph["audit"],
        rank_limit=TOP, rank_nan_means="outside top100, not missing truth",
        margin_definition="truth score minus cutoff score; scaled by top100 score std",
        significance_claim=False, nv_causal_attribution=False,
        reading="Descriptive only: inspect losses/gains, N/V and degree bins, and margins. "
                "No automatic training approval, CLV attribution, significance or new-model success claim.",
        paths=paths))
    print(summary[summary.grouping.eq("all")].to_string(index=False))
    print(correlations.to_string(index=False))
    return paths

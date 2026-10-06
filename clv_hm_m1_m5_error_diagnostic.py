"""Read-only H&M M1 versus M3+M4-B candidate-error diagnostic.

This re-scores the existing seed-43, epoch-300 development checkpoints.  It
does not train, select an epoch, touch the final week, or propose a new M2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import clv_m1_m5_error_diagnostic as features
import clv_m5_nv_rank_diagnostic as movement


VERSION = "clv-hm-m1-m5-candidate-error-diagnostic-v1"
SEED = 43
EPOCH = 300
SPLIT = "hm2y_validation_2020-09-02_08"
ALIGNMENT = "user_value_tendency_recommended_price_alignment"


def _readback_match(metric: str, measured: float, recorded: float) -> tuple[bool, float]:
    # This Spearman correlation is sensitive to GPU top-k tie ordering.  The
    # ranking, hit, weighted-hit and exposure metrics retain the strict gate.
    atol = 1e-5 if metric == ALIGNMENT else 1e-7
    return bool(np.isclose(measured, recorded, rtol=1e-5, atol=atol,
                           equal_nan=True)), atol


def _matching_m1_report(folder: Path, cfg) -> tuple[Path, dict]:
    import lightgcn_clv_m2_training_budget_hm2y as budget

    matches = []
    fields = ("seed", "id_dim", "pref_reg", "batch_size", "lr",
              "negative_count", "epochs", "window_days", "input_days")
    for path in folder.glob("clv_m2_training_budget_hm2y_*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        config = report.get("config", {})
        if any(config.get(field) != getattr(cfg, field) for field in fields):
            continue
        rows = [row for row in report.get("curve", [])
                if row.get("model_id") == budget.M1_MODEL_ID
                and row.get("epoch") == EPOCH]
        if len(rows) == 1:
            matches.append((path, report))
    if len(matches) != 1:
        raise RuntimeError(
            f"H&M M1@300 결과를 하나로 확인할 수 없습니다: {folder}. "
            "재학습하지 않습니다."
        )
    return matches[0]


def _load_compact_checkpoint(folder: Path, pattern: str, *, stage: str,
                             model_id: str, input_hash: str,
                             source_revision: str) -> tuple[dict, dict]:
    import torch
    from clv_run_state import file_sha256

    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"기존 H&M checkpoint를 한 개 찾지 못했습니다: {folder}/{pattern}. "
            "재학습하지 않습니다."
        )
    path = paths[0]
    state = torch.load(path, map_location="cpu", weights_only=False)
    identity = state.get("identity", {})
    expected = {"stage": stage, "model_id": model_id, "seed": SEED,
                "input_hash": input_hash, "source_revision": source_revision}
    if state.get("epoch") != EPOCH or any(identity.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"H&M checkpoint의 epoch/입력/모형/출처가 다릅니다: {path}")
    parameters = state.get("parameter_state")
    if not isinstance(parameters, dict) or not parameters:
        raise RuntimeError(f"H&M 경량 checkpoint 파라미터가 없습니다: {path}")
    return parameters, {"path": str(path), "sha256": file_sha256(path),
                        "epoch": EPOCH, "identity": identity}


def _feature_context(prepared: dict, evaluation_users: np.ndarray) -> dict:
    cached = prepared.get("_hm_candidate_feature_context")
    evaluation_users = np.sort(np.asarray(evaluation_users, np.int64))
    if cached is not None:
        if not np.array_equal(cached["evaluation_users"], evaluation_users):
            raise RuntimeError("H&M 후보특성 cache의 평가고객이 다릅니다")
        return cached

    data = prepared["data"]
    edge_users = np.asarray(data["tr_u"], np.int64)
    edge_items = np.asarray(data["tr_i"], np.int64)
    if len(np.unique(edge_users * data["n_items"] + edge_items)) != len(edge_items):
        raise RuntimeError("H&M 그래프 엣지가 고객-상품 중복을 포함합니다")
    buyers = np.bincount(edge_items, minlength=data["n_items"])

    train = data["train"]
    mask = np.isin(train.u_idx.to_numpy(np.int64), evaluation_users)
    rows = train.loc[mask, ["u_idx", "i_idx", "v"]].copy()
    if rows.empty or rows.u_idx.nunique() != len(evaluation_users):
        raise RuntimeError("H&M 평가고객의 TRAIN 문맥이 누락됐습니다")
    cats = np.asarray(prepared["item_cat"])
    prices = np.asarray(prepared["item_amount_percentile"], float)
    valid_prices = np.asarray(prepared["item_economic_valid"], bool)
    row_items = rows.i_idx.to_numpy(np.int64)
    rows["category"] = cats[row_items]
    rows["spend"] = rows.v.clip(lower=0)
    rows["price_mass"] = rows.spend * np.where(valid_prices[row_items], prices[row_items], 0)
    rows["valid_price_spend"] = rows.spend * valid_prices[row_items]
    context = rows.groupby(["u_idx", "category"], observed=True).agg(
        category_rows=("i_idx", "size"), category_spend=("spend", "sum"))
    totals = rows.groupby("u_idx", observed=True).agg(
        rows=("i_idx", "size"), spend=("spend", "sum"),
        price_mass=("price_mass", "sum"),
        valid_price_spend=("valid_price_spend", "sum"))
    customer_index = context.index.get_level_values(0)
    context["category_row_share"] = (
        context.category_rows / customer_index.map(totals.rows))
    denominator = customer_index.map(totals.spend).to_numpy().clip(min=1e-12)
    context["category_spend_share"] = context.category_spend / denominator
    user_price = totals.price_mass / totals.valid_price_spend.replace(0, np.nan)
    cached = {"evaluation_users": evaluation_users, "buyers": buyers, "cats": cats,
              "prices": prices, "valid_prices": valid_prices,
              "context": context, "user_price": user_price,
              "filtered_train_rows": int(len(rows))}
    prepared["_hm_candidate_feature_context"] = cached
    return cached


def attach_hm_features(candidates: pd.DataFrame, users: pd.DataFrame,
                       prepared: dict) -> pd.DataFrame:
    context = _feature_context(prepared, users.user.unique())
    user_columns = ["seed", "user", "segment", "q_n", "q_v", "clv_valid",
                    "degree", "truth_count"]
    result = candidates.merge(users[user_columns], on=["seed", "user"],
                              how="left", validate="many_to_one")
    if len(result) != len(candidates) or result.segment.isna().any():
        raise RuntimeError("H&M 후보-사용자 매핑 누락")
    item = result.item.to_numpy(np.int64)
    result["category"] = context["cats"][item]
    result["item_buyers"] = context["buyers"][item]
    result["item_amount_percentile"] = np.where(
        context["valid_prices"][item], context["prices"][item], np.nan)
    result["user_amount_position"] = result.user.map(context["user_price"])
    result["price_distance"] = (
        result.item_amount_percentile - result.user_amount_position).abs()
    result = result.merge(
        context["context"][["category_row_share", "category_spend_share"]],
        left_on=["user", "category"], right_index=True, how="left",
        validate="many_to_one")
    columns = ["category_row_share", "category_spend_share"]
    result[columns] = result[columns].fillna(0)
    result["category_seen"] = result.category_row_share.gt(0)
    return result


def _source_files(root: Path, cfg, prepared: dict) -> tuple[dict, pd.DataFrame, list[dict]]:
    import clv_m5_m3_m4_split_nv_hm2y_screen as screen
    from clv_run_state import file_sha256

    m5_folder = root / "results_v3_hm_clv_m5_m3_m4_split_nv_hm2y_s43_v1"
    reports = list((m5_folder / "reports").glob(f"{screen.CODE_VERSION}_*.json"))
    if len(reports) != 1:
        raise RuntimeError(f"H&M M3+M4-B 원본 JSON을 하나로 확인할 수 없습니다: {m5_folder}")
    report_path = reports[0]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("input_hash") != prepared["input_hash"]
            or report.get("split") != SPLIT or report.get("final_test")
            or report.get("holdout") or not np.isclose(report.get("beta"),
                                                       prepared["m5_graph"]["beta"],
                                                       rtol=1e-3, atol=0)):
        raise RuntimeError("H&M M3+M4-B 원본의 입력/분할/그래프가 다릅니다")
    absolute_path = Path(report["result_paths"]["absolute_csv"])
    absolute = pd.read_csv(absolute_path)
    m1_folder = root / "results_v3_hm_clv_m2_training_budget_seed43_v1"
    m1_path, m1_report = _matching_m1_report(m1_folder, cfg)
    sources = [
        {"report": str(report_path), "report_sha256": file_sha256(report_path),
         "absolute": str(absolute_path), "absolute_sha256": file_sha256(absolute_path)},
        {"report": str(m1_path), "report_sha256": file_sha256(m1_path),
         "expected_input_hash": prepared["input_hash"]},
    ]
    return {"m5_folder": m5_folder, "m5_report": report,
            "m1_folder": m1_folder, "m1_report": m1_report}, absolute, sources


def run(root="/content/drive/MyDrive/논문/data") -> dict[str, str]:
    import torch
    import clv_m5_m3_m4_split_nv_hm2y_screen as screen
    import lightgcn_clv_axis_specific_test10 as io
    import lightgcn_clv_hm2y_seed42_common as common
    import lightgcn_clv_m2_training_budget_hm2y as budget
    import lightgcn_clv_m3_centered_value_graph_hm2y as hm2y
    import lightgcn_clv_v3 as v3

    if not torch.cuda.is_available():
        raise RuntimeError("H&M 전체카탈로그 재점수화에는 GPU 런타임이 필요합니다")
    v3.DEVICE = torch.device("cuda")
    root = Path(root)
    out = root / "results_v3_hm_m1_m5_error_diagnostic_v1"
    cfg = screen.configure(out_dir=str(out))
    print("[H&M 진단] 기존 M1/M3+M4-B@300 읽기 · 새 학습 0 · 개발 2020-09-02~08", flush=True)
    prepared = hm2y._prepare(cfg)
    data, cache = prepared["data"], prepared["cache"]
    if (set(data["splits"]) != {"val"} or pd.Timestamp(data["train"].t.max())
            != pd.Timestamp("2020-09-01") or prepared["base_cfg"]["MIN_ITEM_INTER"] != 1
            or prepared["base_cfg"]["EVAL_TEST"] or prepared["base_cfg"]["EVAL_HOLDOUT"]):
        raise RuntimeError("H&M 역사적 개발분할 계약이 다릅니다")
    spec = next(s for s in hm2y.arm_specifications(cfg) if s["arm"] == "value_and_activity")
    graph = hm2y.build_arm_graph(prepared, cfg, spec)
    prepared["m5_graph"] = graph
    sources_info, source_curve, sources = _source_files(root, cfg, prepared)

    degree = np.asarray(prepared["binary_user_degree"], np.int64)
    population = pd.DataFrame({
        "user": np.asarray(cache.users, np.int64), "segment": cache.seg,
        "truth_count": [len(cache.gt[u]) for u in cache.users],
        "q_n": np.asarray(prepared["q_n"])[cache.users],
        "q_v": np.asarray(prepared["q_v"])[cache.users],
        "clv_valid": np.asarray(prepared["clv_valid"], bool)[cache.users],
        "degree": degree[cache.users],
    })
    edges = {}
    for axis, column in (("q_n", "n_bin"), ("q_v", "v_bin")):
        population[column], edges[column] = movement.train_bins(
            prepared[axis], prepared["clv_valid"], cache.users)
    population["degree_bin"], edges["degree_bin"] = movement.train_bins(
        degree, degree > 0, cache.users)
    population["seed"] = SEED

    m1_id, m5_id = budget.M1_MODEL_ID, screen.ARMS["split_nv"]
    checkpoints = [
        ("m1", m1_id, sources_info["m1_folder"],
         f"progress/*/resume/hm2y_m2_training_budget_dev_{m1_id}_s{SEED}_latest.pt",
         "hm2y_m2_training_budget_dev", sources_info["m1_report"]["source_revision"]),
        ("m5", m5_id, sources_info["m5_folder"],
         f"progress/*/resume/hm2y_validation_fixed_epoch_train_{m5_id}_s{SEED}_latest.pt",
         "hm2y_validation_fixed_epoch_train", sources_info["m5_report"]["source_revision"]),
    ]
    scored, recommendations, audits, absolute = {}, [], [], []
    for label, model_id, folder, pattern, stage, revision in checkpoints:
        expected = source_curve[
            source_curve.model_id.eq(model_id) & source_curve.epoch.eq(EPOCH)]
        if len(expected) != 1:
            raise RuntimeError(f"H&M 원본 300epoch 성과가 누락/중복됐습니다: {model_id}")
        parameter_state, source = _load_compact_checkpoint(
            folder, pattern, stage=stage, model_id=model_id,
            input_hash=prepared["input_hash"], source_revision=revision)
        sources.append(source)
        model = (budget._build_model(prepared, cfg, {"kind": "m1"}) if label == "m1"
                 else hm2y._build_model(prepared, cfg, graph))
        common.load_parameter_state(model, parameter_state)
        measured = common.evaluate(model, prepared)
        for metric, value in measured.items():
            reference = float(expected.iloc[0][metric])
            passed, absolute_tolerance = _readback_match(metric, value, reference)
            audits.append({"seed": SEED, "model_id": model_id, "metric": metric,
                           "recorded": reference, "readback": float(value),
                           "absolute_difference": abs(float(value) - reference),
                           "absolute_tolerance": absolute_tolerance,
                           "passed": passed})
        io._atomic_csv(out / "readback.csv", pd.DataFrame(audits))
        if not all(row["passed"] for row in audits):
            raise RuntimeError("H&M checkpoint 재현 차이: readback.csv 확인. 새 학습 없음")
        absolute.append({"seed": SEED, "model_id": model_id, "epoch": EPOCH, **measured})
        recs = []
        scored[label] = movement._score_truth(model, prepared, SEED, recs)
        recommendations.append(pd.DataFrame(recs).assign(model_id=model_id, model=label))
        del model, parameter_state
        torch.cuda.empty_cache()
        print(f"[H&M 진단 완료] {label}: 전체지표 재현·정답순위 기록", flush=True)

    truth, users = movement.movement_tables(
        scored["m1"], scored["m5"], population)
    for k in movement.KS:
        for column, metric in ((f"net_recall@{k}", f"recall@{k}"),
                               (f"net_weight@{k}",
                                f"price_purchase_amount_weighted_hit@{k}")):
            expected_delta = absolute[1][metric] - absolute[0][metric]
            if not np.isclose(users[column].mean(), expected_delta,
                              atol=1e-7, rtol=1e-5):
                raise RuntimeError(f"H&M 정답 이동량과 전체성과 차이가 다릅니다: {column}")
    summary, correlations, transitions = movement.summaries(truth, users)
    frames = {
        "truth_movements": truth, "user_movements": users, "summary": summary,
        "axis_correlations": correlations, "rank_transitions": transitions,
        "absolute": pd.DataFrame(absolute), "readback": pd.DataFrame(audits),
    }
    frames.update(features.enrich_tables(
        truth, users, pd.concat(recommendations, ignore_index=True), prepared,
        feature_attacher=attach_hm_features))
    for key, frame in frames.items():
        frames[key] = frame.rename(
            columns=lambda c: c.replace("_m3", "_m1").replace("m3_", "m1_"))
    paths = {key: str(out / f"{key}.csv") for key in frames}
    for key, frame in frames.items():
        io._atomic_csv(Path(paths[key]), frame)
    paths["json"] = str(out / "result.json")
    context = prepared["_hm_candidate_feature_context"]
    io._atomic_json(Path(paths["json"]), {
        "code_version": VERSION, "seeds": [SEED], "reference_model": "M1",
        "candidate_model": "M3+M4-B", "epoch": EPOCH, "training": False,
        "final_test": False, "holdout": False, "split": SPLIT,
        "input_hash": prepared["input_hash"], "sources": sources,
        "train_bin_edges": edges, "graph_audit": graph["audit"],
        "rank_limit": movement.TOP,
        "rank_nan_means": "outside top100, not missing truth",
        "feature_definitions": {
            "item_buyers": "distinct TRAIN buyers from binary graph edges",
            "item_amount_percentile": "existing TRAIN economic-input percentile; invalid -> NaN",
            "category_row_share": "candidate-category TRAIN rows / all TRAIN rows for that evaluation customer",
            "category_spend_share": "candidate-category positive TRAIN spend / all positive TRAIN spend for that evaluation customer",
            "price_distance": "absolute item percentile minus evaluation customer's positive-spend-weighted TRAIN item percentile",
            "feature_population": "TRAIN context restricted to evaluation customers; item buyers use full TRAIN graph",
        },
        "filtered_train_rows_for_features": context["filtered_train_rows"],
        "significance_claim": False, "nv_causal_attribution": False,
        "reading": "Single historical H&M development seed; descriptive cross-dataset pattern check only. No training, model approval, CLV attribution, significance or final-test claim.",
        "paths": paths,
    })
    print(summary[summary.grouping.eq("all")].to_string(index=False), flush=True)
    print(correlations.to_string(index=False), flush=True)
    return paths


if __name__ == "__main__":
    run()

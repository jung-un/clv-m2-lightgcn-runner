"""TRAIN-only candidate attributes for existing M1/M3+M4 prediction diagnostics.

All feature comparisons are descriptive, not new model inputs or causal claims.
No optimizer, training, checkpoint selection, or last-week test is invoked.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("item_buyers", "item_amount_percentile", "category_row_share",
            "category_spend_share", "price_distance")


def baseline_source(root, seed, cfg, prepared):
    from clv_run_state import file_sha256
    folder = Path(root) / "results_v3_dunnhumby_clv_m2_capacity_search_v1"
    matches = []
    for path in folder.glob(f"arms/*/baseline_m1_bpr_k1_s{seed}.json"):
        report = json.loads(path.read_text())
        if (report.get("id_dim") == cfg.id_dim and report.get("pref_reg") == cfg.pref_reg
                and report.get("seed") == seed and report.get("condition") == "baseline"
                and any(r.get("epoch") == 300 and "metrics" in r for r in report.get("curve", []))):
            checkpoint = folder / "progress" / path.parent.name / "resume" / (
                f"capacity_search_dev_baseline_m1_bpr_k1_s{seed}_latest.pt")
            if checkpoint.is_file():
                matches.append((path, report))
    if len(matches) != 1:
        raise RuntimeError(f"seed{seed} 기존 M1@300 결과/checkpoint를 하나로 확인할 수 없습니다: {folder}. 재학습하지 않습니다.")
    path, report = matches[0]
    # The shared loader checks input hash, source revision, epoch and identity;
    # all recorded metrics are then reproduced against the M5 reference CSV.
    return (("m3", "m1_bpr_k1", folder, "capacity_search_dev", path.parent.name,
             report["source_revision"]),
            dict(report=str(path), report_sha256=file_sha256(path),
                 expected_input_hash=prepared["input_hash"]))


def attach_features(candidates, users, prepared):
    data = prepared["data"]
    train = data["train"]
    cats = np.asarray(data["item_cat"])
    prices = np.asarray(prepared["item_amount_percentile"], float)
    valid_prices = np.asarray(prepared["item_economic_valid"], bool)
    rows = train[["u_idx", "i_idx", "v"]].copy()
    rows["category"] = cats[rows.i_idx.to_numpy()]
    rows["spend"] = rows.v.clip(lower=0)
    rows["price_mass"] = rows.spend * np.where(valid_prices[rows.i_idx], prices[rows.i_idx], 0)
    rows["valid_price_spend"] = rows.spend * valid_prices[rows.i_idx]
    context = rows.groupby(["u_idx", "category"]).agg(
        category_rows=("i_idx", "size"), category_spend=("spend", "sum"))
    totals = rows.groupby("u_idx").agg(rows=("i_idx", "size"), spend=("spend", "sum"),
        price_mass=("price_mass", "sum"), valid_price_spend=("valid_price_spend", "sum"))
    context["category_row_share"] = context.category_rows / context.index.get_level_values(0).map(totals.rows)
    context["category_spend_share"] = context.category_spend / context.index.get_level_values(0).map(totals.spend).to_numpy().clip(min=1e-12)
    buyers = rows.groupby("i_idx").u_idx.nunique().reindex(range(data["n_items"]), fill_value=0)
    user_price = totals.price_mass / totals.valid_price_spend.replace(0, np.nan)
    user_columns = ["seed", "user", "segment", "q_n", "q_v", "clv_valid", "degree", "truth_count"]
    result = candidates.merge(users[user_columns], on=["seed", "user"],
                              how="left", validate="many_to_one")
    if len(result) != len(candidates) or result.segment.isna().any():
        raise RuntimeError("후보-사용자 매핑 누락")
    result["category"] = cats[result.item.to_numpy()]
    result["item_buyers"] = result.item.map(buyers)
    result["item_amount_percentile"] = np.where(valid_prices[result.item], prices[result.item], np.nan)
    result["user_amount_position"] = result.user.map(user_price)
    result["price_distance"] = (result.item_amount_percentile - result.user_amount_position).abs()
    result = result.merge(context[["category_row_share", "category_spend_share"]],
        left_on=["user", "category"], right_index=True, how="left", validate="many_to_one")
    result[["category_row_share", "category_spend_share"]] = result[
        ["category_row_share", "category_spend_share"]].fillna(0)
    result["category_seen"] = result.category_row_share.gt(0)
    return result


def enrich_tables(truth, users, recommendations, prepared, feature_attacher=attach_features):
    truth = feature_attacher(truth, users, prepared)
    recs = feature_attacher(recommendations, users, prepared)
    for k in (10, 20, 50):
        a, b = truth.rank_m3.le(k), truth.rank_m5.le(k)
        truth[f"status@{k}"] = np.select([a & b, ~a & b, a & ~b],
            ["both_hit", "m5_gain", "m5_loss"], default="both_miss")
        # Recover actual recommendation hits, independently of movement flags.
        for label, ranks in (("m1", "rank_m3"), ("m5", "rank_m5")):
            selected = recs[recs.model.eq(label) & recs["rank"].le(k)]
            hits = selected[selected.is_truth]
            expected = truth[truth[ranks].le(k)]
            keys = ["seed", "user", "item"]
            if set(map(tuple, hits[keys].to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
                raise RuntimeError("추천목록 적중과 정답순위 불일치")
            if selected.groupby(["seed", "user"]).size().ne(k).any():
                raise RuntimeError("추천목록 길이 불일치")
    # Both pair-weighted and user-macro summaries prevent prolific buyers
    # silently dominating the apparent candidate-feature differences.
    parts = []
    for k in (10, 20, 50):
        part = truth.rename(columns={f"status@{k}": "group"})
        parts.append(part.assign(k=k)[["seed", "user", "segment", "group", "k", *FEATURES]])
        wrong = recs[recs["rank"].le(k) & ~recs.is_truth].copy()
        wrong["group"] = wrong.model + "_wrong"
        parts.append(wrong.assign(k=k)[["seed", "user", "segment", "group", "k", *FEATURES]])
    long = pd.concat(parts, ignore_index=True)
    long = pd.concat([long, long.assign(segment="all")], ignore_index=True)
    keys = ["seed", "segment", "group", "k"]
    rows = []
    for group, subset in long.groupby(keys):
        macro = subset.groupby("user")[list(FEATURES)].mean()
        for feature in FEATURES:
            rows.append(dict(zip(keys, group)) | dict(feature=feature,
                candidate_pairs=len(subset), users=subset.user.nunique(),
                valid_pairs=int(subset[feature].notna().sum()),
                valid_users=int(macro[feature].notna().sum()),
                pair_mean=subset[feature].mean(), pair_median=subset[feature].median(),
                user_macro_mean=macro[feature].mean()))
    # Within the same user: common misses minus that model's wrong Top-K items.
    paired = []
    for k in (10, 20, 50):
        missing = truth[truth[f"status@{k}"].eq("both_miss")].groupby(
            ["seed", "user", "segment"])[list(FEATURES)].mean()
        for label in ("m1", "m5"):
            wrong = recs[recs.model.eq(label) & recs["rank"].le(k) & ~recs.is_truth].groupby(
                ["seed", "user", "segment"])[list(FEATURES)].mean()
            common = missing.index.intersection(wrong.index)
            delta = (missing.loc[common] - wrong.loc[common]).reset_index()
            paired.append(delta.assign(k=k, reference=label,
                                      definition="common_miss_mean_minus_wrong_mean"))
    return dict(truth_features=truth, recommendations=recs,
                feature_summary=pd.DataFrame(rows),
                paired_user_feature_differences=pd.concat(paired, ignore_index=True))


def run(root="/content/drive/MyDrive/논문/data"):
    from clv_m5_nv_rank_diagnostic import run as diagnostic
    # Fail before expensive data preparation if the existing M1 artifacts are absent.
    for seed in (43, 44):
        folder = Path(root) / "results_v3_dunnhumby_clv_m2_capacity_search_v1"
        if not list(folder.glob(f"progress/*/resume/capacity_search_dev_baseline_m1_bpr_k1_s{seed}_latest.pt")):
            raise FileNotFoundError(f"seed{seed} M1@300 checkpoint가 없습니다: {folder}. 새 학습 없음")
    return diagnostic(root=root, seeds=(43, 44), compare_m1=True)

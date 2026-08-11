"""v3 설계 불변식 테스트. 지키는 것은 '무엇을 넣었나'가 아니라
'리뷰에서 지적된 문제가 되살아나지 않았나'다."""
import io, tokenize
from pathlib import Path

import json

import numpy as np
import pytest
import pandas as pd
import torch

import lightgcn_clv_v3 as V3

_RAW = (Path(__file__).parent / "lightgcn_clv_v3.py").read_text(encoding="utf-8")


def _code_only(text):
    """주석·docstring을 걷어낸 실행 코드만 남긴다. 안 그러면 '제거했다'는 설명 문구가
    '아직 남아있다'로 오판된다.

    docstring 판정은 prev가 INDENT/NEWLINE(논리적 줄바꿈)일 때만 한다. 괄호 안
    줄바꿈은 NL 토큰이라, NL까지 포함하면 dict 리터럴의 문자열 키가 docstring으로
    오인돼 지워진다."""
    out, prev, first = [], tokenize.INDENT, True
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in (tokenize.COMMENT, tokenize.ENCODING):
            continue
        is_doc = tok.type == tokenize.STRING and (first or prev in (tokenize.INDENT,
                                                                    tokenize.NEWLINE))
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            first = False
        if is_doc:
            prev = tok.type
            continue
        out.append(tok.string); prev = tok.type
    return " ".join(out)


SRC = _code_only(_RAW)
NS = SRC.replace(" ", "")            # 토큰 사이 공백 제거본 — 부분문자열 검사용


def _fn(name):
    return SRC.split(f"def {name}")[1].split(" def ")[0]


def _fn_ns(name):
    return _fn(name).replace(" ", "")


# ── 리뷰 지적 1: joint의 λ=0은 baseline이 아니다 ───────────────────────
def test_pref_only_is_the_shared_baseline():
    assert "pref_only" in V3.ARCH_LABEL and "baseline" in V3.ARCH_LABEL["pref_only"]
    assert "baseline 아님" in V3.ARCH_LABEL["joint"]      # joint λ=0은 ablation 표기
    body = _fn("get_or_train").replace(" ", "")
    assert 'get_or_train("pref_only"' in body              # two_stage가 baseline을 재사용
    assert body.count("build_model") >= 3


# ── 리뷰 지적 2: 학습식 == 평가식 ─────────────────────────────────────
def test_train_and_eval_use_the_same_score():
    """λ_eval=1일 때 학습 점수식과 평가 점수식이 정확히 같아야 한다."""
    torch.manual_seed(0)
    n_u, n_i, d, dv = 6, 9, 4, 3
    Up = torch.randn(n_u, d); Ip = torch.randn(n_i, d)
    Uv = torch.randn(n_u, dv); Iv = torch.randn(n_i, dv)
    gate = torch.rand(n_u); lam = 1.0
    u = torch.tensor([0, 2, 5]); i = torch.tensor([1, 4, 8])
    pair = V3.combined_score_pairs(Up, Ip, Uv, Iv, gate, lam, u, i)
    allsc = V3.combined_score_all(Up, Ip, Uv, Iv, gate, lam, u)
    torch.testing.assert_close(pair, allsc[torch.arange(3), i])
    # 평가 경로에 유저별 z-score 정규화가 남아있으면 안 된다
    ev = _fn("evaluate").replace(" ", "")
    assert "keepdim" not in ev and ".std(" not in ev
    assert "combined_score_all" in ev


def test_bpr_loss_uses_combined_score_and_logsigmoid():
    body = _fn("bpr_loss").replace(" ", "")
    assert "combined_score_pairs" in body
    assert "F.logsigmoid" in body and "torch.log(torch.sigmoid" not in body


# ── 리뷰 지적 3: λ_train / λ_eval 분리 ────────────────────────────────
def test_lambda_names_are_separated():
    assert "LAMBDA_TRAIN" in V3.CFG and "LAMBDA_EVAL_SWEEP" in V3.CFG
    assert "LAMBDA_SWEEP" not in V3.CFG        # 옛 이름은 없어야 함
    assert 0.0 in V3.CFG["LAMBDA_EVAL_SWEEP"]


# ── 리뷰 지적 4: λ 선택은 validation에서, 규칙은 코드에 고정 ──────────
def test_lambda_selection_is_validation_only_and_rule_fixed():
    body = _fn("select_lambda")
    assert "val_pwgain10" in body
    # [2026-08-07] 비열등성 가드레일 폐기 — ΔRecall@10 CI는 참고용으로만 계산해
    # NONINFERIORITY_DELTA가 여전히 등장하지만, 필터링(ok=df[...])에는 더 이상 안 쓴다.
    assert "NONINFERIORITY_DELTA" in body
    assert "ok=df[df.passes_noninferiority]" not in body.replace(" ", "")
    assert "ok.empty" not in body
    # 동률이면 더 작은 λ — ascending=[False, True]
    assert "ascending=[False,True]" in body.replace(" ", "")


def test_select_lambda_picks_smallest_on_tie():
    cfg = dict(V3.CFG); cfg["LAMBDA_EVAL_SWEEP"] = [0.0, 0.5, 1.0]; cfg["N_BOOT"] = 50
    rng = np.random.default_rng(0)
    base = rng.random(200) * 0.02
    def mk(pu, pw):
        return {"pu": {"recall": pu}, "agg": {"overall": {10: {"recall": pu.mean(), "revenue": pw}}}}
    # λ=0.5와 1.0이 PWGain 동률 → 더 작은 0.5가 뽑혀야 한다
    per_seed = {s: {0.0: mk(base, 1.0), 0.5: mk(base, 2.0), 1.0: mk(base, 2.0)} for s in (42, 43)}
    base_per_seed = {s: mk(base, 1.0) for s in (42, 43)}      # 외부 pref_only 기준
    lam, tbl = V3.select_lambda(per_seed, base_per_seed, cfg)
    assert lam == 0.5


# ── 리뷰 지적 5: 세그먼트 임계값 고정 ─────────────────────────────────
def test_segment_thresholds_computed_once_from_train():
    clv = np.concatenate([np.linspace(0, 1, 100), [np.nan] * 5])
    lo, hi = V3.segment_thresholds(clv, (0.2, 0.8))
    assert 0.15 < lo < 0.25 and 0.75 < hi < 0.85
    ev = _fn("evaluate")
    assert "quantile" not in ev                 # 평가 안에서 다시 계산하지 않음
    assert "seg_th" in _fn("__init__") or "seg_th" in SRC


# ── 리뷰 지적: compute budget 기록 ────────────────────────────────────
def test_train_phase_records_compute():
    body = _fn("train_phase")
    for key in ["best_epoch", "epochs_run", "updates", "samples", "wall_clock_sec"]:
        assert key in body, f"학습량 기록에 {key} 없음"


# ── 리뷰 지적: 진단값 저장 ────────────────────────────────────────────
def test_score_diagnostics_fields():
    body = _fn("score_diagnostics")
    for key in ["std_s_pref", "std_s_value", "ratio_value_over_pref",
                "mean_norm_U_pref", "mean_norm_I_pref", "mean_norm_U_value", "mean_norm_I_value"]:
        assert key in body


# ── 리뷰 지적: holdout은 켜기 전엔 계산조차 안 함 ─────────────────────
def test_holdout_is_gated():
    assert V3.CFG["EVAL_HOLDOUT"] is False
    # _fn은 중첩 def(build_eval)에서 잘리므로 전체 소스에서 게이트 라인을 직접 확인한다
    assert 'ifcfg["EVAL_HOLDOUT"]:' in NS
    assert 'splits["holdout"]=build_eval' in NS


# ── 리뷰 지적: SELECT_METRIC 실제 사용 ────────────────────────────────
def test_select_metric_is_actually_used():
    body = _fn_ns("train_phase")
    assert 'cfg["SELECT_METRIC"]' in body and 'cfg["SELECT_K"]' in body


# ── 기존 불변식 ───────────────────────────────────────────────────────
def test_no_v2_machinery():
    for dead in ["ACCURACY_EPSILON", "RECALL50_EPSILON", "HR_EPSILON", "DIVERSITY_EPSILON",
                 "VT_TOPK_CKPTS", "EPOCH_SCREEN_LAMBDA", "SHRINKAGE_K", "CLV_GATE_POWER",
                 "HARD_NEG_RATIO"]:
        assert dead not in V3.CFG, f"CFG에 {dead}가 남아있음"
    for dead in ["_passes", "vt_topk", "grid_results"]:
        assert dead not in SRC


def test_gate_is_linear_percentile():
    """percentile_rank 선형(제곱 아님) + 평균 1 정규화."""
    clv = np.array([10.0, 20.0, 30.0, np.nan, 40.0])
    g = V3.build_gate(clv, clv, "clv")
    assert g[3] == 0.0
    raw = np.array([0.25, 0.5, 0.75, 1.0])
    # 정규화 기준은 **유효 유저 4명**의 평균(0.625)이지 전체 5명이 아니다
    np.testing.assert_allclose(np.sort(g[[0, 1, 2, 4]]), raw / raw.mean(), rtol=1e-6)
    assert "**" not in _fn("build_gate")


# ── 2026-08-07: 게이트 모드 + 평균 1 정규화 ──────────────────────────────
def test_gate_modes_all_normalized_to_mean_one():
    """모드마다 평균이 다르면(none=1.0, percentile≈0.5) 같은 λ가 서로 다른 실효
    개입 강도를 뜻해 모드 비교가 스케일 차이를 재게 된다."""
    rng = np.random.default_rng(0)
    clv, vhat = rng.random(500), rng.random(500)
    for mode in ("none", "clv", "vhat"):
        g = V3.build_gate(clv, vhat, mode)
        assert abs(g.mean() - 1.0) < 1e-5, f"{mode} 평균이 1이 아님: {g.mean()}"
    assert V3.CFG["GATE_MODE"] in ("none", "clv", "vhat")


def test_gate_modes_differ_and_use_right_source():
    rng = np.random.default_rng(1)
    clv, vhat = rng.random(300), rng.random(300)
    gn, gc, gv = (V3.build_gate(clv, vhat, m) for m in ("none", "clv", "vhat"))
    assert np.allclose(gn, 1.0)                      # none은 전원 동일
    assert not np.allclose(gc, gv)                   # clv와 vhat은 달라야 함
    # vhat 모드는 vhat 순서를, clv 모드는 clv 순서를 따라야 한다
    assert np.array_equal(np.argsort(gv), np.argsort(vhat))
    assert np.array_equal(np.argsort(gc), np.argsort(clv))
    try:
        V3.build_gate(clv, vhat, "bogus")
        raise AssertionError("알 수 없는 GATE_MODE는 거부해야 한다")
    except ValueError:
        pass


def test_clv_features_returns_vhat():
    """vhat은 clv와 같은 곳에서 계산돼야 정의가 어긋나지 않는다."""
    train = pd.DataFrame({
        "u_idx": [0, 0, 0, 1], "i_idx": [0, 1, 2, 0], "b_raw": [7, 7, 7, 1],
        "t": [0, 0, 0, 5], "v": [20.0, 20.0, 20.0, 90.0], "up": [20.0, 20.0, 20.0, 90.0],
        "cat_idx": [0, 0, 1, 1]})
    x, clv, vhat = V3.clv_features(train, 2, dict(V3.CFG), is_date=False)
    assert vhat.shape == clv.shape == (2,)
    assert np.all(np.isfinite(vhat))                 # 두 유저 모두 train 이력 있음
    assert np.all((vhat >= 0) & (vhat <= 1))         # 백분위 평균이라 0~1


def test_gate_mode_in_ckpt_hash_except_pref_only():
    """GATE_MODE는 가중치를 바꾸므로 해시에 있어야 하지만, pref_only는 λ_train=0이라
    게이트가 학습에 영향을 못 준다 — 넣으면 v3.4식 낭비 재학습이 재현된다."""
    import copy
    c1 = copy.deepcopy(V3.CFG); c1["GATE_MODE"] = "clv"
    c2 = copy.deepcopy(V3.CFG); c2["GATE_MODE"] = "vhat"
    for arch in ("joint_warm", "two_stage", "joint"):
        assert V3.cfg_hash(c1, V3.DCFG, arch, 42) != V3.cfg_hash(c2, V3.DCFG, arch, 42), arch
    assert V3.cfg_hash(c1, V3.DCFG, "pref_only", 42) == V3.cfg_hash(c2, V3.DCFG, "pref_only", 42)


def test_kcore_counts_unique_edges_not_rows():
    """LightGCN 인접행렬은 이진이므로 필터도 고유 엣지를 세야 한다. 반복구매가 많은
    데이터에서 거래행을 세면 아이템 연결도를 과대평가한다."""
    # item A: 한 유저가 10번 반복구매 (행 10, 고유엣지 1)
    # item B: 서로 다른 유저 3명이 1번씩 (행 3, 고유엣지 3)
    tp = pd.DataFrame({
        "u_raw": [1] * 10 + [1, 2, 3],
        "i_raw": ["A"] * 10 + ["B", "B", "B"]})
    keep_u, keep_i, n_edge, _ = V3.kcore_filter(tp, min_u=1, min_i=3)
    assert "A" not in keep_i, "반복구매 10회짜리 단일 유저 아이템이 통과하면 안 됨"
    assert "B" in keep_i
    body = _fn_ns("kcore_filter")
    assert 'drop_duplicates()' in body      # 고유 엣지로 환원한 뒤 세는지
    assert 'tp["i_raw"].value_counts()' not in body   # 거래행을 직접 세면 안 됨


def test_kcore_iterates_until_no_orphan_users():
    """아이템을 지우면 train 이력이 0이 되는 유저가 생긴다 — 한 번만 거르면 남는다."""
    # u=9는 희귀 아이템 Z만 삼 → Z 제거 시 고아가 됨
    tp = pd.DataFrame({
        "u_raw": [1, 2, 3, 1, 2, 3, 9],
        "i_raw": ["A", "A", "A", "B", "B", "B", "Z"]})
    keep_u, keep_i, _, n_iter = V3.kcore_filter(tp, min_u=1, min_i=3)
    assert "Z" not in keep_i
    assert 9 not in keep_u, "아이템 제거로 이력이 0이 된 유저가 남아있음"
    assert n_iter >= 1


def test_kcore_raises_when_result_empty():
    """임계값이 과하면 0-user/0-item으로 진행하다 한참 뒤 엉뚱하게 터진다 — 즉시 중단."""
    tp = pd.DataFrame({"u_raw": [1, 2], "i_raw": ["A", "B"]})
    try:
        V3.kcore_filter(tp, min_u=1, min_i=99)
        raise AssertionError("빈 k-core 결과를 그대로 반환하면 안 된다")
    except ValueError as e:
        assert "비었" in str(e)


def test_kcore_raises_on_nonconvergence():
    """미수렴 부분 결과는 k-core가 아니다 — 경고 후 진행하면 전제가 거짓인 채 실험이 돈다."""
    rng = np.random.default_rng(0)
    n = 300
    tp = pd.DataFrame({"u_raw": rng.integers(0, 40, n), "i_raw": rng.integers(0, 40, n)})
    # 실패 '종류'까지 구분한다 — ValueError(빈 결과)로 통과해버리면 미수렴 검출을 못 본다
    with pytest.raises(RuntimeError, match="수렴하지 않았다"):
        V3.kcore_filter(tp, min_u=3, min_i=3, max_iter=1)   # 일부러 1회로 제한


def test_kcore_result_satisfies_degree_invariant():
    """수렴했다고 끝이 아니라, 반환값이 실제로 조건을 만족하는지 직접 확인한다."""
    rng = np.random.default_rng(1)
    n = 4000
    tp = pd.DataFrame({"u_raw": rng.integers(0, 120, n), "i_raw": rng.integers(0, 200, n)})
    keep_u, keep_i, n_edge, _ = V3.kcore_filter(tp, min_u=3, min_i=5)
    pairs = tp[["u_raw", "i_raw"]].drop_duplicates()
    pairs = pairs[pairs.u_raw.isin(keep_u) & pairs.i_raw.isin(keep_i)]
    assert pairs["i_raw"].value_counts().min() >= 5
    assert pairs["u_raw"].value_counts().min() >= 3
    assert len(pairs) == n_edge


def _tiny_dataset(tmp_path, n_users=40, n_items=25, n_rows=1200, seed=0):
    """prepare_data()를 실제로 돌릴 수 있는 최소 데이터셋(csv 2개 + DCFG)을 만든다."""
    rng = np.random.default_rng(seed)
    tx = pd.DataFrame({
        "uid": rng.integers(0, n_users, n_rows),
        "iid": rng.integers(0, n_items, n_rows),
        "day": rng.integers(0, 60, n_rows),
        "amt": rng.random(n_rows) * 10 + 1,
        "bask": rng.integers(0, 400, n_rows),
        "qty": 1,
    })
    tx_p, meta_p = tmp_path / "tx.csv", tmp_path / "meta.csv"
    tx.to_csv(tx_p, index=False)
    pd.DataFrame({"iid": np.arange(n_items),
                  "cat": [f"C{i % 3}" for i in range(n_items)]}).to_csv(meta_p, index=False)
    dcfg = {"tx_path": str(tx_p), "item_meta_path": str(meta_p),
            "user_col": "uid", "item_col": "iid", "time_col": "day", "value_col": "amt",
            "item_key_col": "iid", "category_col": "cat",
            "basket_col": "bask", "qty_col": "qty", "is_date": False}
    cfg = dict(V3.CFG)
    cfg.update(WINDOW_DAYS=None, VAL_DAYS=7, TEST_DAYS=7, HOLDOUT_DAYS=7,
               MIN_USER_INTER=1, MIN_ITEM_INTER=1, EVAL_HOLDOUT=False)
    return cfg, dcfg


def test_prepare_data_actually_runs_and_split_keys_are_strings(tmp_path):
    """⚠ 이 테스트가 없어서 v3.9의 P0 회귀를 놓쳤다.
    build_eval의 split_key 파라미터가 함수 안에서 pair_key 배열로 덮어써져
    split_stats[ndarray] → TypeError로 데이터 준비 단계에서 항상 죽었는데,
    소스 문자열만 검사하는 테스트는 이를 잡지 못했다. 실제로 호출해야 한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    d = V3.prepare_data(cfg, dcfg)
    ds = d["data_stats"]
    assert set(ds["splits"].keys()) == {"val"}, ds["splits"].keys()   # EVAL_TEST=False
    for k, v in ds["splits"].items():
        assert isinstance(k, str)
        assert {"eval_users", "gt_pairs", "gt_per_user"} <= set(v)
    # data_stats 자체가 JSON 직렬화 가능해야 결과 파일에 들어간다
    json.dumps(ds, default=float)


def test_prepare_data_persists_reserved_boundaries_without_protected_truth(tmp_path):
    cfg, dcfg = _tiny_dataset(tmp_path)
    d = V3.prepare_data(dict(cfg, EVAL_TEST=False, EVAL_HOLDOUT=False), dcfg)
    stats = d["data_stats"]

    assert stats["source"] == {"rows": stats["source"]["rows"], "time_min": 0, "time_max": 59}
    assert stats["analysis_window"]["time_min"] == 0
    assert stats["analysis_window"]["time_max"] == 59
    assert stats["split_boundaries"] == {
        "train": {"start_inclusive": 0, "end_inclusive": 38},
        "val": {"start_exclusive": 38, "end_inclusive": 45},
        "test": {"start_exclusive": 45, "end_inclusive": 52},
        "holdout": {"start_exclusive": 52, "end_inclusive": 59},
    }
    assert stats["split_rows"]["train"] == len(d["train"])
    assert sum(stats["split_rows"].values()) == stats["analysis_window"][
        "rows_after_kcore"
    ]
    assert stats["split_evaluation_status"] == {
        "val": "constructed",
        "test": "not_constructed",
        "holdout": "not_constructed",
    }
    assert set(d["splits"]) == {"val"}
    assert set(stats["splits"]) == {"val"}
    json.dumps(stats)


def test_prepare_data_date_boundaries_are_json_safe(tmp_path):
    cfg, dcfg = _tiny_dataset(tmp_path)
    tx_path = Path(dcfg["tx_path"])
    tx = pd.read_csv(tx_path)
    tx["day"] = (
        pd.Timestamp("2020-01-01") + pd.to_timedelta(tx["day"], unit="D")
    ).dt.strftime("%Y-%m-%d")
    tx.to_csv(tx_path, index=False)
    dcfg["is_date"] = True

    stats = V3.prepare_data(
        dict(cfg, EVAL_TEST=False, EVAL_HOLDOUT=False), dcfg
    )["data_stats"]
    assert stats["source"]["time_min"] == "2020-01-01T00:00:00"
    assert stats["source"]["time_max"] == "2020-02-29T00:00:00"
    assert stats["split_boundaries"]["test"] == {
        "start_exclusive": "2020-02-15T00:00:00",
        "end_inclusive": "2020-02-22T00:00:00",
    }
    assert stats["split_evaluation_status"]["test"] == "not_constructed"
    assert stats["split_evaluation_status"]["holdout"] == "not_constructed"
    json.dumps(stats)


def test_prepare_data_holdout_key_when_enabled(tmp_path):
    cfg, dcfg = _tiny_dataset(tmp_path)
    cfg["EVAL_HOLDOUT"] = True; cfg["EVAL_TEST"] = True
    d = V3.prepare_data(cfg, dcfg)
    assert set(d["data_stats"]["splits"].keys()) == {"val", "test", "holdout"}
    assert d["data_stats"]["split_evaluation_status"] == {
        "val": "constructed",
        "test": "constructed",
        "holdout": "constructed",
    }


def test_prepare_data_respects_min_item_inter(tmp_path):
    """k-core가 실제 파이프라인에서 아이템을 줄이고 통계에 반영되는지."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    lo = V3.prepare_data(dict(cfg, MIN_ITEM_INTER=1), dcfg)["data_stats"]
    hi = V3.prepare_data(dict(cfg, MIN_ITEM_INTER=15), dcfg)["data_stats"]
    assert hi["train_items_after"] <= lo["train_items_after"]
    assert hi["item_drop_rate"] >= lo["item_drop_rate"]
    for key in ("train_rows_after", "row_drop_rate", "train_edges_after"):
        assert key in hi


def test_split_stats_use_internal_lowercase_keys():
    """출력용 라벨('Val    ')을 JSON 키로 쓰면 공백까지 키에 섞인다."""
    assert 'split_stats[split_key]=' in NS
    assert 'split_stats[name.strip()]' not in NS
    assert 'build_eval(df,split_key,label)' in NS
    # split_key와 pair_key는 서로 다른 이름이어야 한다(v3.9 P0 회귀의 원인).
    # 실제 방어는 test_prepare_data_actually_runs_...(런타임 호출)가 하고,
    # 여기서는 이름이 분리됐다는 사실만 확인한다.
    assert 'pair_key=d.u_idx.values' in NS


def test_data_stats_has_row_counts_both_sides():
    """고유엣지 보존율만으로는 반복구매 많은 데이터의 필터 영향을 못 읽는다."""
    for key in ("train_rows_before", "train_rows_after", "row_drop_rate"):
        assert key in NS, f"{key} 없음"


def test_gate_normalized_over_valid_users_only():
    """NaN 유저를 0으로 섞은 채 전체 평균을 1로 맞추면 유효 유저의 실효 강도가
    1/(1-NaN비율)만큼 커진다. mode=none은 0이 없으므로 모드 간 강도가 어긋난다."""
    clv = np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan])
    g = V3.build_gate(clv, clv, "clv")
    valid = ~np.isnan(clv)
    assert abs(g[valid].mean() - 1.0) < 1e-5, f"유효유저 평균이 1이 아님: {g[valid].mean()}"
    assert g[~valid].sum() == 0
    # none 모드의 유효유저 평균과 같아야 모드 비교가 성립
    gn = V3.build_gate(clv, clv, "none")
    assert abs(gn.mean() - g[valid].mean()) < 1e-5


def test_result_hash_includes_gate_mode_for_all_archs():
    """cfg_hash는 pref_only에서 GATE_MODE를 빼므로, result_hash가 따로 넣지 않으면
    pref_only의 none/clv/vhat 실행이 같은 결과 파일을 덮어쓴다."""
    import copy
    c1 = copy.deepcopy(V3.CFG); c1["GATE_MODE"] = "clv"
    c2 = copy.deepcopy(V3.CFG); c2["GATE_MODE"] = "vhat"
    for arch in ("pref_only", "two_stage", "joint_warm", "joint"):
        assert V3.result_hash(c1, V3.DCFG, arch) != V3.result_hash(c2, V3.DCFG, arch), arch


def test_data_stats_recorded():
    """threshold 비교에 필요한 분모(정답쌍·평가유저·degree 분포)가 결과에 남아야 한다."""
    # gt_pairs/eval_users는 중첩 def build_eval 안에 있어 _fn_ns가 못 잡는다 → 전체 소스로 확인
    for key in ["train_edges_before", "train_edges_after", "item_drop_rate",
                "edge_drop_rate", "gt_pairs", "eval_users", "item_degree"]:
        assert key in NS, f"data_stats에 {key} 없음"
    assert '"data_stats":d["data_stats"]' in _fn_ns("main")


# ── 2026-08-09: M3(가치그래프) / M4(CLV-aware 손실) ────────────────────────
def test_graph_modes_change_adjacency(tmp_path):
    """binary는 기존 동작과 동일해야 하고, count/value는 실제로 가중치를 바꿔야 한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    a = V3.prepare_data(dict(cfg, GRAPH_MODE="binary"), dcfg)["adj"].coalesce()
    for mode in ("count", "value", "price", "clv"):
        b = V3.prepare_data(dict(cfg, GRAPH_MODE=mode), dcfg)["adj"].coalesce()
        # shape만 보면 엣지 집합이 바뀌어도 통과한다 — 좌표 자체를 비교한다.
        torch.testing.assert_close(a.indices(), b.indices())
        assert not torch.allclose(a.values(), b.values()), f"{mode}가 가중치를 안 바꿈"
    try:
        V3.prepare_data(dict(cfg, GRAPH_MODE="bogus"), dcfg)
        raise AssertionError("알 수 없는 GRAPH_MODE는 거부해야 한다")
    except ValueError:
        pass


def test_graph_alpha_scales_weights(tmp_path):
    cfg, dcfg = _tiny_dataset(tmp_path)
    lo = V3.prepare_data(dict(cfg, GRAPH_MODE="value", GRAPH_ALPHA=0.5), dcfg)["adj"].coalesce()
    hi = V3.prepare_data(dict(cfg, GRAPH_MODE="value", GRAPH_ALPHA=4.0), dcfg)["adj"].coalesce()
    assert not torch.allclose(lo.values(), hi.values())
    # α=0이면 1 + 0·log(...) = 1 이므로 binary와 같아야 한다
    z = V3.prepare_data(dict(cfg, GRAPH_MODE="value", GRAPH_ALPHA=0.0), dcfg)["adj"].coalesce()
    b = V3.prepare_data(dict(cfg, GRAPH_MODE="binary"), dcfg)["adj"].coalesce()
    torch.testing.assert_close(z.values(), b.values(), rtol=1e-5, atol=1e-6)


def test_graph_clv_mode_actually_uses_clv(tmp_path):
    """clv 변형만 CLV를 직접 참조한다 — count/value는 거래정보만 쓰는 대조군.
    CLV가 달라지면 clv 그래프는 바뀌고 count/value 그래프는 안 바뀌어야 한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    real = V3.prepare_data(dict(cfg, GRAPH_MODE="clv"), dcfg)
    assert "clv" in real and "vhat" in real and "x_val_u" in real   # main과 공유

    # CLV를 뒤집어도 count/value는 불변, clv는 변해야 한다
    orig = V3.clv_features
    try:
        def flipped(train, n_users, c, is_date):
            x, cl, vh = orig(train, n_users, c, is_date)
            return x, -cl, vh                      # CLV 순서를 뒤집는다
        V3.clv_features = flipped
        for mode, should_change in (("count", False), ("value", False), ("clv", True)):
            base = V3.prepare_data(dict(cfg, GRAPH_MODE=mode), dcfg)["adj"].coalesce()
            V3.clv_features = orig
            ref = V3.prepare_data(dict(cfg, GRAPH_MODE=mode), dcfg)["adj"].coalesce()
            V3.clv_features = flipped
            changed = not torch.allclose(base.values(), ref.values())
            assert changed == should_change, f"{mode}: CLV 의존성이 기대와 다름"
    finally:
        V3.clv_features = orig


def test_graph_clv_varies_within_user_in_raw_weights(tmp_path):
    """clv 모드는 **정규화 전 raw 엣지가중치**가 유저 안에서 달라야 한다.
    정규화된 adj로 검사하면 유저별 상수 가중이어도 아이템 degree 차이 때문에 통과한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    d = V3.prepare_data(dict(cfg, GRAPH_MODE="clv"), dcfg)
    eu = d["pos_key"] // d["n_items"]
    w = d["w_edge"]
    assert len(w) == len(eu)
    spreads = [w[eu == u].std() for u in np.unique(eu) if (eu == u).sum() > 1]
    assert len(spreads) > 0, "여러 상품을 산 유저가 없어 검증이 성립하지 않는다"
    assert max(spreads) > 1e-6, "raw 엣지가중치가 유저 내부에서 전부 같음"
    # 대조: 유저별 상수 가중을 넣으면 이 검사가 실패해야 한다(검사가 실효적인지 확인)
    const = np.repeat(np.arange(1.0, len(np.unique(eu)) + 1.0), 1)[
        np.searchsorted(np.unique(eu), eu)]
    assert max(const[eu == u].std() for u in np.unique(eu) if (eu == u).sum() > 1) < 1e-6


def test_graph_price_is_clv_free_control(tmp_path):
    """price는 clv에서 g(CLV_u)만 뺀 대조군 — CLV가 바뀌어도 불변이어야 한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    orig = V3.clv_features
    try:
        def flipped(train, n_users, c, is_date):
            x, cl, vh = orig(train, n_users, c, is_date)
            return x, -cl, vh
        a = V3.prepare_data(dict(cfg, GRAPH_MODE="price"), dcfg)["w_edge"]
        V3.clv_features = flipped
        b = V3.prepare_data(dict(cfg, GRAPH_MODE="price"), dcfg)["w_edge"]
    finally:
        V3.clv_features = orig
    np.testing.assert_allclose(a, b, rtol=1e-6)
    # price와 clv는 서로 달라야 한다(g가 상수 1이 아니므로)
    c = V3.prepare_data(dict(cfg, GRAPH_MODE="clv"), dcfg)["w_edge"]
    assert not np.allclose(a, c), "price와 clv가 같으면 게이트가 작동하지 않는 것"


def test_binary_baseline_is_pure_m1(tmp_path):
    """M3/M4를 켜도 비교기준은 binary+plain이어야 한다 (paired delta가 0이 되면 안 됨)."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    on = dict(cfg, GRAPH_MODE="clv", LOSS_MODE="pair")
    d = V3.prepare_data(on, dcfg)
    d["loss_w"] = np.ones(len(d["tr_u"]), np.float32)      # main()이 채우는 자리
    d_base, cfg_base = V3.binary_baseline(d, on)
    assert d_base is not d
    assert cfg_base["GRAPH_MODE"] == "binary" and cfg_base["LOSS_MODE"] == "plain"
    assert d_base["loss_w"] is None
    ref = V3.prepare_data(dict(cfg, GRAPH_MODE="binary"), dcfg)["adj"].coalesce()
    b = d_base["adj"].coalesce()
    torch.testing.assert_close(b.indices(), ref.indices())
    torch.testing.assert_close(b.values(), ref.values())
    # 순수 M1로 실행 중이면 같은 객체를 돌려줘 불필요한 재계산을 안 한다
    d2 = V3.prepare_data(dict(cfg), dcfg)
    assert V3.binary_baseline(d2, dict(cfg))[0] is d2
    # 기준 체크포인트 해시는 기존 M1과 같아야 한다(재학습 방지)
    for arch in ("pref_only", "two_stage", "joint_warm", "joint"):
        assert V3.cfg_hash(cfg_base, V3.DCFG, arch, 42) == \
               V3.cfg_hash(dict(cfg), V3.DCFG, arch, 42)


def test_eval_test_is_opt_in(tmp_path):
    """EVAL_TEST=False면 test 정답조차 만들지 않는다 — 개발 중 반복 노출 방지."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    assert V3.CFG["EVAL_TEST"] is False, "기본값은 꺼져 있어야 한다"
    off = V3.prepare_data(dict(cfg, EVAL_TEST=False), dcfg)["splits"]
    assert set(off) == {"val"}, f"test가 계산됨: {set(off)}"
    on = V3.prepare_data(dict(cfg, EVAL_TEST=True), dcfg)["splits"]
    assert "test" in on
    # 결과 파일명에도 반영돼야 이전 test 포함 결과를 덮어쓰지 않는다
    a = dict(V3.CFG); b = dict(V3.CFG); b["EVAL_TEST"] = True
    assert V3.result_hash(a, V3.DCFG, "pref_only") != V3.result_hash(b, V3.DCFG, "pref_only")


def test_loss_weight_modes():
    """user는 유저 평균 1, pair는 행 평균 1. plain은 None."""
    rng = np.random.default_rng(0)
    n_u, n_rows = 50, 400
    tr_u = rng.integers(0, n_u, n_rows)
    train = pd.DataFrame({"u_idx": tr_u, "i_idx": rng.integers(0, 30, n_rows),
                          "up": rng.random(n_rows) * 10 + 1})
    clv = rng.random(n_u)
    assert V3.build_loss_weights(train, tr_u, clv, dict(V3.CFG, LOSS_MODE="plain")) is None

    wp = V3.build_loss_weights(train, tr_u, clv, dict(V3.CFG, LOSS_MODE="pair", LOSS_LAMBDA=1.0))
    assert len(wp) == n_rows and abs(wp.mean() - 1.0) < 1e-5      # 행 평균 1

    wu = V3.build_loss_weights(train, tr_u, clv, dict(V3.CFG, LOSS_MODE="user", LOSS_LAMBDA=1.0))
    assert len(wu) == n_rows
    # 같은 유저의 행은 모두 같은 가중치여야 한다(유저 단위 개입)
    for u in np.unique(tr_u):
        assert np.allclose(wu[tr_u == u], wu[tr_u == u][0])
    # 유저 단위로 모으면 평균 1
    per_user = np.array([wu[tr_u == u][0] for u in range(n_u)])
    assert abs(per_user.mean() - 1.0) < 1e-5
    try:
        V3.build_loss_weights(train, tr_u, clv, dict(V3.CFG, LOSS_MODE="bogus"))
        raise AssertionError("알 수 없는 LOSS_MODE는 거부해야 한다")
    except ValueError:
        pass


def test_loss_weight_user_is_constant_within_user(tmp_path):
    """m4_user 가중은 한 유저 안에서 상수, m4_pair는 상수가 아니다.

    ⚠ 이전 버전은 여기서 "m4_user는 순위를 못 바꾼다"는 **부정확한** 서술을 소스에서
    문자열로 확인했다. 유저 상수 가중도 공유 아이템 임베딩을 통해 최종 순위를 바꾼다.
    검증 가능한 사실(가중치가 유저 안에서 상수인지)만 확인한다."""
    cfg, dcfg = _tiny_dataset(tmp_path)
    d = V3.prepare_data(cfg, dcfg)
    tr_u, clv = d["tr_u"], d["clv"]
    wu = V3.build_loss_weights(d["train"], tr_u, clv, dict(cfg, LOSS_MODE="user"))
    wp = V3.build_loss_weights(d["train"], tr_u, clv, dict(cfg, LOSS_MODE="pair"))
    multi = [u for u in np.unique(tr_u) if (tr_u == u).sum() > 1]
    assert multi, "거래가 여러 건인 유저가 없어 검증이 성립하지 않는다"
    assert max(wu[tr_u == u].std() for u in multi) < 1e-6
    assert max(wp[tr_u == u].std() for u in multi) > 1e-6


def test_bpr_loss_applies_sample_weights():
    """가중치를 주면 손실이 달라져야 하고, 전부 1이면 균등 평균과 같아야 한다."""
    torch.manual_seed(0)
    n_u, n_i, n_c = 8, 12, 3
    adj = torch.sparse_coo_tensor(torch.tensor([[0, n_u], [n_u, 0]]),
                                  torch.tensor([0.5, 0.5]),
                                  size=(n_u + n_i, n_u + n_i)).coalesce()
    m = V3.DualSpaceLightGCN(n_u, n_i, n_c,
                             np.random.rand(n_u, 5).astype(np.float32),
                             np.random.rand(n_i, 2).astype(np.float32),
                             np.zeros(n_i, dtype=np.int64), dict(V3.CFG), adj)
    u = torch.tensor([0, 1, 2, 3]); i = torch.tensor([0, 1, 2, 3]); j = torch.tensor([4, 5, 6, 7])
    g = torch.ones(n_u)
    l0, _ = m.bpr_loss(u, i, j, g, 0.0, None)
    l1, _ = m.bpr_loss(u, i, j, g, 0.0, torch.ones(4))
    torch.testing.assert_close(l0, l1)                    # w=1이면 동일
    lw, _ = m.bpr_loss(u, i, j, g, 0.0, torch.tensor([2.0, 0.5, 1.0, 0.5]))
    assert not torch.allclose(l0, lw)                     # 가중치가 실제로 반영됨


def test_m3_m4_in_ckpt_hash_for_all_archs():
    """GATE_MODE와 달리 M3·M4는 pref_only의 가중치도 바꾸므로 예외가 없어야 한다."""
    import copy
    base = copy.deepcopy(V3.CFG)
    # 모드를 켜는 변경은 모든 아키텍처의 해시를 바꿔야 한다
    for key, val in [("GRAPH_MODE", "count"), ("GRAPH_MODE", "value"),
                     ("GRAPH_MODE", "clv"), ("LOSS_MODE", "pair")]:
        c2 = copy.deepcopy(base); c2[key] = val
        for arch in ("pref_only", "two_stage", "joint_warm", "joint"):
            assert V3.cfg_hash(base, V3.DCFG, arch, 42) != V3.cfg_hash(c2, V3.DCFG, arch, 42), \
                f"{key}가 {arch} 해시에 반영되지 않음"
    # 계수는 해당 모드가 켜져 있을 때만 반영된다(꺼져 있으면 학습에 안 쓰이므로)
    for on_key, on_val, coef in [("GRAPH_MODE", "count", "GRAPH_ALPHA"),
                                 ("LOSS_MODE", "pair", "LOSS_LAMBDA")]:
        on = copy.deepcopy(base); on[on_key] = on_val
        on2 = copy.deepcopy(on); on2[coef] = 2.0
        for arch in ("pref_only", "two_stage", "joint_warm", "joint"):
            assert V3.cfg_hash(on, V3.DCFG, arch, 42) != V3.cfg_hash(on2, V3.DCFG, arch, 42), \
                f"{coef}가 {arch} 해시에 반영되지 않음"


def test_m3_m4_defaults_are_neutral():
    """기본값은 M1과 동일해야 한다 — 기존 결과와의 연속성."""
    assert V3.CFG["GRAPH_MODE"] == "binary" and V3.CFG["LOSS_MODE"] == "plain"


def test_default_m3_m4_do_not_invalidate_existing_checkpoints():
    """기본값(binary/plain)이면 학습 결과가 v3.10 이전과 같으므로 해시가 바뀌면 안 된다.
    바뀌면 M3/M4 도입만으로 두 데이터셋의 모든 baseline을 다시 학습하게 된다."""
    import copy
    c = copy.deepcopy(V3.CFG)
    h = V3.cfg_hash(c, V3.DCFG, "pref_only", 42)
    # 기본 모드에서는 α·λ를 바꿔도 학습에 쓰이지 않으므로 해시가 같아야 한다
    for key, val in [("GRAPH_ALPHA", 7.0), ("LOSS_LAMBDA", 7.0)]:
        c2 = copy.deepcopy(V3.CFG); c2[key] = val
        assert V3.cfg_hash(c2, V3.DCFG, "pref_only", 42) == h, f"{key}가 기본모드에서 해시를 바꿈"
    # 반면 모드를 켜면 α·λ가 해시에 반영돼야 한다
    on = copy.deepcopy(V3.CFG); on["GRAPH_MODE"] = "count"
    on2 = copy.deepcopy(on); on2["GRAPH_ALPHA"] = 7.0
    assert V3.cfg_hash(on, V3.DCFG, "pref_only", 42) != V3.cfg_hash(on2, V3.DCFG, "pref_only", 42)


def test_min_item_inter_changes_ckpt_hash():
    """k-core 필터는 아이템 유니버스를 바꾸므로 체크포인트가 달라야 한다."""
    import copy
    c1 = copy.deepcopy(V3.CFG); c1["MIN_ITEM_INTER"] = 1
    c2 = copy.deepcopy(V3.CFG); c2["MIN_ITEM_INTER"] = 10
    assert V3.cfg_hash(c1, V3.DCFG, "pref_only", 42) != V3.cfg_hash(c2, V3.DCFG, "pref_only", 42)


def test_clv_has_no_shrinkage():
    train = pd.DataFrame({
        "u_idx": [0, 0, 0, 1], "i_idx": [0, 1, 2, 0], "b_raw": [7, 7, 7, 1],
        "t": [0, 0, 0, 5], "v": [20.0, 20.0, 20.0, 90.0], "up": [20.0, 20.0, 20.0, 90.0],
        "cat_idx": [0, 0, 0, 1]})
    x, clv, vhat = V3.clv_features(train, 2, dict(V3.CFG), is_date=False)
    assert x.shape == (2, 5)
    assert x[1, 3] > x[0, 3]        # 구매 1건 유저의 AOV가 그대로 반영됨
    assert "SHRINKAGE" not in SRC and "shrink" not in SRC.lower()


def test_value_block_does_not_propagate():
    body = _fn("value_emb")
    assert "sparse" not in body and "adj" not in body
    assert "sparse" in _fn("propagate_pref")


def test_three_segments():
    assert V3.SEG_NAMES == ["저CLV", "중CLV", "고CLV"] and V3.CFG["SEG_EDGES"] == (0.2, 0.8)


def test_pref_only_evaluates_baseline_lambda_only():
    """pref_only는 가치 블록이 미학습(λ=0이라 기울기 0)이라, λ>0 채점은 무작위
    초기화된 임베딩을 주입하는 것이 되어 의미가 없다. λ=0만 평가해야 한다."""
    body = _fn_ns("main")
    assert 'sweep=[0.0]ifarch=="pref_only"else' in body
    assert "forlaminsweep" in body
    assert "cfg[\"LAMBDA_EVAL_SWEEP\"]" not in _fn_ns("select_lambda")   # 실제 평가된 λ만 사용


def test_train_stats_are_plain_python_scalars():
    """체크포인트에 numpy 스칼라가 섞이면 torch.load(weights_only=True)가 거부한다."""
    body = _fn_ns("train_phase")
    assert '"best_epoch":int(best_ep)' in body
    assert '"updates":int(updates)' in body and '"samples":int(samples)' in body
    assert "float(best)" in body


# ── 코드리뷰(2026-08-04) 지적: 정규화 구현이 v2와 달라 선호 임베딩이 붕괴 ────────
def test_regularization_is_batch_l2_not_global_weight_decay():
    """v2는 optimizer에 weight_decay를 주지 않고 BPR loss 안에서 배치 layer-0만 L2했다.
    v3.1이 숫자만 가져와 Adam(weight_decay=)에 꽂은 것이 붕괴 원인이었다."""
    assert V3.CFG["REG_MODE"] == "batch_l2"
    assert V3.CFG["PREF_REG"] == 1e-3        # v2와 같은 값
    assert V3.CFG["VALUE_REG"] == 0.0        # 첫 복구는 가치 정규화 없이
    body = _fn_ns("train_phase")
    # batch_l2 모드에서는 optimizer 감쇠가 0이어야 한다
    assert 'wd=cfg["WD"]ifcfg["REG_MODE"]=="global_wd"else0.0' in body
    assert "weight_decay=wd" in body


def test_batch_l2_only_touches_batch_rows_and_skips_frozen_pref():
    ns_body = _fn_ns("batch_l2")
    assert "self.E_u.weight[u]" in ns_body and "self.E_i.weight[i]" in ns_body
    assert "self.E_i.weight[j]" in ns_body
    assert "self.E_u.weight.requires_grad" in ns_body      # 동결 단계에서는 건너뜀
    assert "VALUE_REG" in ns_body                          # 가치 정규화는 별도 계수


def test_bpr_loss_returns_training_diagnostics():
    """붕괴를 매 epoch 확인할 수 있도록 BPR과 P(pos>neg)를 함께 반환해야 한다."""
    torch.manual_seed(0)
    n_u, n_i, n_c = 8, 12, 3
    x_val = np.random.rand(n_u, 5).astype(np.float32)
    x_it = np.random.rand(n_i, 2).astype(np.float32)
    cat = np.random.randint(0, n_c, n_i).astype(np.int64)
    n = n_u + n_i
    adj = torch.sparse_coo_tensor(torch.tensor([[0, n_u], [n_u, 0]]),
                                  torch.tensor([0.5, 0.5]), size=(n, n)).coalesce()
    m = V3.DualSpaceLightGCN(n_u, n_i, n_c, x_val, x_it, cat, dict(V3.CFG), adj)
    loss, dg = m.bpr_loss(torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5]),
                          torch.rand(n_u), 1.0)
    assert torch.isfinite(loss) and loss.requires_grad
    assert set(dg) == {"bpr", "p_correct"} and 0.0 <= dg["p_correct"] <= 1.0
    assert abs(dg["bpr"] - 0.693) < 0.6          # 초기값이 ln2 근처


def test_score_diagnostics_applies_gate():
    """v3.1은 gate를 인자로 받고도 쓰지 않아 실효 비율을 과대평가했다."""
    body = _fn_ns("score_diagnostics")
    assert "gate[us]" in body and "sv_eff" in body


def test_2x2_diagnostic_exists():
    assert "run_2x2_diagnostic" in dir(V3)
    body = _fn_ns("run_2x2_diagnostic")
    assert '"batch_l2","global_wd"' in body and '"uniform","hard50"' in body


# ── 코드리뷰 2차(2026-08-04) 지적 반영 확인 ──────────────────────────────
def test_comparison_baseline_is_external_pref_only():
    """joint의 λ=0(ablation)이 아니라 외부 pref_only와 비교해야 한다."""
    body = _fn_ns("select_lambda")
    assert "base_per_seed[s]" in body
    assert 'val_per_seed[s][0.0]' not in body          # 자기 λ=0 참조가 남아있으면 안 됨
    m = _fn_ns("main")
    assert 'get_or_train("pref_only"' in m and "base_model" in m
    assert '"role"' in m or "role" in m                # joint λ=0을 ablation으로 표기


def test_bootstrap_treats_seeds_as_repeated_measures():
    """시드를 이어붙이면 같은 유저가 중복돼 CI가 좁아진다. 유저별 시드 평균 후 재표집."""
    body = _fn_ns("paired_bootstrap")
    assert "np.stack(diffs_per_seed)" in body
    assert "d.mean(axis=0)" in body                    # 유저별 시드 평균이 먼저
    assert "per_seed_sd" in body
    # 실제 동작: 동일 diff를 3시드로 주면 1시드일 때와 CI 폭이 비슷해야 한다
    rng = np.random.default_rng(0)
    d = rng.normal(0.01, 0.05, 500)
    one = V3.paired_bootstrap([d], 500)
    three = V3.paired_bootstrap([d, d, d], 500)
    w1 = one["hi"] - one["lo"]; w3 = three["hi"] - three["lo"]
    assert abs(w1 - w3) / w1 < 0.15                    # 표본이 3배로 부풀지 않음


def test_value_mlp_has_no_output_layernorm():
    """출력 LayerNorm이 있으면 가치 점수 크기가 학습과 무관하게 고정돼
    joint에서 학습 시작부터 선호항을 압도한다."""
    m = V3.MLP(5, 8, 4, out_scale=0.01)
    assert isinstance(m.net[-1], torch.nn.Linear)      # 마지막은 Linear로 끝남
    assert not any(isinstance(l, torch.nn.LayerNorm) for l in list(m.net)[3:])
    out = m(torch.randn(64, 5))
    assert out.std() < 0.2                             # 초기 출력이 작아야 함
    assert "VALUE_OUT_SCALE" in V3.CFG


def test_result_hash_covers_evaluation_rules():
    """시드 목록·λ 스윕·δ·K만 바꿔도 결과 파일이 덮어써지면 안 된다."""
    import copy
    c = copy.deepcopy(V3.CFG)
    h0 = V3.result_hash(c, V3.DCFG, "joint")
    for key, val in [("SEED_LIST", [1, 2]), ("LAMBDA_EVAL_SWEEP", [0.0, 9.0]),
                     ("NONINFERIORITY_DELTA", 0.05), ("K_LIST", [5]), ("N_BOOT", 7)]:
        c2 = copy.deepcopy(V3.CFG); c2[key] = val
        assert V3.result_hash(c2, V3.DCFG, "joint") != h0, f"{key}가 해시에 없음"


def test_joint_warm_differs_from_two_stage_only_by_freezing():
    """joint_warm = two_stage와 같은 출발점(pref_only) + 동결 없음.
    warm start 자체는 연구 기여가 아니라, joint이 학습 상한에서 잘려 진 것인지
    구조적으로 진 것인지 가르기 위한 진단용이다."""
    body = _fn_ns("get_or_train")
    # body는 공백 제거본이므로 'elifarch==' 형태로 잘라야 한다
    ts = body.split('elifarch=="two_stage"')[1].split("elifarch==")[0]
    jw = body.split('elifarch=="joint_warm"')[1].split("elifarch==")[0]
    for blk in (ts, jw):
        assert 'get_or_train("pref_only"' in blk        # 같은 출발점
        assert "load_state_dict(base.state_dict())" in blk
    assert "freeze_pref_and_cache()" in ts              # two_stage만 동결
    assert "freeze_pref_and_cache()" not in jw
    assert "model.value_params()" in ts                 # two_stage는 가치만
    assert "list(model.parameters())" in jw             # joint_warm은 전부


def test_ablation_archs_cover_both_joint_variants():
    """선호 블록이 가치항과 함께 학습된 아키텍처는 λ=0이 baseline이 아니다."""
    assert V3.ABLATION_ARCHS == {"joint", "joint_warm"}
    assert "two_stage" not in V3.ABLATION_ARCHS         # 동결이라 λ=0이 곧 pref_only
    assert "baseline 아님" in V3.ARCH_LABEL["joint_warm"]
    m = _fn_ns("main")
    assert "archinABLATION_ARCHS" in m


def test_joint_warm_has_own_checkpoint():
    """arch가 해시에 들어가야 joint과 체크포인트가 안 섞인다."""
    h_jw = V3.cfg_hash(V3.CFG, V3.DCFG, "joint_warm", 42)
    h_j = V3.cfg_hash(V3.CFG, V3.DCFG, "joint", 42)
    h_ts = V3.cfg_hash(V3.CFG, V3.DCFG, "two_stage", 42)
    assert len({h_jw, h_j, h_ts}) == 3


def test_checkpoint_hash_is_stable_across_code_versions():
    """cfg_hash(체크포인트용)는 CODE_VERSION이 바뀌어도 같아야 한다. v3.4에서
    여기 CODE_VERSION을 넣었다가, 학습과 무관한 코드 변경만으로 이미 있는
    pref_only 체크포인트를 못 찾고 48분짜리 재학습이 세 시드 도는 낭비가 났다."""
    h_before = V3.cfg_hash(V3.CFG, V3.DCFG, "pref_only", 42)
    old_version = V3.CODE_VERSION
    try:
        V3.CODE_VERSION = "vX.Y-fake"
        h_after = V3.cfg_hash(V3.CFG, V3.DCFG, "pref_only", 42)
    finally:
        V3.CODE_VERSION = old_version
    assert h_before == h_after


def test_result_hash_still_changes_with_code_version():
    """result_hash(결과 파일용)는 CODE_VERSION이 바뀌면 여전히 달라져야 한다 —
    코드 버전 추적 자체가 필요 없어진 게 아니라 체크포인트에서만 뺀 것이다."""
    h_before = V3.result_hash(V3.CFG, V3.DCFG, "pref_only")
    old_version = V3.CODE_VERSION
    try:
        V3.CODE_VERSION = "vX.Y-fake"
        h_after = V3.result_hash(V3.CFG, V3.DCFG, "pref_only")
    finally:
        V3.CODE_VERSION = old_version
    assert h_before != h_after


# ── 2026-08-09 2차 리뷰: 노출지표 / val delta / 설정 원자성 ──────────────
def test_exposure_stats_math():
    """엔트로피·실효카탈로그·상위점유율이 정의대로 계산되는가."""
    z = V3.exposure_stats(np.zeros(10), 10)
    assert z["n_distinct"] == 0 and z["eff_catalog"] == 0.0
    uni = V3.exposure_stats(np.array([5.0] * 8 + [0.0] * 2), 10)
    assert uni["n_distinct"] == 8
    np.testing.assert_allclose(uni["entropy"], np.log(8), rtol=1e-9)
    np.testing.assert_allclose(uni["eff_catalog"], 8.0, rtol=1e-9)   # 균등 → 상품 수
    conc = V3.exposure_stats(np.array([100.0] + [0.0] * 999), 1000)
    assert conc["eff_catalog"] < 1.001 and conc["top10_share"] == 1.0
    # 절대 개수는 n_items(분모)와 무관해야 한다 — Coverage 착시 방지의 핵심
    e = np.array([3.0, 1.0] + [0.0] * 98)
    assert V3.exposure_stats(e, 100)["n_distinct"] == \
           V3.exposure_stats(e, 100000)["n_distinct"] == 2


def test_flatten_exposes_exposure_fields():
    res = {"overall": {10: {"recall": 0.1}}, "coverage": {10: 0.5}, "gini": {10: 0.9},
           "exposure": {10: V3.exposure_stats(np.array([2.0, 1.0, 0.0]), 3)},
           "value_alignment": 0.3, "seg": {}}
    f = V3.flatten(res)
    for m in ("n_distinct@10", "entropy@10", "eff_catalog@10",
              "top10_share@10", "top100_share@10"):
        assert m in f, f"{m}가 결과 행에 안 실림"


def test_model_id_distinguishes_interventions():
    c = lambda **kw: dict(V3.CFG, **kw)
    assert V3.model_id(c()) == "m1"
    assert V3.model_id(c(GRAPH_MODE="clv")) == "m3_clv"
    assert V3.model_id(c(LOSS_MODE="pair")) == "m4_pair"
    assert V3.model_id(c(GRAPH_MODE="count", LOSS_MODE="user")) == "m3_count_m4_user"
    assert V3.model_id(c(ARCH="two_stage")) == "m2_two_stage"
    ids = {V3.model_id(c(GRAPH_MODE=g)) for g in ("binary", "count", "value", "price", "clv")}
    assert len(ids) == 5


def test_val_delta_is_computed(tmp_path):
    """EVAL_TEST=False에서도 외부 M1 대비 delta가 남아야 한다.
    (이전에는 pu_split이 test/holdout 전용이라 _delta.csv가 통째로 비었다)"""
    src = _fn_ns("main")
    assert 'pu_split["val"]=' in src, "val이 delta 계산 대상에 안 들어감"
    assert 'base_pu_split["val"]=' in src


def test_configure_run_updates_dcfg_and_out_dir():
    """CFG["DATASET"]만 바꾸면 DCFG·OUT_DIR이 어긋난다 — 실제로 재현됐던 사고."""
    import copy
    saved, saved_dcfg = copy.deepcopy(V3.CFG), V3.DCFG
    try:
        V3.configure_run("dunnhumby")
        assert V3.DCFG is V3.SCHEMA["dunnhumby"]
        assert "dunnhumby" in V3.CFG["OUT_DIR"] and "results_v3_hm" not in V3.CFG["OUT_DIR"]
        V3.configure_run("hm", SEED_LIST=[7])
        assert V3.DCFG is V3.SCHEMA["hm"] and V3.CFG["SEED_LIST"] == [7]
        assert "results_v3_hm" in V3.CFG["OUT_DIR"]
        with pytest.raises(AssertionError):
            V3.configure_run("없는데이터셋")
    finally:
        V3.CFG.clear(); V3.CFG.update(saved); V3.DCFG = saved_dcfg


def test_main_asserts_dcfg_out_dir_consistency():
    """configure_run을 우회해도 main() 진입 시 걸려야 한다."""
    body = _fn_ns("main")
    assert 'assertDCFGisSCHEMA[cfg["DATASET"]]' in body
    assert 'cfg["DATASET"]instr(cfg["OUT_DIR"])' in body


def test_run_2x2_does_not_hardcode_test_split():
    """EVAL_TEST=False에서 긴 학습 뒤 AttributeError로 죽던 경로."""
    body = _fn_ns("run_2x2_diagnostic")
    assert 'sp="test"if' in body and 'else"val"' in body   # split을 런타임에 고름
    assert 'df.split==sp' in body                          # 하드코딩된 test 필터 없음

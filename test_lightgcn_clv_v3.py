"""v3 설계 불변식 테스트. 지키는 것은 '무엇을 넣었나'가 아니라
'리뷰에서 지적된 문제가 되살아나지 않았나'다."""
import io, tokenize
from pathlib import Path

import numpy as np
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
    assert "passes_noninferiority" in body and "val_pwgain10" in body
    assert "NONINFERIORITY_DELTA" in body
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
    lam, tbl = V3.select_lambda(per_seed, cfg)
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
    clv = np.array([10.0, 20.0, 30.0, np.nan, 40.0])
    g = V3.clv_gate(clv)
    assert g[3] == 0.0
    np.testing.assert_allclose(np.sort(g[[0, 1, 2, 4]]), [0.25, 0.5, 0.75, 1.0])
    assert "**" not in _fn("clv_gate")


def test_clv_has_no_shrinkage():
    train = pd.DataFrame({
        "u_idx": [0, 0, 0, 1], "i_idx": [0, 1, 2, 0], "b_raw": [7, 7, 7, 1],
        "t": [0, 0, 0, 5], "v": [20.0, 20.0, 20.0, 90.0], "up": [20.0, 20.0, 20.0, 90.0],
        "cat_idx": [0, 0, 0, 1]})
    x, clv = V3.clv_features(train, 2, dict(V3.CFG), is_date=False)
    assert x.shape == (2, 5)
    assert x[1, 3] > x[0, 3]        # 구매 1건 유저의 AOV가 그대로 반영됨
    assert "SHRINKAGE" not in SRC and "shrink" not in SRC.lower()


def test_negative_sampling_is_uniform():
    pos = np.sort(np.array([0 * 50 + 3, 0 * 50 + 7, 1 * 50 + 3], dtype=np.int64))
    neg = V3.sample_negatives(np.zeros(300, dtype=np.int64), 50, pos, np.random.default_rng(0))
    assert not np.isin(neg, [3, 7]).any() and len(np.unique(neg)) > 20
    body = _fn("sample_negatives")
    assert "cat_items" not in body and "hard" not in body.lower()


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

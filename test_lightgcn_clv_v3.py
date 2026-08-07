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
    np.testing.assert_allclose(np.sort(g[[0, 1, 2, 4]]), raw / (raw.sum() / 5), rtol=1e-6)
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

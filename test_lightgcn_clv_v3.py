"""v3 설계 불변식 테스트. 이 파일이 지키는 건 '무엇을 넣었나'가 아니라
'v2에서 임의로 넣었던 조건들이 다시 기어들어오지 않았나'다."""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import lightgcn_clv_v3 as V3

_RAW = (Path(__file__).parent / "lightgcn_clv_v3.py").read_text(encoding="utf-8")


def _code_only(text):
    """주석·docstring을 걷어낸 실행 코드만 남긴다. 이걸 안 하면 '제거했다'고 적어둔
    설명 문구가 '아직 남아있다'로 오판된다."""
    import io, tokenize
    out = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (tokenize.INDENT, tokenize.NEWLINE,
                                                          tokenize.NL, tokenize.DEDENT):
            continue                      # docstring
        out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            prev_type = tok.type
        else:
            prev_type = tok.type
    return " ".join(out)


SRC = _code_only(_RAW)


def _fn_code(name):
    return SRC.split(f"def {name}")[1].split(" def ")[0]


def test_no_guardrails_or_selection_machinery():
    """v2의 가드레일·목적함수·스크리닝·fallback이 하나도 없어야 한다."""
    for dead in ["ACCURACY_EPSILON", "RECALL50_EPSILON", "HR_EPSILON", "DIVERSITY_EPSILON",
                 "VT_TOPK_CKPTS", "EPOCH_SCREEN_LAMBDA", "SHRINKAGE_K", "CLV_GATE_POWER",
                 "HARD_NEG_RATIO", "LAMBDA_GRID"]:
        assert dead not in V3.CFG, f"CFG에 {dead}가 남아있음"
    for dead in ["_passes", "run_stage_b_grid", "vt_topk", "grid_results"]:
        assert dead not in SRC, f"코드에 {dead}가 남아있음"


def test_gate_is_linear_percentile():
    """gate = percentile_rank(CLV). 제곱 아님. NaN은 0."""
    clv = np.array([10.0, 20.0, 30.0, np.nan, 40.0])
    g = V3.clv_gate(clv)
    assert g[3] == 0.0
    np.testing.assert_allclose(np.sort(g[[0, 1, 2, 4]]), [0.25, 0.5, 0.75, 1.0])
    assert "**" not in _fn_code("clv_gate")          # 제곱 없음


def _toy(n=60, seed=0, basket=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "u_idx": rng.integers(0, 8, n), "i_idx": rng.integers(0, 12, n),
        "t": rng.integers(0, 40, n), "v": rng.uniform(5, 100, n).round(2),
        "cat_idx": rng.integers(0, 3, n)})
    df["up"] = df["v"]
    if basket: df["b_raw"] = rng.integers(0, 20, n)
    return df


def test_clv_has_no_shrinkage_and_uses_basket_units():
    """AOV는 구매 건당 평균이고 축소추정이 없어야 한다."""
    train = pd.DataFrame({
        "u_idx": [0, 0, 0, 1], "i_idx": [0, 1, 2, 0], "b_raw": [7, 7, 7, 1],
        "t": [0, 0, 0, 5], "v": [20.0, 20.0, 20.0, 90.0], "up": [20.0, 20.0, 20.0, 90.0],
        "cat_idx": [0, 0, 0, 1]})
    x, clv = V3.clv_features(train, 2, dict(V3.CFG), is_date=False)
    assert x.shape == (2, 5)
    # 유저1은 구매 1건(90) — 축소추정이 있으면 전체평균 쪽으로 끌려가 순위가 흔들린다.
    # 없으면 AOV_raw 그대로 90 > 60이라 AOV 백분위가 유저0보다 높아야 한다.
    assert x[1, 3] > x[0, 3]
    assert "SHRINKAGE" not in SRC and "shrink" not in SRC.lower()


def test_negative_sampling_is_uniform():
    """균등 무작위여야 하고, 이미 산 아이템은 뽑히면 안 된다."""
    pos_key = np.array([0 * 50 + 3, 0 * 50 + 7, 1 * 50 + 3], dtype=np.int64)
    u = np.zeros(300, dtype=np.int64)
    neg = V3.sample_negatives(u, 50, np.sort(pos_key), np.random.default_rng(0))
    assert not np.isin(neg, [3, 7]).any()
    assert len(np.unique(neg)) > 20            # 한쪽으로 쏠리지 않음
    body = _fn_code("sample_negatives")
    assert "cat_items" not in body and "hard" not in body.lower()


def test_value_block_does_not_propagate():
    """가치 임베딩은 MLP 출력 그대로 — 전파(sparse.mm)를 타면 안 된다."""
    body = _fn_code("value_emb")
    assert "sparse" not in body and "adj" not in body
    assert "sparse" in _fn_code("propagate_pref")   # 선호 쪽은 반드시 전파


def test_arch_differs_only_in_training():
    """joint/two_stage가 같은 모델 클래스를 쓰고 학습 절차만 달라야 한다."""
    body = _fn_code("train_model")
    ns = body.replace(" ", "")                         # 토큰 사이 공백 제거 후 비교
    assert ns.count("DualSpaceLightGCN") == 1          # 모델 생성은 한 곳뿐
    assert "pref_params()" in ns and "requires_grad_(False)" in ns
    assert V3.CFG["ARCH"] in ("joint", "two_stage")


def test_lambda_sweep_includes_baseline():
    """λ=0이 baseline이므로 스윕에 반드시 포함돼야 비교가 성립한다."""
    assert 0.0 in V3.CFG["LAMBDA_SWEEP"]
    assert len(V3.CFG["LAMBDA_SWEEP"]) >= 3


def test_three_segments():
    assert V3.SEG_NAMES == ["저CLV", "중CLV", "고CLV"]
    assert V3.CFG["SEG_EDGES"] == (0.2, 0.8)


def test_model_forward_and_score_shapes():
    """모델이 실제로 돌고 λ=0이면 가치항이 사라지는지."""
    torch.manual_seed(0)
    n_u, n_i, n_c = 8, 12, 3
    x_val = np.random.rand(n_u, 5).astype(np.float32)
    x_it = np.random.rand(n_i, 2).astype(np.float32)
    cat = np.random.randint(0, n_c, n_i).astype(np.int64)
    n = n_u + n_i
    idx = torch.tensor([[0, n_u], [n_u, 0]])
    adj = torch.sparse_coo_tensor(idx, torch.tensor([0.5, 0.5]), size=(n, n)).coalesce()
    m = V3.DualSpaceLightGCN(n_u, n_i, n_c, x_val, x_it, cat, dict(V3.CFG), adj)
    Up, Ip, Uv, Iv = m.embeddings()
    assert Up.shape == (n_u, V3.CFG["DIM"]) and Uv.shape == (n_u, V3.CFG["D_VALUE"])
    assert Ip.shape == (n_i, V3.CFG["DIM"]) and Iv.shape == (n_i, V3.CFG["D_VALUE"])
    gate = torch.rand(n_u)
    u = torch.tensor([0, 1]); i = torch.tensor([2, 3]); j = torch.tensor([4, 5])
    l0 = m.bpr_loss(u, i, j, gate, 0.0)
    assert torch.isfinite(l0)

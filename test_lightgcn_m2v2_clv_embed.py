from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "lightgcn_m2v2_clv_embed.py"


def _load_module_upto_cfg():
    """CFG/함수 정의까지만 실행하고 멈춘다. 파일 끝의 'M1 체크포인트 없으면 main()으로
    학습' 가드보다 앞에서 잘라야 한다 — 안 그러면 로컬에 체크포인트가 없는 조합(예:
    dunnhumby)에서 이 테스트가 실제로 전체 학습을 로컬 CPU로 돌려버린다(2026-08-01 발견,
    WINDOW_DAYS=60일 땐 우연히 37초라 안 드러났다가 None으로 바꾸며 36분+ 걸려 발각됨)."""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    src = src.split("# M1 체크포인트(RUN_TAG 지문 기준)가 없으면")[0]
    ns = {"__name__": "test_module"}
    exec(compile(src, str(SCRIPT_PATH), "exec"), ns)
    return ns


def test_cfg_has_phase_keys_and_no_dead_dampen_keys():
    ns = _load_module_upto_cfg()
    cfg = ns["CFG"]
    assert cfg["MODEL_LABEL"] == "M2v2"
    assert cfg["PHASE"] in (1, 2)
    assert "PHASE1_LAMBDA" in cfg
    assert cfg["CLV_GATE_POWER"] >= 1.0
    for dead_key in ["F_BUCKET_EDGES", "F_BUCKET_LABELS", "GATE_N_NEG",
                      "CLV_DAMPEN_GRID", "HIGH_CLV_DAMPEN_GRID", "HIGH_CLV_EPSILON_GRID"]:
        assert dead_key not in cfg, f"CFG에서 {dead_key}는 제거되어야 함"


def test_compute_clv_gate_is_percentile_rank_and_nan_safe():
    ns = _load_module_upto_cfg()
    np = ns["np"]
    compute_clv_gate = ns["compute_clv_gate"]
    clv = np.array([10.0, 20.0, 30.0, np.nan, 40.0], dtype=np.float32)
    gate = compute_clv_gate(clv)
    assert gate[3] == 0.0  # NaN CLV → gate 0
    assert gate[0] < gate[1] < gate[2] < gate[4]  # 순위 보존
    assert gate.min() >= 0.0 and gate.max() <= 1.0


def test_compute_clv_gate_power_suppresses_low_percentile_more_than_high():
    ns = _load_module_upto_cfg()
    np = ns["np"]
    compute_clv_gate = ns["compute_clv_gate"]
    clv = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)  # percentile 0.2/0.4/0.6/0.8/1.0
    g1 = compute_clv_gate(clv, power=1.0)
    g2 = compute_clv_gate(clv, power=2.0)
    np.testing.assert_allclose(g2, g1 ** 2, atol=1e-6)
    lowest_shrink = g1[0] - g2[0]
    highest_shrink = g1[-1] - g2[-1]
    assert lowest_shrink > highest_shrink  # 저CLV가 고CLV보다 더 많이 눌려야 함
    assert g2[0] < g1[0]  # power>1이면 저CLV gate는 줄어듦


def _toy_train(pd, np, seed=0, n=30, with_basket=False):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "u_idx": rng.integers(0, 6, n),
        "i_idx": rng.integers(0, 5, n),
        "t": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
        "v": rng.uniform(10, 100, n).round(2),
        "cat_idx": rng.integers(0, 3, n),
    })
    df["up"] = df["v"]
    if with_basket:
        df["b_raw"] = rng.integers(0, 8, n)
    return df


def test_build_user_features_is_pure_clv_frt_aov_prem_clv():
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    train = _toy_train(pd, np)
    x_val = ns["build_user_features"](train, n_users=6, cfg=dict(ns["CFG"]), is_date=True)
    assert x_val.shape == (6, 5)  # F/T/R + AOV/Prem — CatShare 없음, CLV_p도 없음(v3 폐기)
    assert ((x_val >= 0) & (x_val <= 1)).all()  # 전부 백분위 스케일


def test_F_counts_baskets_not_line_items():
    """F는 "구매 횟수"여야 한다 — 한 장바구니에서 상품 3개를 사면 F=1이지 3이 아니다.
    Dunnhumby는 한 행이 BASKET_ID×PRODUCT_ID 라인이라 행 수를 세면 F가 9배 넘게 부풀고,
    AOV도 "1회 구매당 금액"이 아니라 "상품 라인당 금액"이 된다(2026-08-03 수정)."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    # 유저0: 장바구니 1개에 상품 3라인(총 60) / 유저1: 장바구니 3개에 각 1라인(각 20)
    train = pd.DataFrame({
        "u_idx":  [0, 0, 0, 1, 1, 1],
        "i_idx":  [0, 1, 2, 0, 1, 2],
        "b_raw":  [7, 7, 7, 1, 2, 3],
        "t":      [0, 0, 0, 0, 5, 9],
        "v":      [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        "up":     [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
    })
    g = ns["_user_pct_stats"](train, dict(ns["CFG"]), is_date=False)
    assert g.loc[0, "F"] == 1 and g.loc[1, "F"] == 3     # 라인 수(3,3)가 아니어야 함
    assert g.loc[0, "AOV_raw"] == 60.0                   # 장바구니 합계 (라인 평균 20이 아님)
    assert g.loc[1, "AOV_raw"] == 20.0
    assert g.loc[0, "n_line"] == 3 and g.loc[1, "n_line"] == 3  # Prem 축소추정 분모는 라인 수


def test_no_basket_col_falls_back_to_user_day():
    """H&M은 주문 ID가 없으므로 (고객, 날짜)를 구매 1건으로 본다."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    train = pd.DataFrame({           # 유저0: 같은 날 2라인 + 다른 날 1라인 → F=2
        "u_idx": [0, 0, 0],
        "i_idx": [0, 1, 2],
        "t": [3, 3, 8],
        "v": [10.0, 30.0, 50.0],
        "up": [10.0, 30.0, 50.0],
    })
    g = ns["_user_pct_stats"](train, dict(ns["CFG"]), is_date=False)
    assert g.loc[0, "F"] == 2
    assert g.loc[0, "AOV_raw"] == 45.0   # (40 + 50) / 2


def test_unit_price_divides_by_quantity():
    """단가 up = 라인금액 / 수량. 수량 0 이하(쿠폰 정산행 등)는 1로 간주."""
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    dcfg = dict(ns["SCHEMA"]["dunnhumby"])
    q = dcfg["qty_col"]
    tx = pd.DataFrame({"v": [10.0, 10.0, 10.0], q: [1, 5, 0]})
    up = (tx["v"] / tx[q].clip(lower=1)).astype(np.float32)
    np.testing.assert_allclose(up.values, [10.0, 2.0, 10.0])
    assert ns["SCHEMA"]["hm"]["qty_col"] is None      # H&M price는 이미 단가
    assert ns["SCHEMA"]["hm"]["basket_col"] is None
    assert dcfg["basket_col"] == "BASKET_ID"


def test_item_features_are_price_plus_category_onehot():
    """아이템 특징 = [가격백분위, 카테고리내 가격순위] + CategoryID one-hot(n_cat).
    구매자 CLV 프로파일(5차원)로 교체하는 안은 2026-08-03 실측 악화로 되돌렸다 —
    원핫이 CLV와 무관하더라도 아이템 간 구별 정보를 담고 있어 제거 시 표현력이 무너진다."""
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    n_items, n_cat = 10, 4
    train = _toy_train(pd, np, seed=2, n=200)
    x_item = ns["build_item_features"](train, n_items, n_cat)
    assert x_item.shape == (n_items, 2 + n_cat)
    assert ((x_item >= 0) & (x_item <= 1)).all()
    assert np.isfinite(x_item).all()


def test_value_feature_version_matches_current_feature_defs():
    """2026-08-03에 F/AOV를 구매 건 단위로, Prem·아이템가격을 단가 기준으로 재정의했다.
    차원이 v2와 같아 크래시로 드러나지 않으므로(=옛 체크포인트를 조용히 재사용할 위험)
    반드시 새 번호(5)여야 한다. 특징 정의를 바꾸면 매번 이 값을 올릴 것."""
    ns = _load_module_upto_cfg()
    assert ns["CFG"]["VALUE_FEATURE_VERSION"] == 5
    assert "ITEM_SHRINKAGE_K" not in ns["CFG"]  # v4에서만 쓰던 키, 되돌리며 제거됨


def test_no_gate_f_or_dampen_functions_remain():
    ns = _load_module_upto_cfg()
    for dead_fn in ["compute_fbucket_gate", "compute_gate", "run_dualspace_one_seed"]:
        assert dead_fn not in ns, f"{dead_fn}는 v2에서 제거되어야 함"
    for fn in ["run_dualspace_one_seed_phase1", "run_dualspace_one_seed_phase2", "run_stage_b_grid"]:
        assert fn in ns, f"{fn}는 v2에 있어야 함"


def test_baseline_only_and_force_retrain_flags_exist():
    """baseline 단독 평가 경로와 M1 강제 재학습 플래그. 둘 다 CFG에 있어야 하고,
    run_baseline_only()가 정의돼 있어야 한다(2026-08-04 추가)."""
    ns = _load_module_upto_cfg()
    assert "BASELINE_ONLY" in ns["CFG"]
    assert "FORCE_M1_RETRAIN" in ns["CFG"]
    assert "run_baseline_only" in ns
    assert "_load_m1_pref" in ns          # run_dualspace와 공용으로 뽑아낸 헬퍼


def test_combined_scores_is_baseline_when_lambda_zero():
    """λ=0이면 가치신호가 전혀 안 섞여야 하고, Uv/Iv가 전부 0이어도 NaN이 나오면 안 된다
    (s_val의 표준편차가 0이라 분모 1e-8이 없으면 0/0이 된다)."""
    ns = _load_module_upto_cfg()
    np, torch = ns["np"], ns["torch"]
    n_u, n_i, d = 5, 7, 4
    g = torch.Generator().manual_seed(0)
    U = torch.randn(n_u, d, generator=g); I = torch.randn(n_i, d, generator=g)
    Uv = torch.zeros(n_u, 3); Iv = torch.zeros(n_i, 3)
    gate = np.ones(n_u, dtype=np.float32)
    bu = np.arange(n_u)
    s0 = ns["_combined_scores"](U, I, Uv, Iv, gate, 0.0, bu)
    assert torch.isfinite(s0).all()
    # λ=0 결과 = z^pref 점수만 z-score 정규화한 것과 동일해야 함
    s_rel = U @ I.T
    expect = (s_rel - s_rel.mean(dim=1, keepdim=True)) / (s_rel.std(dim=1, keepdim=True) + 1e-8)
    torch.testing.assert_close(s0, expect)


def test_grid_summary_keeps_every_metric():
    """그리드 요약이 evaluate_combined 결과를 통째로 보존해야 한다. 예전엔 9개 필드만
    남겨서 λ가 NDCG/MAP/Precision/V-NDCG/Novelty/Coverage/Gini/@20에 뭘 하는지
    볼 수 없었다(2026-08-04 수정)."""
    import json, tempfile
    from pathlib import Path as P
    ns = _load_module_upto_cfg()
    mets = ns["_METS"]
    fake = {k: {m: 0.5 for m in mets} for k in (10, 20, 50)}
    res = {"overall": fake, "seg": {k: {"저CLV": {m: 0.1 for m in mets},
                                        "고CLV": {m: 0.2 for m in mets}} for k in (10, 20, 50)},
           "seg_cnt": {"저CLV": 3, "고CLV": 4}, "coverage": {10: 0.3, 20: 0.4, 50: 0.5},
           "gini": {10: 0.6, 20: 0.7, 50: 0.8}, "value_alignment_spearman": 0.42, "n_eval": 7}
    with tempfile.TemporaryDirectory() as td:
        gp = P(td) / "grid_partial_x.pt"
        rows = ns["write_grid_summary"]({(4, 0.0): res, (4, 1.5): res}, gp)
        assert len(rows) == 2
        saved = json.loads((P(td) / "grid_partial_x_summary.json").read_text())
        csv_txt = (P(td) / "grid_partial_x_summary.csv").read_text()
    hdr = csv_txt.splitlines()[0].split(",")
    for col in ["epoch", "lambda", "recall@10", "ndcg@50", "map@20", "coverage@10",
                "gini@50", "value_alignment", "고CLV_revenue@10", "저CLV_precision@20"]:
        assert col in hdr, f"csv에 {col} 열 없음"
    assert len(csv_txt.splitlines()) == 3   # 헤더 + 조합 2개
    r = saved[0]
    assert r["epoch"] == 4 and "lambda" in r
    for m in mets:                                  # 10개 지표 × K 3개가 전부 있어야 함
        for k in ("10", "20", "50"):
            assert m in r["overall"][k], f"overall@{k}에 {m} 없음"
            assert m in r["seg"][k]["고CLV"], f"고CLV@{k}에 {m} 없음"
    for extra in ("coverage", "gini", "value_alignment_spearman", "seg_cnt", "n_eval"):
        assert extra in r, f"{extra} 누락"


def test_regen_grid_summary_flag_exists():
    ns = _load_module_upto_cfg()
    assert "REGEN_GRID_SUMMARY" in ns["CFG"]
    assert "regen_grid_summaries" in ns


def test_run_flags_set_for_m2_rerun():
    """이번 run 설정: M2 전체 파이프라인 + 시드 캐시 무시(그래야 Stage B가 다시 돌아
    전체 지표 요약이 새로 쓰인다). M1은 재학습 불필요(2026-08-04 재현성 확인 완료)."""
    ns = _load_module_upto_cfg()
    cfg = ns["CFG"]
    assert cfg["DATASET"] == "dunnhumby" and cfg["WINDOW_DAYS"] is None
    assert cfg["PHASE"] == 2
    assert cfg["BASELINE_ONLY"] is False
    assert cfg["REGEN_GRID_SUMMARY"] is False
    assert cfg["FORCE_M1_RETRAIN"] is False   # 켜두면 매번 14분씩 헛돎
    assert cfg["FORCE_SEED_RECOMPUTE"] is True


def test_seed_cache_is_bypassed_when_forced():
    """load_or_run_seed가 FORCE_SEED_RECOMPUTE를 실제로 본다(문자열 검사로 충분 —
    실행하려면 모델/데이터가 필요)."""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    body = src.split("def load_or_run_seed")[1].split("\ndef ")[0]
    assert 'not cfg.get("FORCE_SEED_RECOMPUTE")' in body

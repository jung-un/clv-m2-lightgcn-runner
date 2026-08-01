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


def test_build_user_features_is_pure_clv_frt_aov_prem_clv():
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    build_user_features = ns["build_user_features"]
    rng = np.random.default_rng(0)
    n = 30
    train = pd.DataFrame({
        "u_idx": rng.integers(0, 6, n),
        "i_idx": rng.integers(0, 5, n),
        "t": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
        "v": rng.uniform(10, 100, n).round(2),
        "cat_idx": rng.integers(0, 3, n),
    })
    cfg = dict(ns["CFG"])
    x_val, F_u_full = build_user_features(train, n_users=6, cfg=cfg, is_date=True)
    assert x_val.shape == (6, 6)  # F/T/R + AOV/Prem + CLV — CatShare 없음(CLV와 무관해 제거)
    assert F_u_full.shape == (6,)
    assert ((x_val >= 0) & (x_val <= 1)).all()  # 전부 백분위 스케일이어야 함(CLV_p 포함)


def test_clv_column_is_product_of_nhat_vhat():
    """CLV_p가 실제로 N̂×V̂의 백분위인지 — 곱을 명시적으로 주는 게 이 변수의 존재 이유."""
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    rng = np.random.default_rng(1)
    n = 60
    train = pd.DataFrame({
        "u_idx": rng.integers(0, 12, n),
        "i_idx": rng.integers(0, 8, n),
        "t": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"),
        "v": rng.uniform(5, 200, n).round(2),
        "cat_idx": rng.integers(0, 3, n),
    })
    g = ns["_user_pct_stats"](train, dict(ns["CFG"]), True)
    np.testing.assert_allclose(g["CLV_raw"], g["N_hat"] * g["V_hat"], rtol=1e-9)
    np.testing.assert_allclose(g["CLV_p"], g["CLV_raw"].rank(pct=True), rtol=1e-9)


def test_no_gate_f_or_dampen_functions_remain():
    ns = _load_module_upto_cfg()
    for dead_fn in ["compute_fbucket_gate", "compute_gate", "run_dualspace_one_seed"]:
        assert dead_fn not in ns, f"{dead_fn}는 v2에서 제거되어야 함"
    for fn in ["run_dualspace_one_seed_phase1", "run_dualspace_one_seed_phase2", "run_stage_b_grid"]:
        assert fn in ns, f"{fn}는 v2에 있어야 함"

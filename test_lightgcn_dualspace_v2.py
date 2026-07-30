from pathlib import Path

SCRIPT_PATH = Path(__file__).parent / "lightgcn_dualspace_v2.py"


def _load_module_upto_cfg():
    """CFG/DCFG가 정의된 시점까지만 모듈을 로드(맨 아래 results = run_dualspace() 실행은 피함).
    모듈 최상위에 있는 run_dualspace() 즉시실행 라인 때문에, 소스를 읽어서 그 라인만 잘라내고 exec한다."""
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    src = src.split("results = run_dualspace()")[0]
    ns = {"__name__": "test_module"}
    exec(compile(src, str(SCRIPT_PATH), "exec"), ns)
    return ns


def test_cfg_has_consolidated_keys():
    ns = _load_module_upto_cfg()
    cfg = ns["CFG"]
    for key in ["DATASET", "MODEL_LABEL", "SEED", "SEED_LIST",
                "F_BUCKET_EDGES", "F_BUCKET_LABELS", "WINDOW_DAYS",
                "VAL_DAYS", "TEST_DAYS"]:
        assert key in cfg, f"CFG에 {key}가 있어야 함"
    for dead_key in ["MODEL", "GAMMA_INIT", "REG_TARGET", "ITER_FILTER"]:
        assert dead_key not in cfg, f"CFG에서 {dead_key}는 제거되어야 함"


def test_dcfg_no_longer_has_window_or_split_days():
    ns = _load_module_upto_cfg()
    dcfg = ns["DCFG"]
    for removed_key in ["window_months", "val_days", "test_days"]:
        assert removed_key not in dcfg, f"DCFG에서 {removed_key}는 CFG로 이동해야 함"


def test_hm_local_paths_point_to_existing_files():
    ns = _load_module_upto_cfg()
    if ns["IN_COLAB"]:
        return  # 로컬 전용 경로 검증
    dcfg = ns["SCHEMA"]["hm"]
    assert Path(dcfg["tx_path"]).exists(), f"존재하지 않는 경로: {dcfg['tx_path']}"
    assert Path(dcfg["item_meta_path"]).exists(), f"존재하지 않는 경로: {dcfg['item_meta_path']}"


def test_compute_boundaries_reads_days_from_cfg_not_dcfg_is_date_true():
    """dcfg에 val_days/test_days가 아예 없어도(=이제 DCFG의 실제 모습) cfg만으로 동작해야 함."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    compute_boundaries = ns["compute_boundaries"]
    tx = pd.DataFrame({"t": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-31"])})
    cfg = {"VAL_DAYS": 5, "TEST_DAYS": 3}
    dcfg = {"is_date": True}  # val_days/test_days 키 없음 — 있으면 옛 코드로 되돌아간 것
    val_start, test_start = compute_boundaries(tx, cfg, dcfg)
    t_max = tx["t"].max()
    assert test_start == t_max - pd.Timedelta(days=3)
    assert val_start == test_start - pd.Timedelta(days=5)


def test_window_filter_reads_window_days_from_cfg_is_date_true():
    """dcfg에 window_months가 없어도(=이제 DCFG의 실제 모습) cfg["WINDOW_DAYS"] 기준으로 필터링해야 함."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    window_filter = ns["window_filter"]
    tx = pd.DataFrame({"t": pd.to_datetime(["2024-01-01", "2024-01-20", "2024-01-31"])})
    cfg = {"WINDOW_DAYS": 10}
    dcfg = {"is_date": True}  # window_months 키 없음 — 있으면 옛 코드로 되돌아간 것
    out = window_filter(tx, cfg, dcfg)
    assert list(out["t"]) == [pd.Timestamp("2024-01-31")]  # t_max(01-31) - 10일 = 01-21 이후만 생존


def test_window_filter_reads_window_days_from_cfg_is_date_false():
    """dunnhumby처럼 t가 정수 day-index인 경우(is_date=False)도 동일하게 cfg 기준으로 필터링해야 함."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    window_filter = ns["window_filter"]
    tx = pd.DataFrame({"t": [1, 50, 100]})
    cfg = {"WINDOW_DAYS": 10}
    dcfg = {"is_date": False}
    out = window_filter(tx, cfg, dcfg)
    assert list(out["t"]) == [100]  # t_max(100) - 10 = 90 이후만 생존


def test_window_filter_none_keeps_all_rows():
    """WINDOW_DAYS=None이면 전체기간 사용(필터링 없음)."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    window_filter = ns["window_filter"]
    tx = pd.DataFrame({"t": [1, 50, 100]})
    cfg = {"WINDOW_DAYS": None}
    dcfg = {"is_date": False}
    out = window_filter(tx, cfg, dcfg)
    assert list(out["t"]) == [1, 50, 100]


def test_compute_boundaries_reads_days_from_cfg_not_dcfg_is_date_false():
    """dunnhumby처럼 t가 정수 day인 경우(is_date=False)도 동일하게 cfg 기준으로 동작해야 함."""
    ns = _load_module_upto_cfg()
    pd = ns["pd"]
    compute_boundaries = ns["compute_boundaries"]
    tx = pd.DataFrame({"t": [1, 50, 100]})
    cfg = {"VAL_DAYS": 5, "TEST_DAYS": 3}
    dcfg = {"is_date": False}
    val_start, test_start = compute_boundaries(tx, cfg, dcfg)
    t_max = tx["t"].max()
    assert test_start == t_max - 3
    assert val_start == test_start - 5


def _synthetic_tx(pd, np):
    """유저 6명, 아이템 5개, 20건 거래. val/test 분리가 제대로 되는지 확인 가능한 최소 규모."""
    rng = np.random.default_rng(0)
    n = 40
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    return pd.DataFrame({
        "u_raw": rng.integers(0, 6, n),
        "i_raw": rng.integers(0, 5, n).astype(str),
        "t": rng.choice(dates, n),
        "v": rng.uniform(10, 100, n).round(2),
        "cat_raw": rng.integers(0, 3, n),
    })


def test_prepare_data_returns_consistent_shapes(tmp_path):
    ns = _load_module_upto_cfg()
    pd, np = ns["pd"], ns["np"]
    tx = _synthetic_tx(pd, np)
    tx_path = tmp_path / "tx.csv"
    # cat_raw는 merge_category()가 item_meta 쪽에서 만드는 이름과 충돌하므로(스킴상
    # tx 원본에는 카테고리 컬럼이 없어야 정상) CSV에는 쓰지 않는다.
    tx.drop(columns="cat_raw").rename(columns={"u_raw": "customer_id", "i_raw": "article_id",
                                                "t": "t_dat", "v": "price"}).to_csv(tx_path, index=False)
    meta_path = tmp_path / "articles.csv"
    pd.DataFrame({"article_id": tx["i_raw"].unique(),
                  "product_group_name": ["catA"] * tx["i_raw"].nunique()}).to_csv(meta_path, index=False)

    cfg = dict(ns["CFG"]); cfg["WINDOW_DAYS"] = None; cfg["VAL_DAYS"] = 3; cfg["TEST_DAYS"] = 3
    dcfg = dict(ns["DCFG"]); dcfg["tx_path"] = str(tx_path); dcfg["item_meta_path"] = str(meta_path)

    data = ns["prepare_data"](cfg, dcfg)
    for key in ["train", "val_gt", "val_rev", "test_gt", "test_rev", "adj", "pos_key",
                "tr_u", "tr_i", "csr_ptr", "csr_items", "user_pos", "item_cat_arr",
                "cat_items", "n_users", "n_items", "n_cat"]:
        assert key in data, f"prepare_data 반환값에 {key}가 있어야 함"
    assert data["n_users"] > 0 and data["n_items"] > 0
    assert data["csr_ptr"].shape[0] == data["n_users"] + 1
    assert len(data["train"]) > 0

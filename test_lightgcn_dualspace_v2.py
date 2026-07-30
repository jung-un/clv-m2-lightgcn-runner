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


def _reference_score_topk_loop(topk, bu, ks, gt, rev, price_pct, item_nov, cat):
    """기존 evaluate_combined()의 순수 python for문 로직을 그대로 옮긴 참조 구현.
    score_topk()가 이거랑 정확히 같은 숫자를 내야 벡터화가 맞다는 뜻."""
    out = {k: {m: [] for m in ["recall", "precision", "hr", "ndcg", "map", "revenue",
                                 "arp", "novelty", "diversity"]} for k in ks}
    for bi, u in enumerate(bu):
        pos = set(gt[u].tolist()); ur = dict(zip(gt[u].tolist(), rev[u].tolist()))
        pred = topk[bi]
        for k in ks:
            pk = pred[:k]
            hits = [1 if x in pos else 0 for x in pk]
            nh = sum(hits); P = len(pos)
            import math as _m
            dcg = sum(h / _m.log2(r + 2) for r, h in enumerate(hits))
            idcg = sum(1.0 / _m.log2(r + 2) for r in range(min(P, k)))
            ch = s_ap = 0
            for r in range(k):
                if hits[r]:
                    ch += 1; s_ap += ch / (r + 1)
            out[k]["recall"].append(nh / P)
            out[k]["precision"].append(nh / k)
            out[k]["hr"].append(1.0 if nh > 0 else 0.0)
            out[k]["ndcg"].append(dcg / idcg if idcg > 0 else 0.0)
            out[k]["map"].append(s_ap / min(P, k))
            out[k]["revenue"].append(sum(ur.get(it, 0.0) for it in pk if it in pos))
            out[k]["arp"].append(float(price_pct[pk].mean()))
            out[k]["novelty"].append(float(item_nov[pk].mean()))
            valid = [c for c in cat[pk] if c >= 0]
            out[k]["diversity"].append(len(set(valid)) / k if valid else 0.0)
    return out


def test_score_topk_matches_reference_loop():
    ns = _load_module_upto_cfg()
    np = ns["np"]
    rng = np.random.default_rng(1)
    n_items, n_users_eval, max_k = 30, 8, 10
    ks = [5, 10]
    bu = np.arange(n_users_eval)
    topk = np.stack([rng.permutation(n_items)[:max_k] for _ in range(n_users_eval)])
    gt = {u: rng.choice(n_items, size=rng.integers(1, 4), replace=False).astype(np.int32) for u in bu}
    rev = {u: rng.uniform(10, 50, size=len(gt[u])).astype(np.float32) for u in bu}
    price_pct = rng.uniform(0, 1, n_items).astype(np.float32)
    item_nov = rng.uniform(0, 5, n_items).astype(np.float32)
    cat = rng.integers(0, 4, n_items).astype(np.int64)

    ref = _reference_score_topk_loop(topk, bu, ks, gt, rev, price_pct, item_nov, cat)

    pos_key_list, pos_rev_list = [], []
    for u in bu:
        for i, r in zip(gt[u], rev[u]):
            pos_key_list.append(int(u) * n_items + int(i)); pos_rev_list.append(float(r))
    order = np.argsort(pos_key_list)
    pos_key_sorted = np.array(pos_key_list)[order]
    pos_rev_sorted = np.array(pos_rev_list)[order]
    P_arr = np.array([len(gt[u]) for u in bu])

    ideal_rev_cumsum = {}
    for u in bu:
        sorted_rev = np.sort(rev[u])[::-1]
        disc = 1.0 / np.log2(np.arange(2, len(sorted_rev) + 2))
        ideal_rev_cumsum[u] = np.cumsum(sorted_rev * disc)

    out = ns["score_topk"](topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
                            P_arr, price_pct, item_nov, cat, ideal_rev_cumsum)

    for k in ks:
        for m in ["recall", "precision", "hr", "ndcg", "map", "revenue", "arp", "novelty", "diversity"]:
            np.testing.assert_allclose(out[k][m], ref[k][m], rtol=1e-6, atol=1e-8,
                                        err_msg=f"k={k} metric={m} 불일치")


def test_score_topk_zero_ground_truth_user_no_divzero():
    """정답(gt)이 0개인 유저(P=0)가 섞여 있을 때 recall/precision/hr/map/ndcg가
    NaN/inf 없이 정확히 0.0이어야 한다 (분모가 P인 recall/map, k인 precision 등에서
    0-division이 조용히 NaN을 만들어내지 않는지 확인)."""
    ns = _load_module_upto_cfg()
    np = ns["np"]
    rng = np.random.default_rng(2)
    n_items, n_users_eval, max_k = 30, 4, 10
    ks = [5, 10]
    bu = np.arange(n_users_eval)
    topk = np.stack([rng.permutation(n_items)[:max_k] for _ in range(n_users_eval)])
    # user 0: P=0 (빈 gt). 나머지는 평소처럼 1~3개.
    gt = {0: np.array([], dtype=np.int32)}
    gt.update({u: rng.choice(n_items, size=rng.integers(1, 4), replace=False).astype(np.int32)
               for u in bu[1:]})
    rev = {u: rng.uniform(10, 50, size=len(gt[u])).astype(np.float32) for u in bu}
    price_pct = rng.uniform(0, 1, n_items).astype(np.float32)
    item_nov = rng.uniform(0, 5, n_items).astype(np.float32)
    cat = rng.integers(0, 4, n_items).astype(np.int64)

    pos_key_list, pos_rev_list = [], []
    for u in bu:
        for i, r in zip(gt[u], rev[u]):
            pos_key_list.append(int(u) * n_items + int(i)); pos_rev_list.append(float(r))
    order = np.argsort(pos_key_list)
    pos_key_sorted = np.array(pos_key_list)[order]
    pos_rev_sorted = np.array(pos_rev_list)[order]
    P_arr = np.array([len(gt[u]) for u in bu])

    ideal_rev_cumsum = {}
    for u in bu:
        sorted_rev = np.sort(rev[u])[::-1]
        disc = 1.0 / np.log2(np.arange(2, len(sorted_rev) + 2))
        ideal_rev_cumsum[u] = np.cumsum(sorted_rev * disc)

    out = ns["score_topk"](topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
                            P_arr, price_pct, item_nov, cat, ideal_rev_cumsum)

    for k in ks:
        for m in ["recall", "precision", "hr", "ndcg", "map", "revenue", "vndcg",
                   "arp", "novelty", "diversity"]:
            arr = out[k][m]
            assert np.isfinite(arr).all(), f"k={k} metric={m}에 NaN/inf 있음: {arr}"
        # P=0인 유저 0은 맞출 정답 자체가 없으므로 recall/precision/hr/map/ndcg가 0.0이어야 함
        for m in ["recall", "precision", "hr", "map", "ndcg"]:
            assert out[k][m][0] == 0.0, f"k={k} metric={m}: P=0 유저는 0.0이어야 하는데 {out[k][m][0]}"

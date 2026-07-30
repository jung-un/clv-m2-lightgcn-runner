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


def test_build_pos_lookup_and_ideal_rev_cumsum():
    ns = _load_module_upto_cfg()
    np = ns["np"]
    gt = {0: np.array([1, 2], dtype=np.int32), 1: np.array([3], dtype=np.int32)}
    rev = {0: np.array([10.0, 30.0], dtype=np.float32), 1: np.array([5.0], dtype=np.float32)}
    n_items = 10

    pos_key_sorted, pos_rev_sorted = ns["build_pos_lookup"](gt, rev, n_items)
    assert np.all(np.diff(pos_key_sorted) >= 0), "정렬되어 있어야 searchsorted가 맞음"
    # 유저0-아이템2(키=0*10+2=2)의 revenue가 30.0으로 조회되는지
    idx = np.searchsorted(pos_key_sorted, 2)
    assert pos_key_sorted[idx] == 2 and pos_rev_sorted[idx] == 30.0

    cumsum = ns["build_ideal_rev_cumsum"](gt, rev)
    # 유저0: revenue [10,30] 내림차순 정렬 -> [30,10], discount[log2(2),log2(3)]
    expected0 = np.array([30.0 / np.log2(2), 30.0 / np.log2(2) + 10.0 / np.log2(3)])
    np.testing.assert_allclose(cumsum[0], expected0, rtol=1e-6)


def test_evaluate_combined_cache_matches_internal_computation():
    """evaluate_combined()에 pos_lookup/ideal_rev_cumsum을 미리 만들어 넘긴 결과가,
    안 넘겨서(None) 내부에서 build_pos_lookup()/build_ideal_rev_cumsum()을 다시 계산하는
    기존 경로와 정확히 같은 숫자를 내는지 확인 — 캐시는 순수 성능 최적화일 뿐 결과를
    바꾸면 안 된다(run_dualspace_one_seed()의 Stage A/B 그리드가 이 캐시에 의존함)."""
    ns = _load_module_upto_cfg()
    np = ns["np"]; torch = ns["torch"]
    torch.manual_seed(0)

    n_users, n_items, dim = 6, 12, 4
    U_pref = torch.randn(n_users, dim); I_pref = torch.randn(n_items, dim)
    Uv = torch.randn(n_users, dim); Iv = torch.randn(n_items, dim)
    gate_arr = np.ones(n_users)
    rng = np.random.default_rng(0)
    gt = {u: rng.choice(n_items, size=2, replace=False).astype(np.int32) for u in range(n_users)}
    rev = {u: rng.uniform(5, 50, size=2).astype(np.float32) for u in range(n_users)}
    item_meta = dict(price_pct=rng.uniform(0, 1, n_items).astype(np.float32),
                      pop_prob=rng.uniform(0.01, 1, n_items).astype(np.float32),
                      cat=rng.integers(0, 3, n_items))
    user_meta = dict(clv=rng.uniform(0, 100, n_users), vhat=rng.uniform(0, 1, n_users))
    csr_ptr = np.zeros(n_users + 1, dtype=np.int64)  # no purchased-item exclusions
    csr_items = np.array([], dtype=np.int64)
    ks = [5, 10]

    r_internal = ns["evaluate_combined"](U_pref, I_pref, Uv, Iv, gate_arr, 0.5, gt, rev,
                                          item_meta, user_meta, ks, csr_ptr, csr_items,
                                          pos_lookup=None, ideal_rev_cumsum=None)

    pos_lookup = ns["build_pos_lookup"](gt, rev, n_items)
    ideal_rev_cumsum = ns["build_ideal_rev_cumsum"](gt, rev)
    r_cached = ns["evaluate_combined"](U_pref, I_pref, Uv, Iv, gate_arr, 0.5, gt, rev,
                                        item_meta, user_meta, ks, csr_ptr, csr_items,
                                        pos_lookup=pos_lookup, ideal_rev_cumsum=ideal_rev_cumsum)

    for k in ks:
        for m in ns["_METS"]:
            np.testing.assert_allclose(r_internal["overall"][k][m], r_cached["overall"][k][m],
                                        rtol=1e-9, atol=1e-12,
                                        err_msg=f"k={k} metric={m}: cached path diverged from internal-compute path")
    assert r_internal["value_alignment_spearman"] == r_cached["value_alignment_spearman"]


def test_train_value_tower_resumes_from_checkpoint(tmp_path):
    ns = _load_module_upto_cfg()
    np, torch = ns["np"], ns["torch"]
    n_users, n_items = 20, 15
    x_val_u = np.random.default_rng(0).uniform(size=(n_users, 4)).astype(np.float32)
    x_val_i = np.random.default_rng(1).uniform(size=(n_items, 3)).astype(np.float32)
    tr_u = np.random.default_rng(2).integers(0, n_users, 200).astype(np.int64)
    tr_i = np.random.default_rng(3).integers(0, n_items, 200).astype(np.int64)
    pos_key = np.unique(tr_u.astype(np.int64) * n_items + tr_i)
    user_pos = {}
    item_cat_arr = np.random.default_rng(4).integers(0, 3, n_items).astype(np.int64)
    cat_items = {c: np.where(item_cat_arr == c)[0] for c in range(3)}
    val_gt = {u: np.array([int(u % n_items)], dtype=np.int32) for u in range(n_users)}
    csr_ptr = np.zeros(n_users + 1, dtype=np.int64)
    csr_items = np.array([], dtype=np.int32)

    cfg = dict(ns["CFG"]); cfg["VT_MAX_EPOCHS"] = 4; cfg["VT_PATIENCE"] = 100
    cfg["BATCH_SIZE"] = 64; cfg["HARD_NEG_RATIO"] = 0.0; cfg["MLP_HIDDEN"] = 8; cfg["D_VALUE"] = 4
    args = (x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos,
            item_cat_arr, cat_items, val_gt, csr_ptr, csr_items)
    ckpt = tmp_path / "vt_test.pt"

    # 1차: 2 epoch만
    cfg1 = dict(cfg); cfg1["VT_MAX_EPOCHS"] = 2
    ns["train_value_tower"](*args, cfg1, seed=0, ckpt_path=ckpt)
    saved = torch.load(ckpt, weights_only=False)
    assert saved["last_epoch"] == 2

    # 2차: 같은 ckpt_path로 이어서 최대 4 epoch까지
    ns["train_value_tower"](*args, cfg, seed=0, ckpt_path=ckpt)
    saved2 = torch.load(ckpt, weights_only=False)
    assert saved2["last_epoch"] == 4
    assert len(saved2["all_epochs"]) == 4  # 처음 2개 + 이어서 2개

    # 대조군: 중단 없이 한 번에 4 epoch을 도는 참조 실행 (동일 seed/data).
    # epoch 1~2는 중단 시점 이전이므로, 재개했든 안 했든 rng 소비 시퀀스가 완전히
    # 같아 bit-identical해야 한다 (epoch 3~4는 재개 시 rng가 새로 seed되어 이어서
    # 소비되지 않으므로 — train_loop()도 동일한 한계 — 여기서는 비교하지 않는다).
    ckpt_ref = tmp_path / "vt_ref.pt"
    ns["train_value_tower"](*args, cfg, seed=0, ckpt_path=ckpt_ref)
    ref = torch.load(ckpt_ref, weights_only=False)

    for i in (0, 1):
        assert ref["all_epochs"][i]["val_recall10"] == saved2["all_epochs"][i]["val_recall10"]
        ref_state, resumed_state = ref["all_epochs"][i]["state"], saved2["all_epochs"][i]["state"]
        assert set(ref_state.keys()) == set(resumed_state.keys())
        for k in ref_state:
            assert torch.equal(ref_state[k], resumed_state[k]), \
                f"epoch {i + 1} tensor '{k}' diverged between uninterrupted and resumed run"

    # x_val_u/x_val_i는 학습 중 절대 안 바뀌는 static 입력 버퍼 — all_epochs 스냅샷마다
    # 이 두 버퍼가 통째로 복제되면 체크포인트가 epoch 수에 비례해 계속 커진다(리뷰 지적).
    for entry in saved2["all_epochs"]:
        assert "x_val_u" not in entry["state"] and "x_val_i" not in entry["state"]

    # bad/best_epoch/best_val_score이 실제 기록된 val_recall10 히스토리와 정합적인지 —
    # 재개 시 bad가 0으로 리셋되거나 best_epoch/best_val_score가 유실되면 이 불변식이 깨진다.
    def _assert_bookkeeping_consistent(ckpt_dict):
        best_score, best_ep = -1.0, -1
        for e in ckpt_dict["all_epochs"]:
            if e["val_recall10"] > best_score:
                best_score, best_ep = e["val_recall10"], e["epoch"]
        assert ckpt_dict["best_val_score"] == best_score
        assert ckpt_dict["best_epoch"] == best_ep
        assert ckpt_dict["bad"] == ckpt_dict["last_epoch"] - best_ep

    _assert_bookkeeping_consistent(saved)
    _assert_bookkeeping_consistent(saved2)


def test_load_vt_state_tolerates_only_the_two_known_excluded_buffers():
    ns = _load_module_upto_cfg()
    np, torch = ns["np"], ns["torch"]
    model = ns["ValueTower"](
        np.random.default_rng(0).uniform(size=(5, 3)).astype(np.float32),
        np.random.default_rng(1).uniform(size=(4, 2)).astype(np.float32),
        hidden=4, d_value=2)
    full_state = model.state_dict()

    # 정상 경로: x_val_u/x_val_i만 빠진 스냅샷은 조용히 통과해야 한다
    trimmed = {k: v for k, v in full_state.items() if k not in ("x_val_u", "x_val_i")}
    ns["load_vt_state"](model, trimmed)  # raises nothing

    # 회귀 감지 경로: 학습 가능한 파라미터가 빠지면(버퍼가 아닌 진짜 버그) 조용히 넘어가지
    # 않고 바로 에러가 나야 한다 — 안 그러면 그 파라미터가 무작위 초기값에 그대로 남는다
    real_param_key = next(k for k in full_state if k not in ("x_val_u", "x_val_i"))
    broken = {k: v for k, v in full_state.items() if k != real_param_key}
    try:
        ns["load_vt_state"](model, broken)
        assert False, "load_vt_state should have raised on a missing non-buffer key"
    except AssertionError as e:
        assert real_param_key in str(e)

    # 회귀 감지 경로: 낯선 키가 섞여 있어도 에러가 나야 한다
    unexpected = dict(trimmed); unexpected["surprise_key"] = torch.zeros(1)
    try:
        ns["load_vt_state"](model, unexpected)
        assert False, "load_vt_state should have raised on an unexpected key"
    except AssertionError as e:
        assert "surprise_key" in str(e)


def test_stage_b_grid_resumes_partial_progress(tmp_path):
    ns = _load_module_upto_cfg()

    call_log = []

    def fake_eval(gate, lam, gt_, rev_, Uv_, Iv_):
        call_log.append((lam,))
        return {"overall": {10: {"recall": 0.1, "revenue": lam}}}

    grid_path = tmp_path / "grid_partial.pt"
    vt_topk = [{"epoch": 1, "state": {}}, {"epoch": 2, "state": {}}]
    cfg = dict(ns["CFG"])
    cfg["CLV_DAMPEN_GRID"] = [1.0]; cfg["HIGH_CLV_DAMPEN_GRID"] = [1.0]; cfg["LAMBDA_GRID"] = [0, 1]

    # 1차: epoch 1만 계산되도록 vt_topk를 1개짜리로 잘라서 실행.
    # 그리드는 dampen_low=1.0 x dampen_high=1.0 x lam in [0,1] 뿐이고, lam=0은 base_val_res를
    # 재사용하는 지름길이라 _eval을 안 부르므로, epoch당 정확히 1번만 fake_eval이 불려야 한다.
    grid1 = ns["run_stage_b_grid"](vt_topk[:1], cfg, grid_path,
                                    is_low_clv=ns["np"].array([False]), is_high_clv=ns["np"].array([False]),
                                    gate_f=ns["np"].array([1.0]), base_val_res={"overall": {10: {"recall": 0.1, "revenue": 0}}},
                                    _eval=fake_eval)
    n_calls_after_first = len(call_log)
    assert n_calls_after_first == 1

    # 2차: epoch 1+2 전체로 재실행 — epoch 1은 캐시에서 로드되어 fake_eval이 다시 안 불려야 하고,
    # 정확히 epoch2분(1번)만 새로 호출되어야 한다.
    grid2 = ns["run_stage_b_grid"](vt_topk, cfg, grid_path,
                                    is_low_clv=ns["np"].array([False]), is_high_clv=ns["np"].array([False]),
                                    gate_f=ns["np"].array([1.0]), base_val_res={"overall": {10: {"recall": 0.1, "revenue": 0}}},
                                    _eval=fake_eval)
    assert len(call_log) == n_calls_after_first + 1  # epoch1은 재계산 안 됨, epoch2분만 정확히 1번 추가
    assert (1, 0, 1.0, 1.0) in grid2 and (2, 0, 1.0, 1.0) in grid2


def test_stage_b_grid_crash_mid_epoch_does_not_mark_epoch_done(tmp_path):
    """epoch 하나의 내부 그리드(dampen_low×dampen_high×λ)를 다 끝내기 전에 죽으면,
    디스크에 저장된 done_epochs/grid_results에 그 epoch가 (부분적으로도) 남으면 안 된다 —
    저장은 epoch 블록 전체가 끝난 뒤에만 일어나야 재시작 시 안전하게 이어서 돌 수 있다."""
    ns = _load_module_upto_cfg()

    call_count = [0]

    def maybe_crashing_eval(gate, lam, gt_, rev_, Uv_, Iv_):
        call_count[0] += 1
        if call_count[0] == 3:  # epoch1의 2번 호출은 성공, epoch2의 첫 호출(3번째)에서 크래시
            raise RuntimeError("simulated crash mid-epoch")
        return {"overall": {10: {"recall": 0.1, "revenue": lam}}}

    grid_path = tmp_path / "grid_partial_crash.pt"
    vt_topk = [{"epoch": 1, "state": {}}, {"epoch": 2, "state": {}}]
    cfg = dict(ns["CFG"])
    cfg["CLV_DAMPEN_GRID"] = [1.0]; cfg["HIGH_CLV_DAMPEN_GRID"] = [1.0]; cfg["LAMBDA_GRID"] = [0, 1, 2]
    # 그리드는 dampen_low=1.0 x dampen_high=1.0 x lam in [0,1,2] — lam=0은 지름길(호출 없음),
    # lam=1/lam=2만 _eval을 부르므로 epoch당 정확히 2번 호출된다.

    try:
        ns["run_stage_b_grid"](vt_topk, cfg, grid_path,
                                is_low_clv=ns["np"].array([False]), is_high_clv=ns["np"].array([False]),
                                gate_f=ns["np"].array([1.0]), base_val_res={"overall": {10: {"recall": 0.1, "revenue": 0}}},
                                _eval=maybe_crashing_eval)
        assert False, "expected the simulated crash to propagate"
    except RuntimeError:
        pass

    saved = ns["torch"].load(grid_path, weights_only=False)
    assert saved["done_epochs"] == {1}, "epoch1은 완주했으니 저장돼야 하고, epoch2는 크래시로 미완주라 없어야 함"
    assert all(key[0] == 1 for key in saved["grid_results"]), \
        "크래시 시점에 epoch2용으로 이미 계산된 grid_results 항목이 하나도 디스크에 남으면 안 됨"


def test_grid_fingerprint_changes_when_any_of_the_three_grids_change():
    """grid_path의 캐시 파일명은 vt_fingerprint뿐 아니라 grid_fingerprint(LAMBDA_GRID/
    CLV_DAMPEN_GRID/HIGH_CLV_DAMPEN_GRID)에도 의존해야 한다 — 안 그러면 LAMBDA_GRID를
    넓혀 재탐색할 때(로드맵에 있는 다음 단계) 예전 grid_partial 파일의 done_epochs를
    그대로 재사용해버려 새로 추가된 λ/dampen 조합이 조용히 grid_results에서 빠진다."""
    ns = _load_module_upto_cfg()
    base = dict(ns["CFG"])
    base["LAMBDA_GRID"] = [0, 1, 2]; base["CLV_DAMPEN_GRID"] = [1.0, 0.9]; base["HIGH_CLV_DAMPEN_GRID"] = [1.0, 0.9]
    base_fp = ns["grid_fingerprint"](base)

    wider_lambda = dict(base); wider_lambda["LAMBDA_GRID"] = [0, 1, 2, 3]
    assert ns["grid_fingerprint"](wider_lambda) != base_fp

    wider_low = dict(base); wider_low["CLV_DAMPEN_GRID"] = [1.0, 0.9, 0.8]
    assert ns["grid_fingerprint"](wider_low) != base_fp

    wider_high = dict(base); wider_high["HIGH_CLV_DAMPEN_GRID"] = [1.0, 0.9, 0.8]
    assert ns["grid_fingerprint"](wider_high) != base_fp

    # 세 그리드가 동일하면(다른 무관한 CFG 키가 달라도) 같은 지문이어야 한다 —
    # 캐시가 딱 이 세 값의 함수여야지, 다른 키 변화로 불필요하게 무효화되면 안 된다
    unrelated_change = dict(base); unrelated_change["SEED"] = base["SEED"] + 999
    assert ns["grid_fingerprint"](unrelated_change) == base_fp


def test_run_dualspace_skips_completed_seeds(tmp_path):
    ns = _load_module_upto_cfg()
    json = ns["json"]

    calls = []
    def fake_run_one_seed(seed, *a, **kw):
        calls.append(seed)
        return [{"seed": seed, "high_clv_epsilon": 0.0, "best_ep": 1, "best_lam": 0.1,
                 "best_dampen_low": 1.0, "best_dampen_high": 1.0,
                 "intervention_policy": {}, "vt_best_epoch": 1, "vt_best_val_recall": 0.1,
                 "test_base": {"overall": {10: {"recall": 0.1, "revenue": 0.1}},
                               "seg": {10: {"저CLV": {"recall": 0.1, "revenue": 0.1, "arp": 0.1},
                                            "고CLV": {"recall": 0.1, "revenue": 0.1, "arp": 0.1}}},
                               "value_alignment_spearman": 0.1},
                 "test_best": {"overall": {10: {"recall": 0.2, "revenue": 0.2}},
                               "seg": {10: {"저CLV": {"recall": 0.2, "revenue": 0.2, "arp": 0.2},
                                            "고CLV": {"recall": 0.2, "revenue": 0.2, "arp": 0.2}}},
                               "value_alignment_spearman": 0.2},
                 "ci": {"Recall": (0.1, 0, 0.2), "NDCG": (0.1, 0, 0.2), "PWGain": (0.1, 0, 0.2),
                        "ValueAlignment": (0.1, 0, 0.2)}}]

    out_dir = tmp_path
    result1 = ns["load_or_run_seed"](42, out_dir, "M2", "hm", fake_run_one_seed)
    assert calls == [42]
    result2 = ns["load_or_run_seed"](42, out_dir, "M2", "hm", fake_run_one_seed)
    assert calls == [42]  # 두 번째는 파일에서 로드, fake_run_one_seed 재호출 안 됨
    assert result1 == result2


def test_load_or_run_seed_roundtrips_integer_keys(tmp_path):
    ns = _load_module_upto_cfg()
    def fake(seed, *a, **kw):
        return [{"seed": seed, "test_base": {"overall": {10: {"recall": 0.1}}}}]
    r1 = ns["load_or_run_seed"](7, tmp_path, "M2", "hm", fake)
    r2 = ns["load_or_run_seed"](7, tmp_path, "M2", "hm", fake)
    assert 10 in r2[0]["test_base"]["overall"], "재로드 후에도 K값 키는 int 10이어야 함(str '10' 아님)"


def test_load_or_run_seed_roundtrips_coverage_and_gini_integer_keys(tmp_path):
    """test_base/test_best의 coverage·gini도 overall/seg와 똑같이 K(10/20/50) 정수키 dict이므로
    (evaluate_combined()의 반환값 그대로 흘러들어감), round-trip 후에도 int 키여야 한다."""
    ns = _load_module_upto_cfg()
    def fake(seed, *a, **kw):
        return [{"seed": seed, "test_base": {"coverage": {10: 0.5}, "gini": {10: 0.3}}}]
    r1 = ns["load_or_run_seed"](8, tmp_path, "M2", "hm", fake)
    r2 = ns["load_or_run_seed"](8, tmp_path, "M2", "hm", fake)
    assert 10 in r2[0]["test_base"]["coverage"], "coverage 키는 int 10이어야 함(str '10' 아님)"
    assert 10 in r2[0]["test_base"]["gini"], "gini 키는 int 10이어야 함(str '10' 아님)"

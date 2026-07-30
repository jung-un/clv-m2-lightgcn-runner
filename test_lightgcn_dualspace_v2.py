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
    # rng.bit_generator.state가 체크포인트에 저장/복원되므로, epoch 1~2(중단 이전)뿐
    # 아니라 epoch 3~4(재개 이후)까지도 batch 순서/negative sampling이 완전히 같은
    # rng 소비 시퀀스를 따라야 한다 — 전 구간 bit-identical해야 한다(재개 시 rng가
    # 새로 seed돼 다른 시퀀스를 쓰던 예전 버그가 여기서 재현되면 실패한다).
    ckpt_ref = tmp_path / "vt_ref.pt"
    ns["train_value_tower"](*args, cfg, seed=0, ckpt_path=ckpt_ref)
    ref = torch.load(ckpt_ref, weights_only=False)

    for i in range(4):
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


def test_train_value_tower_skips_epoch_when_resuming_already_early_stopped(tmp_path):
    """VT_PATIENCE 조건을 이미 만족한 상태로 저장된 체크포인트를 재개하면, bad를 다시
    검사하지 않고 epoch 하나를 더 도는 낭비(리뷰 지적)가 있으면 안 된다 — resume 즉시
    저장된 best_state로 종료해야 하고, all_epochs/last_epoch가 늘어나선 안 된다."""
    ns = _load_module_upto_cfg()
    np, torch = ns["np"], ns["torch"]
    n_users, n_items = 10, 8
    x_val_u = np.random.default_rng(0).uniform(size=(n_users, 4)).astype(np.float32)
    x_val_i = np.random.default_rng(1).uniform(size=(n_items, 3)).astype(np.float32)

    cfg = dict(ns["CFG"]); cfg["VT_MAX_EPOCHS"] = 10; cfg["VT_PATIENCE"] = 2
    cfg["MLP_HIDDEN"] = 4; cfg["D_VALUE"] = 2; cfg["HARD_NEG_RATIO"] = 0.0

    model = ns["ValueTower"](x_val_u, x_val_i, cfg["MLP_HIDDEN"], cfg["D_VALUE"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                  if k not in ns["VT_SNAPSHOT_EXCLUDED_KEYS"]}
    rng = np.random.default_rng(0)

    ckpt = tmp_path / "vt_already_stopped.pt"
    torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                "last_epoch": 3, "best_epoch": 1, "best_val_score": 0.5,
                "bad": 2, "best_state": best_state,  # bad(2) >= VT_PATIENCE(2) → 이미 early stop 조건 충족
                "all_epochs": [{"epoch": 1, "val_recall10": 0.5, "state": best_state},
                               {"epoch": 2, "val_recall10": 0.3, "state": best_state},
                               {"epoch": 3, "val_recall10": 0.2, "state": best_state}],
                "history": [], "seed": 0, "d_value": cfg["D_VALUE"], "mlp_hidden": cfg["MLP_HIDDEN"],
                "hard_neg_ratio": cfg["HARD_NEG_RATIO"], "rng_state": rng.bit_generator.state},
               ckpt)

    tr_u = np.array([0, 1, 2], dtype=np.int64); tr_i = np.array([0, 1, 2], dtype=np.int64)
    pos_key = np.unique(tr_u * n_items + tr_i)
    user_pos = {}
    item_cat_arr = np.zeros(n_items, dtype=np.int64)
    cat_items = {0: np.arange(n_items)}
    val_gt = {u: np.array([int(u % n_items)], dtype=np.int32) for u in range(n_users)}
    csr_ptr = np.zeros(n_users + 1, dtype=np.int64)
    csr_items = np.array([], dtype=np.int32)

    _, best_ep, best_score, all_epochs = ns["train_value_tower"](
        x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
        val_gt, csr_ptr, csr_items, cfg, seed=0, ckpt_path=ckpt)

    assert best_ep == 1 and best_score == 0.5
    assert len(all_epochs) == 3, "재개 시 epoch가 하나도 더 돌면 안 됨(이미 early stop 조건 충족)"
    saved = torch.load(ckpt, weights_only=False)
    assert saved["last_epoch"] == 3, "체크포인트도 갱신되지 않아야 함(재실행이 없었으므로)"


def test_train_loop_resume_is_rng_deterministic(tmp_path):
    """train_value_tower()와 동일한 gap이 M1 trainer인 train_loop()에도 있었다(리뷰 지적):
    rng가 체크포인트에 저장/복원되지 않으면, 재개한 학습이 안 끊긴 학습과 다른 batch
    순서/negative sampling을 쓰게 된다. 중단(2epoch) 후 재개(→4epoch)한 학습이, 처음부터
    중단 없이 4epoch을 도는 참조 실행과 정확히 같은 loss 히스토리·최종 가중치를 내는지
    (bad/best_epoch 같은 북키핑이 아니라 실제 rng 소비 시퀀스까지) 확인한다."""
    ns = _load_module_upto_cfg()
    np, torch, pd = ns["np"], ns["torch"], ns["pd"]
    n_users, n_items = 20, 15
    rng_data = np.random.default_rng(7)
    train_df = pd.DataFrame({
        "u_idx": rng_data.integers(0, n_users, 300).astype(np.int64),
        "i_idx": rng_data.integers(0, n_items, 300).astype(np.int64),
    })
    adj, pos_key, tr_u, tr_i, csr_ptr, csr_items = ns["build_graph"](train_df, n_users, n_items)
    user_pos = {}
    item_cat_arr = np.zeros(n_items, dtype=np.int64)
    cat_items = {0: np.arange(n_items)}
    val_gt = {u: np.array([int(u % n_items)], dtype=np.int32) for u in range(n_users)}
    val_rev = {u: np.array([1.0], dtype=np.float32) for u in range(n_users)}

    base_cfg = dict(ns["CFG"])
    base_cfg.update(SEED=0, BATCH_SIZE=32, HARD_NEG_RATIO=0.0, DIM=4, N_LAYERS=1,
                     EARLY_STOP=100, EVAL_EVERY=1, EVAL_BATCH=32, K_LIST=[5],
                     SELECT_METRIC="Recall@5", RESUME=True, OUT_DIR=str(tmp_path), WD=1e-4, LR=1e-2)

    def _run(run_tag, epochs):
        c = dict(base_cfg); c["RUN_TAG"] = run_tag; c["EPOCHS"] = epochs
        torch.manual_seed(123)  # 두 실행의 모델 초기가중치를 동일하게 (resume 시엔 어차피 로드로 덮임)
        model = ns["LightGCNCLV"](n_users, n_items, c, adj)
        opt = torch.optim.Adam(model.parameters(), lr=c["LR"])
        history, best_state, best_ep, best_score = ns["train_loop"](
            model, opt, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
            val_gt, val_rev, csr_ptr, csr_items, c)
        return model, history

    _run("resume_tag", 2)                                    # 1차: 2 epoch만
    model_resumed, history_resumed = _run("resume_tag", 4)   # 2차: 같은 ckpt로 이어서 4 epoch
    model_ref, history_ref = _run("ref_tag", 4)               # 대조군: 중단 없이 4 epoch 한 번에

    val_key = f"val_{base_cfg['SELECT_METRIC']}"
    assert len(history_resumed) == len(history_ref) == 4
    for i in range(4):
        assert history_resumed[i]["loss"] == history_ref[i]["loss"], \
            f"epoch {i + 1} loss diverged between uninterrupted and resumed run"
        assert history_resumed[i][val_key] == history_ref[i][val_key], \
            f"epoch {i + 1} val score diverged between uninterrupted and resumed run"

    ref_state, resumed_state = model_ref.state_dict(), model_resumed.state_dict()
    assert set(ref_state.keys()) == set(resumed_state.keys())
    for k in ref_state:
        assert torch.equal(ref_state[k], resumed_state[k]), \
            f"final weight '{k}' diverged between uninterrupted and resumed run"


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


def test_seed_result_fingerprint_changes_when_selection_knobs_change():
    """load_or_run_seed()의 result_*.json 캐시 파일명은 vt_fingerprint/grid_fingerprint뿐
    아니라, 학습에는 영향 없이 선택/평가 단계에서만 쓰이는 나머지 노브(가드레일 epsilon들,
    VT_TOPK_CKPTS, EPOCH_SCREEN_LAMBDA, K_LIST, N_BOOT)에도 의존해야 한다 — 안 그러면
    HIGH_CLV_EPSILON_GRID 같은 걸 바꿔도 grid_fingerprint의 캐시 무효화가 이 바깥쪽(가장
    먼저 확인되는) 캐시에서 완전히 우회당해 예전 result_*.json이 그대로 재사용돼버린다."""
    ns = _load_module_upto_cfg()
    base = dict(ns["CFG"]); dcfg = dict(ns["DCFG"])
    base_fp = ns["seed_result_fingerprint"](base, dcfg, 42)

    wider_eps = dict(base); wider_eps["HIGH_CLV_EPSILON_GRID"] = base["HIGH_CLV_EPSILON_GRID"] + [0.1]
    assert ns["seed_result_fingerprint"](wider_eps, dcfg, 42) != base_fp

    changed_acc_eps = dict(base); changed_acc_eps["ACCURACY_EPSILON"] = base["ACCURACY_EPSILON"] + 0.01
    assert ns["seed_result_fingerprint"](changed_acc_eps, dcfg, 42) != base_fp

    changed_k_list = dict(base); changed_k_list["K_LIST"] = base["K_LIST"] + [100]
    assert ns["seed_result_fingerprint"](changed_k_list, dcfg, 42) != base_fp

    changed_n_boot = dict(base); changed_n_boot["N_BOOT"] = base["N_BOOT"] + 1
    assert ns["seed_result_fingerprint"](changed_n_boot, dcfg, 42) != base_fp

    # vt_fingerprint가 이 지문 안에 접혀 들어가므로, 학습에 영향을 주는 키(WINDOW_DAYS)가
    # 바뀌어도 지문이 달라져야 한다.
    changed_window = dict(base); changed_window["WINDOW_DAYS"] = None
    assert ns["seed_result_fingerprint"](changed_window, dcfg, 42) != base_fp

    # 시드가 다르면 당연히 지문도 달라야 한다
    assert ns["seed_result_fingerprint"](base, dcfg, 43) != base_fp

    # 위 어떤 키와도 무관한 변화는 지문이 그대로여야 한다
    unrelated = dict(base); unrelated["MODEL_LABEL"] = "M9"
    assert ns["seed_result_fingerprint"](unrelated, dcfg, 42) == base_fp


def test_load_or_run_seed_path_reflects_seed_result_fingerprint(tmp_path):
    """seed_result_fingerprint()가 있어도 load_or_run_seed()가 실제로 파일명에 반영하지
    않으면 소용없다 — HIGH_CLV_EPSILON_GRID를 바꾼 뒤 같은 seed로 다시 불렀을 때, 예전
    result_*.json을 재사용하지 않고 run_one_seed_fn을 다시 부르는지 end-to-end로 확인."""
    ns = _load_module_upto_cfg()
    calls = []
    def fake(seed, *a, **kw):
        calls.append(seed)
        return [{"seed": seed}]

    cfg1 = dict(ns["CFG"]); dcfg = dict(ns["DCFG"])
    ns["load_or_run_seed"](42, tmp_path, "M2", "hm", cfg1, dcfg, fake)
    assert calls == [42]
    ns["load_or_run_seed"](42, tmp_path, "M2", "hm", cfg1, dcfg, fake)
    assert calls == [42]  # 설정 불변이면 정상적으로 캐시 재사용

    cfg2 = dict(cfg1); cfg2["HIGH_CLV_EPSILON_GRID"] = cfg1["HIGH_CLV_EPSILON_GRID"] + [0.99]
    ns["load_or_run_seed"](42, tmp_path, "M2", "hm", cfg2, dcfg, fake)
    assert calls == [42, 42], "HIGH_CLV_EPSILON_GRID가 바뀌면 예전 result_*.json을 재사용하면 안 됨"


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
    cfg, dcfg = ns["CFG"], ns["DCFG"]
    result1 = ns["load_or_run_seed"](42, out_dir, "M2", "hm", cfg, dcfg, fake_run_one_seed)
    assert calls == [42]
    result2 = ns["load_or_run_seed"](42, out_dir, "M2", "hm", cfg, dcfg, fake_run_one_seed)
    assert calls == [42]  # 두 번째는 파일에서 로드, fake_run_one_seed 재호출 안 됨
    assert result1 == result2


def test_load_or_run_seed_roundtrips_integer_keys(tmp_path):
    ns = _load_module_upto_cfg()
    def fake(seed, *a, **kw):
        return [{"seed": seed, "test_base": {"overall": {10: {"recall": 0.1}}}}]
    cfg, dcfg = ns["CFG"], ns["DCFG"]
    r1 = ns["load_or_run_seed"](7, tmp_path, "M2", "hm", cfg, dcfg, fake)
    r2 = ns["load_or_run_seed"](7, tmp_path, "M2", "hm", cfg, dcfg, fake)
    assert 10 in r2[0]["test_base"]["overall"], "재로드 후에도 K값 키는 int 10이어야 함(str '10' 아님)"


def test_load_or_run_seed_roundtrips_coverage_and_gini_integer_keys(tmp_path):
    """test_base/test_best의 coverage·gini도 overall/seg와 똑같이 K(10/20/50) 정수키 dict이므로
    (evaluate_combined()의 반환값 그대로 흘러들어감), round-trip 후에도 int 키여야 한다."""
    ns = _load_module_upto_cfg()
    def fake(seed, *a, **kw):
        return [{"seed": seed, "test_base": {"coverage": {10: 0.5}, "gini": {10: 0.3}}}]
    cfg, dcfg = ns["CFG"], ns["DCFG"]
    r1 = ns["load_or_run_seed"](8, tmp_path, "M2", "hm", cfg, dcfg, fake)
    r2 = ns["load_or_run_seed"](8, tmp_path, "M2", "hm", cfg, dcfg, fake)
    assert 10 in r2[0]["test_base"]["coverage"], "coverage 키는 int 10이어야 함(str '10' 아님)"
    assert 10 in r2[0]["test_base"]["gini"], "gini 키는 int 10이어야 함(str '10' 아님)"


def test_end_to_end_smoke_cpu(tmp_path):
    """전체 파이프라인이 CPU + 아주 작은 synthetic 데이터로 에러 없이 한 바퀴
    도는지 확인: prepare_data() -> LightGCNCLV -> train_loop() 1 epoch.
    이전 태스크들 각자는 단위테스트로 통과했지만, 아무도 이 세 함수를 실제로
    이어서 호출해본 적은 없다 — signature 어긋남(인자 순서/누락)이 있다면
    여기서만 걸린다. 실제 성능 검증은 여전히 Colab 몫.

    _synthetic_tx()(유저 6/아이템 5)는 여기서는 일부러 쓰지 않는다 — 그 정도
    밀도면 한 유저가 train 구간에서 아이템 5개를 전부 구매해버려 sample_batch()의
    negative 후보가 바닥나는 (n_items 대비 유저당 구매가 지나치게 많은) 퇴화
    케이스에 걸린다. 이는 sample_batch()의 실제 결함이 아니라 그 함수가 애초에
    전제하는 "이 유저가 안 산 아이템이 최소 하나는 남아있다"는 가정을 fixture가
    깨버린 것이므로, 유저/아이템 개수만 넉넉히 늘려 이 통합테스트 목적(signature
    호환성 확인)에 맞는 밀도로 조정한다."""
    ns = _load_module_upto_cfg()
    pd, np, torch = ns["pd"], ns["np"], ns["torch"]

    rng = np.random.default_rng(0)
    n = 80
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    tx = pd.DataFrame({
        "u_raw": rng.integers(0, 8, n),
        "i_raw": rng.integers(0, 10, n).astype(str),
        "t": rng.choice(dates, n),
        "v": rng.uniform(10, 100, n).round(2),
    })
    tx_path = tmp_path / "tx.csv"
    meta_path = tmp_path / "articles.csv"
    tx.rename(columns={"u_raw": "customer_id", "i_raw": "article_id",
                        "t": "t_dat", "v": "price"}).to_csv(tx_path, index=False)
    pd.DataFrame({"article_id": tx["i_raw"].unique(),
                  "product_group_name": ["catA"] * tx["i_raw"].nunique()}).to_csv(meta_path, index=False)

    cfg = dict(ns["CFG"])
    cfg.update(WINDOW_DAYS=None, VAL_DAYS=3, TEST_DAYS=3, EPOCHS=1, EARLY_STOP=100,
               BATCH_SIZE=8, EVAL_BATCH=8, HARD_NEG_RATIO=0.0, DIM=4, N_LAYERS=1,
               VT_MAX_EPOCHS=1, VT_PATIENCE=100, MLP_HIDDEN=4, D_VALUE=2,
               VT_TOPK_CKPTS=1, LAMBDA_GRID=[0, 0.5], CLV_DAMPEN_GRID=[1.0],
               HIGH_CLV_DAMPEN_GRID=[1.0], HIGH_CLV_EPSILON_GRID=[0.05],
               OUT_DIR=str(tmp_path / "out"), N_BOOT=10, K_LIST=[5], SELECT_METRIC="Recall@5")
    dcfg = dict(ns["DCFG"])
    dcfg["tx_path"] = str(tx_path); dcfg["item_meta_path"] = str(meta_path)
    cfg["RUN_TAG"] = f"M1_{cfg['DATASET']}_s{cfg['SEED']}_{ns['cfg_fingerprint'](cfg, dcfg)}"

    d = ns["prepare_data"](cfg, dcfg)
    model = ns["LightGCNCLV"](d["n_users"], d["n_items"], cfg, d["adj"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    history, best_state, best_ep, best_score = ns["train_loop"](
        model, opt, d["tr_u"], d["tr_i"], d["n_items"], d["pos_key"], d["user_pos"],
        d["item_cat_arr"], d["cat_items"], d["val_gt"], d["val_rev"],
        d["csr_ptr"], d["csr_items"], cfg)

    assert best_state is not None, "1 epoch만 돌아도 best_state가 채워져야 함"
    assert best_ep == 1
    assert len(history) == 1
    assert "user_emb.weight" in best_state and best_state["user_emb.weight"].shape == (d["n_users"], cfg["DIM"])


def test_run_dualspace_one_seed_smoke_via_load_or_run_seed(tmp_path):
    """test_end_to_end_smoke_cpu는 prepare_data -> train_loop까지만 갔다 — run_dualspace_one_seed()
    (Task 3가 csr_ptr/csr_items를 시그니처에 추가한 함수)와 load_or_run_seed()의 *args
    passthrough(Task 8)는 지금까지 어떤 테스트에서도 실제 함수를 대상으로 검증된 적이 없고,
    test_run_dualspace_skips_completed_seeds조차 스텁(fake_run_one_seed)만 썼다(리뷰 지적) —
    인자 순서가 하나라도 어긋나면 TypeError가 아니라 조용히 틀린 값이 들어갈 수 있다.

    prepare_data -> M1 1epoch 학습 -> M1 체크포인트 저장/로드(run_dualspace()와 같은 모양) ->
    load_or_run_seed(..., run_dualspace_one_seed, ...)를 최소 1칸 그리드로 실제 호출한다.
    run_dualspace_one_seed()가 res["overall"][10]/[50]을 하드코딩 인덱싱하므로(Fix 5),
    K_LIST=[10,50]로 두고 n_items>=50으로 맞춘다(torch.topk(scores,50,...)이 n_items<50이면
    에러이므로)."""
    ns = _load_module_upto_cfg()
    pd, np, torch = ns["pd"], ns["np"], ns["torch"]

    rng = np.random.default_rng(0)
    n_users_raw, n_items_raw = 40, 55
    n = 2500
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    tx = pd.DataFrame({
        "u_raw": rng.integers(0, n_users_raw, n),
        "i_raw": rng.integers(0, n_items_raw, n).astype(str),
        "t": rng.choice(dates, n),
        "v": rng.uniform(10, 100, n).round(2),
    })
    tx_path = tmp_path / "tx.csv"
    meta_path = tmp_path / "articles.csv"
    tx.rename(columns={"u_raw": "customer_id", "i_raw": "article_id",
                        "t": "t_dat", "v": "price"}).to_csv(tx_path, index=False)
    pd.DataFrame({"article_id": tx["i_raw"].unique(),
                  "product_group_name": ["catA"] * tx["i_raw"].nunique()}).to_csv(meta_path, index=False)

    CFG, DCFG = ns["CFG"], ns["DCFG"]
    CFG.update(WINDOW_DAYS=None, VAL_DAYS=3, TEST_DAYS=3, EPOCHS=1, EARLY_STOP=100,
               BATCH_SIZE=64, EVAL_BATCH=64, HARD_NEG_RATIO=0.0, DIM=4, N_LAYERS=1,
               VT_MAX_EPOCHS=1, VT_PATIENCE=100, MLP_HIDDEN=4, D_VALUE=2,
               VT_TOPK_CKPTS=1, LAMBDA_GRID=[0, 0.5], CLV_DAMPEN_GRID=[1.0],
               HIGH_CLV_DAMPEN_GRID=[1.0], HIGH_CLV_EPSILON_GRID=[0.05],
               OUT_DIR=str(tmp_path / "out"), N_BOOT=10, K_LIST=[10, 50], SELECT_METRIC="Recall@10")
    DCFG.update(tx_path=str(tx_path), item_meta_path=str(meta_path))
    CFG["RUN_TAG"] = f"M1_{CFG['DATASET']}_s{CFG['SEED']}_{ns['cfg_fingerprint'](CFG, DCFG)}"

    d = ns["prepare_data"](CFG, DCFG)
    model = ns["LightGCNCLV"](d["n_users"], d["n_items"], CFG, d["adj"])
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    history, best_state, best_ep, best_score = ns["train_loop"](
        model, opt, d["tr_u"], d["tr_i"], d["n_items"], d["pos_key"], d["user_pos"],
        d["item_cat_arr"], d["cat_items"], d["val_gt"], d["val_rev"],
        d["csr_ptr"], d["csr_items"], CFG)

    # run_dualspace()가 기대하는 그대로 M1 체크포인트를 저장하고 다시 로드한다
    # (m1_path = OUT_DIR/ckpt_{RUN_TAG}.pt, "best_state" 키로 로드 -> 동결 -> propagate).
    m1_path = Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['RUN_TAG']}.pt"
    m1_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"best_state": best_state, "n_users": d["n_users"], "n_items": d["n_items"],
                "epoch": best_ep, "best_epoch": best_ep, "best_score": best_score,
                "history": history}, m1_path)
    m1_state = torch.load(m1_path, map_location="cpu", weights_only=False)["best_state"]
    pref_model = ns["LightGCNCLV"](d["n_users"], d["n_items"], CFG, d["adj"])
    pref_model.load_state_dict(m1_state)
    for p in pref_model.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        U_pref, I_pref, _, _ = pref_model.propagate()

    item_meta = ns["build_item_meta"](d["train"], d["n_items"])
    clv, vhat = ns["compute_clv_vhat"](d["train"], d["n_users"], CFG, DCFG["is_date"])
    user_meta = dict(clv=clv, vhat=vhat)

    eps_rows = ns["load_or_run_seed"](
        42, CFG["OUT_DIR"], CFG["MODEL_LABEL"], CFG["DATASET"], CFG, DCFG,
        ns["run_dualspace_one_seed"],
        d["train"], d["val_gt"], d["val_rev"], d["test_gt"], d["test_rev"],
        d["n_users"], d["n_items"], d["n_cat"], d["tr_u"], d["tr_i"], d["pos_key"], d["user_pos"],
        d["item_cat_arr"], d["cat_items"], item_meta, user_meta, U_pref, I_pref,
        d["csr_ptr"], d["csr_items"])

    assert len(eps_rows) == len(CFG["HIGH_CLV_EPSILON_GRID"])
    for row in eps_rows:
        assert "test_base" in row and "test_best" in row
        assert 10 in row["test_base"]["overall"] and 10 in row["test_best"]["overall"]

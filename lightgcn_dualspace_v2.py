import os
MP = '/content/drive'
IN_COLAB = os.path.exists('/content') and not os.path.exists('/Users')
if IN_COLAB:
    if not os.path.ismount(MP):
        from google.colab import drive
        drive.mount(MP)
    print('✓ Drive 마운트 확인')
else:
    print('로컬 환경 실행 — Drive 마운트 생략')

import json, math, random, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ═══════════════════════════════════════════════════════════════════
# DCFG: 데이터셋 "고정 사실"만 — 경로/컬럼명/날짜형여부/카테고리컬럼.
# 실험할 때마다 바뀌는 값(윈도우 크기, split 일수 등)은 여기 두지 않고 CFG로 내림.
# ═══════════════════════════════════════════════════════════════════
SCHEMA = {
    "hm": {
        "tx_path": ("/content/drive/MyDrive/논문/data/raw/hm/transactions_train.parquet" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/hm/transactions_train.csv"),
        "item_meta_path": ("/content/drive/MyDrive/논문/data/raw/hm/articles.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/hm/articles.csv"),
        "user_col": "customer_id", "item_col": "article_id",
        "time_col": "t_dat", "value_col": "price",
        "item_key_col": "article_id", "category_col": "product_group_name",
        "is_date": True,
    },
    "dunnhumby": {
        "tx_path": ("/content/drive/MyDrive/논문/data/raw/dunnhumby/transaction_data.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/dunnhumby/dunnhumby_The-Complete-Journey CSV/transaction_data.csv"),
        "item_meta_path": ("/content/drive/MyDrive/논문/data/raw/dunnhumby/product.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/dunnhumby/dunnhumby_The-Complete-Journey CSV/product.csv"),
        "user_col": "household_key", "item_col": "PRODUCT_ID",
        "time_col": "DAY", "value_col": "SALES_VALUE",
        "item_key_col": "PRODUCT_ID", "category_col": "COMMODITY_DESC",
        "is_date": False,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CFG: 실험/모델 하이퍼파라미터 전부 — 이 파일에서 바꿀 값은 전부 여기 있어야 한다.
# ═══════════════════════════════════════════════════════════════════
CFG = {
    # ── 실행 대상 ──
    "DATASET": "hm",          # "hm" | "dunnhumby"
    "MODEL_LABEL": "M2",      # 논문 표기용 이름 (M1과 구분되는 이 dual-space 실험 자체의 이름)
    "SEED": 42,
    "SEED_LIST": [42, 43, 44],  # value tower 다중시드 재현성 확인용

    # ── 데이터 필터링/기간 ──
    "OUT_DIR": None,          # 아래에서 DATASET 확정 후 채움
    "WINDOW_DAYS": 60,        # 최근 N일만 사용 (None=전체기간). 60≈2개월 "기준 세팅". 2년 승격 시 None.
    "VAL_DAYS": 7, "TEST_DAYS": 7,
    "MIN_USER_INTER": 1, "MIN_ITEM_INTER": 1,

    # ── 모델 구조 (z^pref, M1 backbone) ──
    "DIM": 64, "N_LAYERS": 2,

    # ── 학습 ──
    "BATCH_SIZE": 8192, "LR": 5e-4, "WD": 1e-3,
    "EPOCHS": 100, "EARLY_STOP": 20, "EVAL_EVERY": 1, "EVAL_BATCH": 1024,
    "HARD_NEG_RATIO": 0.5,
    "RESUME": True,

    # ── value tower(z^value) ──
    "MLP_HIDDEN": 32, "D_VALUE": 16,
    "VT_MAX_EPOCHS": 60, "VT_PATIENCE": 8,
    "VT_TOPK_CKPTS": 5,        # 결합 PWGain 스크리닝 상위 K개 epoch만 4D 그리드에 포함
    "EPOCH_SCREEN_LAMBDA": 1.0,  # epoch 스크리닝용 대표 λ (그 자체가 최적값은 아님, 순위 매기기용)

    # ── CLV 파생 변수 ──
    "SHRINKAGE_K": 5.0, "PREMIUM_THR": 0.8,
    "F_BUCKET_EDGES": [1, 2, 5, 10],     # right=True: (-inf,1],(1,2],(2,5],(5,10],(10,inf)
    "F_BUCKET_LABELS": ["1회", "2회", "3-5회", "6-10회", "11회+"],
    "GATE_N_NEG": 16,          # F_u 게이트 AUC용 negative 샘플 수

    # ── 평가 ──
    "K_LIST": [10, 20, 50], "SELECT_METRIC": "Recall@10",
    "N_BOOT": 2000,

    # ── 그리드 탐색 ──
    "LAMBDA_GRID": [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0],
    "LOW_CLV_PCTL": 0.2,
    "CLV_DAMPEN_GRID": [0.0, 0.3, 0.6, 1.0],
    "HIGH_CLV_DAMPEN_GRID": [0.0, 0.3, 0.6, 1.0],
    "HIGH_CLV_EPSILON_GRID": [0.0, 0.01, 0.02, 0.05],

    # ── 가드레일 (0.0으로 되돌리지 말 것 — CLAUDE.md §4, λ>0 전체 탈락 버그 재현됨) ──
    "ACCURACY_EPSILON": 0.0,
    "LOW_CLV_EPSILON": 0.0,
    "RECALL50_EPSILON": 0.01,
    "HR_EPSILON": 0.01,
    "DIVERSITY_EPSILON": 0.03,
    "EPS_TOL": 1e-9,
}
CFG["OUT_DIR"] = (f"/content/drive/MyDrive/논문/data/results_{CFG['DATASET']}" if IN_COLAB
                  else f"/Users/jungun/Workspace/논문준비/data/results_{CFG['DATASET']}")
DCFG = SCHEMA[CFG["DATASET"]]

_SUPPORTED_SELECT = {f"{m}@{k}" for m in ["Recall", "Precision", "NDCG", "HitRate", "Revenue"] for k in CFG["K_LIST"]}
assert CFG["SELECT_METRIC"] in _SUPPORTED_SELECT, (
    f"SELECT_METRIC은 학습중 evaluate()가 지원하는 값만 가능합니다: {sorted(_SUPPORTED_SELECT)}")

if not torch.cuda.is_available():
    print("[경고] GPU 미검출 — propagate()가 배치마다 재계산되는 구조라 CPU에서는 매우 느립니다. "
          "Colab 런타임을 GPU로 설정하세요.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cfg_fingerprint(cfg, dcfg):
    """실험 설정이 하나라도 바뀌면 체크포인트 파일명 자체가 달라지게 하는 해시.
    M1(z^pref) 학습 결과에 실제로 영향을 주는 키만 포함한다 — 무관한 키(그리드 탐색용
    LAMBDA_GRID 등)까지 넣으면 그 값만 바꿔도 M1을 헛되이 재학습하게 되므로 일부러 뺀다."""
    keys = ["DIM", "N_LAYERS", "MIN_USER_INTER", "MIN_ITEM_INTER",
            "SHRINKAGE_K", "PREMIUM_THR", "EPOCHS", "BATCH_SIZE", "LR", "WD",
            "SEED", "HARD_NEG_RATIO", "WINDOW_DAYS", "VAL_DAYS", "TEST_DAYS"]
    payload = {k: cfg[k] for k in keys}
    payload.update(category_col=dcfg["category_col"])
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def vt_fingerprint(cfg, dcfg, seed):
    """value tower 전용 설정 지문 (D_VALUE 등 포함, SEED는 인자로 개별 지정)."""
    keys = ["MIN_USER_INTER", "MIN_ITEM_INTER", "SHRINKAGE_K", "PREMIUM_THR",
            "BATCH_SIZE", "LR", "HARD_NEG_RATIO", "D_VALUE", "MLP_HIDDEN",
            "VT_MAX_EPOCHS", "VT_PATIENCE", "WINDOW_DAYS", "VAL_DAYS", "TEST_DAYS"]
    payload = {k: cfg[k] for k in keys}
    payload.update(category_col=dcfg["category_col"], seed=seed)
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


CFG["RUN_TAG"] = f"M1_{CFG['DATASET']}_s{CFG['SEED']}_{cfg_fingerprint(CFG, DCFG)}"


def load_transactions(dcfg):
    tx = pd.read_parquet(dcfg["tx_path"]) if dcfg["tx_path"].endswith(".parquet") else pd.read_csv(dcfg["tx_path"])
    tx = tx.rename(columns={dcfg["user_col"]: "u_raw", dcfg["item_col"]: "i_raw",
                             dcfg["time_col"]: "t", dcfg["value_col"]: "v"})
    if dcfg["is_date"]:
        tx["t"] = pd.to_datetime(tx["t"])
        tx["i_raw"] = tx["i_raw"].astype(str)
    tx = tx.drop_duplicates()
    print(f"원본 {len(tx):,}건 (완전중복 제거 완료)")
    return tx


def window_filter(tx, cfg, dcfg):
    if cfg["WINDOW_DAYS"]:
        t_max = tx["t"].max()
        delta = pd.Timedelta(days=cfg["WINDOW_DAYS"]) if dcfg["is_date"] else cfg["WINDOW_DAYS"]
        tx = tx[tx["t"] >= t_max - delta].copy()
        print(f"최근 {cfg['WINDOW_DAYS']}일 사용: {len(tx):,}건")
    return tx


def merge_category(tx, dcfg):
    meta = pd.read_csv(dcfg["item_meta_path"], dtype={dcfg["item_key_col"]: str} if dcfg["is_date"] else None)
    meta = meta.rename(columns={dcfg["item_key_col"]: "i_raw", dcfg["category_col"]: "cat_raw"})
    tx = tx.merge(meta[["i_raw", "cat_raw"]].drop_duplicates("i_raw"), on="i_raw", how="left")
    tx["cat_raw"] = tx["cat_raw"].fillna("UNKNOWN")
    return tx


def compute_boundaries(tx, cfg, dcfg):
    t_max = tx["t"].max()
    day = lambda n: (pd.Timedelta(days=n) if dcfg["is_date"] else n)
    test_start = t_max - day(cfg["TEST_DAYS"])
    val_start = test_start - day(cfg["VAL_DAYS"])
    return val_start, test_start


def filter_and_index(tx, dcfg, cfg, val_start):
    """MIN_USER_INTER/MIN_ITEM_INTER는 train 구간(t<=val_start)만 기준으로 판단
    (val/test까지 합쳐서 세면 미래 정보로 cold-start를 살리는 누수가 생김)."""
    def train_counts(df):
        tp = df[df["t"] <= val_start]
        return tp["u_raw"].value_counts(), tp["i_raw"].value_counts()

    uc, ic = train_counts(tx)
    keep_u = set(uc[uc >= cfg["MIN_USER_INTER"]].index)
    keep_i = set(ic[ic >= cfg["MIN_ITEM_INTER"]].index)
    # ponytail: CFG["ITER_FILTER"]는 항상 False라 반복 재필터링 분기(구 while 루프)를 제거함.
    # 키 자체는 체크포인트 지문(cfg_fingerprint/vt_fingerprint)에 여전히 들어가므로 CFG에 남겨둠
    # — 여기서 지우면 기존 M1 체크포인트 파일명(해시)이 바뀌어 assert m1_path.exists()가 깨짐.

    tx = tx[tx["u_raw"].isin(keep_u) & tx["i_raw"].isin(keep_i)].copy()
    print(f"필터(train 구간 기준, MIN_USER_INTER={cfg['MIN_USER_INTER']}, "
          f"MIN_ITEM_INTER={cfg['MIN_ITEM_INTER']}) 후: {len(tx):,}건")

    uids = np.sort(tx["u_raw"].unique()); iids = np.sort(tx["i_raw"].unique())
    cats = sorted(tx["cat_raw"].unique())
    u2i = {u: k for k, u in enumerate(uids)}; i2i = {i: k for k, i in enumerate(iids)}
    c2i = {c: k for k, c in enumerate(cats)}
    tx["u_idx"] = tx["u_raw"].map(u2i).astype("int32")
    tx["i_idx"] = tx["i_raw"].map(i2i).astype("int32")
    tx["cat_idx"] = tx["cat_raw"].map(c2i).astype("int32")
    print(f"유저 {len(uids):,} | 아이템 {len(iids):,} | 카테고리({dcfg['category_col']}) {len(cats):,}")
    return tx, len(uids), len(iids), len(cats)


def split_data(tx, val_start, test_start, n_items):
    train = tx[tx["t"] <= val_start].copy()
    val = tx[(tx["t"] > val_start) & (tx["t"] <= test_start)].copy()
    test = tx[tx["t"] > test_start].copy()

    train_users = set(train.u_idx.unique()); train_items = set(train.i_idx.unique())
    train_pair_key = np.unique(train.u_idx.values.astype(np.int64) * n_items + train.i_idx.values)

    def build_eval(df, name):
        d = df[df.u_idx.isin(train_users) & df.i_idx.isin(train_items)]
        key = d.u_idx.values.astype(np.int64) * n_items + d.i_idx.values
        pos = np.clip(np.searchsorted(train_pair_key, key), 0, len(train_pair_key) - 1)
        is_repeat = train_pair_key[pos] == key
        d = d[~is_repeat]                      # 재구매쌍 제거 (교수님 지침)
        agg = d.groupby(["u_idx", "i_idx"], sort=False)["v"].sum().reset_index()
        gt, rev = {}, {}
        for u, g in agg.groupby("u_idx", sort=False):
            gt[u] = g.i_idx.values.astype(np.int32); rev[u] = g.v.values.astype(np.float32)
        print(f"  {name}: 평가유저 {len(gt):,}명, 정답 {len(agg):,}쌍")
        return gt, rev

    val_gt, val_rev = build_eval(val, "Val ")
    test_gt, test_rev = build_eval(test, "Test")
    return train, val_gt, val_rev, test_gt, test_rev


def build_graph(train, n_users, n_items):
    tu = train.u_idx.values.astype(np.int64); ti = train.i_idx.values.astype(np.int64)
    edge_key = np.unique(tu * n_items + ti)
    eu = (edge_key // n_items).astype(np.int64); ei = (edge_key % n_items).astype(np.int64)

    n = n_users + n_items
    rows = np.concatenate([eu, ei + n_users]); cols = np.concatenate([ei + n_users, eu])
    deg = np.bincount(rows, minlength=n).astype(np.float32)
    dinv = np.power(np.maximum(deg, 1), -0.5)
    vals = (dinv[rows] * dinv[cols]).astype(np.float32)
    adj = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])),
                                   torch.from_numpy(vals), size=(n, n)).coalesce().to(DEVICE)

    order = np.argsort(eu, kind="stable")
    csr_items = ei[order].astype(np.int32)
    csr_ptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(np.bincount(eu, minlength=n_users), out=csr_ptr[1:])
    return adj, edge_key, tu, ti, csr_ptr, csr_items


def prepare_data(cfg, dcfg):
    """M1 학습(main())과 Dual-Space 실험(run_dualspace()) 양쪽이 필요로 하는 데이터 준비
    전체를 한 곳에서 수행한다. 이전에는 두 함수가 이 파이프라인을 거의 그대로 복붙해서
    각자 호출했음 — 하나라도 스텝이 바뀌면(예: 필터 조건 수정) 두 군데를 따로 고쳐야
    했던 중복을 여기서 없앤다.

    순서: 원본 로드 → 최근 WINDOW_DAYS만 사용 → 카테고리 merge → val/test 경계 계산 →
    MIN_USER/ITEM_INTER 필터링+인덱싱 → train/val/test 분리 → 그래프(adj) 구축 →
    negative 샘플링에 필요한 부가 인덱스(user_pos, item_cat_arr, cat_items) 생성.

    반환하는 dict는 main()에서는 M1 학습에, run_dualspace()에서는 그대로 value tower
    학습 및 그리드 탐색의 입력으로 재사용된다 — 두 진입점이 정확히 같은 train/val/test
    분리를 보고 있음을 보장하는 것이 이 함수의 핵심 역할이다."""
    tx = load_transactions(dcfg)
    tx = window_filter(tx, cfg, dcfg)
    tx = merge_category(tx, dcfg)
    val_start, test_start = compute_boundaries(tx, cfg, dcfg)
    tx, n_users, n_items, n_cat = filter_and_index(tx, dcfg, cfg, val_start)
    train, val_gt, val_rev, test_gt, test_rev = split_data(tx, val_start, test_start, n_items)
    adj, pos_key, tr_u, tr_i, csr_ptr, csr_items = build_graph(train, n_users, n_items)
    user_pos = train.groupby("u_idx")["i_idx"].apply(lambda s: np.unique(s.values)).to_dict()

    _cm = train.drop_duplicates("i_idx").set_index("i_idx")["cat_idx"]
    item_cat_arr = np.full(n_items, -1, dtype=np.int64)
    item_cat_arr[_cm.index.values] = _cm.values
    cat_items = train.drop_duplicates("i_idx").groupby("cat_idx")["i_idx"].apply(lambda s: s.to_numpy()).to_dict()

    return dict(train=train, val_gt=val_gt, val_rev=val_rev, test_gt=test_gt, test_rev=test_rev,
                adj=adj, pos_key=pos_key, tr_u=tr_u, tr_i=tr_i, csr_ptr=csr_ptr, csr_items=csr_items,
                user_pos=user_pos, item_cat_arr=item_cat_arr, cat_items=cat_items,
                n_users=n_users, n_items=n_items, n_cat=n_cat)


def _user_pct_stats(train, cfg, is_date):
    """유저별 F(구매횟수)/T(활동기간)/R(최근성)/AOV/Prem 원시값 + 백분위 순위.
    build_user_features()와 compute_clv_vhat()가 거의 동일한 계산을 각자 중복 수행하던 걸
    공용 헬퍼로 통합함(ponytail-review 지적) — 수식은 원래 코드와 완전히 동일, 계산 위치만 합침."""
    win_end = train["t"].max()
    span = win_end - train["t"].min()
    win_days = max((span.days if is_date else span), 1)
    prem_flag = (train["v"].rank(pct=True) > cfg["PREMIUM_THR"]).astype("int8")
    k = cfg["SHRINKAGE_K"]

    g = train.assign(_prem=prem_flag).groupby("u_idx").agg(
        F=("v", "count"), first=("t", "min"), last=("t", "max"),
        AOV_raw=("v", "mean"), prem=("_prem", "sum"))
    glob_aov = train["v"].mean(); glob_prem = prem_flag.mean()

    if is_date:
        T = (g["last"] - g["first"]).dt.days
        R = 1 - ((win_end - g["last"]).dt.days / win_days)
    else:
        T = g["last"] - g["first"]
        R = 1 - ((win_end - g["last"]) / win_days)

    g["F_p"], g["T_p"], g["R_p"] = g["F"].rank(pct=True), T.rank(pct=True), R.rank(pct=True)
    AOV = (g.F * g.AOV_raw + k * glob_aov) / (g.F + k)
    Prem = (g.prem + k * glob_prem) / (g.F + k)
    g["AOV_p"], g["Prem_p"] = AOV.rank(pct=True), Prem.rank(pct=True)
    return g


def build_user_features(train, n_users, n_cat, cfg, is_date):
    g = _user_pct_stats(train, cfg, is_date)
    k = cfg["SHRINKAGE_K"]

    x_rep = np.full((n_users, 3), 0.5, np.float32)
    x_rep[g.index.values] = np.stack([g["F_p"].values, g["T_p"].values, g["R_p"].values], axis=1)

    Spend_u = train.groupby("u_idx")["v"].sum()
    cat_spend = (train.groupby(["u_idx", "cat_idx"])["v"].sum()
                 .unstack(fill_value=0.0).reindex(columns=range(n_cat), fill_value=0.0))
    obs_share = cat_spend.div(Spend_u, axis=0).fillna(0.0)
    p_bar = (train.groupby("cat_idx")["v"].sum() / train["v"].sum()).reindex(range(n_cat), fill_value=0.0)
    catshare = obs_share.mul(g["F"], axis=0).add(k * p_bar, axis=1).div(g["F"] + k, axis=0)

    cat_full = np.tile(p_bar.values, (n_users, 1)).astype(np.float32)
    cat_full[catshare.index.values] = catshare.values.astype(np.float32)

    aov_full = np.full(n_users, 0.5, np.float32); prem_full = np.full(n_users, 0.5, np.float32)
    aov_full[g.index.values] = g["AOV_p"].values.astype(np.float32)
    prem_full[g.index.values] = g["Prem_p"].values.astype(np.float32)

    x_val = np.concatenate([aov_full[:, None], prem_full[:, None], cat_full], axis=1)
    F_u_full = np.zeros(n_users, dtype=np.int64)
    F_u_full[g.index.values] = g["F"].values
    print(f"  유저 특징: rep {x_rep.shape} | val(AOV,Prem+CatShare[금액기준]{n_cat}) {x_val.shape}")
    return x_rep.astype(np.float32), x_val.astype(np.float32), F_u_full


def build_item_features(train, n_items, n_cat):
    g = train.groupby("i_idx").agg(med=("v", "median"))
    price_pct = np.full(n_items, 0.5, np.float32)
    price_pct[g.index.values] = g["med"].rank(pct=True).to_numpy(np.float32)

    cat_of_item = train.groupby("i_idx")["cat_idx"].agg(lambda s: s.mode().iat[0])
    within_pct = np.full(n_items, 0.5, np.float32)
    joined = g.join(cat_of_item.rename("cat_idx"))
    for c, sub in joined.groupby("cat_idx"):
        within_pct[sub.index.values] = sub["med"].rank(pct=True).to_numpy(np.float32)

    cat_onehot = np.zeros((n_items, n_cat), np.float32)
    cat_onehot[cat_of_item.index.values, cat_of_item.values] = 1.0

    x_item = np.concatenate([price_pct[:, None], within_pct[:, None], cat_onehot], axis=1)
    print(f"  아이템 특징: val(PricePct,WithinCat+CategoryID{n_cat}) {x_item.shape}")
    return x_item.astype(np.float32)


def build_item_meta(train, n_items):
    pop = np.bincount(train["i_idx"].values.astype(np.int64), minlength=n_items).astype(np.float64)
    pop_prob = pop / max(pop.sum(), 1.0)
    med = train.groupby("i_idx")["v"].median()
    price_pct = np.full(n_items, 0.5, np.float64)
    price_pct[med.index.values] = med.rank(pct=True).values
    cat = np.full(n_items, -1, np.int64)
    cmap = train.groupby("i_idx")["cat_idx"].agg(lambda s: s.mode().iat[0])
    cat[cmap.index.values] = cmap.values
    return dict(price_pct=price_pct, pop_prob=pop_prob, cat=cat)


def compute_clv_vhat(train, n_users, cfg, is_date):
    """평가/세그먼트 전용 CLV — 모델 입력(CatShare 등)과 독립적으로, 원시 행동변수로만 산출."""
    g = _user_pct_stats(train, cfg, is_date)
    n_hat = g[["F_p", "T_p", "R_p"]].mean(axis=1)
    v_hat = g[["AOV_p", "Prem_p"]].mean(axis=1)
    clv_full = np.full(n_users, np.nan); vhat_full = np.full(n_users, 0.5, np.float32)
    clv_full[g.index.values] = (n_hat * v_hat).values
    vhat_full[g.index.values] = v_hat.values.astype(np.float32)
    return clv_full, vhat_full


class SideMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        return self.net(x)


class LightGCNCLV(nn.Module):
    # ponytail: 이전에는 cfg["MODEL"] != "M1"일 때 유저/아이템 side 정보(CLV 파생 변수)를
    # layer-0 임베딩에 가산 주입하는 분기(use_side, x_rep/x_val_u/x_val_i, mlp_rep/val_u/val_i,
    # gamma_rep/val_u/val_i)가 있었음(구 M2 원본 가산주입 방식). 이 스크립트는 MODEL이 항상
    # "M1"으로 고정돼 있어(z^pref는 이 클래스가 아니라 별도 ValueTower가 Dual-Space로 담당)
    # 그 분기가 한 번도 실행되지 않는 죽은 코드였음 — 삭제. 원본 구현은
    # lightgcn_clv_exp_colab_emb2.ipynb(emb2)에 그대로 보존되어 있음.
    def __init__(self, n_users, n_items, cfg, adj):
        super().__init__()
        self.n_users, self.n_items = n_users, n_items
        self.n_layers = cfg["N_LAYERS"]
        self.adj = adj
        D = cfg["DIM"]

        self.user_emb = nn.Embedding(n_users, D); self.item_emb = nn.Embedding(n_items, D)
        nn.init.normal_(self.user_emb.weight, std=0.1); nn.init.normal_(self.item_emb.weight, std=0.1)
        self.side_keys = []  # train_loop()의 로그 출력이 참조 — 항상 빈 리스트(side 주입 없음)

    def propagate(self):
        eu, ei = self.user_emb.weight, self.item_emb.weight
        x = torch.cat([eu, ei], dim=0); out = x
        for _ in range(self.n_layers):
            x = torch.sparse.mm(self.adj, x); out = out + x
        out = out / (self.n_layers + 1)
        return out[:self.n_users], out[self.n_users:], eu, ei

    def bpr_loss(self, u, pos, neg, wd):
        # ponytail: REG_TARGET="effective"/"id" 분기 제거 — side 주입이 없어(위 참고) eu0/ei0가
        # 항상 user_emb.weight/item_emb.weight와 같은 값이라 두 옵션이 동일했음. CFG["REG_TARGET"]
        # 키 자체는 체크포인트 지문(cfg_fingerprint) 안정성 때문에 CFG에는 그대로 남겨둠(값 미사용).
        U, I, eu0, ei0 = self.propagate()
        eu, ep, en = U[u], I[pos], I[neg]
        bpr = -F.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
        reg = (eu0[u].norm(2).pow(2) + ei0[pos].norm(2).pow(2) + ei0[neg].norm(2).pow(2)) / len(u)
        return bpr + wd * reg


@torch.no_grad()
def evaluate(model, gt, rev, ks, csr_ptr, csr_items, batch):
    model.eval()
    U, I, _, _ = model.propagate()
    n_items = I.shape[0]
    users = np.fromiter(gt.keys(), dtype=np.int64)
    max_k = max(ks)
    pos_key_sorted, pos_rev_sorted = build_pos_lookup(gt, rev, n_items)
    ideal_rev_cumsum = build_ideal_rev_cumsum(gt, rev)
    P_arr = np.zeros(int(users.max()) + 1, dtype=np.int64)
    for u in users: P_arr[u] = len(gt[u])
    price_pct = np.zeros(n_items); item_nov = np.zeros(n_items); cat_arr = np.full(n_items, -1)

    acc = {f"{m}@{k}": 0.0 for k in ks for m in ["Recall", "Precision", "NDCG", "HitRate", "Revenue"]}
    for s in range(0, len(users), batch):
        bu = users[s:s + batch]
        scores = U[torch.from_numpy(bu).to(DEVICE)] @ I.T
        rr, cc = [], []
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u + 1]
            if b > a:
                rr.append(np.full(b - a, bi, np.int64)); cc.append(csr_items[a:b].astype(np.int64))
        if rr:
            scores[torch.from_numpy(np.concatenate(rr)).to(DEVICE),
                   torch.from_numpy(np.concatenate(cc)).to(DEVICE)] = -np.inf
        topk = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
        res = score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
                          P_arr, price_pct, item_nov, cat_arr, ideal_rev_cumsum)
        for k in ks:
            acc[f"Recall@{k}"] += res[k]["recall"].sum()
            acc[f"Precision@{k}"] += res[k]["precision"].sum()
            acc[f"NDCG@{k}"] += res[k]["ndcg"].sum()
            acc[f"HitRate@{k}"] += res[k]["hr"].sum()
            acc[f"Revenue@{k}"] += res[k]["revenue"].sum()
    n = len(users)
    model.train()
    return {m: v / n for m, v in acc.items()}


_METS = ["recall", "precision", "ndcg", "hr", "map", "revenue", "vndcg", "arp", "novelty", "diversity"]

def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    c = np.cumsum(x); return float((n + 1 - 2 * (c / c[-1]).sum()) / n)

def _spearman(a, b):
    """scipy 사용 — 동순위(tie)를 평균순위로 정확히 처리."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
               P_arr, price_pct, item_nov, cat_arr, ideal_rev_cumsum):
    """evaluate()/evaluate_combined()/evaluate_combined_peruser()가 공유하는 유일한
    지표계산 함수. 이전에는 이 셋이 유저별 python for문으로 Recall/NDCG/HR 등을
    서로 다른 스타일로 중복 구현했었다 — 여기서는 배치 전체를 numpy 벡터 연산으로
    한 번에 계산한다.

    핵심 트릭: "히트 여부"를 python의 set 멤버십 검사 대신, build_graph()가 이미 쓰는
    것과 같은 "u*n_items+i를 정렬한 키 배열 + searchsorted" 패턴으로 판정한다.
    topk[bi, r]이 유저 bu[bi]의 실제 정답(ground truth)인지를 배치 전체에 대해
    한 번의 searchsorted 호출로 알아내고, 동시에 그 정답의 revenue 값까지 같은
    인덱스로 gather한다 (하나의 조회로 히트여부와 revenue를 동시에 얻음).

    파라미터:
      topk: [batch, max_k] 추천 아이템 인덱스 (이미 구매한 아이템은 -inf 마스킹된 상태의 topk)
      bu: [batch] 이 배치의 유저 인덱스
      ks: 평가할 K 목록 (예: [10, 20, 50]) — max(ks) <= topk.shape[1]이어야 함
      pos_key_sorted, pos_rev_sorted: 정답 집합 전체를 u*n_items+i로 인코딩해 정렬한
        키 배열과, 같은 순서의 revenue 배열 (호출부가 한 번만 만들어 재사용)
      P_arr: [n_users] 유저별 정답 개수(0-indexed 유저ID로 바로 조회 가능한 배열)
      price_pct, item_nov, cat_arr: [n_items] 아이템별 가격백분위/novelty/카테고리
      ideal_rev_cumsum: {user_idx: np.ndarray} 유저별 "정답 revenue를 내림차순 정렬 후
        NDCG discount를 곱해 누적합한 배열" — V-NDCG의 분모(idcgv)를 매번 다시 정렬하지
        않고 인덱싱만으로 얻기 위한 사전계산 캐시(그리드 탐색 내내 gt/rev가 고정이므로
        이 캐시도 seed 하나당 한 번만 만들면 됨).

    반환: {k: {metric_name: np.ndarray[batch]}} — 합산/평균은 호출부가 한다.
    """
    batch, max_k = topk.shape
    keys = bu[:, None].astype(np.int64) * n_items + topk.astype(np.int64)
    idx = np.clip(np.searchsorted(pos_key_sorted, keys), 0, len(pos_key_sorted) - 1)
    is_hit = pos_key_sorted[idx] == keys
    gain = np.where(is_hit, pos_rev_sorted[idx], 0.0)

    disc_full = 1.0 / np.log2(np.arange(2, max_k + 2))
    P_batch = P_arr[bu]

    price_row = price_pct[topk]; nov_row = item_nov[topk]; cat_row = cat_arr[topk]

    out = {}
    for k in ks:
        hit_k = is_hit[:, :k]; gain_k = gain[:, :k]; disc_k = disc_full[:k]
        nh = hit_k.sum(axis=1)
        Pk = np.minimum(P_batch, k)
        idcg = np.cumsum(disc_k)[Pk - 1]
        idcg = np.where(Pk > 0, idcg, 0.0)
        dcg = (hit_k * disc_k).sum(axis=1)
        cum_hits = np.cumsum(hit_k, axis=1)
        ranks = np.arange(1, k + 1)
        map_num = (cum_hits * hit_k / ranks).sum(axis=1)
        idcgv = np.array([ideal_rev_cumsum[u][min(len(ideal_rev_cumsum[u]), k) - 1]
                           if len(ideal_rev_cumsum[u]) > 0 else 0.0 for u in bu])
        dcgv = (gain_k * disc_k).sum(axis=1)

        # diversity: top-k 슬라이스(랭크 0..k-1)에 대해서만 카테고리 정렬해 distinct count.
        # (주의: max_k 전체를 한 번 정렬해서 [:k]로 자르면 "top-k 안의 서로 다른 카테고리 수"가
        # 아니라 "max_k개 중 카테고리값이 가장 작은 k개"가 돼버려 k < max_k일 때 값이 달라진다
        # — 반드시 k마다 cat_row[:, :k]를 새로 정렬해야 함.)
        cat_k = cat_row[:, :k]
        sorted_cat = np.sort(cat_k, axis=1)
        changed = np.concatenate([np.ones((batch, 1), dtype=bool), sorted_cat[:, 1:] != sorted_cat[:, :-1]], axis=1)
        valid_cat = sorted_cat >= 0
        n_valid = valid_cat.sum(axis=1)
        diversity = np.where(n_valid > 0, (changed & valid_cat).sum(axis=1) / k, 0.0)

        out[k] = dict(
            recall=np.where(P_batch > 0, nh / np.maximum(P_batch, 1), 0.0),
            precision=nh / k,
            hr=(nh > 0).astype(np.float64),
            ndcg=np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0),
            map=np.where(Pk > 0, map_num / np.maximum(Pk, 1), 0.0),
            revenue=gain_k.sum(axis=1),
            vndcg=np.where(idcgv > 0, dcgv / np.maximum(idcgv, 1e-12), 0.0),
            arp=price_row[:, :k].mean(axis=1),
            novelty=nov_row[:, :k].mean(axis=1),
            diversity=diversity,
        )
    return out


def build_pos_lookup(gt, rev, n_items):
    """gt/rev(유저→정답아이템, 유저→revenue)를 score_topk()가 쓰는
    "정렬된 u*n_items+i 키 배열 + 같은 순서 revenue 배열"로 변환."""
    keys, revs = [], []
    for u, items in gt.items():
        keys.append(u * n_items + items.astype(np.int64))
        revs.append(rev[u])
    keys = np.concatenate(keys); revs = np.concatenate(revs).astype(np.float64)
    order = np.argsort(keys, kind="stable")
    return keys[order], revs[order]


def build_ideal_rev_cumsum(gt, rev):
    """유저별 '정답 revenue를 내림차순 정렬 후 NDCG discount를 곱해 누적합'한 배열.
    V-NDCG의 이상적(ideal) DCG를 매 grid 호출마다 다시 정렬하지 않고 인덱싱만으로
    꺼내 쓰기 위한 사전계산 — gt/rev가 고정된 seed 하나당 한 번만 계산하면 된다."""
    out = {}
    for u, items in gt.items():
        sorted_rev = np.sort(rev[u])[::-1]
        disc = 1.0 / np.log2(np.arange(2, len(sorted_rev) + 2))
        out[u] = np.cumsum(sorted_rev * disc)
    return out


def sample_batch(u_arr, pos_arr, n_items, pos_key, user_pos, item_cat_arr, cat_items,
                  rng, hard_ratio=0.5, max_try=50):
    # 주의: train에서 관측된 positive가 negative로 뽑히지 않도록만 보장합니다.
    # 미관측 future positive(=진짜 false negative)까지 배제하는 건 아닙니다 —
    # implicit feedback 데이터의 근본적 한계로, 여기서는 해소되지 않습니다.
    n = len(u_arr)
    neg = np.empty(n, dtype=np.int64)
    is_hard = rng.random(n) < hard_ratio
    pos_cats = item_cat_arr[pos_arr]

    def draw_hard(idxs):
        for j in idxs:
            c = pos_cats[j]
            cand = cat_items.get(c)
            if cand is None or len(cand) < 2:
                neg[j] = rng.integers(0, n_items)
            else:
                neg[j] = cand[rng.integers(0, len(cand))]

    draw_hard(np.where(is_hard)[0])
    idx_rand = np.where(~is_hard)[0]
    neg[idx_rand] = rng.integers(0, n_items, size=len(idx_rand))

    u64 = u_arr.astype(np.int64)
    bad = np.zeros(n, dtype=bool)
    for _ in range(max_try):
        key = u64 * n_items + neg
        pos = np.clip(np.searchsorted(pos_key, key), 0, len(pos_key) - 1)
        bad = pos_key[pos] == key
        if not bad.any():
            return neg
        bad_idx = np.where(bad)[0]
        draw_hard(bad_idx[is_hard[bad_idx]])
        bad_rand = bad_idx[~is_hard[bad_idx]]
        if len(bad_rand):
            neg[bad_rand] = rng.integers(0, n_items, size=len(bad_rand))
    for j in np.where(bad)[0]:
        u = u_arr[j]
        mask = np.ones(n_items, dtype=bool)
        mask[user_pos.get(u, [])] = False
        neg[j] = rng.choice(np.flatnonzero(mask))
    return neg


def train_loop(model, opt, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
                val_gt, val_rev, csr_ptr, csr_items, cfg):
    ckpt_dir = Path(cfg["OUT_DIR"]); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"ckpt_{cfg['RUN_TAG']}.pt"
    rng = np.random.default_rng(cfg["SEED"])
    n_train = len(tr_u); n_batch = math.ceil(n_train / cfg["BATCH_SIZE"])
    best_score, best_ep, best_state, bad = -1.0, -1, None, 0
    history, start_ep = [], 1

    if cfg["RESUME"] and ckpt.exists():
        st = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if st["n_users"] == model.n_users and st["n_items"] == model.n_items:
            model.load_state_dict(st["last_state"]); opt.load_state_dict(st["opt"])
            start_ep = st["epoch"] + 1; best_score, best_ep = st["best_score"], st["best_epoch"]
            best_state, history = st["best_state"], st["history"]
            print(f"[RESUME] epoch {st['epoch']}까지 복원 (RUN_TAG={cfg['RUN_TAG']}이 설정 지문까지 일치)")

    for ep in range(start_ep, cfg["EPOCHS"] + 1):
        t0 = time.time(); perm = rng.permutation(n_train); tot = 0.0
        for b in range(n_batch):
            idx = perm[b * cfg["BATCH_SIZE"]:(b + 1) * cfg["BATCH_SIZE"]]
            bu, bi = tr_u[idx], tr_i[idx]
            bn = sample_batch(bu, bi, n_items, pos_key, user_pos, item_cat_arr, cat_items, rng,
                               hard_ratio=cfg["HARD_NEG_RATIO"])
            loss = model.bpr_loss(torch.from_numpy(bu.astype(np.int64)).to(DEVICE),
                                   torch.from_numpy(bi.astype(np.int64)).to(DEVICE),
                                   torch.from_numpy(bn).to(DEVICE), cfg["WD"])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        avg_loss = tot / n_batch
        rec = {"epoch": ep, "loss": avg_loss, "sec": time.time() - t0}

        if ep % cfg["EVAL_EVERY"] == 0:
            vm = evaluate(model, val_gt, val_rev, cfg["K_LIST"], csr_ptr, csr_items, cfg["EVAL_BATCH"])
            score = vm[cfg["SELECT_METRIC"]]
            gtxt = ""
            if model.side_keys:
                gtxt = " γ " + " ".join(f"{k}={float(getattr(model, f'gamma_{k}')):.3f}" for k in model.side_keys)
            star = ""
            if score > best_score:
                best_score, best_ep, bad = score, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                star = " ★"
            else:
                bad += 1
            print(f"ep {ep:3d} | loss {avg_loss:.4f} | val {cfg['SELECT_METRIC']} {score:.5f}{gtxt} "
                  f"| {rec['sec']:.0f}s{star}")
            rec.update({f"val_{k}": v for k, v in vm.items()})
            torch.save({"last_state": model.state_dict(), "opt": opt.state_dict(), "epoch": ep,
                        "best_state": best_state, "best_epoch": best_ep, "best_score": best_score,
                        "history": history + [rec], "n_users": model.n_users, "n_items": model.n_items},
                       ckpt)
            if cfg["EARLY_STOP"] and bad >= cfg["EARLY_STOP"]:
                history.append(rec); print("early stop"); break
        history.append(rec)
    return history, best_state, best_ep, best_score


def main():
    """M1을 학습/재개. 이미 체크포인트가 있으면 즉시 복원되고 끝남."""
    set_seed(CFG["SEED"])
    print(f"DATASET={CFG['DATASET']} | DEVICE={DEVICE} | RUN_TAG={CFG['RUN_TAG']}")

    d = prepare_data(CFG, DCFG)
    model = LightGCNCLV(d["n_users"], d["n_items"], CFG, d["adj"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    history, best_state, best_ep, best_score = train_loop(
        model, opt, d["tr_u"], d["tr_i"], d["n_items"], d["pos_key"], d["user_pos"],
        d["item_cat_arr"], d["cat_items"], d["val_gt"], d["val_rev"],
        d["csr_ptr"], d["csr_items"], CFG)
    model.load_state_dict(best_state)
    print(f"완료 — best epoch {best_ep}, val {CFG['SELECT_METRIC']} {best_score:.5f}")
    return model


# ═══════════════════════════════════════════════════════════════════
# Dual-Space (M2) 실험
# ═══════════════════════════════════════════════════════════════════


class ValueTower(nn.Module):
    def __init__(self, x_val_u, x_val_i, hidden, d_value):
        super().__init__()
        self.register_buffer("x_val_u", torch.from_numpy(x_val_u))
        self.register_buffer("x_val_i", torch.from_numpy(x_val_i))
        self.mlp_u = SideMLP(x_val_u.shape[1], hidden, d_value)
        self.mlp_i = SideMLP(x_val_i.shape[1], hidden, d_value)

    def encode(self):
        zu = F.normalize(self.mlp_u(self.x_val_u), dim=1)
        zi = F.normalize(self.mlp_i(self.x_val_i), dim=1)
        return zu, zi

    def bpr_loss(self, u, pos, neg):
        zu, zi = self.encode()
        eu, ep, en = zu[u], zi[pos], zi[neg]
        pos_score = (eu * ep).sum(-1)
        neg_score = (eu * en).sum(-1)
        return -F.logsigmoid(pos_score - neg_score).mean()


@torch.no_grad()
def evaluate_value_only(model, gt, csr_ptr, csr_items, k=10, batch=1024):
    """value tower 자체의 Recall — 조기종료 판단용 (z^pref와 무관)."""
    model.eval()
    zu, zi = model.encode()
    users = np.fromiter(gt.keys(), dtype=np.int64)
    total = 0.0
    for s in range(0, len(users), batch):
        bu = users[s:s + batch]
        scores = zu[torch.from_numpy(bu).to(DEVICE)] @ zi.T
        rr, cc = [], []
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u + 1]
            if b > a:
                rr.append(np.full(b - a, bi, np.int64)); cc.append(csr_items[a:b].astype(np.int64))
        if rr:
            scores[torch.from_numpy(np.concatenate(rr)).to(DEVICE),
                   torch.from_numpy(np.concatenate(cc)).to(DEVICE)] = -np.inf
        topk = scores.topk(k, dim=1).indices.cpu().numpy()
        for bi, u in enumerate(bu):
            g = gt[u]; gset = set(g.tolist())
            hit = sum(1 for x in topk[bi] if x in gset)
            total += hit / len(g)
    model.train()
    return total / len(users)


def train_value_tower(x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos,
                       item_cat_arr, cat_items, val_gt, csr_ptr, csr_items, cfg, seed,
                       ckpt_path=None):
    """value tower 자체 Recall(조기종료 판단용)로 학습을 모니터링한다. 단, 최종 채택
    epoch는 이 함수가 정하지 않는다 — VT 단독 Recall이 좋은 epoch가 M1과 결합했을 때도
    PWGain 기준 최적이라는 보장이 없기 때문(리뷰 지적 반영). value tower는 아주 작은
    MLP라 저장 비용이 낮으므로, "VT 단독 Recall 상위 K개"가 아니라 조기종료 전까지
    학습된 모든 epoch의 state를 보관해서 반환한다 — 어떤 epoch를 결합 그리드에 넣을지는
    호출부(run_dualspace_one_seed)가 실제 결합 PWGain으로 다시 스크리닝한다.
    z^pref와 파라미터 전혀 공유 안 함.

    M1 train_loop()과 동일한 저장/재개 패턴: ckpt_path에 매 epoch 끝날 때마다
    현재까지 진행상황(epoch 번호, 모델/옵티마이저 state, best 기록, all_epochs 스냅샷)을
    저장하고, 시작할 때 그 파일이 있으면 이어서 학습한다. Colab 세션이 VT_MAX_EPOCHS(최대
    60)를 다 돌기 전에 끊겨도 처음부터 다시 돌 필요가 없다."""
    set_seed(seed)
    model = ValueTower(x_val_u, x_val_i, cfg["MLP_HIDDEN"], cfg["D_VALUE"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    rng = np.random.default_rng(seed)
    n_train = len(tr_u); n_batch = math.ceil(n_train / cfg["BATCH_SIZE"])
    best_score, best_ep, best_state, bad = -1.0, -1, None, 0
    all_epochs = []  # [{"epoch":ep, "val_recall10":val_score, "state":state_dict}] — 전 epoch 보관
    history = []
    start_ep = 1

    if ckpt_path and Path(ckpt_path).exists():
        st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model_state"]); opt.load_state_dict(st["opt_state"])
        start_ep = st["last_epoch"] + 1
        best_score, best_ep, bad = st["best_val_score"], st["best_epoch"], st["bad"]
        all_epochs = st["all_epochs"]; history = st["history"]
        best_state = st["best_state"]
        print(f"[VT RESUME seed{seed}] epoch {st['last_epoch']}까지 복원, epoch {start_ep}부터 재개")

    for ep in range(start_ep, cfg["VT_MAX_EPOCHS"] + 1):
        perm = rng.permutation(n_train); tot = 0.0
        for b in range(n_batch):
            idx = perm[b*cfg["BATCH_SIZE"]:(b+1)*cfg["BATCH_SIZE"]]
            bu, bi = tr_u[idx], tr_i[idx]
            bn = sample_batch(bu, bi, n_items, pos_key, user_pos, item_cat_arr, cat_items, rng,
                               hard_ratio=cfg["HARD_NEG_RATIO"])
            loss = model.bpr_loss(torch.from_numpy(bu.astype(np.int64)).to(DEVICE),
                                   torch.from_numpy(bi.astype(np.int64)).to(DEVICE),
                                   torch.from_numpy(bn).to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        val_score = evaluate_value_only(model, val_gt, csr_ptr, csr_items, k=10)
        # x_val_u/x_val_i are static input-feature buffers (never trained, identical across
        # every epoch) — excluding them keeps each all_epochs snapshot small. They're already
        # present on `model` at construction time, so load_state_dict(..., strict=False) below
        # is exactly correct, not a workaround.
        state_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k not in ("x_val_u", "x_val_i")}
        star = ""
        if val_score > best_score:
            best_score, best_ep, bad = val_score, ep, 0
            best_state = state_snapshot
            star = " ★"
        else:
            bad += 1
        all_epochs.append({"epoch": ep, "val_recall10": val_score, "state": state_snapshot})
        history.append({"epoch": ep, "loss": tot/n_batch, "val_recall@10": val_score})
        print(f"[value tower seed{seed}] ep {ep:3d} loss {tot/n_batch:.4f} val_R@10 {val_score:.5f}{star}")

        if ckpt_path:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "opt_state": opt.state_dict(),
                        "last_epoch": ep, "best_epoch": best_ep, "best_val_score": best_score,
                        "bad": bad, "best_state": best_state, "all_epochs": all_epochs,
                        "history": history, "seed": seed, "d_value": cfg["D_VALUE"],
                        "mlp_hidden": cfg["MLP_HIDDEN"], "hard_neg_ratio": cfg["HARD_NEG_RATIO"]},
                       ckpt_path)

        if cfg["VT_PATIENCE"] and bad >= cfg["VT_PATIENCE"]:
            print("  early stop"); break

    model.load_state_dict(best_state, strict=False)  # best_state has no x_val_u/x_val_i (see above); already correct on model
    if ckpt_path:
        print(f"  저장 → {ckpt_path} (VT 단독최고 epoch={best_ep}, 총 {len(all_epochs)}개 epoch 보관)")
    return model, best_ep, best_score, all_epochs


def compute_fbucket_gate(train, val_gt, x_val_u, item_cat_arr_train, F_u, user_pos, n_neg=16, seed=0):
    """CatShare vs 단순 HasBought의 AUC gap을 train→val에서 즉석 산출 (하드코딩 제거).
    off-by-one 없이 right=True로 구간화.
    negative 샘플링 시 해당 유저의 알려진 positive(train + val)는 제외한다 — 그렇지
    않으면 무작위로 뽑힌 negative 아이템이 실제로는 그 유저가 산 적 있는 아이템이라
    label=0으로 잘못 표시되어 AUC 추정이 오염될 수 있다. n_neg도 4→기본 16으로 늘려
    seed에 따른 gap 추정 변동을 줄인다.
    주의(val 정보 사용 범위 확인): 여기서 val_gt를 negative 제외 집합에 포함하는 건
    "validation 진단(F_u 게이트 산출)" 목적에서만이다 — 실제 value tower 학습 negative
    sampler인 sample_batch()는 train만으로 구성된 user_pos만 받으며 val 정보를 전혀
    쓰지 않는다(run_dualspace()에서 user_pos = train.groupby(...)로 생성됨). 두 샘플러의
    val 접근 범위를 혼동하지 않도록 여기 명시해 둔다."""
    catshare = x_val_u[:, 2:]
    n_cat = catshare.shape[1]
    hasbought = np.zeros_like(catshare)
    seen = train.groupby("u_idx")["cat_idx"].unique()
    for u, cats in seen.items():
        hasbought[u, cats] = 1.0

    rng = np.random.default_rng(seed)
    all_items = train["i_idx"].unique()
    known_pos = {u: set(user_pos.get(u, [])) | set(val_gt.get(u, [])) for u in val_gt}
    rows_u, rows_c, rows_label = [], [], []
    for u, items in val_gt.items():
        upos = known_pos[u]
        for i in items:
            c = item_cat_arr_train[i]
            if c < 0: continue
            rows_u.append(u); rows_c.append(c); rows_label.append(1)
            n_drawn = 0
            n_tries = 0
            while n_drawn < n_neg and n_tries < n_neg * 5:
                n_tries += 1
                neg_i = rng.choice(all_items)
                if neg_i in upos:
                    continue  # 알려진 positive는 negative로 쓰지 않음
                nc = item_cat_arr_train[neg_i]
                if nc < 0: continue
                rows_u.append(u); rows_c.append(nc); rows_label.append(0)
                n_drawn += 1

    rows_u = np.array(rows_u); rows_c = np.array(rows_c); rows_label = np.array(rows_label)
    bucket_idx = np.digitize(F_u[rows_u], CFG["F_BUCKET_EDGES"], right=True)  # off-by-one 수정
    labels = CFG["F_BUCKET_LABELS"]
    gaps = np.zeros(len(labels))
    print("  F_u 구간별 실측 gap (train→val 즉석 산출):")
    for b in range(len(labels)):
        mask = bucket_idx == b
        if mask.sum() < 50 or rows_label[mask].sum() == 0:
            print(f"    {labels[b]:6s}: 표본 부족, gap=0")
            continue
        sn = hasbought[rows_u[mask], rows_c[mask]]
        ss = catshare[rows_u[mask], rows_c[mask]]
        try:
            an = roc_auc_score(rows_label[mask], sn)
            asr = roc_auc_score(rows_label[mask], ss)
            gaps[b] = max(asr - an, 0.0)
            print(f"    {labels[b]:6s}: n={mask.sum():,} AUC_naive={an:.4f} AUC_catshare={asr:.4f} gap={gaps[b]:+.4f}")
        except ValueError:
            print(f"    {labels[b]:6s}: AUC 계산 불가(단일 클래스), gap=0")
    return gaps / max(gaps.max(), 1e-8)


def compute_gate(F_u_arr, gate_lookup):
    idx = np.digitize(F_u_arr, CFG["F_BUCKET_EDGES"], right=True)  # off-by-one 수정
    return gate_lookup[idx].astype(np.float32)


@torch.no_grad()
def _combined_scores(U_pref, I_pref, Uv, Iv, gate_arr, lam_base, bu):
    uid_t = torch.tensor(bu, dtype=torch.long, device=DEVICE)
    s_rel = U_pref[uid_t] @ I_pref.T
    s_val = Uv[uid_t] @ Iv.T
    s_rel = (s_rel - s_rel.mean(dim=1, keepdim=True)) / (s_rel.std(dim=1, keepdim=True) + 1e-8)
    s_val = (s_val - s_val.mean(dim=1, keepdim=True)) / (s_val.std(dim=1, keepdim=True) + 1e-8)
    g = torch.tensor(gate_arr[bu], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    return s_rel + lam_base * g * s_val


@torch.no_grad()
def evaluate_combined(U_pref, I_pref, Uv, Iv, gate_arr, lam_base,
                       gt, rev, item_meta, user_meta, ks, csr_ptr, csr_items, batch=1024,
                       clv_lo_th=None, clv_hi_th=None, pos_lookup=None, ideal_rev_cumsum=None):
    # clv_lo_th/clv_hi_th를 명시하지 않으면(기존 동작 유지) 이번에 평가되는 유저 집합만으로
    # 임계값을 다시 계산합니다 — 단, run_dualspace_one_seed에서는 게이트 마스크(is_low_clv)와
    # 완전히 동일한 전역 임계값을 항상 넘겨서, "누구를 dampen했는지"와 "누구를 저CLV로 채점하는지"가
    # 어긋나지 않도록 합니다 (어긋나면 dampen=0으로도 제약이 항상 실패하는 버그가 생김).
    # pos_lookup/ideal_rev_cumsum: Stage A/B 그리드에서 매 호출마다 gt/rev로부터 다시 만들지
    # 않고 run_dualspace_one_seed()가 seed당 한 번 만든 걸 넘겨받아 재사용하기 위한 선택적 캐시.
    # None이면 기존처럼 함수 내부에서 계산(하위호환).
    n_items = I_pref.shape[0]
    price_pct, pop_prob, cat = item_meta["price_pct"], item_meta["pop_prob"], item_meta["cat"]
    item_nov = -np.log2(pop_prob + 1e-12)
    clv, vhat = user_meta["clv"], user_meta["vhat"]
    med_clv = np.nanmedian(clv)
    users = list(gt.keys())
    uclv = np.array([clv[u] if not np.isnan(clv[u]) else med_clv for u in users])
    lo_th = clv_lo_th if clv_lo_th is not None else np.nanquantile(uclv, 0.2)
    hi_th = clv_hi_th if clv_hi_th is not None else np.nanquantile(uclv, 0.8)
    seg_of = {u: ("저CLV" if c <= lo_th else ("고CLV" if c >= hi_th else "중간")) for u, c in zip(users, uclv)}
    segs = ["저CLV", "고CLV"]; seg_cnt = {s: sum(1 for u in users if seg_of[u] == s) for s in segs}

    pos_key_sorted, pos_rev_sorted = pos_lookup if pos_lookup is not None else build_pos_lookup(gt, rev, n_items)
    if ideal_rev_cumsum is None:
        ideal_rev_cumsum = build_ideal_rev_cumsum(gt, rev)
    users_arr = np.array(users)
    P_arr = np.zeros(int(users_arr.max()) + 1, dtype=np.int64)
    for u in users: P_arr[u] = len(gt[u])

    overall = {k: {m: 0.0 for m in _METS} for k in ks}
    seg_acc = {k: {s: {m: 0.0 for m in _METS} for s in segs} for k in ks}
    expo = {k: np.zeros(n_items) for k in ks}
    cal_v, cal_p = [], []
    max_k = max(ks); k0 = ks[0]
    seg_arr = np.array([seg_of[u] for u in users])

    for s in range(0, len(users), batch):
        bu = users_arr[s:s+batch]
        scores = _combined_scores(U_pref, I_pref, Uv, Iv, gate_arr, lam_base, bu)
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u+1]
            if b > a: scores[bi, csr_items[a:b]] = -1e9
        topk = scores.topk(max_k, dim=1).indices.cpu().numpy()
        res = score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
                          P_arr, price_pct, item_nov, cat, ideal_rev_cumsum)
        bseg = seg_arr[s:s+batch]
        for k in ks:
            for m in _METS: overall[k][m] += res[k][m].sum()
            for sg in segs:
                mask = bseg == sg
                if mask.any():
                    for m in _METS: seg_acc[k][sg][m] += res[k][m][mask].sum()
            for bi in range(len(bu)):
                expo[k][topk[bi, :k]] += 1
            if k == k0:
                cal_v.extend(vhat[u] for u in bu); cal_p.extend(res[k]["arp"].tolist())

    n = len(users)
    for k in ks:
        for m in _METS: overall[k][m] /= n
        for s in segs:
            if seg_cnt[s]:
                for m in _METS: seg_acc[k][s][m] /= seg_cnt[s]
    coverage = {k: float((expo[k]>0).sum())/n_items for k in ks}
    gini_expo = {k: _gini(expo[k]) for k in ks}
    value_alignment = _spearman(cal_v, cal_p)   # 이름 변경: calibration → value_alignment_spearman
    return dict(overall=overall, seg=seg_acc, seg_cnt=seg_cnt, coverage=coverage,
                gini=gini_expo, value_alignment_spearman=value_alignment, n_eval=n)


@torch.no_grad()
def evaluate_combined_peruser(U_pref, I_pref, Uv, Iv, gate_arr, lam_base,
                               gt, rev, item_meta, user_meta, csr_ptr, csr_items, k=10, batch=1024,
                               pos_lookup=None, ideal_rev_cumsum=None):
    """유저별 Recall/NDCG/Revenue/ARP 배열 (bootstrap CI 용). users 순서 고정 보장.
    csr_ptr/csr_items를 인자로 명시 전달 — 모듈 전역(csr_ptr_global 등)에 암묵적으로
    의존하면 재현성·독립 실행이 깨지기 쉬우므로 함수 시그니처에 드러낸다."""
    users = np.array(sorted(gt.keys()))
    n = len(users)
    price_pct = item_meta["price_pct"]; item_nov = -np.log2(item_meta["pop_prob"] + 1e-12)
    cat = item_meta["cat"]; vhat = user_meta["vhat"]
    n_items = I_pref.shape[0]
    pos_key_sorted, pos_rev_sorted = pos_lookup if pos_lookup is not None else build_pos_lookup(gt, rev, n_items)
    if ideal_rev_cumsum is None:
        ideal_rev_cumsum = build_ideal_rev_cumsum(gt, rev)
    P_arr = np.zeros(int(users.max()) + 1, dtype=np.int64)
    for u in users: P_arr[u] = len(gt[u])

    recall_arr = np.zeros(n); ndcg_arr = np.zeros(n); revenue_arr = np.zeros(n); arp_arr = np.zeros(n)
    idx_map = {u: i for i, u in enumerate(users)}

    for s in range(0, n, batch):
        bu = users[s:s+batch]
        scores = _combined_scores(U_pref, I_pref, Uv, Iv, gate_arr, lam_base, bu)
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u+1]
            if b > a: scores[bi, csr_items[a:b]] = -1e9
        topk = scores.topk(k, dim=1).indices.cpu().numpy()
        res = score_topk(topk, bu, [k], pos_key_sorted, pos_rev_sorted, n_items,
                          P_arr, price_pct, item_nov, cat, ideal_rev_cumsum)[k]
        for bi, u in enumerate(bu):
            j = idx_map[u]
            recall_arr[j] = res["recall"][bi]; ndcg_arr[j] = res["ndcg"][bi]
            revenue_arr[j] = res["revenue"][bi]; arp_arr[j] = res["arp"][bi]
    vhat_arr = np.array([vhat[u] for u in users])
    return dict(users=users, recall=recall_arr, ndcg=ndcg_arr, revenue=revenue_arr,
                arp=arp_arr, vhat=vhat_arr)


def bootstrap_mean_diff_ci(arr_a, arr_b, n_boot=2000, seed=0):
    """arr_b - arr_a (같은 유저 순서) 평균 차이의 95% 부트스트랩 CI."""
    rng = np.random.default_rng(seed)
    n = len(arr_a)
    diff = arr_b - arr_a
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diff[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def bootstrap_spearman_diff_ci(vhat_arr, arp_a, arp_b, n_boot=2000, seed=0):
    """유저별 (V̂, ARP) 쌍을 부트스트랩 재추출해 Spearman(V̂,ARP) 차이의 95% CI.
    점추정값(중심값)은 부트스트랩 표본들의 평균이 아니라 실제 관측 데이터에서
    계산한 차이여야 한다 (버그 수정: 이전 버전은 diffs.mean()을 반환해 부트스트랩
    분포의 평균을 점추정값으로 잘못 사용했음 — CI 폭 계산에는 문제가 없었지만
    중심값 자체가 관측값과 다를 수 있었음)."""
    observed = _spearman(vhat_arr, arp_b) - _spearman(vhat_arr, arp_a)
    rng = np.random.default_rng(seed)
    n = len(vhat_arr)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = _spearman(vhat_arr[idx], arp_a[idx])
        sb = _spearman(vhat_arr[idx], arp_b[idx])
        diffs[i] = sb - sa
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(observed), float(lo), float(hi)


csr_ptr_global = None
csr_items_global = None

def run_dualspace_one_seed(seed, train, val_gt, val_rev, test_gt, test_rev,
                            n_users, n_items, n_cat, tr_u, tr_i, pos_key, user_pos,
                            item_cat_arr, cat_items, item_meta, user_meta,
                            U_pref, I_pref):
    """이 함수는 seed 하나에 대해 고CLV Recall 보호수준(ε_high) 스윕 전체를 실행하고,
    ε_high별 결과를 리스트로 반환한다(단일 dict가 아님 — 호출부가 스윕 비교표를 만든다).
    탐색 비용이 큰 (epoch×λ×dampen_low×dampen_high) 그리드는 ε_high와 무관하므로 딱 한 번만
    계산하고, ε_high별로는 그 결과를 재필터링만 한다(추가 재학습·재평가 없음)."""
    global csr_ptr_global, csr_items_global
    set_seed(seed)  # ── 수정 #1: 재현성 확보 ──
    csr_ptr, csr_items = csr_ptr_global, csr_items_global  # 이하 명시적으로만 사용(전역 직접참조 지양)

    x_rep, x_val_u, F_u = build_user_features(train, n_users, n_cat, CFG, DCFG["is_date"])
    x_val_i = build_item_features(train, n_items, n_cat)

    vt_ckpt = Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['MODEL_LABEL']}_vt_{CFG['DATASET']}_s{seed}_{vt_fingerprint(CFG, DCFG, seed)}.pt"
    value_model, vt_best_ep, vt_best_val, vt_all_epochs = train_value_tower(
        x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
        val_gt, csr_ptr, csr_items, CFG, seed, ckpt_path=vt_ckpt)

    gate_f = compute_fbucket_gate(train, val_gt, x_val_u, item_cat_arr, F_u, user_pos,
                                   n_neg=CFG["GATE_N_NEG"], seed=seed)
    gate_f = compute_gate(F_u, gate_f)

    # ── CLV 세그먼트 유저 마스크(저CLV/고CLV): evaluate_combined의 세그먼트 정의(하위 20%/상위 20%)와
    #    반드시 "동일한 전역 임계값"을 써야 함 — 전체 n_users 기준으로 한 번만 계산해서
    #    게이트 구성과 세그먼트 채점 양쪽에 동일하게 넘긴다. (버그 수정: 이전 버전은
    #    게이트는 전체 유저 기준 임계값을, evaluate_combined 내부는 val/test로 평가되는
    #    유저 부분집합만으로 다시 계산한 별도 임계값을 썼음 — 두 "저CLV" 정의가 어긋나서
    #    dampen=0으로 완전히 꺼도 evaluate_combined이 "저CLV"라 판단하는 유저 중 일부는
    #    실제로는 안 꺼진 상태였고, 그 결과 제약이 항상 실패해 λ=0만 선택되는 문제가 있었음.)
    clv = user_meta["clv"]
    clv_valid = clv[~np.isnan(clv)]
    clv_lo_th = np.quantile(clv_valid, CFG["LOW_CLV_PCTL"])
    clv_hi_th = np.quantile(clv_valid, 1 - CFG["LOW_CLV_PCTL"])
    is_low_clv = np.where(np.isnan(clv), False, clv <= clv_lo_th)
    is_high_clv = np.where(np.isnan(clv), False, clv >= clv_hi_th)

    # ── Stage A/B 그리드(seed당 evaluate_combined() 약 800회 호출)가 매번 gt/rev로부터
    #    pos_key_sorted/ideal_rev_cumsum을 다시 만들지 않도록, val/test 각각 seed당 한 번만
    #    캐시를 만들어 _eval()이 재사용한다 (gt_가 val_gt인지 test_gt인지로 어느 캐시를 쓸지 결정).
    pos_lookup = build_pos_lookup(val_gt, val_rev, n_items)
    ideal_rev_cumsum_val = build_ideal_rev_cumsum(val_gt, val_rev)
    pos_lookup_test = build_pos_lookup(test_gt, test_rev, n_items)
    ideal_rev_cumsum_test = build_ideal_rev_cumsum(test_gt, test_rev)

    def _eval(gate, lam, gt_, rev_, Uv_, Iv_):
        is_val = gt_ is val_gt
        return evaluate_combined(U_pref, I_pref, Uv_, Iv_, gate, lam, gt_, rev_, item_meta, user_meta,
                                  CFG["K_LIST"], csr_ptr, csr_items,
                                  clv_lo_th=clv_lo_th, clv_hi_th=clv_hi_th,
                                  pos_lookup=pos_lookup if is_val else pos_lookup_test,
                                  ideal_rev_cumsum=ideal_rev_cumsum_val if is_val else ideal_rev_cumsum_test)

    # ── baseline(λ=0)은 gate/dampen/Uv/Iv 값과 완전히 무관하다: _combined_scores에서
    #    lam_base=0이면 s_val 항 전체가 사라지므로 value tower의 어떤 체크포인트를
    #    쓰든 결과가 같다. 따라서 baseline은 (아무 체크포인트로) 딱 한 번만 계산한다.
    with torch.no_grad():
        Uv0, Iv0 = value_model.encode()
    base_val_res = _eval(gate_f, 0, val_gt, val_rev, Uv0, Iv0)
    base_val_recall = base_val_res["overall"][10]["recall"]
    base_val_low_recall = base_val_res["seg"][10]["저CLV"]["recall"]
    base_val_low_revenue = base_val_res["seg"][10]["저CLV"]["revenue"]
    base_val_high_recall = base_val_res["seg"][10]["고CLV"]["recall"]
    base_val_recall50 = base_val_res["overall"][50]["recall"]
    base_val_hr10 = base_val_res["overall"][10]["hr"]
    base_val_div10 = base_val_res["overall"][10]["diversity"]

    # ── Stage A: VT epoch 스크리닝. "VT 단독 Recall 상위 K개"가 아니라 "결합 후 대표 λ에서의
    #    PWGain 상위 K개"로 뽑는다(리뷰 지적: VT 단독 기준으로 topK를 뽑으면, 결합 시 최적인
    #    epoch가 VT 단독 순위에서 밀려 그리드에 아예 못 들어올 수 있음). EPOCH_SCREEN_LAMBDA는
    #    스크리닝 전용 대표값이며 그 자체가 최종 채택 λ는 아니다 — 이후 4D 그리드가 λ 자체도
    #    다시 탐색한다. 전체 epoch를 다 이 4D 그리드에 넣으면 계산량이 지나치게 커지므로,
    #    이 1차 스크리닝으로 후보를 좁히는 절충을 택했다(완전한 unconstrained joint 탐색은 아님).
    screen_lam = CFG["EPOCH_SCREEN_LAMBDA"]
    screen_rows = []
    for ck in vt_all_epochs:
        value_model.load_state_dict(ck["state"], strict=False)  # x_val_u/x_val_i excluded from snapshot, already correct on value_model
        with torch.no_grad():
            Uv_s, Iv_s = value_model.encode()
        res_s = _eval(gate_f, screen_lam, val_gt, val_rev, Uv_s, Iv_s)
        screen_rows.append((ck["epoch"], ck["state"], res_s["overall"][10]["revenue"]))
    screen_rows.sort(key=lambda t: t[2], reverse=True)
    screened = screen_rows[:CFG["VT_TOPK_CKPTS"]]
    vt_topk = [{"epoch": e, "state": st} for e, st, _ in screened]
    print(f"\n[seed {seed}] VT epoch 스크리닝(대표 λ={screen_lam}, dampen 없음, 전체 {len(vt_all_epochs)}개 중 "
          f"결합 PWGain 상위 {len(vt_topk)}개 채택): {[c['epoch'] for c in vt_topk]} "
          f"(VT 단독 최고 epoch={vt_best_ep})")

    # ── Stage B: (epoch × dampen_low × dampen_high × λ) 그리드. ε_high와 무관하므로 한 번만 계산 ──
    #    가드레일: @10만 보호하면 @20/@50, HitRate, Diversity가 몰래 훼손될 수 있어
    #    Recall@50 · HitRate@10 · Diversity@10에도 비열등 제약(기본 1%/1%/3% 손실까지 허용)을 건다.
    #    EPS_TOL: ε=0(손실 불허)일 때 부동소수점 오차로 억울하게 탈락하지 않도록 하는 절대 허용치.
    eps = CFG["ACCURACY_EPSILON"]
    eps_low = CFG["LOW_CLV_EPSILON"]
    eps_r50 = CFG["RECALL50_EPSILON"]
    eps_hr = CFG["HR_EPSILON"]
    eps_div = CFG["DIVERSITY_EPSILON"]
    tol = CFG["EPS_TOL"]

    grid_results = {}
    for ck in vt_topk:
        ep_id = ck["epoch"]
        value_model.load_state_dict(ck["state"], strict=False)  # same as above
        with torch.no_grad():
            Uv_c, Iv_c = value_model.encode()
        for dampen_low in CFG["CLV_DAMPEN_GRID"]:
            for dampen_high in CFG["HIGH_CLV_DAMPEN_GRID"]:
                gate_seg = np.where(is_low_clv, dampen_low, np.where(is_high_clv, dampen_high, 1.0))
                gate_d = gate_f * gate_seg
                for lam in CFG["LAMBDA_GRID"]:
                    key = (ep_id, lam, dampen_low, dampen_high)
                    if lam == 0:
                        grid_results[key] = base_val_res  # λ=0은 epoch/dampen 무관, 재사용
                        continue
                    grid_results[key] = _eval(gate_d, lam, val_gt, val_rev, Uv_c, Iv_c)

    def _passes(res, eps_high_):
        r = res["overall"][10]["recall"]
        low_r = res["seg"][10]["저CLV"]["recall"]; low_rev = res["seg"][10]["저CLV"]["revenue"]
        high_r = res["seg"][10]["고CLV"]["recall"]
        r50 = res["overall"][50]["recall"]; hr10 = res["overall"][10]["hr"]; div10 = res["overall"][10]["diversity"]
        return (r >= base_val_recall * (1 - eps) - tol and
                low_r >= base_val_low_recall * (1 - eps_low) - tol and
                low_rev >= base_val_low_revenue * (1 - eps_low) - tol and
                high_r >= base_val_high_recall * (1 - eps_high_) - tol and
                r50 >= base_val_recall50 * (1 - eps_r50) - tol and
                hr10 >= base_val_hr10 * (1 - eps_hr) - tol and
                div10 >= base_val_div10 * (1 - eps_div) - tol)

    n_total = len(grid_results)
    print(f"[seed {seed}] validation 그리드 총 {n_total}개 조합 계산 완료 "
          f"(기준 Recall@10={base_val_recall:.4f}, 저CLV_R={base_val_low_recall:.4f}, "
          f"저CLV_PWGain={base_val_low_revenue:.5f}, 고CLV_R={base_val_high_recall:.4f}, "
          f"Recall@50={base_val_recall50:.4f}, HR@10={base_val_hr10:.4f}, Diversity@10={base_val_div10:.4f})")
    print(f"  고CLV Recall 보호수준(ε_high) 스윕: {CFG['HIGH_CLV_EPSILON_GRID']} "
          f"— '저CLV와 동일하게 완전 보호(0%)'가 항상 유일한 답은 아닐 수 있어(고CLV는 원래 "
          f"'가치극대화' 목표이므로 Recall 소폭 손실을 감내하고 PWGain을 취하는 것도 설계 의도에 부합할 "
          f"수 있음) 여러 보호수준의 트레이드오프를 함께 비교한다.")

    eps_rows = []
    for eps_high in CFG["HIGH_CLV_EPSILON_GRID"]:
        candidates = [key for key, res in grid_results.items() if _passes(res, eps_high)]
        if not candidates:
            fallback_ep = vt_best_ep if vt_best_ep in [c["epoch"] for c in vt_topk] else vt_topk[0]["epoch"]
            candidates = [(fallback_ep, 0, 1.0, 1.0)]
        best_ep, best_lam, best_dampen_low, best_dampen_high = max(
            candidates, key=lambda k: grid_results[k]["overall"][10]["revenue"])
        gate_arr = gate_f * np.where(is_low_clv, best_dampen_low, np.where(is_high_clv, best_dampen_high, 1.0))
        selected_state = next(ck["state"] for ck in vt_topk if ck["epoch"] == best_ep)
        value_model.load_state_dict(selected_state, strict=False)  # same as above
        with torch.no_grad():
            Uv, Iv = value_model.encode()

        sel_res = grid_results[(best_ep, best_lam, best_dampen_low, best_dampen_high)]
        r10 = sel_res["overall"][10]; low = sel_res["seg"][10]["저CLV"]; high = sel_res["seg"][10]["고CLV"]
        print(f"  [ε_high={eps_high:.0%}] 통과 {len(candidates)}/{n_total} → 선택 ep={best_ep} λ={best_lam} "
              f"d_low={best_dampen_low} d_high={best_dampen_high} | val_Recall={r10['recall']:.4f} "
              f"val_PWGain={r10['revenue']:.5f} 저CLV_R={low['recall']:.4f} 고CLV_R={high['recall']:.4f}")
        if best_dampen_low == 0.0:
            print("    ※ 저CLV dampen=0.0 → 저CLV 비개입 정책(개선 아님, 개입 제외)")
        if best_dampen_high == 0.0:
            print("    ※ 고CLV dampen=0.0 → 고CLV 비개입 정책(고CLV PWGain 이득도 함께 사라졌을 가능성)")

        # ── test는 선택된 (epoch, λ, dampen_low, dampen_high) 조합 하나만 최종 1회 평가 ──
        test_res = _eval(gate_arr, best_lam, test_gt, test_rev, Uv, Iv)
        base_test_res = _eval(gate_arr, 0, test_gt, test_rev, Uv, Iv)

        # ── 수정 #10: bootstrap CI ──
        pu_base = evaluate_combined_peruser(U_pref, I_pref, Uv, Iv, gate_arr, 0,
                                             test_gt, test_rev, item_meta, user_meta, csr_ptr, csr_items, k=10)
        pu_best = evaluate_combined_peruser(U_pref, I_pref, Uv, Iv, gate_arr, best_lam,
                                             test_gt, test_rev, item_meta, user_meta, csr_ptr, csr_items, k=10)
        assert np.array_equal(pu_base["users"], pu_best["users"]), "유저 순서 불일치"

        ci = {}
        for name, key in [("Recall", "recall"), ("NDCG", "ndcg"), ("PWGain", "revenue")]:
            mean_d, lo, hi = bootstrap_mean_diff_ci(pu_base[key], pu_best[key], n_boot=CFG["N_BOOT"], seed=seed)
            ci[name] = (mean_d, lo, hi)
        va_mean, va_lo, va_hi = bootstrap_spearman_diff_ci(pu_base["vhat"], pu_base["arp"], pu_best["arp"],
                                                            n_boot=CFG["N_BOOT"], seed=seed)
        ci["ValueAlignment"] = (va_mean, va_lo, va_hi)

        # 세그먼트별 개입 여부(저/중/고) — 두 세그먼트 dampen이 모두 0이면 사실상 중간 CLV에만 개입한 정책
        intervention_policy = {
            "low_clv": "on" if (best_lam > 0 and best_dampen_low > 0.0) else "off",
            "mid_clv": "on" if best_lam > 0 else "off",
            "high_clv": "on" if (best_lam > 0 and best_dampen_high > 0.0) else "off",
        }

        tb = base_test_res["seg"][10]; tt = test_res["seg"][10]
        print(f"    test: 전체 Recall {base_test_res['overall'][10]['recall']:.4f}→{test_res['overall'][10]['recall']:.4f} "
              f"PWGain {base_test_res['overall'][10]['revenue']:.5f}→{test_res['overall'][10]['revenue']:.5f} | "
              f"저CLV R {tb['저CLV']['recall']:.4f}→{tt['저CLV']['recall']:.4f} "
              f"PWGain {tb['저CLV']['revenue']:.5f}→{tt['저CLV']['revenue']:.5f} | "
              f"고CLV R {tb['고CLV']['recall']:.4f}→{tt['고CLV']['recall']:.4f} "
              f"PWGain {tb['고CLV']['revenue']:.5f}→{tt['고CLV']['revenue']:.5f} | "
              f"개입정책 low={intervention_policy['low_clv']} mid={intervention_policy['mid_clv']} "
              f"high={intervention_policy['high_clv']}")

        eps_rows.append(dict(
            seed=seed, high_clv_epsilon=eps_high,
            best_ep=best_ep, best_lam=best_lam,
            best_dampen_low=best_dampen_low, best_dampen_high=best_dampen_high,
            intervention_policy=intervention_policy,
            vt_best_epoch=vt_best_ep, vt_best_val_recall=vt_best_val,
            test_base=base_test_res, test_best=test_res, ci=ci,
        ))

    # ── 가장 엄격한 설정(ε_high 그리드의 첫 값 — 사용자가 처음 선택한 "완전 보호") 하나에 대해서만
    #    K=10/20/50 전체 지표와 bootstrap CI를 상세 출력한다. 나머지 ε_high는 위 압축 로그로 충분.
    #    PWGain(가격가중 적중 proxy) 주의: H&M의 price는 실제 통화 매출액이 아니라 상대적
    #    가격 스케일이다. PWGain = (1/|U|) Σ_u Σ_{i∈TopK(u)} 1[i∈GT_u]·price_i — "평가 사용자당
    #    적중상품의 정규화 price 합계"이며, "실제 매출 증가/증분매출/수익성 개선"으로 표현해서는
    #    안 되고 "가격가중 추천 적중값이 증가했다"로만 서술해야 한다.
    default_row = eps_rows[0]
    print(f"\n[seed {seed}] 상세 지표 (ε_high={default_row['high_clv_epsilon']:.0%} 기준 — "
          f"VT epoch={default_row['best_ep']}, λ={default_row['best_lam']}, "
          f"저CLV dampen={default_row['best_dampen_low']}, 고CLV dampen={default_row['best_dampen_high']})")

    def _print_full(label, res):
        print(f"  --- {label} ---")
        for k in CFG["K_LIST"]:
            m = res["overall"][k]
            print(f"   @{k:<2d} Recall {m['recall']:.4f} Prec {m['precision']:.4f} NDCG {m['ndcg']:.4f} "
                  f"HR {m['hr']:.4f} MAP {m['map']:.4f} PWGain {m['revenue']:.5f} V-NDCG {m['vndcg']:.4f} "
                  f"ARP {m['arp']:.3f} Novelty {m['novelty']:.3f} Divers {m['diversity']:.3f} "
                  f"Coverage {res['coverage'][k]:.4f} Gini {res['gini'][k]:.3f}")
        print(f"   ValueAlignment(Spearman): {res['value_alignment_spearman']:.3f}")

    _print_full("λ=0 (baseline)", default_row["test_base"])
    _print_full(f"λ={default_row['best_lam']} (선택됨)", default_row["test_best"])

    print("  95% bootstrap CI (test @10, 선택-baseline; PWGain=price-weighted gain proxy, 실제 매출 아님):")
    for name, (m, lo, hi) in default_row["ci"].items():
        sig = "" if lo <= 0 <= hi else "  ← 유의(0 미포함)"
        print(f"    Δ{name}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}")

    return eps_rows


def run_dualspace():
    d = prepare_data(CFG, DCFG)
    train, val_gt, val_rev, test_gt, test_rev = d["train"], d["val_gt"], d["val_rev"], d["test_gt"], d["test_rev"]
    n_users, n_items, n_cat = d["n_users"], d["n_items"], d["n_cat"]
    tr_u, tr_i, pos_key = d["tr_u"], d["tr_i"], d["pos_key"]
    user_pos, item_cat_arr, cat_items = d["user_pos"], d["item_cat_arr"], d["cat_items"]
    csr_ptr, csr_items, adj = d["csr_ptr"], d["csr_items"], d["adj"]

    global csr_ptr_global, csr_items_global
    csr_ptr_global, csr_items_global = csr_ptr, csr_items

    # ── M1은 항상 CFG["RUN_TAG"] 하나의 체크포인트만 가리킨다 (MODEL 키가 없어졌으므로
    #    이전처럼 m1_cfg = {**CFG, "MODEL":"M1"}로 별도 dict를 만들 필요가 없음 —
    #    CFG 자체가 이미 M1 전용 설정이다) ──
    m1_path = Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['RUN_TAG']}.pt"
    assert m1_path.exists(), f"M1 체크포인트 없음: {m1_path}\n먼저 main()으로 M1을 학습하세요."
    m1_state = torch.load(m1_path, map_location=DEVICE, weights_only=False)["best_state"]
    pref_model = LightGCNCLV(n_users, n_items, CFG, adj).to(DEVICE)
    pref_model.load_state_dict(m1_state)
    for p in pref_model.parameters(): p.requires_grad_(False)
    with torch.no_grad():
        U_pref, I_pref, _, _ = pref_model.propagate()
    print(f"✓ M1 체크포인트 로드 완료 ({m1_path.name}), z^pref 동결")

    item_meta = build_item_meta(train, n_items)
    clv, vhat = compute_clv_vhat(train, n_users, CFG, DCFG["is_date"])
    user_meta = dict(clv=clv, vhat=vhat)

    # ── Value Tower 3-seed robustness test conditional on fixed M1(seed=CFG['SEED']) ──
    # 주의: M1(z^pref)은 항상 CFG['SEED'](=42) 체크포인트 하나만 사용한다. SEED_LIST의
    # 42/43/44는 value tower 초기화·negative sampling에만 적용되는 시드다. 즉 아래 결과는
    # "전체 모델(M1+value tower) 다중 시드"가 아니라 "고정 M1 위에서 value tower만 다중
    # 시드로 재현성을 확인한 것"이다. 전체 모델 다중 시드를 주장하려면 M1도 seed
    # 42/43/44로 각각 재학습해 같은 시드의 value tower와 결합해야 한다(추가 M1 학습 필요,
    # 미실행).
    eps_grid = CFG["HIGH_CLV_EPSILON_GRID"]
    all_results = []  # all_results[seed_idx][eps_idx] = 결과 dict (seed × ε_high)
    for seed in CFG["SEED_LIST"]:
        res = run_dualspace_one_seed(seed, train, val_gt, val_rev, test_gt, test_rev,
                                      n_users, n_items, n_cat, tr_u, tr_i, pos_key, user_pos,
                                      item_cat_arr, cat_items, item_meta, user_meta, U_pref, I_pref)
        all_results.append(res)

    print(f"\n{'='*100}")
    print(f"Value Tower {len(CFG['SEED_LIST'])}-seed robustness test conditional on fixed M1(seed={CFG['SEED']})")
    print(f"({CFG['MODEL_LABEL']}, 세그먼트별 차등 개입정책(dampen_low/dampen_high) 적용 — "
          f"dampen=0은 '개선'이 아니라 해당 세그먼트에 가치 개입을 아예 하지 않았다는 뜻)")
    print(f"{'='*100}")
    for ei, eps_high in enumerate(eps_grid):
        print(f"\n  [ε_high={eps_high:.0%}] seed별 결과:")
        print(f"  {'seed':>5} {'VTep':>5} {'best_λ':>8} {'d_low':>6} {'d_high':>7} {'ΔRecall':>10} "
              f"{'ΔPWGain':>11} {'ΔValueAlign':>12} {'저CLV_R':>13} {'고CLV_R':>13}")
        for seed_res in all_results:
            r = seed_res[ei]
            dr = r["ci"]["Recall"][0]; drev = r["ci"]["PWGain"][0]; dva = r["ci"]["ValueAlignment"][0]
            b_lo = r["test_base"]["seg"][10]["저CLV"]; t_lo = r["test_best"]["seg"][10]["저CLV"]
            b_hi = r["test_base"]["seg"][10]["고CLV"]; t_hi = r["test_best"]["seg"][10]["고CLV"]
            print(f"  {r['seed']:>5} {r['best_ep']:>5} {r['best_lam']:>8} {r['best_dampen_low']:>6} "
                  f"{r['best_dampen_high']:>7} {dr:>+10.5f} {drev:>+11.5f} {dva:>+12.4f} "
                  f"{b_lo['recall']:.4f}→{t_lo['recall']:.4f} {b_hi['recall']:.4f}→{t_hi['recall']:.4f}")

    # ── 고CLV Recall 보호수준(ε_high) 비교표: 시드 평균±SD (사용자 요청 — 여러 보호정도 비교분석) ──
    print(f"\n{'='*100}\n고CLV Recall 보호수준(ε_high) 비교 — 시드 평균 ± SD\n{'='*100}")
    print(f"  {'ε_high':>7} {'ΔRecall(전체)':>16} {'ΔPWGain(전체)':>16} {'고CLV_R b→t':>16} "
          f"{'고CLV_PWGain b→t':>20} {'d_low(seed간)':>14} {'d_high(seed간)':>14}")
    for ei, eps_high in enumerate(eps_grid):
        rows_i = [seed_res[ei] for seed_res in all_results]
        dr_arr = np.array([r["ci"]["Recall"][0] for r in rows_i])
        drev_arr = np.array([r["ci"]["PWGain"][0] for r in rows_i])
        high_b_r = np.mean([r["test_base"]["seg"][10]["고CLV"]["recall"] for r in rows_i])
        high_t_r = np.mean([r["test_best"]["seg"][10]["고CLV"]["recall"] for r in rows_i])
        high_b_rev = np.mean([r["test_base"]["seg"][10]["고CLV"]["revenue"] for r in rows_i])
        high_t_rev = np.mean([r["test_best"]["seg"][10]["고CLV"]["revenue"] for r in rows_i])
        dlows = [r["best_dampen_low"] for r in rows_i]; dhighs = [r["best_dampen_high"] for r in rows_i]
        print(f"  {eps_high:>6.0%} {dr_arr.mean():>+8.5f}±{dr_arr.std():.5f} "
              f"{drev_arr.mean():>+8.5f}±{drev_arr.std():.5f} "
              f"{high_b_r:.4f}→{high_t_r:.4f}      {high_b_rev:.5f}→{high_t_rev:.5f}       "
              f"{dlows}      {dhighs}")
    print("  ※ PWGain = price-weighted gain proxy(추천 적중상품 price 합의 사용자 평균). "
          "H&M price는 실통화 매출액이 아니므로 실제 비즈니스 매출/증분매출로 해석 금지.")
    print("  ※ 위 시드간 변동성은 '고정 M1(seed=%d) 조건부' value tower 강건성이며, "
          "전체 모델의 불확실성이 아님." % CFG["SEED"])
    print("  ※ 탐색적 결과 주의: H&M test는 이번 연구 과정에서 이전 버전들의 결과를 보며 모델/제약을 "
          "반복 수정해온 과정에 이미 노출되어 있어, 엄밀한 의미의 최초 1회 confirmatory test가 아니다. "
          "본 결과는 개발/탐색 결과로 규정하고, 최종 확증은 지금 확정한 설정(λ grid·제약·gate 정의)을 "
          "그대로 고정해 Dunnhumby에서 재검증하는 것을 권장한다.")

    out_path = Path(CFG["OUT_DIR"]) / f"result_{CFG['MODEL_LABEL']}_{CFG['DATASET']}_multiseed.json"
    payload = []
    for seed_res in all_results:
        for r in seed_res:
            payload.append({
                "seed": r["seed"], "high_clv_epsilon": r["high_clv_epsilon"],
                "best_lambda": r["best_lam"], "vt_selected_epoch": r["best_ep"],
                "vt_standalone_best_epoch": r["vt_best_epoch"], "vt_standalone_best_val_recall": r["vt_best_val_recall"],
                "best_low_clv_dampen": r["best_dampen_low"], "best_high_clv_dampen": r["best_dampen_high"],
                "intervention_policy": r["intervention_policy"],
                "test_base_recall10": r["test_base"]["overall"][10]["recall"],
                "test_best_recall10": r["test_best"]["overall"][10]["recall"],
                "test_base_pwgain10": r["test_base"]["overall"][10]["revenue"],
                "test_best_pwgain10": r["test_best"]["overall"][10]["revenue"],
                "test_base_value_alignment": r["test_base"]["value_alignment_spearman"],
                "test_best_value_alignment": r["test_best"]["value_alignment_spearman"],
                "ci": {k: {"mean": v[0], "lo": v[1], "hi": v[2]} for k, v in r["ci"].items()},
                "segment": {
                    sg: {
                        "base_recall10": r["test_base"]["seg"][10][sg]["recall"],
                        "best_recall10": r["test_best"]["seg"][10][sg]["recall"],
                        "base_pwgain10": r["test_base"]["seg"][10][sg]["revenue"],
                        "best_pwgain10": r["test_best"]["seg"][10][sg]["revenue"],
                        "base_arp": r["test_base"]["seg"][10][sg]["arp"],
                        "best_arp": r["test_best"]["seg"][10][sg]["arp"],
                    } for sg in ["저CLV", "고CLV"]
                },
            })
    meta = {
        "model_label": CFG["MODEL_LABEL"],
        "m1_seed_fixed": CFG["SEED"],
        "value_tower_seeds": CFG["SEED_LIST"],
        "high_clv_epsilon_grid": eps_grid,
        "pwgain_formula": "PWGain@K = mean_u sum_{i in TopK(u)} 1[i in GT_u] * price_i (평가 사용자당 적중상품 price 합의 평균)",
        "note_multiseed": "M1(z^pref)은 고정 seed 체크포인트 하나. 시드는 value tower(초기화+negative sampling)에만 적용됨 — 전체 모델 다중 시드 아님.",
        "note_pwgain": "revenue/pwgain 필드는 가격가중 추천 적중값(price-weighted gain proxy)이며 실제 통화 매출·증분매출·수익성 지표가 아님. "
                       "허용 표현: '가격가중 추천 적중값이 증가했다'. 금지 표현: 실제 매출 증가/고객당 매출 증가/증분 매출/수익성 개선.",
        "note_low_clv": "best_low_clv_dampen=0.0인 행은 저CLV에 가치 개입을 적용하지 않은 것(비개입 정책)이며, 저CLV 지표 유지가 '개선'을 의미하지 않음.",
        "note_high_clv_epsilon": "고CLV Recall 보호수준(ε_high)을 0/1/2/5%로 스윕한 비교결과 — 원 설계의도(저CLV=손실회피, 고CLV=가치극대화)상 "
                                  "고CLV에서 Recall을 소폭 내주고 PWGain을 취하는 트레이드오프가 반드시 배제해야 할 문제는 아닐 수 있어, "
                                  "단일 정책을 강제하지 않고 비교표로 남김.",
        "note_epoch_screening": "value tower 전체 epoch를 저장하되, 4D 그리드에 넣을 상위 VT_TOPK_CKPTS개는 VT 단독 Recall이 아니라 "
                                 "대표 λ(EPOCH_SCREEN_LAMBDA)에서의 결합 PWGain으로 스크리닝함 — 완전한 unconstrained joint 탐색은 아니며 "
                                 "스크리닝 λ가 고정된 절충안임.",
        "note_exploratory": "본 H&M 결과는 연구 과정에서 이전 버전 결과를 보며 반복적으로 모델/제약을 수정해 온 개발 결과이며, "
                             "엄밀한 최초 1회 confirmatory test가 아님. 최종 확증은 여기서 고정한 설정을 그대로 Dunnhumby에 "
                             "적용해 재검증하는 것을 권장.",
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": payload}, f, indent=2, default=float, ensure_ascii=False)
    print(f"\n저장 → {out_path}")
    return all_results


# M1이 이미 학습되어 있으면(체크포인트 존재) 아래 main()은 즉시 복원되고 끝납니다.
# main()  # 필요시 주석 해제 (M1을 처음부터 학습해야 하는 경우)
results = run_dualspace()

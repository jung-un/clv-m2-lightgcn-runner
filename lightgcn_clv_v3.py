"""LightGCN + CLV 이중공간(v3) — 처음부터 다시 쓴 버전.

설계 원칙: **"임베딩을 분리하고 CLV에 따라 반영 정도를 조정한다" 외에는 아무 조건도 넣지 않는다.**

    S(u,i) = <z_u^pref, z_i^pref> + λ · gate(u) · <z_u^value, z_i^value>
    gate(u) = percentile_rank(CLV_u)            # 0~1, 선형
    CLV_u   = N̂_u × V̂_u
    N̂_u     = mean(F_p, T_p, R_p)                # 전부 백분위
    V̂_u     = mean(AOV_p, Prem_p)                # 축소추정 없음

v2(lightgcn_m2v2_clv_embed.py)에서 **제거한 것** — 전부 이전 담당자가 임의로 넣었던 장치다:
  - 가드레일 4종(Recall@10 무손실 / Recall@50 / HR / Diversity 허용손실)
  - 목적함수(가드레일 통과 조합 중 가격가중 적중값 최대인 것을 선택)
  - Stage A epoch 스크리닝(대표 λ로 VT epoch을 5개로 추림)
  - 후보가 없을 때 λ=0으로 떨어지는 fallback
  - AOV/Prem 축소추정(shrinkage)
  - gate의 제곱(power=2.0)
  - hard negative 샘플링(50%)
→ λ는 고르지 않고 **스윕 전체를 결과로 보고**한다. λ=0이 곧 baseline이므로 비교도 자동이다.

v2에서 **그대로 가져온 것** — 이미 검증·디버깅이 끝난 부분이다:
  - 데이터 처리(완전중복 제거, 구매 건=BASKET_ID/(고객,날짜), 단가=금액÷수량,
    train 구간 기준 필터, 재구매쌍 제거)
  - 지표 계산(score_topk 계열, Recall/Precision/NDCG/HR/MAP/가격가중적중값/
    V-NDCG/평균가격백분위/Novelty/Diversity/Coverage/Gini/정렬도)

두 아키텍처를 CFG["ARCH"] 하나로 전환해 비교한다. 차이는 **학습 절차 하나뿐**이다:
  - "joint"     : LightGCN 1회 학습. 선호·가치 블록이 같은 BPR 손실로 동시에 학습됨.
  - "two_stage" : 선호 블록만 먼저 학습(=순수 LightGCN) → 동결 → 가치 블록만 학습.
가치 임베딩의 정체(CLV 변수 MLP), gate, λ, 입력 특징은 두 안이 완전히 동일하다.
"""
import os, json, math, time, random, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

IN_COLAB = os.path.exists("/content")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════
# SCHEMA: 데이터셋의 "고정 사실"만 (경로/컬럼명/날짜형여부)
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
        # 주문 식별자가 없고 시간 해상도가 날짜뿐이라 (고객, 날짜)를 구매 1건으로 본다.
        # price가 이미 상품 1개당 단가라 나눌 수량이 없다.
        "basket_col": None, "qty_col": None,
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
        # 한 행 = BASKET_ID × PRODUCT_ID 구매 라인(라인이 장바구니의 약 9.4배).
        # QUANTITY≠1인 행이 약 21%라 SALES_VALUE는 단가가 아니다.
        "basket_col": "BASKET_ID", "qty_col": "QUANTITY",
        "is_date": False,
    },
}


# ═══════════════════════════════════════════════════════════════════
# CFG
#   [기본]  = LightGCN 원논문/기존 실험에서 그대로 가져온 값
#   [선택]  = 이번 설계에서 사용자가 명시적으로 정한 값
#   [임의]  = 정하지 않으면 코드가 안 돌아가서 부득이 고른 값 (바꾸려면 여기만 수정)
# ═══════════════════════════════════════════════════════════════════
CFG = {
    # ── 실행 대상 ──
    "DATASET": "hm",              # [선택] "hm" | "dunnhumby"
    "ARCH": "joint",              # [선택] "joint" | "two_stage" — 이 둘을 비교하는 게 목적
    "MODEL_LABEL": "v3",
    "SEED_LIST": [42, 43, 44],    # [기본] 3시드

    # ── 데이터 ──
    "OUT_DIR": None,              # 아래에서 DATASET 확정 후 채움
    "WINDOW_DAYS": 60,            # [선택] hm=60(2개월), dunnhumby=None(전체 ~2년)
    "VAL_DAYS": 7, "TEST_DAYS": 7,          # [기본]
    "MIN_USER_INTER": 1, "MIN_ITEM_INTER": 1,  # [기본] 필터 없음

    # ── 선호 임베딩(z^pref) = 표준 LightGCN ──
    "DIM": 64, "N_LAYERS": 2,     # [기본] 기존 실험값 유지

    # ── 가치 임베딩(z^value) = CLV 변수 MLP, 그래프 전파 없음 ──
    "D_VALUE": 16,                # [임의] 가치 임베딩 차원
    "MLP_HIDDEN": 32,             # [임의] MLP 은닉 차원
    "CAT_EMB_DIM": 16,            # [임의] 카테고리 ID 임베딩 차원(원핫 대신)

    # ── 학습 (LightGCN 관례 + 기존 실험값) ──
    "BATCH_SIZE": 8192, "LR": 5e-4, "WD": 1e-3,   # [기본] 기존값 유지
    "EPOCHS": 100, "EARLY_STOP": 20,              # [기본]
    "SELECT_METRIC": "recall@10",                 # [기본] 학습 조기종료 기준
    "EVAL_BATCH": 1024,                           # [기본] 평가 시 유저 배치
    # negative 샘플링은 **균등 무작위**(LightGCN 원논문). v2의 hard negative 50%는 제거.
    "LAMBDA_TRAIN": 1.0,          # [임의] 학습 중 결합 강도. λ는 채점 시점 파라미터라
                                  # 학습은 한 번만 하고 아래 LAMBDA_SWEEP으로 채점한다.

    # ── CLV 파생 변수 ──
    "PREMIUM_THR": 0.8,           # [임의] 단가 상위 20%를 "고가 상품"으로 판정
    # 축소추정(shrinkage) 없음 — 지도교수님 자료 정의 그대로

    # ── λ 스윕: 고르지 않고 전부 보고한다 ──
    "LAMBDA_SWEEP": [0.0, 0.1, 0.5, 1.0, 2.0],   # [임의] 값 목록

    # ── 평가 ──
    "K_LIST": [10, 20, 50],       # [기본]
    "N_BOOT": 2000,               # [기본] 부트스트랩 반복
    "SEG_EDGES": (0.2, 0.8),      # [선택] 하위20% / 중위60% / 상위20%
}
CFG["OUT_DIR"] = (f"/content/drive/MyDrive/논문/data/results_v3_{CFG['DATASET']}" if IN_COLAB
                  else f"/Users/jungun/Workspace/논문준비/data/results_v3_{CFG['DATASET']}")
DCFG = SCHEMA[CFG["DATASET"]]

_METS = ["recall", "precision", "ndcg", "hr", "map", "revenue", "vndcg", "arp", "novelty", "diversity"]
SEG_NAMES = ["저CLV", "중CLV", "고CLV"]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def run_tag(cfg, dcfg, seed):
    """설정이 바뀌면 체크포인트 파일명이 달라지게 하는 해시."""
    keys = ["DATASET", "ARCH", "DIM", "N_LAYERS", "D_VALUE", "MLP_HIDDEN", "CAT_EMB_DIM",
            "BATCH_SIZE", "LR", "WD", "EPOCHS", "EARLY_STOP", "WINDOW_DAYS",
            "VAL_DAYS", "TEST_DAYS", "MIN_USER_INTER", "MIN_ITEM_INTER",
            "PREMIUM_THR", "LAMBDA_TRAIN"]
    payload = {k: cfg[k] for k in keys}
    payload.update(category_col=dcfg["category_col"], seed=seed)
    h = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"v3_{cfg['ARCH']}_{cfg['DATASET']}_s{seed}_{h}"


# ═══════════════════════════════════════════════════════════════════
# 데이터 — v2에서 그대로 이식 (검증 완료)
# ═══════════════════════════════════════════════════════════════════
def load_transactions(dcfg):
    """원본 로드 + 파생 컬럼 2개.
    b_raw = 구매 건(장바구니) 식별자. 없으면 만들지 않고 이후 (u_idx, t)로 묶는다.
    up    = 상품 1개당 단가 = 라인 금액 / 수량(0 이하는 1로 간주)."""
    tx = pd.read_parquet(dcfg["tx_path"]) if dcfg["tx_path"].endswith(".parquet") else pd.read_csv(dcfg["tx_path"])
    ren = {dcfg["user_col"]: "u_raw", dcfg["item_col"]: "i_raw",
           dcfg["time_col"]: "t", dcfg["value_col"]: "v"}
    if dcfg["basket_col"]:
        ren[dcfg["basket_col"]] = "b_raw"
    tx = tx.rename(columns=ren)
    if dcfg["is_date"]:
        tx["t"] = pd.to_datetime(tx["t"])
        tx["i_raw"] = tx["i_raw"].astype(str)
    tx = tx.drop_duplicates()
    q = dcfg["qty_col"]
    tx["up"] = (tx["v"] / tx[q].clip(lower=1) if q else tx["v"]).astype(np.float32)
    unit = "BASKET_ID" if dcfg["basket_col"] else "(고객,날짜)"
    print(f"원본 {len(tx):,}건 (완전중복 제거 완료) | 구매 1건 단위={unit} | 단가=금액/{q or 1}")
    return tx


def prepare_data(cfg, dcfg):
    tx = load_transactions(dcfg)
    if cfg["WINDOW_DAYS"]:
        t_max = tx["t"].max()
        delta = pd.Timedelta(days=cfg["WINDOW_DAYS"]) if dcfg["is_date"] else cfg["WINDOW_DAYS"]
        tx = tx[tx["t"] >= t_max - delta].copy()
        print(f"최근 {cfg['WINDOW_DAYS']}일 사용: {len(tx):,}건")

    meta = pd.read_csv(dcfg["item_meta_path"], dtype={dcfg["item_key_col"]: str} if dcfg["is_date"] else None)
    meta = meta.rename(columns={dcfg["item_key_col"]: "i_raw", dcfg["category_col"]: "cat_raw"})
    tx = tx.merge(meta[["i_raw", "cat_raw"]].drop_duplicates("i_raw"), on="i_raw", how="left")
    tx["cat_raw"] = tx["cat_raw"].fillna("UNKNOWN")

    t_max = tx["t"].max()
    day = lambda n: (pd.Timedelta(days=n) if dcfg["is_date"] else n)
    test_start = t_max - day(cfg["TEST_DAYS"])
    val_start = test_start - day(cfg["VAL_DAYS"])

    # 필터는 train 구간만 기준으로 — val/test까지 세면 미래 정보로 cold-start를 살리는 누수
    tp = tx[tx["t"] <= val_start]
    uc, ic = tp["u_raw"].value_counts(), tp["i_raw"].value_counts()
    keep_u = set(uc[uc >= cfg["MIN_USER_INTER"]].index)
    keep_i = set(ic[ic >= cfg["MIN_ITEM_INTER"]].index)
    tx = tx[tx["u_raw"].isin(keep_u) & tx["i_raw"].isin(keep_i)].copy()
    print(f"필터(train 구간 기준) 후: {len(tx):,}건")

    uids = np.sort(tx["u_raw"].unique()); iids = np.sort(tx["i_raw"].unique())
    cats = sorted(tx["cat_raw"].unique())
    tx["u_idx"] = tx["u_raw"].map({u: k for k, u in enumerate(uids)}).astype("int32")
    tx["i_idx"] = tx["i_raw"].map({i: k for k, i in enumerate(iids)}).astype("int32")
    tx["cat_idx"] = tx["cat_raw"].map({c: k for k, c in enumerate(cats)}).astype("int32")
    n_users, n_items, n_cat = len(uids), len(iids), len(cats)
    print(f"유저 {n_users:,} | 아이템 {n_items:,} | 카테고리({dcfg['category_col']}) {n_cat:,}")

    train = tx[tx["t"] <= val_start].copy()
    val = tx[(tx["t"] > val_start) & (tx["t"] <= test_start)].copy()
    test = tx[tx["t"] > test_start].copy()

    train_users = set(train.u_idx.unique()); train_items = set(train.i_idx.unique())
    train_pair_key = np.unique(train.u_idx.values.astype(np.int64) * n_items + train.i_idx.values)

    def build_eval(df, name):
        d = df[df.u_idx.isin(train_users) & df.i_idx.isin(train_items)]
        key = d.u_idx.values.astype(np.int64) * n_items + d.i_idx.values
        pos = np.clip(np.searchsorted(train_pair_key, key), 0, len(train_pair_key) - 1)
        d = d[train_pair_key[pos] != key]          # 재구매쌍 제거 (교수님 지침)
        agg = d.groupby(["u_idx", "i_idx"], sort=False)["v"].sum().reset_index()
        gt, rev = {}, {}
        for u, g in agg.groupby("u_idx", sort=False):
            gt[u] = g.i_idx.values.astype(np.int32); rev[u] = g.v.values.astype(np.float32)
        print(f"  {name}: 평가유저 {len(gt):,}명, 정답 {len(agg):,}쌍")
        return gt, rev

    val_gt, val_rev = build_eval(val, "Val ")
    test_gt, test_rev = build_eval(test, "Test")

    # 그래프(선호 임베딩 전파용) — 표준 LightGCN 정규화 인접행렬
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

    return dict(train=train, val_gt=val_gt, val_rev=val_rev, test_gt=test_gt, test_rev=test_rev,
                adj=adj, pos_key=edge_key, tr_u=tu, tr_i=ti, csr_ptr=csr_ptr, csr_items=csr_items,
                n_users=n_users, n_items=n_items, n_cat=n_cat)


# ═══════════════════════════════════════════════════════════════════
# CLV — 지도교수님 자료 정의 그대로 (축소추정 없음)
# ═══════════════════════════════════════════════════════════════════
def clv_features(train, n_users, cfg, is_date):
    """반환: x_val_u [n_users, 5] (F_p,T_p,R_p,AOV_p,Prem_p), clv [n_users] (=N̂×V̂).

    F/AOV는 **구매 건(장바구니) 단위**다. 한 행은 주문이 아니라 상품 라인이므로
    행 수를 세면 F가 '구매 횟수'가 아니라 '산 상품 개수'가 된다.
    Prem은 **단가** 기준 — 라인 금액으로 재면 대량구매가 고가구매로 오인된다.
    """
    win_end = train["t"].max()
    span = win_end - train["t"].min()
    win_days = max((span.days if is_date else span), 1)

    bkeys = ["u_idx", "b_raw"] if "b_raw" in train.columns else ["u_idx", "t"]
    basket = train.groupby(bkeys, sort=False).agg(bval=("v", "sum"), btime=("t", "max"))
    gb = basket.groupby(level="u_idx", sort=False)
    g = pd.DataFrame({"F": gb.size(), "first": gb["btime"].min(),
                      "last": gb["btime"].max(), "AOV": gb["bval"].mean()})

    prem_flag = (train["up"].rank(pct=True) > cfg["PREMIUM_THR"]).astype("int8")
    g = g.join(train.assign(_p=prem_flag).groupby("u_idx")["_p"].agg(prem="sum", n_line="count"))
    g["Prem"] = g["prem"] / g["n_line"]          # 축소추정 없이 관측 비율 그대로

    if is_date:
        T = (g["last"] - g["first"]).dt.days
        R = 1 - ((win_end - g["last"]).dt.days / win_days)
    else:
        T = g["last"] - g["first"]
        R = 1 - ((win_end - g["last"]) / win_days)

    g["F_p"] = g["F"].rank(pct=True); g["T_p"] = T.rank(pct=True); g["R_p"] = R.rank(pct=True)
    g["AOV_p"] = g["AOV"].rank(pct=True); g["Prem_p"] = g["Prem"].rank(pct=True)
    g["N_hat"] = g[["F_p", "T_p", "R_p"]].mean(axis=1)
    g["V_hat"] = g[["AOV_p", "Prem_p"]].mean(axis=1)
    g["CLV"] = g["N_hat"] * g["V_hat"]

    x = np.full((n_users, 5), 0.5, np.float32)
    x[g.index.values] = g[["F_p", "T_p", "R_p", "AOV_p", "Prem_p"]].values.astype(np.float32)
    clv = np.full(n_users, np.nan)
    clv[g.index.values] = g["CLV"].values
    print(f"  유저 가치 입력: [F_p,T_p,R_p,AOV_p,Prem_p] {x.shape} (축소추정 없음)")
    return x, clv


def clv_gate(clv):
    """gate(u) = percentile_rank(CLV_u). 선형 0~1. CLV가 없는 유저(train 이력 없음)는 0."""
    s = pd.Series(clv)
    gate = s.rank(pct=True).to_numpy(np.float32)
    gate[np.isnan(clv)] = 0.0
    return gate


def item_value_features(train, n_items):
    """아이템 가치 입력 = [가격백분위, 카테고리내 가격순위] + 카테고리 ID(임베딩용).
    가격은 라인 금액이 아니라 **단가(up)** 중앙값 기준."""
    g = train.groupby("i_idx").agg(med=("up", "median"))
    price_pct = np.full(n_items, 0.5, np.float32)
    price_pct[g.index.values] = g["med"].rank(pct=True).to_numpy(np.float32)

    cat_of = train.groupby("i_idx")["cat_idx"].agg(lambda s: s.mode().iat[0])
    within = np.full(n_items, 0.5, np.float32)
    joined = g.join(cat_of.rename("cat_idx"))
    for c, sub in joined.groupby("cat_idx"):
        within[sub.index.values] = sub["med"].rank(pct=True).to_numpy(np.float32)

    cat_arr = np.zeros(n_items, dtype=np.int64)
    cat_arr[cat_of.index.values] = cat_of.values
    x = np.stack([price_pct, within], axis=1)
    print(f"  아이템 가치 입력: [가격백분위, 카테고리내 가격순위] {x.shape} + 카테고리 임베딩")
    return x.astype(np.float32), cat_arr


def item_meta(train, n_items):
    """평가용 아이템 부가정보 (가격백분위/인기도/카테고리)."""
    pop = np.bincount(train["i_idx"].values.astype(np.int64), minlength=n_items).astype(np.float64)
    med = train.groupby("i_idx")["up"].median()
    price_pct = np.full(n_items, 0.5, np.float64)
    price_pct[med.index.values] = med.rank(pct=True).values
    cat = np.full(n_items, -1, np.int64)
    cmap = train.groupby("i_idx")["cat_idx"].agg(lambda s: s.mode().iat[0])
    cat[cmap.index.values] = cmap.values
    return dict(price_pct=price_pct, pop_prob=pop / max(pop.sum(), 1.0), cat=cat)


# ═══════════════════════════════════════════════════════════════════
# 모델
# ═══════════════════════════════════════════════════════════════════
class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim), nn.LeakyReLU(0.2))
    def forward(self, x):
        return self.net(x)


class DualSpaceLightGCN(nn.Module):
    """선호 공간과 가치 공간을 분리한 LightGCN.

    z^pref : 자유 임베딩 + **표준 LightGCN 전파**(협업 공간)
    z^value: CLV 변수/가격 속성 MLP 출력, **전파 없음**(속성 공간)

    가치 블록에 전파를 태우지 않는 이유는 두 가지다. (1) 전파를 태우면 유저의 가치
    임베딩이 "이 유저가 산 물건들의 평균"이 되는데 그건 z^pref가 이미 하는 일이라
    같은 협업 신호를 두 번 학습하게 된다. (2) two_stage 쪽 가치 블록은 구조상 MLP라
    전파가 없으므로, joint만 태우면 두 아키텍처가 두 가지 점에서 달라져 비교가
    해석되지 않는다.
    """
    def __init__(self, n_users, n_items, n_cat, x_val_u, x_item, item_cat, cfg, adj):
        super().__init__()
        self.n_users, self.n_items, self.cfg, self.adj = n_users, n_items, cfg, adj
        self.E_u = nn.Embedding(n_users, cfg["DIM"])
        self.E_i = nn.Embedding(n_items, cfg["DIM"])
        nn.init.normal_(self.E_u.weight, std=0.1); nn.init.normal_(self.E_i.weight, std=0.1)

        self.cat_emb = nn.Embedding(n_cat, cfg["CAT_EMB_DIM"])
        nn.init.normal_(self.cat_emb.weight, std=0.1)
        self.mlp_u = MLP(x_val_u.shape[1], cfg["MLP_HIDDEN"], cfg["D_VALUE"])
        self.mlp_i = MLP(x_item.shape[1] + cfg["CAT_EMB_DIM"], cfg["MLP_HIDDEN"], cfg["D_VALUE"])

        self.register_buffer("x_val_u", torch.from_numpy(x_val_u))
        self.register_buffer("x_item", torch.from_numpy(x_item))
        self.register_buffer("item_cat", torch.from_numpy(item_cat))

    def pref_params(self):
        return list(self.E_u.parameters()) + list(self.E_i.parameters())

    def value_params(self):
        return (list(self.cat_emb.parameters()) + list(self.mlp_u.parameters())
                + list(self.mlp_i.parameters()))

    def propagate_pref(self):
        """표준 LightGCN: 층별 임베딩의 평균."""
        x = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        out = x
        for _ in range(self.cfg["N_LAYERS"]):
            x = torch.sparse.mm(self.adj, x)
            out = out + x
        out = out / (self.cfg["N_LAYERS"] + 1)
        return out[:self.n_users], out[self.n_users:]

    def value_emb(self):
        zu = self.mlp_u(self.x_val_u)
        zi = self.mlp_i(torch.cat([self.x_item, self.cat_emb(self.item_cat)], dim=1))
        return zu, zi

    def embeddings(self):
        Up, Ip = self.propagate_pref()
        Uv, Iv = self.value_emb()
        return Up, Ip, Uv, Iv

    def bpr_loss(self, u, i, j, gate_t, lam):
        Up, Ip, Uv, Iv = self.embeddings()
        gu = gate_t[u].unsqueeze(1)
        su = Up[u]; sv = Uv[u] * (lam * gu)
        pos = (su * Ip[i]).sum(1) + (sv * Iv[i]).sum(1)
        neg = (su * Ip[j]).sum(1) + (sv * Iv[j]).sum(1)
        return -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()


# ═══════════════════════════════════════════════════════════════════
# 지표 계산 — v2에서 그대로 이식
# ═══════════════════════════════════════════════════════════════════
def _gini(x):
    x = np.sort(np.asarray(x, dtype=np.float64)); n = len(x)
    if n == 0 or x.sum() == 0: return 0.0
    c = np.cumsum(x); return float((n + 1 - 2 * (c / c[-1]).sum()) / n)


def _spearman(a, b):
    if len(a) < 3: return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if np.isnan(r) else float(r)


def build_pos_lookup(gt, rev, n_items):
    keys, revs = [], []
    for u, items in gt.items():
        keys.append(u * n_items + items.astype(np.int64)); revs.append(rev[u])
    keys = np.concatenate(keys); revs = np.concatenate(revs).astype(np.float64)
    order = np.argsort(keys, kind="stable")
    return keys[order], revs[order]


def build_ideal_rev_cumsum(gt, rev):
    out = {}
    for u in gt:
        sorted_rev = np.sort(rev[u])[::-1]
        disc = 1.0 / np.log2(np.arange(2, len(sorted_rev) + 2))
        out[u] = np.cumsum(sorted_rev * disc)
    return out


def score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
               P_arr, price_pct, item_nov, cat_arr, ideal_rev_cumsum):
    """배치 전체를 numpy 벡터 연산으로 채점. 반환 {k: {metric: [batch]}}."""
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
        idcg = np.where(Pk > 0, np.cumsum(disc_k)[np.maximum(Pk, 1) - 1], 0.0)
        dcg = (hit_k * disc_k).sum(axis=1)
        cum_hits = np.cumsum(hit_k, axis=1)
        map_num = (cum_hits * hit_k / np.arange(1, k + 1)).sum(axis=1)
        idcgv = np.array([ideal_rev_cumsum[u][min(len(ideal_rev_cumsum[u]), k) - 1]
                           if len(ideal_rev_cumsum[u]) > 0 else 0.0 for u in bu])
        dcgv = (gain_k * disc_k).sum(axis=1)
        cat_k = cat_row[:, :k]
        div = np.array([len(np.unique(row)) / k for row in cat_k])
        out[k] = {
            "recall": np.where(P_batch > 0, nh / np.maximum(P_batch, 1), 0.0),
            "precision": nh / k,
            "ndcg": np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0),
            "hr": (nh > 0).astype(np.float64),
            "map": np.where(Pk > 0, map_num / np.maximum(Pk, 1), 0.0),
            "revenue": gain_k.sum(axis=1),
            "vndcg": np.where(idcgv > 0, dcgv / np.maximum(idcgv, 1e-12), 0.0),
            "arp": price_row[:, :k].mean(axis=1),
            "novelty": nov_row[:, :k].mean(axis=1),
            "diversity": div,
        }
    return out


@torch.no_grad()
def evaluate(model, lam, gate_t, gt, rev, meta, clv, ks, csr_ptr, csr_items, cfg,
             seg_edges, per_user=False):
    """λ 하나로 채점. per_user=True면 부트스트랩용 유저별 배열도 함께 반환."""
    Up, Ip, Uv, Iv = model.embeddings()
    n_items = Ip.shape[0]
    price_pct, pop_prob, cat = meta["price_pct"], meta["pop_prob"], meta["cat"]
    item_nov = -np.log2(pop_prob + 1e-12)

    users = list(gt.keys())
    users_arr = np.array(users)
    med_clv = np.nanmedian(clv)
    uclv = np.array([clv[u] if not np.isnan(clv[u]) else med_clv for u in users])
    lo, hi = np.nanquantile(uclv, seg_edges[0]), np.nanquantile(uclv, seg_edges[1])
    seg_arr = np.where(uclv <= lo, "저CLV", np.where(uclv >= hi, "고CLV", "중CLV"))
    seg_cnt = {s: int((seg_arr == s).sum()) for s in SEG_NAMES}

    pos_key_sorted, pos_rev_sorted = build_pos_lookup(gt, rev, n_items)
    ideal = build_ideal_rev_cumsum(gt, rev)
    P_arr = np.zeros(int(users_arr.max()) + 1, dtype=np.int64)
    for u in users: P_arr[u] = len(gt[u])

    overall = {k: {m: 0.0 for m in _METS} for k in ks}
    seg_acc = {k: {s: {m: 0.0 for m in _METS} for s in SEG_NAMES} for k in ks}
    expo = {k: np.zeros(n_items) for k in ks}
    cal_v, cal_p = [], []
    pu = {m: [] for m in _METS} if per_user else None
    max_k, k0 = max(ks), ks[0]

    for s0 in range(0, len(users), cfg["EVAL_BATCH"]):
        bu = users_arr[s0:s0 + cfg["EVAL_BATCH"]]
        ut = torch.as_tensor(bu, dtype=torch.long, device=DEVICE)
        s_rel = Up[ut] @ Ip.T
        s_val = Uv[ut] @ Iv.T
        s_rel = (s_rel - s_rel.mean(1, keepdim=True)) / (s_rel.std(1, keepdim=True) + 1e-8)
        s_val = (s_val - s_val.mean(1, keepdim=True)) / (s_val.std(1, keepdim=True) + 1e-8)
        scores = s_rel + lam * gate_t[ut].unsqueeze(1) * s_val
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u + 1]
            if b > a: scores[bi, csr_items[a:b]] = -1e9
        topk = scores.topk(max_k, dim=1).indices.cpu().numpy()
        res = score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
                         P_arr, price_pct, item_nov, cat, ideal)
        bseg = seg_arr[s0:s0 + cfg["EVAL_BATCH"]]
        for k in ks:
            for m in _METS:
                overall[k][m] += res[k][m].sum()
                for sg in SEG_NAMES:
                    mask = bseg == sg
                    if mask.any(): seg_acc[k][sg][m] += res[k][m][mask].sum()
            for bi in range(len(bu)): expo[k][topk[bi, :k]] += 1
        if per_user:
            for m in _METS: pu[m].append(res[k0][m])
        cal_v.extend(uclv[s0:s0 + cfg["EVAL_BATCH"]]); cal_p.extend(res[k0]["arp"].tolist())

    n = len(users)
    for k in ks:
        for m in _METS:
            overall[k][m] /= n
            for sg in SEG_NAMES:
                if seg_cnt[sg]: seg_acc[k][sg][m] /= seg_cnt[sg]
    out = dict(overall=overall, seg=seg_acc, seg_cnt=seg_cnt, n_eval=n,
               coverage={k: float((expo[k] > 0).sum()) / n_items for k in ks},
               gini={k: _gini(expo[k]) for k in ks},
               value_alignment=_spearman(cal_v, cal_p))
    if per_user:
        out["per_user"] = {m: np.concatenate(pu[m]) for m in _METS}
    return out


# ═══════════════════════════════════════════════════════════════════
# 학습
# ═══════════════════════════════════════════════════════════════════
def sample_negatives(u_arr, n_items, pos_key, rng, max_try=50):
    """균등 무작위 negative 샘플링 (LightGCN 원논문). v2의 hard negative는 제거."""
    n = len(u_arr)
    neg = rng.integers(0, n_items, size=n)
    u64 = u_arr.astype(np.int64)
    for _ in range(max_try):
        key = u64 * n_items + neg
        pos = np.clip(np.searchsorted(pos_key, key), 0, len(pos_key) - 1)
        bad = pos_key[pos] == key
        if not bad.any(): break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))
    return neg


def train_phase(model, params, d, gate_t, lam_train, cfg, seed, tag, meta, clv):
    """BPR로 params만 학습. val recall@10(λ=lam_train 기준)로 조기종료."""
    if not params:
        return model
    opt = torch.optim.Adam(params, lr=cfg["LR"], weight_decay=cfg["WD"])
    rng = np.random.default_rng(seed)
    tr_u, tr_i, pos_key = d["tr_u"], d["tr_i"], d["pos_key"]
    n_train = len(tr_u); n_batch = math.ceil(n_train / cfg["BATCH_SIZE"])
    best, best_ep, best_state, bad = -1.0, -1, None, 0

    for ep in range(1, cfg["EPOCHS"] + 1):
        model.train(); t0 = time.time()
        perm = rng.permutation(n_train); tot = 0.0
        for b in range(n_batch):
            idx = perm[b * cfg["BATCH_SIZE"]:(b + 1) * cfg["BATCH_SIZE"]]
            bu, bi = tr_u[idx], tr_i[idx]
            bj = sample_negatives(bu, d["n_items"], pos_key, rng)
            loss = model.bpr_loss(torch.as_tensor(bu, dtype=torch.long, device=DEVICE),
                                  torch.as_tensor(bi, dtype=torch.long, device=DEVICE),
                                  torch.as_tensor(bj, dtype=torch.long, device=DEVICE),
                                  gate_t, lam_train)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        model.eval()
        r = evaluate(model, lam_train, gate_t, d["val_gt"], d["val_rev"], meta, clv,
                     [10], d["csr_ptr"], d["csr_items"], cfg, CFG["SEG_EDGES"])
        score = r["overall"][10]["recall"]
        star = ""
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            star = " ★"
        else:
            bad += 1
        print(f"  [{tag}] ep {ep:3d} | loss {tot/n_batch:.4f} | val recall@10 {score:.5f} "
              f"| {time.time()-t0:.0f}s{star}")
        if bad >= cfg["EARLY_STOP"]:
            print(f"  [{tag}] early stop"); break
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  [{tag}] 완료 — best epoch {best_ep}, val recall@10 {best:.5f}")
    return model


def train_model(d, gate_t, x_val_u, x_item, item_cat, meta, clv, cfg, seed):
    """ARCH에 따라 학습 절차만 달라진다. 모델 구조·입력·gate·λ는 완전히 동일."""
    set_seed(seed)
    model = DualSpaceLightGCN(d["n_users"], d["n_items"], d["n_cat"],
                              x_val_u, x_item, item_cat, cfg, d["adj"]).to(DEVICE)
    if cfg["ARCH"] == "joint":
        # 선호·가치 블록을 같은 BPR 손실로 동시에 학습 (LightGCN 1회 학습)
        train_phase(model, list(model.parameters()), d, gate_t, cfg["LAMBDA_TRAIN"],
                    cfg, seed, "joint", meta, clv)
    elif cfg["ARCH"] == "two_stage":
        # 1단계: 선호 블록만 (λ=0 → 순수 LightGCN). 가치 블록은 점수에 관여하지 않는다.
        train_phase(model, model.pref_params(), d, gate_t, 0.0, cfg, seed, "stage1-pref", meta, clv)
        # 2단계: 선호 블록 동결, 가치 블록만
        for p in model.pref_params(): p.requires_grad_(False)
        train_phase(model, model.value_params(), d, gate_t, cfg["LAMBDA_TRAIN"],
                    cfg, seed, "stage2-value", meta, clv)
    else:
        raise ValueError(f"ARCH는 'joint' 또는 'two_stage'만 가능: {cfg['ARCH']}")
    return model


# ═══════════════════════════════════════════════════════════════════
# λ 스윕 + 부트스트랩
# ═══════════════════════════════════════════════════════════════════
def bootstrap_ci(base_pu, lam_pu, n_boot, seed):
    """λ=0 대비 차이의 95% 신뢰구간 (유저 단위 부트스트랩)."""
    rng = np.random.default_rng(seed)
    n = len(base_pu)
    diff = lam_pu - base_pu
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def flatten(res):
    """중첩 결과를 recall@10 / 고CLV_revenue@10 식 열로 평탄화."""
    out = {}
    for k, mets in res["overall"].items():
        for m, v in mets.items(): out[f"{m}@{k}"] = v
    for name in ("coverage", "gini"):
        for k, v in res[name].items(): out[f"{name}@{k}"] = v
    out["value_alignment"] = res["value_alignment"]
    for k, segs in res["seg"].items():
        for sg, mets in segs.items():
            for m, v in mets.items(): out[f"{sg}_{m}@{k}"] = v
    return out


def run_seed(seed, d, gate_t, x_val_u, x_item, item_cat, meta, clv, cfg):
    tag = run_tag(cfg, DCFG, seed)
    ckpt = Path(cfg["OUT_DIR"]) / f"ckpt_{tag}.pt"
    model = DualSpaceLightGCN(d["n_users"], d["n_items"], d["n_cat"],
                              x_val_u, x_item, item_cat, cfg, d["adj"]).to(DEVICE)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        print(f"[seed {seed}] 체크포인트 로드 ({ckpt.name})")
    else:
        model = train_model(d, gate_t, x_val_u, x_item, item_cat, meta, clv, cfg, seed)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt)
        print(f"[seed {seed}] 저장 → {ckpt}")
    model.eval()

    rows, base_pu = [], None
    for lam in cfg["LAMBDA_SWEEP"]:
        r = evaluate(model, lam, gate_t, d["test_gt"], d["test_rev"], meta, clv,
                     cfg["K_LIST"], d["csr_ptr"], d["csr_items"], cfg,
                     cfg["SEG_EDGES"], per_user=True)
        pu = r.pop("per_user")
        if lam == 0.0:
            base_pu = pu
        row = {"seed": seed, "arch": cfg["ARCH"], "lambda": lam, **flatten(r)}
        if base_pu is not None:
            for m in ["recall", "ndcg", "revenue", "arp"]:
                mean, lo, hi = bootstrap_ci(base_pu[m], pu[m], cfg["N_BOOT"], seed)
                row[f"d_{m}_mean"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = mean, lo, hi
        rows.append(row)
        sig = ""
        if base_pu is not None and lam > 0:
            mean, lo, hi = bootstrap_ci(base_pu["revenue"], pu["revenue"], cfg["N_BOOT"], seed)
            sig = f" | Δ가격가중적중값 {mean:+.5f} [{lo:+.5f}, {hi:+.5f}]" + \
                  ("  ← 유의" if lo > 0 or hi < 0 else "")
        print(f"  λ={lam:<5} recall@10 {r['overall'][10]['recall']:.6f} "
              f"| 가격가중적중값@10 {r['overall'][10]['revenue']:.6f} "
              f"| 정렬도 {r['value_alignment']:+.4f}{sig}")
    return rows


def main():
    print(f"DATASET={CFG['DATASET']} | ARCH={CFG['ARCH']} | DEVICE={DEVICE}")
    d = prepare_data(CFG, DCFG)
    x_val_u, clv = clv_features(d["train"], d["n_users"], CFG, DCFG["is_date"])
    x_item, item_cat = item_value_features(d["train"], d["n_items"])
    meta = item_meta(d["train"], d["n_items"])
    gate = clv_gate(clv)
    gate_t = torch.from_numpy(gate).to(DEVICE)
    print(f"  gate(u)=percentile_rank(CLV): 평균 {gate.mean():.3f}, "
          f"하위20% 평균 {gate[gate <= np.quantile(gate, .2)].mean():.3f}")

    all_rows = []
    for seed in CFG["SEED_LIST"]:
        print(f"\n{'='*80}\nseed {seed} | ARCH={CFG['ARCH']}\n{'='*80}")
        all_rows += run_seed(seed, d, gate_t, x_val_u, x_item, item_cat, meta, clv, CFG)

    out = Path(CFG["OUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    csv_path = out / f"result_v3_{CFG['ARCH']}_{CFG['DATASET']}.csv"
    df.to_csv(csv_path, index=False, float_format="%.6f")
    with open(str(csv_path).replace(".csv", ".json"), "w") as f:
        json.dump({"cfg": {k: v for k, v in CFG.items() if k != "OUT_DIR"},
                   "note_revenue": "revenue = 가격가중 추천 적중값(price-weighted gain proxy), 실제 매출 아님",
                   "note_arp": "arp = 추천 상품의 평균 가격 백분위(인기도 기반 ARP와 다른 지표)",
                   "rows": all_rows}, f, indent=2, default=float, ensure_ascii=False)
    print(f"\n저장 → {csv_path}")

    print(f"\n{'='*80}\nλ 스윕 요약 (3시드 평균, test @10)\n{'='*80}")
    g = df.groupby("lambda")[["recall@10", "ndcg@10", "revenue@10", "arp@10",
                              "diversity@10", "coverage@10", "value_alignment"]].mean()
    print(g.to_string(float_format=lambda x: f"{x:.6f}"))
    return df


if __name__ == "__main__":
    main()

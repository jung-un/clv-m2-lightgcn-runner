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
    "DATASET": "dunnhumby",   # "hm" | "dunnhumby"
    "MODEL_LABEL": "M2v2",    # v1("M2")과 구분 — CLV 임베딩 재설계 버전
    "SEED": 42,
    "SEED_LIST": [42, 43, 44],  # value tower 다중시드 재현성 확인용

    # ── 실행 단계 ──
    "PHASE": 2,                # 1=개념검증(고정 epoch/λ 1회 평가), 2=λ 그리드 본탐색
    "PHASE1_LAMBDA": 1.0,      # 1단계 고정 λ (epoch은 VT 단독 최고를 씀)

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

    # ── 평가 ──
    "K_LIST": [10, 20, 50], "SELECT_METRIC": "Recall@10",
    "N_BOOT": 2000,
    "LOW_CLV_PCTL": 0.2,       # 세그먼트 리포팅/가드레일 경계 — 게이트 계산과는 무관(리포팅 전용)
    "CLV_GATE_POWER": 2.0,     # gate(u)=percentile_rank(CLV_u)**power. power=1이면 저CLV(하위
                               # 20%)도 gate가 최대 0.2까지 나와 LOW_CLV_EPSILON=0.0 가드레일을
                               # 거의 항상 위반한다(2026-07-31 실측). power>1로 저CLV를 더 세게
                               # 누르되(0.2**2=0.04) 여전히 연속함수(계단 없음)로 유지.

    # ── phase2 그리드 탐색 (dampen 없음 — gate가 CLV percentile로 고정 결정되므로) ──
    "LAMBDA_GRID": [0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0],

    # ── 가드레일 (전체 종합 성과 기준 — 세그먼트별 무손실 요구는 2026-08-01 제거함) ──
    # RECALL50/HR/DIVERSITY를 0.0으로 되돌리지 말 것 — CLAUDE.md §4, λ>0 전체 탈락 버그 재현됨
    "ACCURACY_EPSILON": 0.0,
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
assert {10, 50} <= set(CFG["K_LIST"]), "K_LIST must include 10 and 50 for the @10/@50 guardrails"
assert CFG["PHASE"] in (1, 2), "PHASE는 1(개념검증) 또는 2(그리드 본탐색)만 가능"

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


def grid_fingerprint(cfg):
    """Stage B(phase2) 그리드 캐시가 실제로 의존하는 하이퍼파라미터만 지문화한다.
    v1과 달리 dampen 그리드가 없다(gate가 CLV percentile로 고정 결정) — LAMBDA_GRID와
    VT_TOPK_CKPTS/EPOCH_SCREEN_LAMBDA(Stage A 스크리닝이 vt_topk에 어떤 epoch를 뽑는지를
    바꿔, 캐시된 epoch 키가 안 맞으면 StopIteration 크래시로 이어짐)만 있으면 된다.
    LOW_CLV_PCTL은 세그먼트 리포팅/가드레일 임계값에 영향을 주므로 포함한다.
    CLV_GATE_POWER는 gate_arr 자체를 바꿔 그리드의 모든 (epoch,λ) 조합의 결합점수에
    영향을 주므로 포함한다.
    cfg["RUN_TAG"]도 포함 — M1이 재학습되면(RUN_TAG 변경) 이 캐시도 자동 무효화."""
    keys = ["LAMBDA_GRID", "VT_TOPK_CKPTS", "EPOCH_SCREEN_LAMBDA", "LOW_CLV_PCTL", "CLV_GATE_POWER"]
    payload = {k: cfg[k] for k in keys}
    payload["RUN_TAG"] = cfg["RUN_TAG"]
    s = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def seed_result_fingerprint(cfg, dcfg, seed):
    """load_or_run_seed()의 최종 결과 캐시 지문. PHASE/PHASE1_LAMBDA(phase1 결과 자체를
    바꿈)와 가드레일 epsilon들·K_LIST·N_BOOT(phase2 선택/리포팅에 영향)를 포함한다.
    CLV_GATE_POWER는 phase1/phase2 둘 다의 결합점수 자체를 바꾸므로 포함한다."""
    keys = ["PHASE", "PHASE1_LAMBDA", "ACCURACY_EPSILON", "RECALL50_EPSILON",
            "HR_EPSILON", "DIVERSITY_EPSILON", "EPS_TOL", "K_LIST", "N_BOOT", "LOW_CLV_PCTL",
            "CLV_GATE_POWER"]
    payload = {k: cfg[k] for k in keys}
    own = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]
    combined = f"{vt_fingerprint(cfg, dcfg, seed)}_{grid_fingerprint(cfg)}_{own}"
    return hashlib.md5(combined.encode()).hexdigest()[:8]


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
    # ponytail: 예전에 CFG["ITER_FILTER"]가 항상 False라 반복 재필터링 분기(구 while 루프)를
    # 제거함 — 그 키 자체도 이미 CFG/cfg_fingerprint에서 삭제됨(죽은 코드 삭제 흔적일 뿐).

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
    """z^value 입력 특징. v1과의 차이: 반복거래축(F_p/T_p/R_p, CLV의 N̂ 성분)을
    가치축(AOV_p/Prem_p/CatShare, V̂ 성분) 앞에 이어붙인다 — CLV를 구성하는 두 축
    전부를 z^value가 직접 학습하도록(v1은 V̂ 관련 변수만 썼음). 단일 MLP가 이
    확장된 벡터를 통째로 받는다(축별 분리 MLP 아님)."""
    g = _user_pct_stats(train, cfg, is_date)
    k = cfg["SHRINKAGE_K"]

    ftr_full = np.full((n_users, 3), 0.5, np.float32)
    ftr_full[g.index.values] = np.stack([g["F_p"].values, g["T_p"].values, g["R_p"].values], axis=1)

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

    x_val_u = np.concatenate([ftr_full, aov_full[:, None], prem_full[:, None], cat_full], axis=1)
    F_u_full = np.zeros(n_users, dtype=np.int64)
    F_u_full[g.index.values] = g["F"].values
    print(f"  유저 특징: val(F,T,R+AOV,Prem+CatShare[금액기준]{n_cat}) {x_val_u.shape}")
    return x_val_u.astype(np.float32), F_u_full


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
    """M1(z^pref) backbone. 순수 협업신호만으로 유저/아이템 임베딩을 학습하는
    표준 LightGCN이다 — 이름에 "CLV"가 들어있지만 이 클래스 자체는 CLV 정보를
    전혀 쓰지 않는다(과거에는 side-injection 방식(M2 원본, 실패)이 이 클래스
    안에 있었으나 전부 제거됨, lightgcn_clv_exp_colab_emb2.ipynb에 원본 보존).

    CLV를 반영하는 부분(z^value)은 이 클래스가 아니라 ValueTower가 완전히 별도로
    담당한다(Dual-Space 구조) — 두 임베딩 공간은 학습 중 파라미터를 전혀 공유하지
    않고, 최종적으로 run_dualspace_one_seed()의 _combined_scores()에서 점수만
    더해진다. 이 클래스는 M1 학습(main())과 Dual-Space의 z^pref 동결 로드
    (run_dualspace()) 양쪽에서 동일하게 쓰인다.

    구조: user_emb/item_emb(layer-0) → N_LAYERS번 정규화 인접행렬(adj)과 곱해
    이웃 정보를 전파 → 모든 레이어 출력의 평균이 최종 임베딩(LightGCN 원 논문의
    핵심 아이디어: 비선형 변환 없이 단순 평균).
    """
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
        # 항상 user_emb.weight/item_emb.weight와 같은 값이라 두 옵션이 동일했음(CFG["REG_TARGET"]
        # 키 자체도 이미 CFG/cfg_fingerprint에서 삭제됨).
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
            if "rng_state" in st:  # 재개 시에도 배치순서/negative sampling이 이어지도록 rng 상태 복원
                rng.bit_generator.state = st["rng_state"]
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
                        "history": history + [rec], "n_users": model.n_users, "n_items": model.n_items,
                        "rng_state": rng.bit_generator.state},
                       ckpt)
            if cfg["EARLY_STOP"] and bad >= cfg["EARLY_STOP"]:
                history.append(rec); print("early stop"); break
        history.append(rec)
    return history, best_state, best_ep, best_score


def main():
    """M1(z^pref)을 학습하거나, 이미 학습되어 있으면(체크포인트 존재) 즉시 복원하고
    끝낸다. prepare_data()로 데이터를 준비한 뒤 LightGCNCLV를 표준 BPR loss로
    학습한다 — CLV 정보는 전혀 관여하지 않는 순수 협업필터링 학습이다.

    이 함수가 만드는 체크포인트(ckpt_{RUN_TAG}.pt)는 run_dualspace()가 z^pref로
    그대로 가져다 쓴다 — 이 함수를 다시 실행할 필요가 있는 경우는: (1) 아직 한
    번도 학습 안 한 경우, (2) CFG의 cfg_fingerprint 대상 키(DIM/EPOCHS/WINDOW_DAYS
    등)를 바꿔서 RUN_TAG 자체가 달라진 경우, (3) 2년 전체 데이터로 최종 검증할 때
    (WINDOW_DAYS=None으로 바꾸면 자동으로 새 RUN_TAG가 되어 새로 학습됨) 뿐이다."""
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
    """z^value — CLV(가치) 신호만으로 학습되는 독립 임베딩 공간. LightGCNCLV(z^pref)와
    파라미터를 전혀 공유하지 않고(Dual-Space), 오직 유저/아이템의 "가치 특징"
    (x_val_u: AOV/Prem/CatShare, x_val_i: 가격백분위/카테고리내순위)만 입력으로
    받는 얕은 MLP(SideMLP) 두 개로 구성된다.

    z^pref는 M1에서 이미 학습되어 완전히 동결된 채로 쓰이는 반면, z^value는
    run_dualspace() 안에서 이 클래스로 매번 새로 학습된다(seed별로 독립).
    encode()가 반환하는 (zu, zi)는 L2 정규화되어 있어 내적이 코사인 유사도와
    같아진다 — 결합 시(_combined_scores) z^pref 점수와 스케일을 맞추기 위한
    zscore 정규화와 별개로, z^value 자체의 내부 스케일을 안정시키는 역할.
    """
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


VT_SNAPSHOT_EXCLUDED_KEYS = {"x_val_u", "x_val_i"}  # static input buffers, never saved per-epoch


def load_vt_state(model, state):
    """value tower 저장 스냅샷에는 x_val_u/x_val_i(static 입력 버퍼)가 의도적으로 빠져있으므로
    strict=False로 로드한다. 하지만 그 외 키가 빠지거나 낯선 키가 섞여 있으면(예: 나중에
    ValueTower에 파라미터가 추가됐는데 저장 로직이 안 따라간 경우) 그건 진짜 버그이므로
    조용히 무작위 초기값으로 남기지 않고 바로 에러를 낸다."""
    result = model.load_state_dict(state, strict=False)
    assert set(result.missing_keys) <= VT_SNAPSHOT_EXCLUDED_KEYS and not result.unexpected_keys, \
        f"VT state_dict mismatch: missing={result.missing_keys} unexpected={result.unexpected_keys}"


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
        if "rng_state" in st:  # 재개 시에도 배치순서/negative sampling이 이어지도록 rng 상태 복원
            rng.bit_generator.state = st["rng_state"]
        print(f"[VT RESUME seed{seed}] epoch {st['last_epoch']}까지 복원, epoch {start_ep}부터 재개")
        if cfg["VT_PATIENCE"] and bad >= cfg["VT_PATIENCE"]:
            print(f"  [VT RESUME seed{seed}] 이미 early stop 조건 충족(bad={bad}>=PATIENCE={cfg['VT_PATIENCE']}) "
                  f"— epoch 재실행 없이 기존 best_state(epoch={best_ep})로 즉시 종료 (총 {len(all_epochs)}개 epoch 보관됨)")
            load_vt_state(model, best_state)
            return model, best_ep, best_score, all_epochs

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
        # present on `model` at construction time, so load_vt_state() (strict except for these
        # two known keys) below is exactly correct, not a workaround.
        state_snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if k not in VT_SNAPSHOT_EXCLUDED_KEYS}
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
                        "mlp_hidden": cfg["MLP_HIDDEN"], "hard_neg_ratio": cfg["HARD_NEG_RATIO"],
                        "rng_state": rng.bit_generator.state},
                       ckpt_path)

        if cfg["VT_PATIENCE"] and bad >= cfg["VT_PATIENCE"]:
            print("  early stop"); break

    load_vt_state(model, best_state)  # best_state has no x_val_u/x_val_i (see above); already correct on model
    if ckpt_path:
        print(f"  저장 → {ckpt_path} (VT 단독최고 epoch={best_ep}, 총 {len(all_epochs)}개 epoch 보관)")
    return model, best_ep, best_score, all_epochs


def compute_clv_gate(clv, power=1.0):
    """gate(u) = percentile_rank(CLV_u)**power — CLV가 낮을수록 z^value 반영을 자동으로
    줄이고, 높을수록 그대로 반영한다. v1의 gate_F(u)(F_u 구간별 CatShare 신뢰도 기반)와
    dampen_low/dampen_high(저/고CLV 3단계 계단식)를 이 연속함수 하나로 대체한다.

    근거: 마스터 보고서 §7.2에서 λ를 키울수록 저CLV 유저의 Recall/PWGain이 함께
    나빠지는 게 실측됐다 — 저CLV에게 가치 임베딩을 세게 반영하는 건 실제 손해다.
    percentile_rank는 저CLV 유저의 gate를 0에 가깝게 만들어 이 손해를 구조적으로 막는다.

    power(2026-07-31 추가): power=1이면 저CLV 리포팅 경계(하위 20%)에 속한 유저도
    gate가 최대 0.2까지 나와 LOW_CLV_EPSILON=0.0 가드레일을 실측상 거의 항상 위반한다
    (Recall@10은 top-10 경계에서 순위가 한 칸만 밀려도 hit→miss로 뒤집히는 이분법
    지표라, gate=0.2 정도의 미세한 개입도 걸림돌이 됨). power>1을 주면 0~1 사이 값이
    거듭제곱되어 저CLV(작은 값)는 훨씬 더 0에 가깝게, 고CLV(1에 가까운 값)는 상대적으로
    덜 줄어든다(예: power=2일 때 0.2**2=0.04, 0.95**2=0.9025) — 계단함수 없이 곡선
    형태만 바꾸는 것이라 "CLV 한 등수 차이로 gate가 급변" 문제는 생기지 않는다.

    관측 없는 유저(train 거래 없어 CLV가 NaN, compute_clv_vhat 참고)는 percentile
    계산에서 제외하고 gate=0으로 고정한다(가장 보수적인 값)."""
    valid = ~np.isnan(clv)
    gate = np.zeros_like(clv, dtype=np.float32)
    gate[valid] = pd.Series(clv[valid]).rank(pct=True).to_numpy(np.float32) ** power
    return gate


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
    """유저별 Recall/NDCG/PWGain/ARP 배열 (bootstrap CI 용). users 순서 고정 보장."""
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


def run_stage_b_grid(vt_topk, cfg, grid_path, gate_arr, base_val_res, _eval,
                      value_model=None, val_gt=None, val_rev=None):
    """(epoch × λ) 그리드를 계산한다. v1과 달리 dampen 차원이 없다 — gate(u)가
    compute_clv_gate()로 이미 고정 결정되므로, 세그먼트별로 곱할 별도 dampen을
    탐색할 필요가 없어졌다(그리드가 5×11=55개로 줄어듦, v1은 5×11×4×4=880개).
    epoch(vt_topk의 원소) 하나를 끝낼 때마다 grid_results를 grid_path에 저장하고,
    시작할 때 이미 저장된 파일이 있으면 그 안의 키들은 건너뛴다(v1과 동일한 재개 패턴)."""
    grid_results = {}
    done_epochs = set()
    if Path(grid_path).exists():
        saved = torch.load(grid_path, weights_only=False)
        grid_results = saved["grid_results"]; done_epochs = saved["done_epochs"]
        print(f"  [Stage B RESUME] epoch {sorted(done_epochs)}는 이미 계산됨, 이어서 진행")

    for ck in vt_topk:
        ep_id = ck["epoch"]
        if ep_id in done_epochs:
            continue
        if value_model is not None:
            load_vt_state(value_model, ck["state"])
            with torch.no_grad():
                Uv_c, Iv_c = value_model.encode()
        else:
            Uv_c, Iv_c = None, None
        for lam in cfg["LAMBDA_GRID"]:
            key = (ep_id, lam)
            if lam == 0:
                grid_results[key] = base_val_res  # λ=0은 epoch 무관, 재사용
                continue
            grid_results[key] = _eval(gate_arr, lam, val_gt, val_rev, Uv_c, Iv_c)
        done_epochs.add(ep_id)
        Path(grid_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"grid_results": grid_results, "done_epochs": done_epochs}, grid_path)
        print(f"  [Stage B] epoch {ep_id} 블록 완료 및 저장 ({len(done_epochs)}/{len(vt_topk)} epoch)")

    # 캐시에서 재개해 아무것도 새로 안 돌려도, 사람이 (epoch,λ)별 val 지표를 직접
    # 비교할 수 있도록 읽기 쉬운 요약을 항상 남긴다(.pt는 텍스트로 못 읽음).
    summary_rows = []
    for (ep_id, lam), res in sorted(grid_results.items()):
        o10, o50, seg10 = res["overall"][10], res["overall"][50], res["seg"][10]
        summary_rows.append({
            "epoch": ep_id, "lambda": lam,
            "val_recall10": o10["recall"], "val_recall50": o50["recall"],
            "val_pwgain10": o10["revenue"], "val_hr10": o10["hr"], "val_diversity10": o10["diversity"],
            "저CLV_recall10": seg10["저CLV"]["recall"], "저CLV_pwgain10": seg10["저CLV"]["revenue"],
            "고CLV_recall10": seg10["고CLV"]["recall"], "고CLV_pwgain10": seg10["고CLV"]["revenue"],
        })
    summary_path = Path(str(grid_path).replace(".pt", "_summary.json"))
    with open(summary_path, "w") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    print(f"  [Stage B] 그리드 {len(summary_rows)}개 조합 λ별 val 지표 요약 → {summary_path}")
    return grid_results


def run_dualspace_one_seed_phase1(seed, train, val_gt, val_rev, test_gt, test_rev,
                                    n_users, n_items, n_cat, tr_u, tr_i, pos_key, user_pos,
                                    item_cat_arr, cat_items, item_meta, user_meta,
                                    U_pref, I_pref, csr_ptr, csr_items):
    """1단계 개념검증: Stage A(epoch 스크리닝)/Stage B(λ 그리드) 둘 다 건너뛰고,
    고정값(epoch=VT 단독학습 최고, λ=CFG["PHASE1_LAMBDA"]) 하나로만 z^value를
    결합해 baseline(λ=0)과 비교한다. 지표 계산 깊이(세그먼트별+bootstrap CI)는
    phase2와 동일 — 생략하는 건 여러 epoch·λ 조합을 비교하는 그리드 탐색뿐이다."""
    set_seed(seed)
    x_val_u, F_u = build_user_features(train, n_users, n_cat, CFG, DCFG["is_date"])
    x_val_i = build_item_features(train, n_items, n_cat)

    vt_ckpt = Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['MODEL_LABEL']}_vt_{CFG['DATASET']}_s{seed}_{vt_fingerprint(CFG, DCFG, seed)}.pt"
    value_model, vt_best_ep, vt_best_val, vt_all_epochs = train_value_tower(
        x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
        val_gt, csr_ptr, csr_items, CFG, seed, ckpt_path=vt_ckpt)

    gate_arr = compute_clv_gate(user_meta["clv"], power=CFG["CLV_GATE_POWER"])

    clv = user_meta["clv"]; clv_valid = clv[~np.isnan(clv)]
    clv_lo_th = np.quantile(clv_valid, CFG["LOW_CLV_PCTL"])
    clv_hi_th = np.quantile(clv_valid, 1 - CFG["LOW_CLV_PCTL"])

    pos_lookup_test = build_pos_lookup(test_gt, test_rev, n_items)
    ideal_rev_cumsum_test = build_ideal_rev_cumsum(test_gt, test_rev)

    def _eval_test(gate, lam, Uv_, Iv_):
        return evaluate_combined(U_pref, I_pref, Uv_, Iv_, gate, lam, test_gt, test_rev,
                                  item_meta, user_meta, CFG["K_LIST"], csr_ptr, csr_items,
                                  clv_lo_th=clv_lo_th, clv_hi_th=clv_hi_th,
                                  pos_lookup=pos_lookup_test, ideal_rev_cumsum=ideal_rev_cumsum_test)

    best_ckpt = next(ck for ck in vt_all_epochs if ck["epoch"] == vt_best_ep)
    load_vt_state(value_model, best_ckpt["state"])
    with torch.no_grad():
        Uv, Iv = value_model.encode()

    lam = CFG["PHASE1_LAMBDA"]
    test_res = _eval_test(gate_arr, lam, Uv, Iv)
    base_test_res = _eval_test(gate_arr, 0, Uv, Iv)

    pu_base = evaluate_combined_peruser(U_pref, I_pref, Uv, Iv, gate_arr, 0,
                                         test_gt, test_rev, item_meta, user_meta, csr_ptr, csr_items, k=10)
    pu_best = evaluate_combined_peruser(U_pref, I_pref, Uv, Iv, gate_arr, lam,
                                         test_gt, test_rev, item_meta, user_meta, csr_ptr, csr_items, k=10)
    assert np.array_equal(pu_base["users"], pu_best["users"]), "유저 순서 불일치"

    ci = {}
    for name, key in [("Recall", "recall"), ("NDCG", "ndcg"), ("PWGain", "revenue")]:
        mean_d, lo, hi = bootstrap_mean_diff_ci(pu_base[key], pu_best[key], n_boot=CFG["N_BOOT"], seed=seed)
        ci[name] = (mean_d, lo, hi)
    va_mean, va_lo, va_hi = bootstrap_spearman_diff_ci(pu_base["vhat"], pu_base["arp"], pu_best["arp"],
                                                        n_boot=CFG["N_BOOT"], seed=seed)
    ci["ValueAlignment"] = (va_mean, va_lo, va_hi)

    tb = base_test_res["seg"][10]; tt = test_res["seg"][10]
    print(f"\n[seed {seed}, PHASE 1 개념검증] VT epoch={vt_best_ep}, λ={lam}")
    print(f"  test: 전체 Recall {base_test_res['overall'][10]['recall']:.4f}→{test_res['overall'][10]['recall']:.4f} "
          f"PWGain {base_test_res['overall'][10]['revenue']:.5f}→{test_res['overall'][10]['revenue']:.5f} "
          f"ValueAlignment {base_test_res['value_alignment_spearman']:.3f}→{test_res['value_alignment_spearman']:.3f} | "
          f"저CLV R {tb['저CLV']['recall']:.4f}→{tt['저CLV']['recall']:.4f} | "
          f"고CLV R {tb['고CLV']['recall']:.4f}→{tt['고CLV']['recall']:.4f}")
    print("  95% bootstrap CI (test @10, 선택-baseline; PWGain=price-weighted gain proxy, 실제 매출 아님):")
    for name, (m, lo, hi) in ci.items():
        sig = "" if lo <= 0 <= hi else "  ← 유의(0 미포함)"
        print(f"    Δ{name}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}")

    return [dict(seed=seed, phase=1, vt_best_epoch=vt_best_ep, lam=lam,
                 test_base=base_test_res, test_best=test_res, ci=ci)]


def run_dualspace_one_seed_phase2(seed, train, val_gt, val_rev, test_gt, test_rev,
                                    n_users, n_items, n_cat, tr_u, tr_i, pos_key, user_pos,
                                    item_cat_arr, cat_items, item_meta, user_meta,
                                    U_pref, I_pref, csr_ptr, csr_items):
    """2단계 본탐색: Stage A(epoch 스크리닝) + Stage B(λ 그리드, dampen 없음) 전부 돌려서
    validation에서 최적 (epoch, λ)를 고르고 test에서 1회 평가한다. phase1에서 개선이
    확인된 뒤에만 CFG["PHASE"]=2로 명시 전환해서 쓴다(자동 트리거 아님)."""
    set_seed(seed)
    x_val_u, F_u = build_user_features(train, n_users, n_cat, CFG, DCFG["is_date"])
    x_val_i = build_item_features(train, n_items, n_cat)

    vt_ckpt = Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['MODEL_LABEL']}_vt_{CFG['DATASET']}_s{seed}_{vt_fingerprint(CFG, DCFG, seed)}.pt"
    value_model, vt_best_ep, vt_best_val, vt_all_epochs = train_value_tower(
        x_val_u, x_val_i, tr_u, tr_i, n_items, pos_key, user_pos, item_cat_arr, cat_items,
        val_gt, csr_ptr, csr_items, CFG, seed, ckpt_path=vt_ckpt)

    gate_arr = compute_clv_gate(user_meta["clv"], power=CFG["CLV_GATE_POWER"])

    clv = user_meta["clv"]; clv_valid = clv[~np.isnan(clv)]
    clv_lo_th = np.quantile(clv_valid, CFG["LOW_CLV_PCTL"])
    clv_hi_th = np.quantile(clv_valid, 1 - CFG["LOW_CLV_PCTL"])

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

    with torch.no_grad():
        Uv0, Iv0 = value_model.encode()
    base_val_res = _eval(gate_arr, 0, val_gt, val_rev, Uv0, Iv0)
    base_val_recall = base_val_res["overall"][10]["recall"]
    base_val_recall50 = base_val_res["overall"][50]["recall"]
    base_val_hr10 = base_val_res["overall"][10]["hr"]
    base_val_div10 = base_val_res["overall"][10]["diversity"]

    screen_lam = CFG["EPOCH_SCREEN_LAMBDA"]
    screen_rows = []
    for ck in vt_all_epochs:
        load_vt_state(value_model, ck["state"])
        with torch.no_grad():
            Uv_s, Iv_s = value_model.encode()
        res_s = _eval(gate_arr, screen_lam, val_gt, val_rev, Uv_s, Iv_s)
        screen_rows.append((ck["epoch"], ck["state"], res_s["overall"][10]["revenue"]))
    screen_rows.sort(key=lambda t: t[2], reverse=True)
    screened = screen_rows[:CFG["VT_TOPK_CKPTS"]]
    vt_topk = [{"epoch": e, "state": st} for e, st, _ in screened]
    print(f"\n[seed {seed}, PHASE 2] VT epoch 스크리닝(대표 λ={screen_lam}, 전체 {len(vt_all_epochs)}개 중 "
          f"결합 PWGain 상위 {len(vt_topk)}개 채택): {[c['epoch'] for c in vt_topk]}")

    eps = CFG["ACCURACY_EPSILON"]
    eps_r50 = CFG["RECALL50_EPSILON"]; eps_hr = CFG["HR_EPSILON"]; eps_div = CFG["DIVERSITY_EPSILON"]
    tol = CFG["EPS_TOL"]

    grid_path = Path(CFG["OUT_DIR"]) / (f"grid_partial_{CFG['MODEL_LABEL']}_{CFG['DATASET']}_s{seed}_"
                                         f"{vt_fingerprint(CFG, DCFG, seed)}_{grid_fingerprint(CFG)}.pt")
    grid_results = run_stage_b_grid(vt_topk, CFG, grid_path, gate_arr, base_val_res, _eval,
                                     value_model=value_model, val_gt=val_gt, val_rev=val_rev)

    def _passes(res):
        """전체 종합 성과 기준 가드레일만 적용한다(2026-08-01) — 세그먼트별(저CLV) 무손실
        요구는 뺐다. 사용자가 "결국 보는 것은 저CLV/고CLV 세그먼트가 아니라 전체 종합 성과"
        라고 명시적으로 방향을 정함 — 이전에는 저CLV Recall/PWGain 완전 무손실을 요구했었다."""
        r = res["overall"][10]["recall"]
        r50 = res["overall"][50]["recall"]; hr10 = res["overall"][10]["hr"]; div10 = res["overall"][10]["diversity"]
        return (r >= base_val_recall * (1 - eps) - tol and
                r50 >= base_val_recall50 * (1 - eps_r50) - tol and
                hr10 >= base_val_hr10 * (1 - eps_hr) - tol and
                div10 >= base_val_div10 * (1 - eps_div) - tol)

    vt_topk_epochs = {c["epoch"] for c in vt_topk}
    candidates = [key for key, res in grid_results.items() if key[0] in vt_topk_epochs and _passes(res)]
    if not candidates:
        fallback_ep = vt_best_ep if vt_best_ep in vt_topk_epochs else vt_topk[0]["epoch"]
        candidates = [(fallback_ep, 0)]
    best_ep, best_lam = max(candidates, key=lambda k: grid_results[k]["overall"][10]["revenue"])
    selected_state = next(ck["state"] for ck in vt_topk if ck["epoch"] == best_ep)
    load_vt_state(value_model, selected_state)
    with torch.no_grad():
        Uv, Iv = value_model.encode()

    sel_res = grid_results[(best_ep, best_lam)]
    r10 = sel_res["overall"][10]
    print(f"  통과 {len(candidates)}/{len(grid_results)} → 선택 ep={best_ep} λ={best_lam} "
          f"val_Recall={r10['recall']:.4f} val_PWGain={r10['revenue']:.5f}")

    test_res = _eval(gate_arr, best_lam, test_gt, test_rev, Uv, Iv)
    base_test_res = _eval(gate_arr, 0, test_gt, test_rev, Uv, Iv)

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

    tb = base_test_res["seg"][10]; tt = test_res["seg"][10]
    print(f"    test: 전체 Recall {base_test_res['overall'][10]['recall']:.4f}→{test_res['overall'][10]['recall']:.4f} "
          f"PWGain {base_test_res['overall'][10]['revenue']:.5f}→{test_res['overall'][10]['revenue']:.5f} | "
          f"저CLV R {tb['저CLV']['recall']:.4f}→{tt['저CLV']['recall']:.4f} | "
          f"고CLV R {tb['고CLV']['recall']:.4f}→{tt['고CLV']['recall']:.4f}")
    print("  95% bootstrap CI (test @10, 선택-baseline; PWGain=price-weighted gain proxy, 실제 매출 아님):")
    for name, (m, lo, hi) in ci.items():
        sig = "" if lo <= 0 <= hi else "  ← 유의(0 미포함)"
        print(f"    Δ{name}: {m:+.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}")

    return [dict(seed=seed, phase=2, best_ep=best_ep, best_lam=best_lam,
                 vt_best_epoch=vt_best_ep, vt_best_val_recall=vt_best_val,
                 test_base=base_test_res, test_best=test_res, ci=ci)]


def load_or_run_seed(seed, out_dir, model_label, dataset, cfg, dcfg, run_one_seed_fn, *args, **kwargs):
    """시드 하나의 최종 결과(eps_rows)를 개별 파일로 저장/로드한다. 이미 이 시드가
    끝나있으면(파일 존재) run_one_seed_fn을 다시 부르지 않는다 — SEED_LIST 3개를
    순서대로 도는데 세 번째 시드 도중 세션이 끊기면, 이전에는 첫 두 시드까지
    포함해서 처음부터 다시 돌아야 했다.

    이 캐시는 재개 체인에서 가장 바깥쪽(=가장 먼저 확인되는) 캐시라, 파일명에
    seed_result_fingerprint(cfg, dcfg, seed)를 접어 넣어야 한다 — 안 그러면 WINDOW_DAYS나
    가드레일 epsilon, K_LIST 등을 바꿔도 이전 설정으로 만든 result_*.json을 그대로 다시
    반환해버려서, grid_fingerprint가 막으려던 stale-cache 재사용이 이 바깥 캐시에서
    되살아난다."""
    path = Path(out_dir) / (f"result_{model_label}_{dataset}_s{seed}_"
                             f"{seed_result_fingerprint(cfg, dcfg, seed)}.json")
    if path.exists():
        print(f"[시드 RESUME] seed {seed}는 이미 완료됨 ({path}) — 재계산 없이 로드")
        with open(path) as f:
            eps_rows = json.load(f)["eps_rows"]
        for row in eps_rows:
            for section in ("test_base", "test_best"):
                if section in row:
                    for outer_key in ("overall", "seg", "coverage", "gini"):
                        if outer_key in row[section]:
                            row[section][outer_key] = {int(k): v for k, v in row[section][outer_key].items()}
            if "ci" in row:  # JSON은 (mean, lo, hi) 튜플도 리스트로 바꾸므로 원래 타입으로 복원
                row["ci"] = {k: tuple(v) for k, v in row["ci"].items()}
        return eps_rows
    eps_rows = run_one_seed_fn(seed, *args, **kwargs)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"seed": seed, "eps_rows": eps_rows}, f, default=float, ensure_ascii=False)
    return eps_rows


def run_dualspace():
    """전체 파이프라인의 최상위 진입점. prepare_data()로 데이터를 한 번 준비하고,
    M1(z^pref) 체크포인트를 로드/동결한 뒤, CFG["SEED_LIST"]의 각 시드에 대해
    run_dualspace_one_seed()를 실행한다(load_or_run_seed()를 통해 이미 끝난
    시드는 건너뜀). 모든 시드가 끝나면 시드간 비교표를 출력하고 최종 결과를
    result_{MODEL_LABEL}_{DATASET}_multiseed.json으로 저장한다.

    주의: M1(z^pref)은 항상 CFG["SEED"](기본 42) 체크포인트 하나만 쓴다.
    SEED_LIST의 42/43/44는 value tower(z^value) 초기화·negative sampling에만
    적용되는 시드다 — 즉 이 함수의 결과는 "전체 모델 다중시드"가 아니라
    "고정 M1 위에서 z^value만 다중시드로 재현성을 확인한 것"이다."""
    d = prepare_data(CFG, DCFG)
    train, val_gt, val_rev, test_gt, test_rev = d["train"], d["val_gt"], d["val_rev"], d["test_gt"], d["test_rev"]
    n_users, n_items, n_cat = d["n_users"], d["n_items"], d["n_cat"]
    tr_u, tr_i, pos_key = d["tr_u"], d["tr_i"], d["pos_key"]
    user_pos, item_cat_arr, cat_items = d["user_pos"], d["item_cat_arr"], d["cat_items"]
    csr_ptr, csr_items, adj = d["csr_ptr"], d["csr_items"], d["adj"]

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
    run_seed_fn = run_dualspace_one_seed_phase1 if CFG["PHASE"] == 1 else run_dualspace_one_seed_phase2
    all_results = []  # all_results[seed_idx] = [단일 결과 dict] (phase1/phase2 둘 다 리스트 길이 1)
    for seed in CFG["SEED_LIST"]:
        res = load_or_run_seed(seed, CFG["OUT_DIR"], CFG["MODEL_LABEL"], CFG["DATASET"], CFG, DCFG,
                                run_seed_fn, train, val_gt, val_rev, test_gt, test_rev,
                                n_users, n_items, n_cat, tr_u, tr_i, pos_key, user_pos,
                                item_cat_arr, cat_items, item_meta, user_meta, U_pref, I_pref,
                                csr_ptr, csr_items)
        all_results.append(res)

    print(f"\n{'='*100}")
    print(f"M2v2 (PHASE {CFG['PHASE']}) — {len(CFG['SEED_LIST'])}-seed 결과, fixed M1(seed={CFG['SEED']})")
    print(f"{'='*100}")
    for seed_res in all_results:
        r = seed_res[0]
        dr = r["ci"]["Recall"][0]; drev = r["ci"]["PWGain"][0]; dva = r["ci"]["ValueAlignment"][0]
        lam_desc = f"lam={r['lam']}" if r["phase"] == 1 else f"best_ep={r['best_ep']} best_lam={r['best_lam']}"
        print(f"  seed={r['seed']} {lam_desc} ΔRecall={dr:+.5f} ΔPWGain={drev:+.5f} ΔValueAlign={dva:+.4f}")
    print("  ※ PWGain = price-weighted gain proxy. H&M price는 실통화 매출액이 아니므로 "
          "실제 비즈니스 매출/증분매출로 해석 금지.")
    print("  ※ 탐색적 결과 주의: 최종 확증은 여기서 고정한 설정을 그대로 Dunnhumby에서 재검증하는 것을 권장.")

    out_path = Path(CFG["OUT_DIR"]) / f"result_{CFG['MODEL_LABEL']}_{CFG['DATASET']}_multiseed.json"
    payload = []
    for seed_res in all_results:
        r = seed_res[0]
        row = {
            "seed": r["seed"], "phase": r["phase"],
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
        }
        if r["phase"] == 1:
            row.update(vt_best_epoch=r["vt_best_epoch"], lam=r["lam"])
        else:
            row.update(best_ep=r["best_ep"], best_lam=r["best_lam"],
                       vt_best_epoch=r["vt_best_epoch"], vt_best_val_recall=r["vt_best_val_recall"])
        payload.append(row)
    meta = {
        "model_label": CFG["MODEL_LABEL"], "phase": CFG["PHASE"],
        "m1_seed_fixed": CFG["SEED"], "value_tower_seeds": CFG["SEED_LIST"],
        "pwgain_formula": "PWGain@K = mean_u sum_{i in TopK(u)} 1[i in GT_u] * price_i",
        "note_multiseed": "M1(z^pref)은 고정 seed 체크포인트 하나. 시드는 value tower에만 적용됨.",
        "note_pwgain": "revenue/pwgain 필드는 가격가중 추천 적중값(price-weighted gain proxy)이며 "
                       "실제 통화 매출·증분매출·수익성 지표가 아님.",
        "note_gate": "gate(u) = percentile_rank(CLV_u) — v1의 gate_F(u)/dampen_low/dampen_high를 "
                     "대체. F/T/R을 z^value 입력에 포함(v1은 AOV/Prem/CatShare만).",
        "note_exploratory": "본 H&M 결과는 개발/탐색 결과이며 confirmatory test가 아님.",
    }
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "results": payload}, f, indent=2, default=float, ensure_ascii=False)
    print(f"\n저장 → {out_path}")
    return all_results


# M1 체크포인트(RUN_TAG 지문 기준)가 없으면 자동으로 먼저 학습한다 — CFG를 바꿔서
# RUN_TAG(지문)가 달라지면(예: WINDOW_DAYS 변경) 이 체크가 자동으로 재학습을 트리거한다.
# 이미 존재하면 main() 자체가 즉시 복원만 하고 끝나므로 항상 호출해도 안전하다.
if not (Path(CFG["OUT_DIR"]) / f"ckpt_{CFG['RUN_TAG']}.pt").exists():
    print(f"[runner] M1 체크포인트 없음(RUN_TAG={CFG['RUN_TAG']}) — main()으로 먼저 학습")
    main()
results = run_dualspace()

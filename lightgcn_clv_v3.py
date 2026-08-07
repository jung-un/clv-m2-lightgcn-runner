"""LightGCN + CLV 이중공간(v3).

    S(u,i) = <z_u^pref, z_i^pref> + λ · gate(u) · <z_u^value, z_i^value>     (raw dot product)
    CLV_u   = N̂_u × V̂_u
    N̂_u     = mean(F_p, T_p, R_p)                # 전부 백분위
    V̂_u     = mean(AOV_p, Prem_p)                # 축소추정 없음

    gate(u) = GATE_MODE에 따라 (전부 유효 유저 평균 1로 정규화, 선형·제곱 아님)
        none : 1                       게이트 없음 대조군
        clv  : percentile_rank(CLV_u)  기본값
        vhat : percentile_rank(V̂_u)    거래금액축만 — N̂는 가격 정보를 담지 않는다는
                                       EDA 근거(build_gate 주석 참고)

"임베딩을 분리하고 CLV에 따라 반영 정도를 조정한다" 외의 조건은 넣지 않는다.
v2에 있던 가드레일 4종·목적함수·epoch 스크리닝·λ=0 fallback·축소추정·gate 제곱·
hard negative는 전부 없다.

═══ 아키텍처 3종 ═══
  pref_only  : 순수 LightGCN. λ=0으로 선호 블록만 학습. **공통 baseline**이며
               two_stage·joint_warm의 출발점으로 재사용된다(시드당 한 번만 학습).
  two_stage  : pref_only 체크포인트를 불러 선호 블록 **동결** → 가치 블록만 학습.
  joint_warm : pref_only 체크포인트에서 출발하되 **동결하지 않고** 둘 다 학습.
               two_stage와의 차이는 "동결하느냐" 하나뿐이다.
  joint      : 별도 random init에서 선호·가치 블록을 처음부터 함께 학습.

  ⚠ joint_warm/joint은 총 학습량이 다르다 — joint_warm은 pref_only 학습분을 물려받는다.
    warm start는 표준 관행(사전학습→미세조정)이며 그 자체가 연구 기여는 아니다.
    joint이 학습 상한(95~99/100)에서 잘려 진 것인지 구조적으로 진 것인지 가르기 위한
    진단 목적으로 추가했다.

⚠ joint의 λ_eval=0은 baseline이 아니다. 선호 임베딩이 이미 가치항의 존재에 맞춰
  학습됐으므로 "joint-ablation"이며, 결과 표기에서도 그렇게 부른다. 두 아키텍처의
  공통 비교 기준은 언제나 pref_only다.

═══ 점수식 통일 ═══
  학습(BPR)과 평가가 combined_score_* 한 쌍만 쓴다. 둘 다 raw dot product이므로
  λ_train=1과 λ_eval=1이 정확히 같은 점수를 만든다. v2의 유저별 z-score 정규화는
  제거했다 — 그건 z^pref가 별도 모델에서 동결돼 와 상대 스케일이 임의였던 v2의
  사정 때문이었고, v3에서는 모델이 스케일을 직접 학습하므로 평가에서 지우면
  학습 결과를 버리는 셈이 된다. 대신 학습된 상대 스케일을 진단값으로 저장한다.

═══ λ 절차 ═══
  LAMBDA_TRAIN      : 학습 중 결합 강도(BPR loss 안에 들어간다 = 학습된 파라미터에 영향)
  LAMBDA_EVAL_SWEEP : 채점 시점 스윕. 실험 전에 고정하며 실행 중 바꾸지 않는다.

  주 결과(primary)   : validation 3시드 평균으로 **아키텍처별 공통 λ 하나**를 고르고
                      (select_lambda의 규칙은 코드에 사전 고정), 그 λ 하나만 test에서
                      baseline 대비 통계 검정한다. 시드별 λ 선택은 하지 않는다.
  민감도(secondary)  : test의 λ 전체 곡선은 사전 선언된 descriptive 분석으로만 보고한다.
                      "최적 λ"라 부르지 않고 λ별 유의 라벨도 붙이지 않는다.

═══ 최종 확증용 홀드아웃 ═══
  기존 test 구간은 개발 과정에서 여러 번 확인됐다. 마지막 HOLDOUT_DAYS 구간은
  EVAL_HOLDOUT=True로 명시적으로 켜기 전까지 **계산조차 하지 않는다**.
"""
import os, json, math, time, random, hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

CODE_VERSION = "v3.8"          # 결과 파일에 기록 — 코드가 바뀌면 올릴 것
IN_COLAB = os.path.exists("/content")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SCHEMA = {
    "hm": {
        "tx_path": ("/content/drive/MyDrive/논문/data/raw/hm/transactions_train.parquet" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/hm/transactions_train.csv"),
        "item_meta_path": ("/content/drive/MyDrive/논문/data/raw/hm/articles.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/hm/articles.csv"),
        "user_col": "customer_id", "item_col": "article_id",
        "time_col": "t_dat", "value_col": "price",
        "item_key_col": "article_id", "category_col": "product_group_name",
        # 주문 식별자가 없고 시간 해상도가 날짜뿐 → (고객, 날짜)를 구매 1건으로 본다.
        # price가 이미 상품 1개당 단가라 나눌 수량이 없다.
        "basket_col": None, "qty_col": None, "is_date": True,
    },
    "dunnhumby": {
        "tx_path": ("/content/drive/MyDrive/논문/data/raw/dunnhumby/transaction_data.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/dunnhumby/dunnhumby_The-Complete-Journey CSV/transaction_data.csv"),
        "item_meta_path": ("/content/drive/MyDrive/논문/data/raw/dunnhumby/product.csv" if IN_COLAB
                    else "/Users/jungun/Workspace/논문준비/data/dunnhumby/dunnhumby_The-Complete-Journey CSV/product.csv"),
        "user_col": "household_key", "item_col": "PRODUCT_ID",
        "time_col": "DAY", "value_col": "SALES_VALUE",
        "item_key_col": "PRODUCT_ID", "category_col": "COMMODITY_DESC",
        # 한 행 = BASKET_ID × PRODUCT_ID 라인(라인이 장바구니의 약 9.4배).
        # QUANTITY≠1인 행이 약 21%라 SALES_VALUE는 단가가 아니다.
        "basket_col": "BASKET_ID", "qty_col": "QUANTITY", "is_date": False,
    },
}

# ═══════════════════════════════════════════════════════════════════
# CFG   [기본]=LightGCN/기존 실험값  [선택]=사용자가 정함  [임의]=부득이 고른 값
# ═══════════════════════════════════════════════════════════════════
CFG = {
    "DATASET": "hm",                  # [선택] "hm" | "dunnhumby"
    "ARCH": "pref_only",              # [선택] "pref_only" | "two_stage" | "joint_warm" | "joint"
    "SEED_LIST": [42, 43, 44],        # [기본]

    # ── 데이터 (시간순: train | val | test | holdout) ──
    "OUT_DIR": None,
    "WINDOW_DAYS": 60,                # [선택] hm=60, dunnhumby=None(전체)
    "VAL_DAYS": 7, "TEST_DAYS": 7,    # [기본]
    "HOLDOUT_DAYS": 7,                # [선택] 최종 확증 전용. 아래 플래그 전엔 계산 안 함
    "EVAL_HOLDOUT": False,            # ⚠ 논문 최종 확증 때 딱 한 번만 True
    # [2026-08-07] MIN_ITEM_INTER=1은 Dunnhumby(유저 2,500 vs 아이템 90,785, 아이템당
    # 상호작용 중앙값 3)에서 학습 불가능한 아이템 임베딩을 6만 개 넘게 만들어 baseline이
    # 인기도로 붕괴하는 원인으로 의심된다. 10-core는 LightGCN 원논문 등 표준 관행이고
    # Dunnhumby 기준 아이템 71%를 버려도 상호작용은 93.7%가 남는다. Phase 1에서 1/10/20을
    # pref_only로 비교해 확정할 것.
    # ⚠ 바꾸면 아이템 유니버스와 평가 정답쌍이 함께 달라져 Recall이 기계적으로 오른다
    #   (Dunnhumby k=10에서 test 정답 84.4%만 잔존). threshold 간 비교는 반드시
    #   결과 JSON의 data_stats(정답쌍 수·평가유저 수)와 함께 읽을 것.
    # 필수 조건은 **한 데이터셋 안에서 M1~M5가 같은 값을 쓰는 것**이다. 두 데이터셋에
    #   같은 숫자를 강제할지는 별개의 방법론 결정이며(데이터 구조가 다름), 다르게 쓸
    #   경우 절대 지표를 데이터셋 간 직접 비교하지 말고 각 데이터셋 내 baseline 대비
    #   변화로만 비교할 것.
    "MIN_USER_INTER": 1, "MIN_ITEM_INTER": 1,   # [기본]

    # ── 임베딩 ──
    "DIM": 64, "N_LAYERS": 2,         # [기본] z^pref = 표준 LightGCN
    "D_VALUE": 16,                    # [임의] 가치 임베딩 차원
    "MLP_HIDDEN": 32,                 # [임의] MLP 은닉
    "CAT_EMB_DIM": 16,                # [임의] 카테고리 임베딩(원핫 대신)
    # 가치 MLP 마지막 Linear의 초기화 스케일. 출력 LayerNorm을 없앤 뒤 이 값으로
    # 초기 가치 점수를 작게 눌러둔다 — 학습 시작부터 가치항이 선호항을 압도하면
    # joint에서 선호공간이 학습되지 못한다(epoch 0 비율을 로그로 확인할 것).
    "VALUE_OUT_SCALE": 0.01,          # [임의]

    # ── 학습 ──
    "BATCH_SIZE": 8192, "LR": 5e-4,               # [기본]
    # 정규화 — v2와 **같은 방식**으로 되돌린 것. v2는 optimizer에 weight_decay를 주지 않고
    # BPR loss 안에서 배치에 등장한 layer-0 임베딩만 L2했다(LightGCN 공식 구현도 동일).
    # v3.1에서 숫자(1e-3)만 가져와 Adam(weight_decay=)에 꽂은 것이 붕괴의 원인이었다 —
    # 전역 감쇠는 배치에 안 들어간 30만 개 행까지 매 스텝 깎아 선호 임베딩을 0으로 만든다.
    "REG_MODE": "batch_l2",       # [선택] "batch_l2"(기본) | "global_wd"(진단용 재현)
    "PREF_REG": 1e-3,             # [기본] 배치 layer-0 L2 계수. v2와 동일한 값
    "VALUE_REG": 0.0,             # [선택] 가치 파라미터 L2. 첫 복구는 0으로 두고 해석
    "WD": 1e-3,                   # REG_MODE="global_wd"일 때만 쓰임(붕괴 재현 진단용)
    "NEG_MODE": "uniform",        # [선택] "uniform"(LightGCN 기본) | "hard50"(v2 방식, ablation)
    "EPOCHS": 100, "EARLY_STOP": 20,              # [기본] 단계별 상한 100, 수렴은 조기종료가 결정
    "SELECT_METRIC": "recall",                    # [기본] 조기종료 기준 지표
    "SELECT_K": 10,                               # [기본] 그 지표의 K
    "EVAL_BATCH": 1024,                           # [기본]
    # negative 샘플링 = 균등 무작위(LightGCN 원논문). v2의 hard negative는 제거.

    # ── λ ──
    "LAMBDA_TRAIN": 1.0,                            # [임의] BPR loss 안에 들어감
    # [2026-08-07] joint_warm의 PWGain@10이 λ=2.0(스윕 최댓값)까지 계속 단조증가해
    # 아직 정점을 못 봤음 — 4.0/8.0까지 넓혀서 어디서 꺾이는지 확인한다.
    "LAMBDA_EVAL_SWEEP": [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0], # [임의] 실험 전 고정, 실행 중 변경 금지
    # [2026-08-07] select_lambda()의 비열등성 가드레일 폐기 — Recall@10 하나만 보호하는
    # 필터라 Recall@20/50 손실을 못 걸렀고, 이 논문에서 Recall 자체가 핵심 지표도 아님.
    # NONINFERIORITY_DELTA는 이제 선택에 안 쓰이고 참고용 출력(ΔRecall 신뢰구간)에만 남는다.
    "NONINFERIORITY_DELTA": 0.01,                   # [임의] 1% — 참고용, 선택 기준 아님

    # ── CLV ──
    "PREMIUM_THR": 0.8,               # [임의] 단가 상위 20%를 고가 상품으로
    # 축소추정 없음 — 지도교수님 자료 정의 그대로

    # 게이트 = 유저별 가치항 개입 강도. 모두 유저 평균 1로 정규화된다(build_gate 참고).
    #   none : g(u)=1        게이트 없음. 개입 강도를 유저별로 조절하지 않는 대조군
    #   clv  : percentile(CLV = N̂×V̂)   현재 방식
    #   vhat : percentile(V̂ = mean(AOV_p,Prem_p))  거래금액축만
    # [2026-08-07 EDA 근거] 유저 가격선호와의 Spearman이 세 조건에서 일관되게
    #   V̂ > CLV > N̂ 이다 (H&M 60일 .580/.450/.090, H&M 2년 .563/.272/-.058,
    #   Dunnhumby .789/.501/-.058). N̂는 어디서도 가격 정보를 담지 않으므로
    #   CLV=N̂×V̂ 를 게이트로 쓰면 가격 신호가 무관한 축에 희석된다. 관측창이 길수록
    #   N̂ 비중이 커져 희석이 심해진다(2년 H&M에서 V̂와 CLV 격차가 2배).
    #   → 게이트의 목적(아이템 가격속성과의 매칭 강도)과 지표를 일치시키는 것이지
    #     데이터셋별로 공식을 맞추는 것이 아니다.
    "GATE_MODE": "clv",               # [선택] "none" | "clv" | "vhat"

    # ── 평가 ──
    "K_LIST": [10, 20, 50],           # [기본]
    "N_BOOT": 2000,                   # [기본]
    "SEG_EDGES": (0.2, 0.8),          # [선택] 하위20%/중위60%/상위20%
                                      #        임계값은 train 전체 고객에서 한 번 계산해 고정
}
CFG["OUT_DIR"] = (f"/content/drive/MyDrive/논문/data/results_v3_{CFG['DATASET']}" if IN_COLAB
                  else f"/Users/jungun/Workspace/논문준비/data/results_v3_{CFG['DATASET']}")
DCFG = SCHEMA[CFG["DATASET"]]

_METS = ["recall", "precision", "ndcg", "hr", "map", "revenue", "vndcg", "arp", "novelty", "diversity"]
SEG_NAMES = ["저CLV", "중CLV", "고CLV"]
ARCH_LABEL = {"pref_only": "LightGCN baseline", "two_stage": "two-stage",
              "joint_warm": "joint warm-start (λ=0은 ablation, baseline 아님)",
              "joint": "joint from scratch (λ=0은 ablation, baseline 아님)"}
# λ=0이 baseline이 아닌 아키텍처 — 선호 블록이 가치항과 함께 학습돼 이미 적응했다.
ABLATION_ARCHS = {"joint", "joint_warm"}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def cfg_hash(cfg, dcfg, arch, seed):
    """**체크포인트 파일명** 전용 해시 — 학습 결과(가중치)에 실제로 영향을 주는
    설정만 넣는다. 기간/λ_train/차원을 바꿨을 때 이전 체크포인트를 덮어쓰지 않게 한다.

    ⚠ CODE_VERSION은 넣지 않는다. v3.4에서 여기에 CODE_VERSION을 넣었다가, 학습과
    무관한 변경(joint_warm 아키텍처 추가)만으로도 pref_only 해시가 바뀌어 이미 있는
    체크포인트를 못 찾고 48분짜리 학습을 세 시드 다시 돌리는 낭비가 났다. 코드 버전
    추적은 result_hash()에서 결과 파일명으로만 한다 — 체크포인트는 가중치가 실제로
    달라질 때만 새 파일이 되어야 한다."""
    keys = ["DATASET", "DIM", "N_LAYERS", "D_VALUE", "MLP_HIDDEN", "CAT_EMB_DIM",
            "BATCH_SIZE", "LR", "REG_MODE", "PREF_REG", "VALUE_REG", "WD", "NEG_MODE",
            "EPOCHS", "EARLY_STOP", "SELECT_METRIC", "SELECT_K",
            "WINDOW_DAYS", "VAL_DAYS", "TEST_DAYS", "HOLDOUT_DAYS",
            "MIN_USER_INTER", "MIN_ITEM_INTER", "PREMIUM_THR", "LAMBDA_TRAIN",
            # GATE_MODE는 bpr_loss의 gate를 통해 가중치를 직접 바꾸므로 학습 설정이다.
            # 빠뜨리면 모드가 다른 실행이 같은 체크포인트를 재사용해 결과가 오염된다.
            "GATE_MODE"]
    payload = {k: cfg[k] for k in keys}
    # 단, pref_only는 λ_train=0이라 가치항이 손실에 전혀 들어가지 않는다 → 게이트가
    # 가중치에 영향을 못 준다. 그런데도 해시에 넣으면 GATE_MODE만 바꿔도 pref_only를
    # 세 시드 다시 학습하게 되어 v3.4의 낭비가 그대로 재현된다. two_stage/joint_warm이
    # 이 체크포인트를 warm start로 재사용하므로 영향 범위도 크다.
    if arch == "pref_only":
        payload.pop("GATE_MODE")
    payload.update(category_col=dcfg["category_col"], arch=arch, seed=seed)
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:8]


def result_hash(cfg, dcfg, arch):
    """**결과 파일명** 전용 해시. 학습 설정(cfg_hash)뿐 아니라 시드 목록·λ 스윕·
    비열등 δ·평가 K·부트스트랩 횟수·세그먼트 경계까지 넣는다. v3.2까지는 첫 시드의
    학습 해시만 써서, 시드 목록이나 평가 규칙만 바꾸면 이전 결과를 덮어썼다."""
    # GATE_MODE는 **모든 아키텍처에서** 결과 해시에 넣는다. cfg_hash는 pref_only에서
    # GATE_MODE를 빼므로(가중치에 영향 없음), 그대로 두면 pref_only의 none/clv/vhat
    # 실행이 같은 결과 파일명을 써서 JSON의 CFG·진단값이 서로 덮어써진다.
    payload = {"gate_mode": cfg["GATE_MODE"],
               "train": cfg_hash(cfg, dcfg, arch, cfg["SEED_LIST"][0]),
               "seeds": cfg["SEED_LIST"], "lambda_eval": cfg["LAMBDA_EVAL_SWEEP"],
               "delta": cfg["NONINFERIORITY_DELTA"], "k_list": cfg["K_LIST"],
               "n_boot": cfg["N_BOOT"], "seg_edges": list(cfg["SEG_EDGES"]),
               "eval_holdout": cfg["EVAL_HOLDOUT"], "code": CODE_VERSION}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


# ═══════════════════════════════════════════════════════════════════
# 데이터
# ═══════════════════════════════════════════════════════════════════
def load_transactions(dcfg):
    """b_raw = 구매 건(장바구니) 식별자(없으면 만들지 않고 이후 (u_idx,t)로 묶음).
    up = 상품 1개당 단가 = 라인 금액 / 수량(0 이하는 1로 간주)."""
    tx = pd.read_parquet(dcfg["tx_path"]) if dcfg["tx_path"].endswith(".parquet") else pd.read_csv(dcfg["tx_path"])
    ren = {dcfg["user_col"]: "u_raw", dcfg["item_col"]: "i_raw",
           dcfg["time_col"]: "t", dcfg["value_col"]: "v"}
    if dcfg["basket_col"]:
        ren[dcfg["basket_col"]] = "b_raw"
    tx = tx.rename(columns=ren)
    if dcfg["is_date"]:
        tx["t"] = pd.to_datetime(tx["t"]); tx["i_raw"] = tx["i_raw"].astype(str)
    tx = tx.drop_duplicates()
    q = dcfg["qty_col"]
    tx["up"] = (tx["v"] / tx[q].clip(lower=1) if q else tx["v"]).astype(np.float32)
    unit = "BASKET_ID" if dcfg["basket_col"] else "(고객,날짜)"
    print(f"원본 {len(tx):,}건 (완전중복 제거 완료) | 구매 1건 단위={unit} | 단가=금액/{q or 1}")
    return tx


def kcore_filter(tp, min_u, min_i, max_iter=20):
    """train 구간에서 **고유 (유저,아이템) 엣지** 기준 k-core를 수렴할 때까지 적용.

    ① 거래행이 아니라 고유 엣지를 센다.
       M1 LightGCN 인접행렬은 이진이라, 한 가구가 같은 상품을 20번 사도 그래프
       degree는 1이다. 거래행을 세면 반복구매가 많은 데이터에서 아이템 연결도를
       과대평가한다 — Dunnhumby는 엣지당 평균 1.84행이고, 실측으로 MIN_ITEM_INTER=10
       에서 3,331개 아이템이 구매 가구 10곳 미만인데 통과했다(그중 103개는 구매 가구가
       단 한 곳). 반복구매 '횟수'를 신호로 쓰는 건 M3(가치그래프)의 역할이고,
       M1의 아이템 eligibility 필터는 그래프 구조와 같은 기준을 써야 한다.

    ② 한 번만 거르지 않고 번갈아 반복한다.
       아이템을 지우면 train 이력이 0이 되는 유저가 생긴다. 그 유저는 그래프 degree 0,
       CLV/V̂ = NaN, 평가에서는 제외되는데 유저 인덱스에는 남아 게이트 정규화에 0으로
       섞인다. 유저·아이템 조건이 동시에 만족될 때까지 돌려야 진짜 k-core다.
    """
    pairs = tp[["u_raw", "i_raw"]].drop_duplicates()
    it = 0
    for it in range(1, max_iter + 1):
        n_before = len(pairs)
        ic = pairs["i_raw"].value_counts()
        pairs = pairs[pairs["i_raw"].isin(set(ic.index[ic >= min_i]))]
        uc = pairs["u_raw"].value_counts()
        pairs = pairs[pairs["u_raw"].isin(set(uc.index[uc >= min_u]))]
        if len(pairs) == n_before:
            break
    else:
        print(f"  ⚠ k-core가 {max_iter}회 안에 수렴하지 않음 — 임계값을 확인할 것")
    return set(pairs["u_raw"].unique()), set(pairs["i_raw"].unique()), len(pairs), it


def prepare_data(cfg, dcfg):
    """시간순 분할: train | val | test | holdout(가장 최근).
    holdout은 EVAL_HOLDOUT=True일 때만 평가에 쓰인다(그 전엔 정답조차 만들지 않음)."""
    tx = load_transactions(dcfg)
    if cfg["WINDOW_DAYS"]:
        t_max0 = tx["t"].max()
        delta = pd.Timedelta(days=cfg["WINDOW_DAYS"]) if dcfg["is_date"] else cfg["WINDOW_DAYS"]
        tx = tx[tx["t"] >= t_max0 - delta].copy()
        print(f"최근 {cfg['WINDOW_DAYS']}일 사용: {len(tx):,}건")

    meta = pd.read_csv(dcfg["item_meta_path"], dtype={dcfg["item_key_col"]: str} if dcfg["is_date"] else None)
    meta = meta.rename(columns={dcfg["item_key_col"]: "i_raw", dcfg["category_col"]: "cat_raw"})
    tx = tx.merge(meta[["i_raw", "cat_raw"]].drop_duplicates("i_raw"), on="i_raw", how="left")
    tx["cat_raw"] = tx["cat_raw"].fillna("UNKNOWN")

    t_max = tx["t"].max()
    day = lambda n: (pd.Timedelta(days=n) if dcfg["is_date"] else n)
    hold_start = t_max - day(cfg["HOLDOUT_DAYS"])
    test_start = hold_start - day(cfg["TEST_DAYS"])
    val_start = test_start - day(cfg["VAL_DAYS"])
    print(f"분할 경계: train ≤ {val_start} < val ≤ {test_start} < test ≤ {hold_start} < holdout")

    tp = tx[tx["t"] <= val_start]
    n_row0, n_u0, n_i0 = len(tp), tp["u_raw"].nunique(), tp["i_raw"].nunique()
    n_edge0 = len(tp[["u_raw", "i_raw"]].drop_duplicates())
    keep_u, keep_i, n_edge1, n_iter = kcore_filter(
        tp, cfg["MIN_USER_INTER"], cfg["MIN_ITEM_INTER"])
    tx = tx[tx["u_raw"].isin(keep_u) & tx["i_raw"].isin(keep_i)].copy()
    print(f"필터(train 고유엣지 k-core, {n_iter}회 반복 수렴) 후: {len(tx):,}건 | "
          f"유저 {n_u0:,}→{len(keep_u):,} 아이템 {n_i0:,}→{len(keep_i):,} "
          f"엣지 {n_edge0:,}→{n_edge1:,}")
    data_stats = {
        "min_user_inter": cfg["MIN_USER_INTER"], "min_item_inter": cfg["MIN_ITEM_INTER"],
        "kcore_iters": n_iter,
        "train_rows_before": n_row0, "train_edges_before": n_edge0,
        "train_users_before": n_u0, "train_items_before": n_i0,
        "train_edges_after": n_edge1,
        "train_users_after": len(keep_u), "train_items_after": len(keep_i),
        "user_drop_rate": 1 - len(keep_u) / max(n_u0, 1),
        "item_drop_rate": 1 - len(keep_i) / max(n_i0, 1),
        "edge_drop_rate": 1 - n_edge1 / max(n_edge0, 1),
    }

    uids = np.sort(tx["u_raw"].unique()); iids = np.sort(tx["i_raw"].unique())
    cats = sorted(tx["cat_raw"].unique())
    tx["u_idx"] = tx["u_raw"].map({u: k for k, u in enumerate(uids)}).astype("int32")
    tx["i_idx"] = tx["i_raw"].map({i: k for k, i in enumerate(iids)}).astype("int32")
    tx["cat_idx"] = tx["cat_raw"].map({c: k for k, c in enumerate(cats)}).astype("int32")
    n_users, n_items, n_cat = len(uids), len(iids), len(cats)
    print(f"유저 {n_users:,} | 아이템 {n_items:,} | 카테고리({dcfg['category_col']}) {n_cat:,}")

    train = tx[tx["t"] <= val_start].copy()
    train_users = set(train.u_idx.unique()); train_items = set(train.i_idx.unique())
    train_pair_key = np.unique(train.u_idx.values.astype(np.int64) * n_items + train.i_idx.values)

    def build_eval(df, name):
        d = df[df.u_idx.isin(train_users) & df.i_idx.isin(train_items)]
        key = d.u_idx.values.astype(np.int64) * n_items + d.i_idx.values
        pos = np.clip(np.searchsorted(train_pair_key, key), 0, len(train_pair_key) - 1)
        d = d[train_pair_key[pos] != key]           # 재구매쌍 제거 (교수님 지침)
        agg = d.groupby(["u_idx", "i_idx"], sort=False)["v"].sum().reset_index()
        gt, rev = {}, {}
        for u, g in agg.groupby("u_idx", sort=False):
            gt[u] = g.i_idx.values.astype(np.int32); rev[u] = g.v.values.astype(np.float32)
        print(f"  {name}: 평가유저 {len(gt):,}명, 정답 {len(agg):,}쌍 "
              f"(유저당 {len(agg)/max(len(gt),1):.2f})")
        # MIN_ITEM_INTER를 올리면 희귀 아이템이 정답에서도 사라져 Recall이 기계적으로
        # 오른다. 이 분모를 남겨두지 않으면 "모델이 좋아진 것"과 "어려운 정답이 없어진 것"
        # 을 구분할 수 없다 — threshold 간 비교의 필수 전제다.
        split_stats[name.strip()] = {
            "eval_users": len(gt), "gt_pairs": len(agg),
            "gt_per_user": len(agg) / max(len(gt), 1)}
        return gt, rev

    split_stats = {}
    splits = {"val": build_eval(tx[(tx.t > val_start) & (tx.t <= test_start)], "Val    "),
              "test": build_eval(tx[(tx.t > test_start) & (tx.t <= hold_start)], "Test   ")}
    if cfg["EVAL_HOLDOUT"]:
        splits["holdout"] = build_eval(tx[tx.t > hold_start], "Holdout")
    else:
        print("  Holdout: 계산 안 함 (EVAL_HOLDOUT=False — 최종 확증 때만 켤 것)")
    data_stats["splits"] = split_stats
    data_stats.update(n_users=n_users, n_items=n_items, n_categories=n_cat)

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

    # NEG_MODE="hard50" ablation에서만 쓰이는 카테고리 인덱스
    _cm = train.drop_duplicates("i_idx").set_index("i_idx")["cat_idx"]
    item_cat_arr = np.zeros(n_items, dtype=np.int64)
    item_cat_arr[_cm.index.values] = _cm.values
    cat_items = {int(c): g.to_numpy() for c, g in
                 train.drop_duplicates("i_idx").groupby("cat_idx")["i_idx"]}

    # 학습 그래프의 아이템 degree 분포 — MIN_ITEM_INTER 효과를 사후 확인하는 근거
    ideg = np.bincount(ei, minlength=n_items)
    data_stats["item_degree"] = {
        "mean": float(ideg.mean()), "median": float(np.median(ideg)),
        "p10": float(np.percentile(ideg, 10)), "p90": float(np.percentile(ideg, 90)),
        "min": int(ideg.min()), "max": int(ideg.max()),
        "n_degree_lt_5": int((ideg < 5).sum()), "n_degree_lt_10": int((ideg < 10).sum())}

    return dict(train=train, splits=splits, adj=adj, pos_key=edge_key, tr_u=tu, tr_i=ti,
                csr_ptr=csr_ptr, csr_items=csr_items, item_cat=item_cat_arr, cat_items=cat_items,
                n_users=n_users, n_items=n_items, n_cat=n_cat, data_stats=data_stats)


# ═══════════════════════════════════════════════════════════════════
# CLV
# ═══════════════════════════════════════════════════════════════════
def clv_features(train, n_users, cfg, is_date):
    """반환: x_val_u [n_users,5] = (F_p,T_p,R_p,AOV_p,Prem_p), clv = N̂×V̂, vhat = V̂.
    F/AOV는 구매 건(장바구니) 단위, Prem은 단가 기준. 축소추정 없음.
    vhat은 GATE_MODE="vhat"에서만 쓰이지만, clv와 같은 곳에서 계산해야 정의가
    어긋나지 않으므로 항상 함께 반환한다(train 없는 유저는 둘 다 nan)."""
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
    g["Prem"] = g["prem"] / g["n_line"]

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
    vhat = np.full(n_users, np.nan)
    vhat[g.index.values] = g["V_hat"].values
    print(f"  유저 가치 입력 [F_p,T_p,R_p,AOV_p,Prem_p] {x.shape} (축소추정 없음)")
    return x, clv, vhat


def build_gate(clv, vhat, mode):
    """gate(u) = 유저별 가치항 개입 강도. **모든 모드를 유저 평균 1로 정규화한다.**

      none : 1                        (게이트 없음 대조군)
      clv  : percentile_rank(CLV_u)   (현재 방식)
      vhat : percentile_rank(V̂_u)     (거래금액축만)

    정규화가 없으면 모드마다 평균이 달라진다 — none은 1.0인데 percentile_rank는
    구성상 약 0.5다. 그러면 같은 λ가 모드별로 실효 개입 강도 2배 차이를 뜻하게 되어,
    모드 비교가 게이트 '구조'가 아니라 '스케일' 차이를 재게 된다. λ 스윕이 일부
    흡수하지만 그리드가 성겨서 완전히 상쇄되지 않는다. 평균 1로 맞춰야 λ가 모드 간
    같은 의미를 갖는다.

    CLV(또는 V̂)를 못 구한 유저는 0 — 아는 게 없는 유저에게 가치 개입을 하지 않는다.

    ⚠ 정규화는 **유효 유저(개입 대상)만** 기준으로 한다. 전체 배열 평균을 1로 맞추면
    NaN 유저가 0으로 섞여 유효 유저의 실효 강도가 1/(1-NaN비율)만큼 커진다
    (NaN 20%면 유효 평균 1.25). mode="none"은 NaN 유저도 1이라 0이 없으므로,
    전체 평균 기준으로 맞추면 none과 clv/vhat의 실효 강도가 애초에 어긋난다.
    """
    if mode == "none":
        gate = np.ones(len(clv), np.float32)
        valid = np.ones(len(clv), bool)
    elif mode in ("clv", "vhat"):
        src = clv if mode == "clv" else vhat
        valid = ~np.isnan(src)
        gate = pd.Series(src).rank(pct=True).to_numpy(np.float32)
        gate[~valid] = 0.0
    else:
        raise ValueError(f"GATE_MODE={mode!r} — none|clv|vhat 중 하나여야 한다")
    if not valid.any():
        raise ValueError("유효 CLV 유저가 없어 게이트를 정규화할 수 없다")
    m = float(gate[valid].mean())
    if m <= 0:
        raise ValueError(f"유효 유저 게이트 평균이 {m} — 정규화 불가")
    gate = (gate / m).astype(np.float32)
    print(f"  게이트 mode={mode}: 유효유저 평균 {gate[valid].mean():.4f}(정규화 기준) "
          f"전체 평균 {gate.mean():.4f} min {gate[valid].min():.4f} max {gate.max():.4f} "
          f"비개입(0) 유저 {int((~valid).sum()):,}명")
    return gate


def segment_thresholds(clv, edges):
    """train 전체 고객 기준으로 한 번 계산해 고정. val/test/holdout·모든 λ가 같은 임계값을
    쓴다 — 평가 집단마다 다시 계산하면 val의 고CLV와 test의 고CLV가 다른 고객군이 된다."""
    valid = clv[~np.isnan(clv)]
    lo, hi = np.quantile(valid, edges[0]), np.quantile(valid, edges[1])
    print(f"  세그먼트 임계값(train 전체 {len(valid):,}명 고정): 저≤{lo:.4f} < 중 < 고≥{hi:.4f}")
    return float(lo), float(hi)


def item_value_features(train, n_items):
    """[가격백분위, 카테고리내 가격순위] + 카테고리 ID. 가격은 단가(up) 중앙값 기준."""
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
    x = np.stack([price_pct, within], axis=1).astype(np.float32)
    print(f"  아이템 가치 입력 [가격백분위, 카테고리내 가격순위] {x.shape} + 카테고리 임베딩")
    return x, cat_arr


def item_meta(train, n_items):
    pop = np.bincount(train["i_idx"].values.astype(np.int64), minlength=n_items).astype(np.float64)
    med = train.groupby("i_idx")["up"].median()
    price_pct = np.full(n_items, 0.5, np.float64)
    price_pct[med.index.values] = med.rank(pct=True).values
    cat = np.full(n_items, -1, np.int64)
    cmap = train.groupby("i_idx")["cat_idx"].agg(lambda s: s.mode().iat[0])
    cat[cmap.index.values] = cmap.values
    return dict(price_pct=price_pct, pop_prob=pop / max(pop.sum(), 1.0), cat=cat)


# ═══════════════════════════════════════════════════════════════════
# 모델 — 점수식은 combined_score_* 한 쌍만 쓴다 (학습·평가 동일)
# ═══════════════════════════════════════════════════════════════════
def combined_score_pairs(Up, Ip, Uv, Iv, gate, lam, u, i):
    """(u,i) 쌍의 점수. BPR 학습용. raw dot product."""
    return (Up[u] * Ip[i]).sum(1) + lam * gate[u] * (Uv[u] * Iv[i]).sum(1)


def combined_score_all(Up, Ip, Uv, Iv, gate, lam, u):
    """유저 배치 × 전체 아이템 점수 [B, n_items]. 평가용. 위와 같은 정의여야 한다."""
    return Up[u] @ Ip.T + (lam * gate[u]).unsqueeze(1) * (Uv[u] @ Iv.T)


class MLP(nn.Module):
    """마지막 층은 LayerNorm/활성화 없이 Linear로 끝내고 작게 초기화한다.

    v3.1~3.2는 출력에 LayerNorm이 있어 가치 임베딩의 크기가 학습과 무관하게 단위
    스케일로 고정됐다. 그 결과 학습 시작 시점부터 가치 점수가 선호 점수보다 약 100배
    커서, λ_train=1로 함께 학습하는 joint에서는 BPR이 처음부터 가치공간에 지배되고
    선호공간이 학습될 기회를 잃는다. 학습 후 진단으로는 늦으므로 구조에서 막는다.
    작게 시작해두면 필요할 때 모델이 스스로 키울 수 있다."""
    def __init__(self, in_dim, hidden, out_dim, out_scale=0.01):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, out_dim))
        nn.init.normal_(self.net[-1].weight, std=out_scale)
        nn.init.zeros_(self.net[-1].bias)
    def forward(self, x):
        return self.net(x)


class DualSpaceLightGCN(nn.Module):
    """z^pref: 자유 임베딩 + 표준 LightGCN 전파(협업 공간).
    z^value: CLV 변수/가격 속성 MLP 출력, **전파 없음**(속성 공간).

    가치 블록에 전파를 태우지 않는 이유: (1) 태우면 유저 가치 임베딩이 "산 물건들의
    평균"이 되는데 그건 z^pref가 이미 하는 일이라 같은 협업 신호를 중복 학습한다.
    (2) two_stage 쪽은 구조상 MLP라 전파가 없어, joint만 태우면 두 아키텍처가 두 가지
    점에서 달라져 비교가 해석되지 않는다."""

    def __init__(self, n_users, n_items, n_cat, x_val_u, x_item, item_cat, cfg, adj):
        super().__init__()
        self.n_users, self.n_items, self.cfg, self.adj = n_users, n_items, cfg, adj
        self.E_u = nn.Embedding(n_users, cfg["DIM"]); self.E_i = nn.Embedding(n_items, cfg["DIM"])
        nn.init.normal_(self.E_u.weight, std=0.1); nn.init.normal_(self.E_i.weight, std=0.1)
        self.cat_emb = nn.Embedding(n_cat, cfg["CAT_EMB_DIM"])
        nn.init.normal_(self.cat_emb.weight, std=0.1)
        self.mlp_u = MLP(x_val_u.shape[1], cfg["MLP_HIDDEN"], cfg["D_VALUE"], cfg["VALUE_OUT_SCALE"])
        self.mlp_i = MLP(x_item.shape[1] + cfg["CAT_EMB_DIM"], cfg["MLP_HIDDEN"],
                         cfg["D_VALUE"], cfg["VALUE_OUT_SCALE"])
        self.register_buffer("x_val_u", torch.from_numpy(x_val_u))
        self.register_buffer("x_item", torch.from_numpy(x_item))
        self.register_buffer("item_cat", torch.from_numpy(item_cat))
        self._pref_cache = None          # 동결 학습(stage2) 중 전파 재계산을 피하기 위한 캐시

    def pref_params(self):
        return list(self.E_u.parameters()) + list(self.E_i.parameters())

    def value_params(self):
        return (list(self.cat_emb.parameters()) + list(self.mlp_u.parameters())
                + list(self.mlp_i.parameters()))

    def propagate_pref(self):
        x = torch.cat([self.E_u.weight, self.E_i.weight], dim=0)
        out = x
        for _ in range(self.cfg["N_LAYERS"]):
            x = torch.sparse.mm(self.adj, x)
            out = out + x
        out = out / (self.cfg["N_LAYERS"] + 1)
        return out[:self.n_users], out[self.n_users:]

    def freeze_pref_and_cache(self):
        """선호 블록을 동결하고 전파 결과를 한 번만 계산해 둔다. stage2가 매 배치
        희소행렬 곱을 다시 하지 않도록 하는 것 — 동결됐으므로 값이 변하지 않는다."""
        for p in self.pref_params():
            p.requires_grad_(False)
        with torch.no_grad():
            self._pref_cache = tuple(t.detach() for t in self.propagate_pref())

    def pref_emb(self):
        return self._pref_cache if self._pref_cache is not None else self.propagate_pref()

    def value_emb(self):
        zu = self.mlp_u(self.x_val_u)
        zi = self.mlp_i(torch.cat([self.x_item, self.cat_emb(self.item_cat)], dim=1))
        return zu, zi

    def embeddings(self, need_value=True):
        Up, Ip = self.pref_emb()
        if not need_value:      # stage1은 λ=0이라 가치항이 점수에 기여하지 않는다
            return Up, Ip, None, None
        Uv, Iv = self.value_emb()
        return Up, Ip, Uv, Iv

    def batch_l2(self, u, i, j, need_value):
        """배치에 등장한 **layer-0** 선호 임베딩만 L2 (v2·LightGCN 공식 구현과 동일).
        전역 weight decay와 달리 배치에 안 들어온 행은 건드리지 않는다 — 30만 유저
        규모에서 전역 감쇠를 걸면 선호 임베딩이 통째로 0으로 수축한다(v3.1의 붕괴 원인).

        선호 블록이 동결된 단계(two_stage의 stage2)에서는 기울기가 없으므로 건너뛴다.
        가치 쪽은 layer-0 임베딩이라는 게 없는 MLP라 파라미터 L2로 따로 둔다(기본 0)."""
        cfg = self.cfg
        reg = torch.zeros((), device=self.E_u.weight.device)
        if cfg["PREF_REG"] > 0 and self.E_u.weight.requires_grad:
            reg = reg + cfg["PREF_REG"] * (
                self.E_u.weight[u].pow(2).sum()
                + self.E_i.weight[i].pow(2).sum()
                + self.E_i.weight[j].pow(2).sum()) / len(u)
        if need_value and cfg["VALUE_REG"] > 0:
            vp = [q for q in self.value_params() if q.requires_grad]
            if vp:
                reg = reg + cfg["VALUE_REG"] * sum(q.pow(2).sum() for q in vp)
        return reg

    def bpr_loss(self, u, i, j, gate, lam):
        """반환 (loss, 진단값). 진단값은 붕괴를 매 epoch 눈으로 확인하기 위한 것."""
        need_value = lam != 0.0
        Up, Ip, Uv, Iv = self.embeddings(need_value=need_value)
        if not need_value:
            pos = (Up[u] * Ip[i]).sum(1); neg = (Up[u] * Ip[j]).sum(1)
        else:
            pos = combined_score_pairs(Up, Ip, Uv, Iv, gate, lam, u, i)
            neg = combined_score_pairs(Up, Ip, Uv, Iv, gate, lam, u, j)
        bpr = -F.logsigmoid(pos - neg).mean()
        loss = bpr + (self.batch_l2(u, i, j, need_value)
                      if self.cfg["REG_MODE"] == "batch_l2" else 0.0)
        with torch.no_grad():
            diag = {"bpr": float(bpr), "p_correct": float((pos > neg).float().mean())}
        return loss, diag

    @torch.no_grad()
    def score_diagnostics(self, gate, n_sample=512, seed=0):
        """학습된 상대 스케일 진단 — λ=1이 실제로 어느 정도의 개입인지 확인용."""
        Up, Ip, Uv, Iv = self.embeddings()
        rng = np.random.default_rng(seed)
        us = torch.as_tensor(rng.choice(self.n_users, min(n_sample, self.n_users), replace=False),
                             dtype=torch.long, device=Up.device)
        sp = (Up[us] @ Ip.T)
        # gate를 곱한 **실효** 가치 점수로 비교해야 λ=1이 실제로 어느 정도의 개입인지 알 수 있다.
        # v3.1에서는 gate를 인자로 받고도 쓰지 않아 비율이 과대평가됐다.
        sv_eff = gate[us].unsqueeze(1) * (Uv[us] @ Iv.T)
        return {"std_s_pref": float(sp.std()), "std_s_value": float(sv_eff.std()),
                "ratio_value_over_pref": float(sv_eff.std() / (sp.std() + 1e-12)),
                "mean_norm_U_pref": float(Up.norm(dim=1).mean()),
                "mean_norm_I_pref": float(Ip.norm(dim=1).mean()),
                "mean_norm_U_value": float(Uv.norm(dim=1).mean()),
                "mean_norm_I_value": float(Iv.norm(dim=1).mean())}


# ═══════════════════════════════════════════════════════════════════
# 지표
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
        sr = np.sort(rev[u])[::-1]
        out[u] = np.cumsum(sr * (1.0 / np.log2(np.arange(2, len(sr) + 2))))
    return out


def score_topk(topk, bu, ks, pos_key_sorted, pos_rev_sorted, n_items,
               P_arr, price_pct, item_nov, cat_arr, ideal_rev_cumsum):
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
        nh = hit_k.sum(axis=1); Pk = np.minimum(P_batch, k)
        idcg = np.where(Pk > 0, np.cumsum(disc_k)[np.maximum(Pk, 1) - 1], 0.0)
        dcg = (hit_k * disc_k).sum(axis=1)
        map_num = (np.cumsum(hit_k, axis=1) * hit_k / np.arange(1, k + 1)).sum(axis=1)
        idcgv = np.array([ideal_rev_cumsum[u][min(len(ideal_rev_cumsum[u]), k) - 1]
                           if len(ideal_rev_cumsum[u]) > 0 else 0.0 for u in bu])
        dcgv = (gain_k * disc_k).sum(axis=1)
        div = np.array([len(np.unique(r)) / k for r in cat_row[:, :k]])
        out[k] = {"recall": np.where(P_batch > 0, nh / np.maximum(P_batch, 1), 0.0),
                  "precision": nh / k,
                  "ndcg": np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0),
                  "hr": (nh > 0).astype(np.float64),
                  "map": np.where(Pk > 0, map_num / np.maximum(Pk, 1), 0.0),
                  "revenue": gain_k.sum(axis=1),
                  "vndcg": np.where(idcgv > 0, dcgv / np.maximum(idcgv, 1e-12), 0.0),
                  "arp": price_row[:, :k].mean(axis=1),
                  "novelty": nov_row[:, :k].mean(axis=1),
                  "diversity": div}
    return out


class EvalCache:
    """split 하나에 대해 재사용되는 조회표. λ마다·epoch마다 다시 만들지 않는다."""
    def __init__(self, gt, rev, clv, seg_th, n_items):
        self.gt, self.rev = gt, rev
        self.users = np.array(list(gt.keys()))
        self.pos_key, self.pos_rev = build_pos_lookup(gt, rev, n_items)
        self.ideal = build_ideal_rev_cumsum(gt, rev)
        self.P_arr = np.zeros(int(self.users.max()) + 1, dtype=np.int64)
        for u in gt: self.P_arr[u] = len(gt[u])
        med = np.nanmedian(clv)
        uclv = np.array([clv[u] if not np.isnan(clv[u]) else med for u in self.users])
        lo, hi = seg_th
        self.uclv = uclv
        self.seg = np.where(uclv <= lo, "저CLV", np.where(uclv >= hi, "고CLV", "중CLV"))
        self.seg_cnt = {s: int((self.seg == s).sum()) for s in SEG_NAMES}


@torch.no_grad()
def evaluate(model, lam, gate_t, cache, meta, ks, csr_ptr, csr_items, cfg, per_user=False):
    Up, Ip, Uv, Iv = model.embeddings()
    n_items = Ip.shape[0]
    price_pct, pop_prob, cat = meta["price_pct"], meta["pop_prob"], meta["cat"]
    item_nov = -np.log2(pop_prob + 1e-12)

    overall = {k: {m: 0.0 for m in _METS} for k in ks}
    seg_acc = {k: {s: {m: 0.0 for m in _METS} for s in SEG_NAMES} for k in ks}
    expo = {k: np.zeros(n_items) for k in ks}
    cal_v, cal_p = [], []
    pu = {m: [] for m in _METS} if per_user else None
    max_k, k0 = max(ks), ks[0]

    for s0 in range(0, len(cache.users), cfg["EVAL_BATCH"]):
        bu = cache.users[s0:s0 + cfg["EVAL_BATCH"]]
        ut = torch.as_tensor(bu, dtype=torch.long, device=DEVICE)
        scores = combined_score_all(Up, Ip, Uv, Iv, gate_t, lam, ut)
        for bi, u in enumerate(bu):
            a, b = csr_ptr[u], csr_ptr[u + 1]
            if b > a: scores[bi, csr_items[a:b]] = -1e9
        topk = scores.topk(max_k, dim=1).indices.cpu().numpy()
        res = score_topk(topk, bu, ks, cache.pos_key, cache.pos_rev, n_items,
                         cache.P_arr, price_pct, item_nov, cat, cache.ideal)
        bseg = cache.seg[s0:s0 + cfg["EVAL_BATCH"]]
        for k in ks:
            for m in _METS:
                overall[k][m] += res[k][m].sum()
                for sg in SEG_NAMES:
                    msk = bseg == sg
                    if msk.any(): seg_acc[k][sg][m] += res[k][m][msk].sum()
            for bi in range(len(bu)): expo[k][topk[bi, :k]] += 1
        if per_user:
            for m in _METS: pu[m].append(res[k0][m])
        cal_v.extend(cache.uclv[s0:s0 + cfg["EVAL_BATCH"]]); cal_p.extend(res[k0]["arp"].tolist())

    n = len(cache.users)
    for k in ks:
        for m in _METS:
            overall[k][m] /= n
            for sg in SEG_NAMES:
                if cache.seg_cnt[sg]: seg_acc[k][sg][m] /= cache.seg_cnt[sg]
    out = dict(overall=overall, seg=seg_acc, seg_cnt=cache.seg_cnt, n_eval=n,
               coverage={k: float((expo[k] > 0).sum()) / n_items for k in ks},
               gini={k: _gini(expo[k]) for k in ks},
               value_alignment=_spearman(cal_v, cal_p))
    if per_user:
        out["per_user"] = {m: np.concatenate(pu[m]) for m in _METS}
    return out


# ═══════════════════════════════════════════════════════════════════
# 학습
# ═══════════════════════════════════════════════════════════════════
def sample_negatives(u_arr, pos_arr, n_items, pos_key, rng, mode="uniform",
                     item_cat=None, cat_items=None, max_try=50):
    """negative 샘플링. 기본은 **균등 무작위**(LightGCN 원논문).

    mode="hard50"은 v2가 쓰던 방식으로, 절반을 양성 아이템과 같은 카테고리에서 뽑는다.
    기본값이 아니라 **ablation 전용**이다 — 정규화 수정과 동시에 바꾸면 무엇이 효과를
    냈는지 가릴 수 없어, 정상 baseline 복구를 확인한 뒤 따로 비교한다."""
    n = len(u_arr)
    neg = rng.integers(0, n_items, size=n)
    if mode == "hard50":
        hard = rng.random(n) < 0.5
        for k in np.where(hard)[0]:
            cand = cat_items.get(int(item_cat[pos_arr[k]]))
            if cand is not None and len(cand) > 1:
                neg[k] = cand[rng.integers(0, len(cand))]
    elif mode != "uniform":
        raise ValueError(f"NEG_MODE: {mode}")

    u64 = u_arr.astype(np.int64)
    for _ in range(max_try):
        key = u64 * n_items + neg
        pos = np.clip(np.searchsorted(pos_key, key), 0, len(pos_key) - 1)
        bad = pos_key[pos] == key
        if not bad.any(): break
        neg[bad] = rng.integers(0, n_items, size=int(bad.sum()))   # 재추첨은 균등으로
    return neg


def train_phase(model, params, d, gate_t, lam_train, cfg, seed, tag, val_cache, meta):
    """BPR로 params만 학습. 조기종료가 수렴점을 결정한다(상한 EPOCHS).
    학습량(업데이트 수·샘플 수·시간)을 함께 기록해 아키텍처 간 비교에 쓴다."""
    # REG_MODE="batch_l2"면 optimizer에 감쇠를 주지 않는다(정규화는 loss 안에서).
    wd = cfg["WD"] if cfg["REG_MODE"] == "global_wd" else 0.0
    opt = torch.optim.Adam(params, lr=cfg["LR"], weight_decay=wd)
    rng = np.random.default_rng(seed)
    tr_u, tr_i, pos_key = d["tr_u"], d["tr_i"], d["pos_key"]
    n_train = len(tr_u); n_batch = math.ceil(n_train / cfg["BATCH_SIZE"])
    best, best_ep, best_state, bad = -1.0, -1, None, 0
    updates, samples, t_start = 0, 0, time.time()

    for ep in range(1, cfg["EPOCHS"] + 1):
        model.train(); t0 = time.time()
        perm = rng.permutation(n_train); tot = tot_bpr = tot_pc = 0.0
        for b in range(n_batch):
            idx = perm[b * cfg["BATCH_SIZE"]:(b + 1) * cfg["BATCH_SIZE"]]
            bu, bi = tr_u[idx], tr_i[idx]
            bj = sample_negatives(bu, bi, d["n_items"], pos_key, rng, cfg["NEG_MODE"],
                                  d["item_cat"], d["cat_items"])
            loss, dg = model.bpr_loss(torch.as_tensor(bu, dtype=torch.long, device=DEVICE),
                                      torch.as_tensor(bi, dtype=torch.long, device=DEVICE),
                                      torch.as_tensor(bj, dtype=torch.long, device=DEVICE),
                                      gate_t, lam_train)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); tot_bpr += dg["bpr"]; tot_pc += dg["p_correct"]
            updates += 1; samples += len(idx)
        model.eval()
        r = evaluate(model, lam_train, gate_t, val_cache, meta, [cfg["SELECT_K"]],
                     d["csr_ptr"], d["csr_items"], cfg)
        score = r["overall"][cfg["SELECT_K"]][cfg["SELECT_METRIC"]]
        star = ""
        if score > best:
            best, best_ep, bad = score, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            star = " ★"
        else:
            bad += 1
        # 붕괴를 매 epoch 눈으로 확인 — BPR은 0.693에서 시작해 내려가야 하고,
        # P(pos>neg)는 0.5에서 올라가야 하며, layer-0 norm이 0으로 수축하면 안 된다.
        with torch.no_grad():
            nu = float(model.E_u.weight.norm(dim=1).mean())
            ni = float(model.E_i.weight.norm(dim=1).mean())
        print(f"  [{tag}] ep {ep:3d} | loss {tot/n_batch:.4f} bpr {tot_bpr/n_batch:.4f} "
              f"P(pos>neg) {tot_pc/n_batch:.3f} | ‖E_u‖ {nu:.4f} ‖E_i‖ {ni:.4f} | "
              f"val {cfg['SELECT_METRIC']}@{cfg['SELECT_K']} {score:.5f} | {time.time()-t0:.0f}s{star}")
        if bad >= cfg["EARLY_STOP"]:
            print(f"  [{tag}] early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    # 값은 전부 순수 파이썬 타입으로 — numpy 스칼라가 섞이면 체크포인트를
    # torch.load(weights_only=True)로 되읽을 수 없다.
    stats = {"phase": tag, "best_epoch": int(best_ep), "epochs_run": int(ep),
             f"best_val_{cfg['SELECT_METRIC']}@{cfg['SELECT_K']}": float(best),
             "updates": int(updates), "samples": int(samples),
             "wall_clock_sec": round(time.time() - t_start, 1)}
    print(f"  [{tag}] 완료 — best ep {best_ep}/{ep}, val {best:.5f}, "
          f"업데이트 {updates:,}회, 샘플 {samples:,}건, {stats['wall_clock_sec']:.0f}s")
    return stats


def build_model(d, x_val_u, x_item, item_cat, cfg):
    return DualSpaceLightGCN(d["n_users"], d["n_items"], d["n_cat"],
                             x_val_u, x_item, item_cat, cfg, d["adj"]).to(DEVICE)


def get_or_train(arch, seed, d, gate_t, x_val_u, x_item, item_cat, meta, val_cache, cfg):
    """arch별 학습. pref_only 체크포인트는 two_stage가 stage2 초기값으로 재사용한다."""
    out = Path(cfg["OUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
    ck = out / f"ckpt_{arch}_{cfg['DATASET']}_s{seed}_{cfg_hash(cfg, DCFG, arch, seed)}.pt"
    if ck.exists():
        model = build_model(d, x_val_u, x_item, item_cat, cfg)
        blob = torch.load(ck, map_location=DEVICE)
        model.load_state_dict(blob["state"])
        print(f"[{arch} s{seed}] 체크포인트 로드 ({ck.name})")
        return model, blob["train_stats"]

    set_seed(seed)
    if arch == "pref_only":
        model = build_model(d, x_val_u, x_item, item_cat, cfg)
        stats = [train_phase(model, model.pref_params(), d, gate_t, 0.0, cfg, seed,
                             "pref_only", val_cache, meta)]
    elif arch == "two_stage":
        # stage1 = pref_only 체크포인트를 그대로 재사용 (없으면 여기서 학습됨)
        base, base_stats = get_or_train("pref_only", seed, d, gate_t, x_val_u, x_item,
                                        item_cat, meta, val_cache, cfg)
        model = build_model(d, x_val_u, x_item, item_cat, cfg)
        model.load_state_dict(base.state_dict())
        model.freeze_pref_and_cache()
        stats = list(base_stats) + [train_phase(model, model.value_params(), d, gate_t,
                                                cfg["LAMBDA_TRAIN"], cfg, seed,
                                                "two_stage-value", val_cache, meta)]
    elif arch == "joint_warm":
        # two_stage와 같은 출발점이지만 **동결하지 않는다**. 차이는 freeze 한 줄뿐.
        base, base_stats = get_or_train("pref_only", seed, d, gate_t, x_val_u, x_item,
                                        item_cat, meta, val_cache, cfg)
        model = build_model(d, x_val_u, x_item, item_cat, cfg)
        model.load_state_dict(base.state_dict())
        stats = list(base_stats) + [train_phase(model, list(model.parameters()), d, gate_t,
                                                cfg["LAMBDA_TRAIN"], cfg, seed,
                                                "joint_warm", val_cache, meta)]
    elif arch == "joint":
        model = build_model(d, x_val_u, x_item, item_cat, cfg)   # 별도 random init
        stats = [train_phase(model, list(model.parameters()), d, gate_t,
                             cfg["LAMBDA_TRAIN"], cfg, seed, "joint", val_cache, meta)]
    else:
        raise ValueError(f"ARCH: {arch}")

    torch.save({"state": model.state_dict(), "train_stats": stats}, ck)
    print(f"[{arch} s{seed}] 저장 → {ck}")
    return model, stats


# ═══════════════════════════════════════════════════════════════════
# λ 선택 (validation) — 규칙을 코드에 사전 고정
# ═══════════════════════════════════════════════════════════════════
def paired_bootstrap(diffs_per_seed, n_boot, seed=0):
    """유저 단위 paired bootstrap. **시드는 독립 표본이 아니다.**

    diffs_per_seed: [n_seeds][n_users] — 시드별 (모델 − baseline) 유저별 차이.
    같은 유저가 시드마다 반복되므로 이어붙여 재표집하면 표본 수가 부풀어 신뢰구간이
    실제보다 좁아진다(v3.2까지의 버그). 유저별로 시드 평균을 먼저 낸 뒤 **유저를**
    재표집한다. 시드 간 변동은 per_seed_mean/sd로 따로 보고한다."""
    d = np.stack(diffs_per_seed)                 # [S, U] — 같은 split이면 유저 순서 동일
    dbar = d.mean(axis=0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(dbar), size=(n_boot, len(dbar)))
    means = dbar[idx].mean(axis=1)
    return {"mean": float(dbar.mean()),
            "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)),
            "per_seed_mean": [float(x) for x in d.mean(axis=1)],
            "per_seed_sd": float(d.mean(axis=1).std(ddof=1)) if len(d) > 1 else 0.0}


def select_lambda(val_per_seed, base_per_seed, cfg):
    """아키텍처별 공통 λ 하나를 validation에서 고른다. **시드별 선택은 하지 않는다.**

    규칙 (2026-08-07 변경 — 비열등성 가드레일 폐기):
      validation PWGain@10(시드 평균)이 최대인 λ. 동률이면 더 작은 λ.

    v3.5까지는 "CI하한(ΔRecall@10) ≥ -δ × baseline"을 만족하는 λ만 후보로 남긴
    뒤 그중 PWGain@10을 극대화했다. 하지만 이 가드레일은 Recall@10 하나만 보호할 뿐
    Recall@20/50 같은 다른 지표 손실은 전혀 못 걸러냈고(joint_warm λ=2.0에서 실측 확인),
    이 논문은 애초에 Recall 자체를 핵심 지표로 삼지 않는다는 결론이 나서 폐기했다.
    ΔRecall@10과 그 신뢰구간은 여전히 계산해 참고용으로 출력하되 선택에는 안 쓴다.
    """
    lams = sorted(next(iter(val_per_seed.values())).keys())
    # 비교 기준은 **언제나 외부 pref_only**다. joint의 λ=0은 선호 임베딩이 이미 가치항에
    # 적응한 ablation이라 baseline이 아니다(v3.2까지 자기 λ=0과 비교하던 것을 고침).
    base_recall = np.mean([base_per_seed[s]["agg"]["overall"][10]["recall"] for s in base_per_seed])
    delta_abs = cfg["NONINFERIORITY_DELTA"] * base_recall
    rows = []
    for lam in lams:
        ci = paired_bootstrap([val_per_seed[s][lam]["pu"]["recall"] - base_per_seed[s]["pu"]["recall"]
                               for s in val_per_seed], cfg["N_BOOT"])
        pw = np.mean([val_per_seed[s][lam]["agg"]["overall"][10]["revenue"] for s in val_per_seed])
        rows.append({"lambda": lam, "d_recall_mean": ci["mean"], "d_recall_ci_lo": ci["lo"],
                     "d_recall_seed_sd": ci["per_seed_sd"],
                     "ref_would_pass_old_gate": ci["lo"] >= -delta_abs, "val_pwgain10": float(pw)})
    df = pd.DataFrame(rows)
    print(f"\n[λ 선택] 가드레일 없음 — val PWGain@10 최대인 λ를 그대로 선택한다. "
          f"(참고: 구 비열등 기준선 -{delta_abs:.6f} = -{cfg['NONINFERIORITY_DELTA']:.1%} × baseline {base_recall:.6f})")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    best = df.sort_values(["val_pwgain10", "lambda"], ascending=[False, True]).iloc[0]
    print(f"  → 선택 λ = {best['lambda']} (val PWGain@10 최대, 후보 {len(df)}개)")
    return float(best["lambda"]), df


def flatten(res):
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


# ═══════════════════════════════════════════════════════════════════
def run_2x2_diagnostic():
    """정규화 × negative 샘플링 2×2를 pref_only·시드 1개로 돌려 붕괴 원인을 분리한다.

      배치L2 + uniform  : 정상 복구 여부 확인 (기대되는 정상 조건)
      전역WD + uniform  : v3.1의 실패 재현
      전역WD + hard50   : hard negative가 붕괴를 가리는지
      배치L2 + hard50   : v2에 가장 가까운 조건

    네 조합의 차이가 정규화 축에서 나오면 원인이 정규화라는 뜻이다."""
    base = dict(CFG)
    rows = []
    for reg in ("batch_l2", "global_wd"):
        for neg in ("uniform", "hard50"):
            CFG.update(base); CFG.update(ARCH="pref_only", REG_MODE=reg, NEG_MODE=neg,
                                         SEED_LIST=[base["SEED_LIST"][0]])
            print(f"\n{'#'*84}\n#  진단: REG_MODE={reg} | NEG_MODE={neg}\n{'#'*84}")
            df = main()
            r = df[(df.split == "test") & (df["lambda"] == 0.0)].iloc[0]
            rows.append({"reg": reg, "neg": neg, "recall@10": r["recall@10"],
                         "ndcg@10": r["ndcg@10"], "coverage@10": r["coverage@10"]})
    CFG.update(base)
    out = pd.DataFrame(rows)
    print(f"\n{'='*84}\n2×2 진단 요약 (pref_only, test @10)\n{'='*84}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return out


def main():
    cfg, arch = CFG, CFG["ARCH"]
    print(f"DATASET={cfg['DATASET']} | ARCH={arch} ({ARCH_LABEL[arch]}) | "
          f"DEVICE={DEVICE} | CODE={CODE_VERSION}")
    d = prepare_data(cfg, DCFG)
    x_val_u, clv, vhat = clv_features(d["train"], d["n_users"], cfg, DCFG["is_date"])
    x_item, item_cat = item_value_features(d["train"], d["n_items"])
    meta = item_meta(d["train"], d["n_items"])
    gate = build_gate(clv, vhat, cfg["GATE_MODE"]); gate_t = torch.from_numpy(gate).to(DEVICE)
    # 세그먼트는 게이트와 무관하게 **항상 CLV 기준**이다 — CLV는 분석변수(어느 고객군에서
    # 이득/손실이 났는지)이고, 게이트는 개입변수라 역할이 다르다. GATE_MODE를 바꿔도
    # 세그먼트 정의가 따라 바뀌면 모드 간 세그먼트 결과를 비교할 수 없다.
    seg_th = segment_thresholds(clv, cfg["SEG_EDGES"])
    caches = {name: EvalCache(gt, rev, clv, seg_th, d["n_items"])
              for name, (gt, rev) in d["splits"].items()}

    # pref_only는 가치 블록이 학습되지 않는다(λ=0이라 기울기가 0). 그 상태로 λ>0을
    # 채점하면 **무작위 초기화된** 가치 임베딩을 주입하는 셈이라 의미가 없다.
    # baseline은 λ=0 하나만 평가한다.
    sweep = [0.0] if arch == "pref_only" else cfg["LAMBDA_EVAL_SWEEP"]
    if arch == "pref_only":
        print("  (pref_only는 가치 블록이 미학습이므로 λ=0만 평가한다)")

    train_stats, diagnostics = {}, {}
    val_per_seed, base_per_seed, test_rows = {}, {}, []
    pu_split, base_pu_split = {}, {}       # split → λ → seed → 유저별 지표
    for seed in cfg["SEED_LIST"]:
        print(f"\n{'='*84}\nseed {seed} | ARCH={arch}\n{'='*84}")
        model, stats = get_or_train(arch, seed, d, gate_t, x_val_u, x_item, item_cat,
                                    meta, caches["val"], cfg)
        model.eval()
        train_stats[seed] = stats
        diagnostics[seed] = model.score_diagnostics(gate_t)
        rat = diagnostics[seed]["ratio_value_over_pref"]
        warn = ""
        if arch != "pref_only" and not (0.1 <= rat <= 10):
            warn = ("  ⚠ 두 점수의 스케일 차가 큽니다 — λ=1이 의도보다 훨씬 강한(또는 약한) "
                    "개입일 수 있으니 LAMBDA_EVAL_SWEEP 범위를 이 비율에 맞춰 재검토할 것")
        dg = diagnostics[seed]
        print(f"  진단: std(s_pref)={dg['std_s_pref']:.4f} "
              f"std(gate·s_value)={dg['std_s_value']:.4f} 비율={rat:.4f}")
        print(f"        layer-0 평균 norm: ‖U_pref‖={dg['mean_norm_U_pref']:.4f} "
              f"‖I_pref‖={dg['mean_norm_I_pref']:.4f} | "
              f"‖U_value‖={dg['mean_norm_U_value']:.4f} ‖I_value‖={dg['mean_norm_I_value']:.4f}")
        if dg["std_s_pref"] < 1e-3:
            print("        ⚠ std(s_pref)가 0에 가깝습니다 — 선호 임베딩 붕괴 의심")
        if warn:
            print(f"      {warn.strip()}")

        # 비교 기준 = **외부 pref_only(같은 시드)**. arch가 pref_only면 자기 자신.
        if arch == "pref_only":
            base_model = model
        else:
            base_model, _ = get_or_train("pref_only", seed, d, gate_t, x_val_u, x_item,
                                         item_cat, meta, caches["val"], cfg)
            base_model.eval()

        val_per_seed[seed] = {}
        for lam in sweep:
            r = evaluate(model, lam, gate_t, caches["val"], meta, cfg["K_LIST"],
                         d["csr_ptr"], d["csr_items"], cfg, per_user=True)
            val_per_seed[seed][lam] = {"pu": r.pop("per_user"), "agg": r}
        rb = evaluate(base_model, 0.0, gate_t, caches["val"], meta, cfg["K_LIST"],
                      d["csr_ptr"], d["csr_items"], cfg, per_user=True)
        base_per_seed[seed] = {"pu": rb.pop("per_user"), "agg": rb}

        for split in [s for s in ("test", "holdout") if s in caches]:
            rb2 = evaluate(base_model, 0.0, gate_t, caches[split], meta, cfg["K_LIST"],
                           d["csr_ptr"], d["csr_items"], cfg, per_user=True)
            base_pu_split.setdefault(split, {})[seed] = rb2.pop("per_user")
            for lam in sweep:
                r = evaluate(model, lam, gate_t, caches[split], meta, cfg["K_LIST"],
                             d["csr_ptr"], d["csr_items"], cfg, per_user=True)
                pu_split.setdefault(split, {}).setdefault(lam, {})[seed] = r.pop("per_user")
                label = "ablation" if (arch in ABLATION_ARCHS and lam == 0.0) else "model"
                test_rows.append({"seed": seed, "arch": arch, "split": split,
                                  "lambda": lam, "role": label, **flatten(r)})

    # Δ는 **모든 시드를 모아** 유저 단위로 한 번에 계산한다(시드는 독립 표본이 아님).
    delta_rows = []
    for split, per_lam in pu_split.items():
        for lam, per_seed in per_lam.items():
            row = {"arch": arch, "split": split, "lambda": lam}
            for m in ("recall", "ndcg", "revenue", "arp"):
                ci = paired_bootstrap([per_seed[s][m] - base_pu_split[split][s][m]
                                       for s in sorted(per_seed)], cfg["N_BOOT"])
                row[f"d_{m}_mean"], row[f"d_{m}_lo"], row[f"d_{m}_hi"] = ci["mean"], ci["lo"], ci["hi"]
                row[f"d_{m}_seed_sd"] = ci["per_seed_sd"]
            delta_rows.append(row)
    delta_df = pd.DataFrame(delta_rows)

    sel_lambda, sel_table = select_lambda(val_per_seed, base_per_seed, cfg)

    out = Path(cfg["OUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
    stem = f"result_{arch}_{cfg['DATASET']}_{result_hash(cfg, DCFG, arch)}"
    df = pd.DataFrame(test_rows)
    df.to_csv(out / f"{stem}.csv", index=False, float_format="%.6f")
    delta_df.to_csv(out / f"{stem}_delta.csv", index=False, float_format="%.6f")
    val_rows = [{"seed": s, "arch": arch, "split": "val", "lambda": lam,
                 **flatten(v["agg"])} for s in val_per_seed for lam, v in val_per_seed[s].items()]
    pd.DataFrame(val_rows).to_csv(out / f"{stem}_val.csv", index=False, float_format="%.6f")
    with open(out / f"{stem}.json", "w") as f:
        json.dump({"code_version": CODE_VERSION,
                   "cfg": {k: v for k, v in cfg.items() if k != "OUT_DIR"},
                   "arch_label": ARCH_LABEL[arch],
                   "selected_lambda_from_validation": sel_lambda,
                   "baseline_for_comparison": "pref_only (same seed)",
                   "delta_table": delta_df.to_dict("records"),
                   "lambda_selection_table": sel_table.to_dict("records"),
                   "train_stats": train_stats, "score_diagnostics": diagnostics,
                   "segment_thresholds": {"low": seg_th[0], "high": seg_th[1],
                                          "note": "train 전체 고객 기준 고정"},
                   # MIN_ITEM_INTER를 바꿔 비교할 때 분모가 함께 변한다(희귀 아이템이
                   # 정답에서도 빠져 Recall이 기계적으로 오름). 이 블록 없이는 "모델이
                   # 좋아진 것"과 "어려운 정답이 사라진 것"을 구분할 수 없다.
                   "data_stats": d["data_stats"],
                   "gate": {"mode": cfg["GATE_MODE"],
                            "note": "유효 유저(개입 대상) 평균 1로 정규화. "
                                    "정규화가 없으면 모드별 평균 차이가 λ의 의미를 바꾼다."},
                   "note_primary": "주 검정은 validation에서 선택된 λ 하나에만 적용한다.",
                   "note_secondary": "test의 λ 곡선은 사전 선언된 민감도 분석이며 "
                                     "'최적 λ'가 아니고 λ별 유의 라벨을 붙이지 않는다.",
                   "note_revenue": "revenue = 가격가중 추천 적중값, 실제 매출 아님",
                   "note_arp": "arp = 추천 상품의 평균 가격 백분위(인기도 기반 ARP와 다름)",
                   }, f, indent=2, default=float, ensure_ascii=False)
    print(f"\n저장 → {out / (stem + '.csv')}\n     → {out / (stem + '_val.csv')}")

    print(f"\n{'='*84}\n[주 결과] validation 선택 λ = {sel_lambda} — test\n"
          f"  비교 기준: 외부 pref_only(같은 시드). joint의 λ=0은 ablation으로 별도 보고.\n{'='*84}")
    prim = df[(df.split == "test") & (df["lambda"] == sel_lambda)]
    bl = np.mean([base_pu_split["test"][s]["recall"].mean() for s in base_pu_split["test"]])
    print(f"  pref_only Recall@10 (3시드 평균) {bl:.6f}")
    for m in ("recall@10", "ndcg@10", "revenue@10", "arp@10", "value_alignment"):
        print(f"  {m:<18} 선택 λ에서 {prim[m].mean():.6f}")
    dsel = delta_df[(delta_df.split == "test") & (delta_df["lambda"] == sel_lambda)]
    if len(dsel):
        r = dsel.iloc[0]
        for m in ("recall", "ndcg", "revenue", "arp"):
            print(f"  Δ{m:<8} {r[f'd_{m}_mean']:+.6f} [{r[f'd_{m}_lo']:+.6f}, {r[f'd_{m}_hi']:+.6f}]"
                  f"  (시드 간 sd {r[f'd_{m}_seed_sd']:.6f})")
    if arch in ABLATION_ARCHS:
        abl = df[(df.split == "test") & (df["lambda"] == 0.0)]
        print(f"\n  [참고] {arch} λ=0 ablation Recall@10 {abl['recall@10'].mean():.6f} "
              f"— baseline이 아니라 가치항을 끈 {arch} 자신")

    print(f"\n{'='*84}\n[민감도] test λ 곡선 — 사전 선언된 descriptive 분석 (유의 라벨 없음)\n{'='*84}")
    g = df[df.split == "test"].groupby("lambda")[
        ["recall@10", "ndcg@10", "revenue@10", "arp@10", "diversity@10",
         "coverage@10", "value_alignment"]].mean()
    print(g.to_string(float_format=lambda x: f"{x:.6f}"))
    return df


if __name__ == "__main__":
    main()

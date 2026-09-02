# NGCF CLV Level-Composition-Price Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** LightGCN에서 상대적으로 가장 유망했던 고정 historical-CLV 수준·N/V 구성·가격 좌표를 NGCF의 layer-0 입력으로 옮기고, 동일 차원 및 degree-matched shuffle 대조군과 seed 42 개발구간에서 비교한다.

**Architecture:** 공통 binary user-item graph와 plain BPR을 유지한다. NGCF는 `E + A_hat E`와 `E ⊙ A_hat E`를 학습형 선형변환으로 결합하며, 실제 CLV arm에만 ID(64)+CLV 관계(2)+가격(1)의 67차원 layer-0 표현을 사용한다. CLV 사용자 입력은 고정 관측값이지만 이를 사용하는 NGCF 변환과 상품 투영은 같은 optimizer에서 추천손실로 공동학습한다.

**Tech Stack:** Python, PyTorch sparse tensors, NumPy, pandas, pytest, Jupyter/Colab JSON.

---

### Task 1: NGCF 표현 모델

**Files:**
- Create: `ngcf_clv_level_composition_price_model.py`
- Create: `test_ngcf_clv_level_composition_price.py`

1. 작은 정규화 그래프를 만드는 테스트 fixture를 작성한다.
2. NGCF@64/67의 layer-0 차원, 실제 CLV의 64+3 차원, CLV 사용자 좌표와 상품 가격 좌표를 검증한다.
3. NGCF propagation이 두 학습형 메시지 경로를 사용하고 layer 0·1·2를 concatenate하는지 검증한다.
4. 한 번의 BPR backward에서 ID, NGCF 변환, 상품 관계 투영, 가격 혼합 파라미터가 모두 gradient를 받는지 검증한다.
5. M3 edge weighting, M4 sample weighting, 외부 score residual 입력을 거부하는 테스트를 추가한다.
6. 모델 구현 후 `python -m pytest -q test_ngcf_clv_level_composition_price.py`를 실행한다.

### Task 2: Seed-42 Dunnhumby 개발구간 실행기

**Files:**
- Create: `ngcf_clv_level_composition_price_screen.py`
- Modify: `test_ngcf_clv_level_composition_price.py`

1. 고정 설정을 dataclass로 만들고 train≤683/eval 684–690, test/holdout 미구성, binary graph, uniform negative, unweighted BPR, MIN_ITEM_INTER=1을 preflight에 명시한다.
2. 기존 검증된 전처리·CLV·상품 가격 입력과 baseline 표시 행만 재사용한다.
3. `ngcf_m1_64`, `ngcf_m1_67`, 실제 CLV M2, degree-matched shuffled CLV M2 네 arm을 동일 seed와 fixed 100 epochs로 학습한다.
4. arm별 checkpoint 재개·캐시, 전체/CLV 구간 지표, Top-10 변경, gradient 및 표현 진단을 저장한다.
5. 실제 CLV가 NGCF@67과 shuffle을 모두 이겨야 하는 사전 판정 규칙을 구현한다.
6. config 보호와 shuffle 보존성에 대한 국소 테스트를 실행한다.

### Task 3: Colab 진입점과 실행 전 검증

**Files:**
- Create: `clv_m2_ngcf_level_composition_price_dunnhumby_colab.ipynb`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

1. 저장소 checkout, Drive mount, preflight 출력, 실행, 핵심 표 출력 셀만 가진 작은 Colab notebook을 작성한다.
2. notebook JSON과 Python 문법을 검사하고 전체 국소 테스트를 실행한다.
3. 보호 split이 만들어지지 않고 네 arm과 판정 규칙이 preflight에 정확히 기록되는지 확인한다.
4. 구현 사실과 아직 결과가 없는 상태를 `RESEARCH_STATUS.md`에 확정 사실/다음 실험으로 구분해 기록한다.
5. 변경 파일만 commit하고 `feat/m2-joint-nv-lightgcn` 브랜치에 push한다.

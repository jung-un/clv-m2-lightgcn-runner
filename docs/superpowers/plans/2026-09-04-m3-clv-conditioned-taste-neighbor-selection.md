# CLV-Conditioned Taste-Neighbor M3 Implementation Plan

> **For Codex:** Execute this plan task by task. Keep the fixed seed-42, train-only precheck, and no-holdout safeguards intact.

**Goal:** 구매취향 TF-IDF 상위 100명 안에서 historical CLV 수준·N/V 구성 유사도로 최종 이웃 20명을 선택하고, 그 관계를 LightGCN 2층 사용자 전파에 넣는 M3 파일럿과 Colab을 구현한다.

**Architecture:** 새 그래프 모듈이 train-only 이진 구매행렬, TF-IDF 후보 관계, CLV/셔플/degree 최종 관계 연산자를 만든다. 새 모델 모듈은 M1의 방향별 전파를 그대로 계산한 뒤 사용자 2층에만 고정 `gamma=0.075` 이웃 메시지를 결합한다. 다중시점 사전진단이 통과할 때만 네 M3 arm을 하나의 BPR 학습 루프로 실행한다.

**Tech Stack:** Python, NumPy, pandas, SciPy sparse, PyTorch, unittest, Colab notebook JSON

---

### Task 1: 그래프 관계의 불변조건을 테스트로 고정

**Files:**
- Create: `test_clv_m3_clv_taste_neighbor_graph.py`
- Create: `clv_m3_clv_taste_neighbor_graph.py`

1. CLV 분위·N/V 구성좌표, tuple shuffle 보존, 취향 후보 제한, arm별 동일 행질량·이웃 수, 자기 엣지 제거를 검사하는 실패 테스트를 작성한다.
2. `python -m unittest test_clv_m3_clv_taste_neighbor_graph.py -v`를 실행해 새 모듈 부재로 실패하는지 확인한다.
3. 이진 TF-IDF Top-100 후보와 `preference`, `actual_clv`, `clv_shuffle`, `degree_relation` Top-20 연산자를 구현한다.
4. 같은 명령으로 테스트를 통과시킨다.

### Task 2: LightGCN 내부 전파를 테스트로 고정

**Files:**
- Create: `test_clv_m3_clv_taste_neighbor_model.py`
- Create: `clv_m3_clv_taste_neighbor_model.py`

1. `gamma=0`과 무효 관계행의 정확한 M1 복귀, 상품 표현 불변, 사용자 2층 결합식, plain BPR gradient를 검사하는 실패 테스트를 작성한다.
2. 새 모델 모듈 부재로 실패하는지 확인한다.
3. 두 방향 이진 LightGCN 연산자와 arm별 사용자 이웃 연산자를 받는 2-layer 모델을 구현한다.
4. 모델 테스트를 통과시킨다.

### Task 3: train-only 다중시점 사전진단과 성능 runner 구현

**Files:**
- Create: `test_lightgcn_clv_m3_clv_taste_neighbor.py`
- Create: `lightgcn_clv_m3_clv_taste_neighbor_diagnostic.py`
- Create: `lightgcn_clv_m3_clv_taste_neighbor.py`

1. 고정 설정, 다섯 train-only 기준시점, paired Candidate Recall@100 판정, 실제/shuffle 이웃변경률, 학습 차단을 테스트한다.
2. 테스트 실패를 확인한 뒤 진단 runner를 구현한다.
3. 호환 M1 재사용, 네 arm의 동일 seed·초기화·고정 100 epoch 학습, 단일 test 평가, 여섯 정확도 지표 비교를 구현한다.
4. 관련 단위 테스트와 모듈 import 검사를 통과시킨다.

### Task 4: Colab과 연구상태 갱신

**Files:**
- Create: `clv_m3_clv_taste_neighbor_dunnhumby_colab.ipynb`
- Modify: `/Users/jungun/Workspace/논문준비/RESEARCH_STATUS.md`

1. Google Drive 연결, 저장소 clone/fetch, 고정 source commit checkout, 모듈 캐시 제거, 사전진단, 조건부 GPU 학습 순서의 Colab을 만든다.
2. M3 새 가설·고정값·학습 차단 규칙을 마스터 상태문서에 기록한다.
3. notebook JSON/compile과 전체 관련 테스트를 새로 실행한다.
4. 소스·테스트를 먼저 커밋하고 해당 commit을 notebook에 고정한 뒤 notebook·상태문서를 커밋한다.
5. `feat/m2-joint-nv-lightgcn` 브랜치에 push하고 Colab 링크를 제공한다.

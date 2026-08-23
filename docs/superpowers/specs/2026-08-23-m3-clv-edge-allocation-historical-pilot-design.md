# M3 historical CLV 엣지 배분 그래프 1시드 파일럿 설계

## 1. 목적과 연구 위치

이 실험은 M3의 그래프 구조·전파 개입을 검토하는 사후 탐색적 파일럿이다.
M1의 학습자료·고유 사용자–아이템 엣지 집합·임베딩·plain BPR·uniform
negative sampling은 유지하고, 상품이 사용자에게서 받는 전파계수의 상대적
구성만 historical CLV proxy와 엣지 관계특이성으로 바꾼다. M2 표현이나 M4
손실가중은 사용하지 않는다.

최종 Dunnhumby test는 이미 분석에 노출됐으므로 이 모델의 선택이나 평가에
재사용하지 않는다. 빠른 1시드 판단은 기존의 과거 개발구간인 1~683일 학습,
684~690일 평가에서 수행한다. 이 결과는 탐색적 백테스트이며 최종 확증으로
소급하지 않는다.

## 2. 파일럿 범위

- 데이터셋: Dunnhumby
- 학습구간: DAY 1~683
- 평가구간: DAY 684~690
- seed: 42
- 학습: 100 epoch 고정
- validation, early stopping, epoch 선택: 없음
- 모형: M1과 제안 M3 두 개
- 신규상품 평가: 학습구간의 모든 `(user,item)` 쌍을 평가 정답에서 제외
- `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`
- M1 checkpoint는 입력 manifest·분할·seed·차원·optimizer·epoch가 모두
  일치할 때만 기존 historical backtest 결과에서 fail-closed로 재사용한다.

## 3. 고객가치와 관계특이성

모든 값은 학습구간에서만 계산한다.

\[
N_u=\text{사용자 }u\text{의 서로 다른 장바구니 수},\qquad
V_u=\text{사용자 }u\text{의 평균 장바구니금액}
\]

\[
C_u=N_uV_u
\]

`C_u`는 미래 CLV 예측값이 아니라 학습기간 historical CLV proxy다.

사용자 `u`가 상품 `i`를 포함한 서로 다른 장바구니 수를 `f_ui`, 상품 `i`의
학습 구매고객 수를 `d_i`, 학습 사용자를 `|U|`라 한다. 가격을 사용하지 않는
TF-IDF 형태의 관계특이성을 다음과 같이 계산한다.

\[
r_{ui}=\log(1+f_{ui})\log\left(\frac{|U|+1}{d_i+1}\right)
\]

사용자 내부에서 합이 1이 되도록 배분비를 만든다.

\[
a_{ui}=\frac{r_{ui}}{\sum_{j\in N(u)}r_{uj}}
\]

분모가 0이면 해당 사용자의 고유 엣지에 균등 배분한다. 엣지별 CLV 배분값은

\[
b_{ui}=C_u a_{ui},\qquad \sum_{i\in N(u)}b_{ui}=C_u
\]

이다. 따라서 한 사용자의 CLV를 모든 엣지에 반복 부여하지 않고, 반복구매는
강화하되 전체적으로 흔한 상품은 할인한 관계에 나누어 배분한다.

## 4. LightGCN 전파계수

M1의 대칭 정규화 계수를

\[
\hat A=D^{-1/2}AD^{-1/2}
\]

라 한다. 각 상품 `i`의 학습 구매자 안에서 `b_ui`를 중간순위 백분위
`q_i(b_ui)`로 변환하고 다음 양의 계수를 사용한다.

\[
c_{ui}=0.5+q_i(b_{ui})
\]

상품이 받는 전체 메시지 질량을 M1과 같게 유지한다.

\[
\widetilde A_{iu}
=\hat A_{iu}c_{ui}
\frac{\sum_v\hat A_{iv}}
{\sum_v\hat A_{iv}c_{vi}}
\]

따라서 모든 상품에 대해

\[
\sum_u\widetilde A_{iu}=\sum_u\hat A_{iu}
\]

가 성립한다. 구매자가 한 명뿐인 상품은 자동으로 M1 계수와 같아진다.
별도의 `alpha`, `beta`, `lambda`, clipping 또는 성능 기반 강도 선택은 없다.

- `user <- item`: M1의 `\hat A` 그대로 사용
- `item <- user`: `\widetilde A` 사용
- 최종 점수·plain BPR·optimizer: M1과 동일

## 5. 사전 고정 판독

파일럿은 다음을 모두 만족할 때만 후속 검증 후보로 남긴다.

1. Recall/NDCG @10·20·50가 각각 같은 seed M1의 99% 이상
2. 가격·구매금액 가중 적중값@10이 M1보다 큼
3. 추천 상품 평균 가격 백분위가 M1의 97~103%
4. Top-10 서로 다른 추천상품 수가 M1의 95% 이상
5. Top-10 노출점유율 증가가 1%p 이하

seed 하나에서는 표준편차·신뢰구간·일반화·유의성을 주장하지 않는다. 지표가
미달해도 모든 불리한 결과를 저장하고 정상 종료한다.

파일럿이 통과하면 수식과 설정을 변경하지 않은 채 다음 단계로 확장한다.

- 동일 분할의 seed 42~51 반복 또는 H&M 외부 적용
- CLV 내용효과 식별을 위한 관계강도-only, N-only, V-only,
  고객 간 CLV 순열 대조군

제안모형이 이러한 대조군을 넘기 전에는 `N×V`로 정의한 CLV의 고유 효과를
주장하지 않는다.

## 6. 구현 경계와 검증

예정 구현은 다음 세 단위로 나눈다.

- 그래프 모듈: 관계특이성·CLV 배분·상품별 백분위·질량보존 계수 계산
- runner: historical split, M1 재사용 검증, 고정 100 epoch 학습, 평가·저장
- Colab: source SHA 고정, Drive 입출력, preflight, 실행 및 결과표

필수 자동검사는 다음과 같다.

- M1과 M3의 정렬된 unique edge index 완전 일치
- 사용자별 `sum_i a_ui=1`, `sum_i b_ui=C_u`
- 모든 전파계수의 유한성·양수성
- 상품별 수신계수 합이 M1과 수치허용오차 내 일치
- degree-1 상품의 M3 계수가 M1과 일치
- 학습쌍의 평가 정답·Top-K 유입 차단
- validation·최종 test·holdout 미구성
- 고정 seed·100 epoch·plain BPR·uniform negative sampling 불변

예정 파일명은 내부 구현 구분에만 사용하고 논문에서는 새 모형명을 만들지
않으며 수식과 정의로 풀어 쓴다.

- `clv_m3_edge_allocation_graph.py`
- `lightgcn_clv_m3_edge_allocation_backtest.py`
- `clv_m3_edge_allocation_backtest_colab.ipynb`

# M3-CLV-NV 구성형 가치 그래프 설계

## 1. 목적과 연구 위치

본 실험은 M3(그래프 구조·전파 개입)의 1차 탐색 실험이다. M1의 이진
사용자–아이템 그래프를 유지하되, historical CLV proxy가 형성된 N(거래활동)
축과 V(거래당 가치) 축을 사용자–아이템 관계 강도와 각각 결합해 엣지 가중치를
구성한다. 임베딩 초기값이나 BPR 손실가중을 바꾸지 않으므로 M2·M4와 구분한다.

기존 `GRAPH_MODE=clv`의 `CLV_u × item_price_i`는 사용하지 않는다. 고CLV 고객에게
고가 상품을 강화하는 이전 가정으로 돌아가기 때문이다.

## 2. 비교 모형과 실행 범위

- 데이터: Dunnhumby 전체 학습창
- 시드: 42
- 평가: validation only
- M1: binary graph, plain BPR, uniform negative sampling
- M3: M1과 같은 임베딩 차원·전파층·학습예산·BPR·negative sampling,
  CLV-NV 구성형 엣지 가중치만 적용
- test·holdout은 정답을 구성하지 않는다.
- 1차에서는 N-only, V-only, shuffled-user 대조군을 학습하지 않는다.

## 3. 사용자 CLV N/V 구성축

모든 입력은 학습구간에서만 계산한다.

\[
N_u = \frac{\max(B_u-1,0)}{\max(T_u,1)}
\]

- \(B_u\): 고객의 서로 다른 장바구니 수
- \(T_u\): 고객 관찰기간
- \(N_u\): 관찰기간으로 보정한 반복거래율

\[
V_u = \operatorname{MeanBasketValue}_u
\]

\[
q_N(u)=\operatorname{Percentile}(N_u),\qquad
q_V(u)=\operatorname{Percentile}(V_u)
\]

`rankdata(method="average")`로 동점자에 같은 백분위를 부여한다. 이 값은 미래
CLV 예측값이 아니라 학습구간의 historical CLV 구성요인이다.

## 4. 엣지 단위 N/V 관계강도

### 4.1 N축: 반복구매 관계

\[
n_{ui}=\log\left(1+\max(cnt_{ui}-1,0)\right)
\]

\(cnt_{ui}\)는 사용자 \(u\)가 아이템 \(i\)를 구매한 서로 다른 장바구니 수다.
한 번만 구매한 엣지는 N축 반복구매 증거가 0이다. 반복구매가 있는 엣지만
사용자 내부 백분위로 변환하고, 그 외 엣지는 0으로 둔다.

\[
r^N_{ui}=\operatorname{WithinUserPercentile}(n_{ui})
\]

### 4.2 V축: 거래당 가치 맥락

\[
v_{ui}=\operatorname{Mean}\left(
\text{장바구니 총액}\mid i\text{가 포함된 }u\text{의 장바구니}
\right)
\]

\[
r^V_{ui}=\operatorname{WithinUserPercentile}(v_{ui})
\]

상품 단가를 직접 사용하지 않는다. 저가 상품이라도 큰 장바구니에서 일관되게
함께 구매됐다면 V축 관계강도가 높을 수 있다.

## 5. CLV-NV 구성형 엣지 가중치

\[
c^N_{ui}=2q_N(u)r^N_{ui},\qquad
c^V_{ui}=2q_V(u)r^V_{ui}
\]

각 축을 전체 학습 엣지 평균 1로 정규화한다.

\[
\bar c^a_{ui}=\frac{c^a_{ui}}
{\operatorname{Mean}_{(u,i)\in E}(c^a_{ui})+\epsilon},
\quad a\in\{N,V\}
\]

\[
w^{raw}_{ui}=1+\frac{1}{2}
\left(\bar c^N_{ui}+\bar c^V_{ui}\right)
\]

\[
w_{ui}=\operatorname{Clip}\left(
\frac{w^{raw}_{ui}}{\operatorname{Mean}_{E}(w^{raw})},
0.25,4.0
\right)
\]

이 가중치로 대칭 정규화 인접행렬을 구성한다.

\[
\hat A_{NV}=D_{NV}^{-1/2}A_{NV}D_{NV}^{-1/2}
\]

LightGCN 전파와 최종 내적 점수, plain BPR은 M1과 동일하다. 별도의
`alpha`나 가중치 스윕을 두지 않는다.

## 6. 저장 및 성공 판독

두 모형 모두 Recall/NDCG @10·20·50, 가격·구매금액 가중 적중값 @10·20·50,
ARP, Coverage, 추천 상품 절대 개수, 노출 엔트로피, effective catalog,
top-10/top-100 노출점유율, value alignment를 저장한다. M1 대비 절대차와
상대차를 표로 출력한다.

1차 seed 42 validation 통과 조건은 다음과 같다.

1. Recall/NDCG @10·20·50 각각 M1의 99% 이상
2. 가격·구매금액 가중 적중값@10이 M1보다 높음

Coverage와 노출분산 지표는 함께 해석하지만 성과를 강제하는 학습 제약이나
통과조건으로 사용하지 않는다. 1차가 통과하면 N-only, V-only, shuffled-user 대조군과
다중 시드를 순차적으로 실행한다.

## 7. 불변식과 실패 방지

- M1과 M3의 unique edge index는 정확히 같아야 한다.
- 모든 가중치는 유한·양수여야 하며 clip 범위를 벗어나지 않아야 한다.
- 사용자별 within-user 랭크는 학습 정보만 사용한다.
- M3의 BPR 학습행·negative sampling·학습예산은 M1과 동일해야 한다.
- test·holdout 정답을 구성하는 경로는 실험 시작 전에 차단한다.

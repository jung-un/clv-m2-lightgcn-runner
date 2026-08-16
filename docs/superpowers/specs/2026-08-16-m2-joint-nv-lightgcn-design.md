# M2 Joint N/V LightGCN Design

## 목적

CLV를 완성된 LightGCN 점수 밖에서 더하지 않고, CLV의 두 구성 메커니즘인 거래활동(N)과 거래가치(V)를 LightGCN의 초기 사용자·아이템 표현에 내장한다. ID, N, V 공간은 분리하여 CLV 총량이 같아도 가치 형성 방식이 다른 고객에게 다른 추천을 할 수 있게 한다.

## 고정 원칙

- 하나의 모델, 하나의 학습 루프, 하나의 optimizer를 사용한다.
- 사전학습 CLV encoder, 동결 M1, stop-gradient, 후처리 residual을 사용하지 않는다.
- binary graph, uniform negative sampling, 균일 BPR을 유지한다.
- CLV 보조손실이나 표본 가중치를 추가하지 않는다. 이 실험은 M2만 변경한다.
- test/holdout은 validation 개발 단계에서 구성하거나 평가하지 않는다.

## 모델

\[
E_u^{(0)}=[E_u^{ID}\Vert \gamma_N g_N(u)h_u^N\Vert \gamma_V g_V(u)h_u^V]
\]

\[
E_i^{(0)}=[E_i^{ID}\Vert h_i^N\Vert h_i^V]
\]

- `ID`: 자유 사용자·아이템 embedding. 기존 선호 학습 용량을 줄이지 않는다.
- `N`: 비계약형 CLV 거래과정의 관측량인 반복거래 횟수
  `x=max(basket_count-1,0)`, 첫 거래 이후 마지막 거래시점
  `t_x=max((observed_days-1)-recency_days,0)`, 고객 관찰기간
  `T=observed_days-1`, 평균 거래간격과 각 validity mask를 아이템의
  재구매율·재구매 간격에 연결하는 공간이다.
- `V`: 거래당 평균금액 `avg_basket_value`와 validity mask를 아이템의
  가격대·카테고리 내 가격대·거래내 가치비중과 연결하는 공간이다.
- `gamma_N`, `gamma_V`: 양수인 전역 학습 강도. 거의 0에서 시작하여 ID 경로를 우선 확보한다.
- `g_N`, `g_V`: train 이력에서 고정 계산한 축 백분위 기반 gate. 학습형 gate는 기본모형에 넣지 않는다.

사용자 N/V 입력은 기존 거래이력에서 산출하지만, 과거의
`[F_p,T_p,R_p]` / `[AOV_p,Prem_p]` 백분위 묶음으로 되돌아가지 않는다.
특히 `Premium`은 V축에서 제외한다. 위 변수는 미래 예측치 `N_hat/V_hat`이
아니라 CLV 구성과정에 관련된 고정 관측변수다. 이 관측변수를 N/V
embedding network에 직접 넣고 추천 BPR로 공동 학습한다.

결합된 초기 표현 전체를 동일한 binary LightGCN 인접행렬로 전파한 뒤, 최종 사용자와 아이템 표현의 내적 하나로 점수를 계산한다. 따라서 CLV 경로도 추천 BPR의 gradient를 직접 받는다.

## 빠른 1차 비교

- `m1`: 순수 LightGCN.
- `joint_nv`: 정상 N/V 사용자 표현.

1차에는 이 두 모형만 실행한다. `joint_shuffled_user`와
`joint_constant_user`는 제안모형의 성과 가능성이 확인된 뒤 식별 검증에서만
실행한다.

## 1차 실험

H&M 60일과 Dunnhumby 전체 기간(약 704일)을 각각 seed 42,
validation-only로 순차 실행한다. Recall/NDCG, 가격·구매금액 가중
적중, ARP, Coverage, 추천 상품 절대개수, 노출 entropy/effective catalog,
상위 상품 노출점유율을 같이 저장한다. 인접 적용 전에 다른 정규화나
손실함수 변경은 하지 않는다.

동시에 공식 validation/test가 아닌 **train 내부 시점**만 사용해 변수
타당성을 진단한다. H&M 60일은 14일 입력→7일 미래, Dunnhumby는 365일
입력→90일 미래 rolling anchor를 사용한다. N 행동점수와 미래 거래횟수,
V 행동점수와 미래 거래당 평균금액, N×V 행동점수와 미래 총거래금액의
Spearman 상관 및 N/V 사분면별 미래값을 저장한다. 이 진단을 통과하기
전에는 행동점수를 미래 `N_hat/V_hat`으로 부르지 않는다.

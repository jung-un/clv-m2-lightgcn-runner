# M2 협업경로 보호형 CLV–상품 관계 표현 설계

## 1. 연구질문과 범위

이 설계는 M1~M5 중 **M2 임베딩·표현 개입**이다. 기존 2층 LightGCN의
협업경로가 CLV 보조경로와 공동학습되는 과정에서 약해졌던 문제를 줄이면서,
historical CLV proxy의 N/V 구성과 상품 가격속성을 작은 후보별 관계표현으로
함께 학습하면 신규상품 정확도와 가격·구매금액 가중 적중값을 동시에 개선할
수 있는지 검정한다.

그래프는 binary, negative sampling은 uniform, 표본가중은 없으며, 목적함수는
기존 plain BPR와 sampled ID L2 하나만 사용한다. 모든 학습 파라미터는 하나의
optimizer와 학습루프에서 동시에 갱신한다.

## 2. 기존 결과에 대한 직접 대응

1. 공동학습 checkpoint의 ID-only가 M1보다 약해졌으므로, 보조경로에서
   협업표현으로 향하는 gradient를 차단한다.
2. `q_C`와 `q_N-q_V`가 거의 같은 문맥방향을 만들었으므로, 사용자 조건은
   `[q_N,q_V,q_C]`를 분리 입력한다.
3. 직전 후보에는 아이템 경제속성이 없었으므로, 상품 전체 가격 백분위를
   전용 1차원 관계로 사용한다.
4. 가격관계가 자유 혼합에서 소멸한 전례가 있으므로 가격 몫 `beta`와 방향,
   최대 개입강도를 사전 고정한다.
5. 학습형 전체 축 강도 `s_N,s_V`와 독립 N/V 상품점수는 사용하지 않는다.

## 3. 표준 협업경로

사용자와 상품의 64차원 자유 ID 임베딩을 기존 binary 그래프에서 2층
LightGCN으로 전파한다.

\[
z=\frac{1}{3}\left(E^{(0)}+E^{(1)}+E^{(2)}\right)
\]

사용자·상품 표현을 각각 `z_u`, `z_i`로 쓴다. 이 경로는 최종 BPR 점수의
직접 항으로 학습된다.

## 4. 3차원 CLV 조건부 협업관계

보조경로에서는 `z_u,z_i`의 값은 사용하되, 이 경로의 gradient가 ID
임베딩과 LightGCN 전파로 되돌아가지 않게 `stop-gradient`를 적용한다.

\[
b_u=\operatorname{Unit}(P_u\,\operatorname{sg}(z_u)),\qquad
b_i=\operatorname{Unit}(P_i\,\operatorname{sg}(z_i))
\]

\[
m_u=1+\delta\tanh\left(W_C[q_N(u),q_V(u),q_C(u)]\right)
\]

\[
r_u=\operatorname{Unit}(b_u\odot m_u),\qquad r_i=b_i
\]

- `P_u,P_i`: `64→3` 공동학습 투영
- `W_C`: `3→3` 공동학습 조건변환
- `delta=0.25`: CLV가 각 관계좌표를 바꿀 수 있는 최대 범위
- `sg`: 보조경로가 ID 표현을 직접 다시 쓰는 것을 막지만 `P_u,P_i,W_C`는
  동일 BPR gradient로 학습된다.

CLV만으로 상품표현을 만들 수 없고, `z_u,z_i`가 있어야 관계가 생긴다.

## 5. 1차원 가격 적합성

사용자 V 수준과 상품 전체 가격 백분위를 중심화해 사용한다.

\[
v_u=2q_V(u)-1,\qquad p_i=2PricePct_i-1
\]

가격축의 부호를 유지하면서 BPR이 제한된 범위에서 크기를 보정할 수 있도록
다음 양수계수를 사용한다.

\[
\kappa(a)=\frac{1+\epsilon\tanh(a)}{1+\epsilon},\qquad \epsilon=0.5
\]

따라서 `kappa`는 `[1/3,1]` 범위이고 부호가 뒤집히거나 최대 예산을 초과하지
않는다. `a`는 다른 파라미터와 함께 학습한다. 가격 입력이 없는 상품이나 V
입력이 유효하지 않은 사용자는 해당 좌표를 0으로 둔다.

## 6. 고정 CLV 개입범위와 최종 표현

전체 historical CLV 수준은 제한된 범위에서만 보조표현의 강도를 바꾼다.

\[
g_C(u)=1+\eta(q_C(u)-0.5),\qquad \eta=0.5
\]

따라서 유효 사용자의 `g_C`는 `[0.75,1.25]`다. 최종 표현은 다음과 같다.

\[
\widetilde z_u=
\left[
z_u\;\middle|\;
\sqrt{\rho(1-\beta)}g_C(u)r_u\;\middle|\;
\sqrt{\rho\beta}g_C(u)\kappa(a)v_u
\right]
\]

\[
\widetilde z_i=
\left[
z_i\;\middle|\;
\sqrt{\rho(1-\beta)}r_i\;\middle|\;
\sqrt{\rho\beta}p_i
\right]
\]

\[
S(u,i)=\widetilde z_u^\top\widetilde z_i
\]

고정값은 `rho=0.05`, `beta=0.25`, `delta=0.25`, `eta=0.5`다. 별도의
외부 보정이나 재정렬은 수행하지 않는다.

## 7. 학습계약

- 신규상품 추천 과업, 학습 pair 후보 제외
- `MIN_ITEM_INTER=1`
- ID 차원 64, 관계 차원 3, 가격 차원 1
- LightGCN 2층, 0·1·2층 평균
- binary graph, uniform negative sampling
- plain BPR + 기존 sampled ID L2
- 표본가중·새 손실항·사전학습·동결 없음
- 하나의 optimizer와 학습루프
- 100 epoch 고정, epoch 선택 없음
- 최종 test와 holdout 생성·평가 없음

## 8. 빠른 개발실험

- 데이터: Dunnhumby
- seed: 42
- 학습: day 1~683
- 평가: day 684~690 신규상품 역사적 개발분할
- matched `rho=0` arm과 active M2를 동일 초기화·난수순서로 각각 학습
- 외부 저장 M1@64는 표시용으로만 사용

긍정 판정은 다음을 모두 만족할 때다.

1. 전체 Recall/NDCG@10·20·50이 matched `rho=0`의 99% 이상
2. 가격·구매금액 가중 적중값@10이 matched `rho=0`보다 증가
3. 고CLV Recall@10과 NDCG@10이 모두 증가
4. 공동학습 ID-only가 matched `rho=0` 핵심 정확도의 99.5% 이상
5. 고CLV Top-10 집합이 실제로 변경

단일 seed 탐색이므로 유의성이나 두 데이터 일반화를 주장하지 않는다.

## 9. 필수 진단

- jointly-trained ID-only와 full의 전체·CLV 구간 성과
- 관계·가격 점수 각각의 `std(component)/std(S_ID)`
- 전체 보조점수의 실제 영향력과 Top-10·50 변경률
- ID 임베딩 gradient norm과 보조투영 gradient norm
- `q_N,q_V,q_C` 각각과 사용자 관계좌표/점수의 상관
- V 사분위별 추천상품 평균 가격 백분위
- 학습된 `kappa(a)`와 사전 범위 준수 여부
- 가격축을 제외한 뷰, 관계축을 제외한 뷰

active M2가 1차 판정을 통과한 경우에만 shuffled-user CLV, 동일 총차원 M1,
여러 seed, H&M 2년 순으로 확장한다.

## 10. 한계와 정직한 서술

이 구조는 CLV 보조좌표를 LightGCN 이웃 전파에 싣지 않는다. 하지만 완성된
M1을 동결한 뒤 외부에서 점수를 더하는 방식도 아니다. ID 임베딩과 보조표현은
같은 forward의 하나의 내적과 같은 BPR로 처음부터 공동학습된다.

`stop-gradient`는 ID 경로를 M1과 수치적으로 동일하게 보장하지 않는다.
보조점수가 BPR margin을 바꾸므로 ID 경로의 학습궤적은 여전히 달라질 수 있다.
따라서 공동학습 ID-only 진단으로 보호 여부를 직접 판정한다.

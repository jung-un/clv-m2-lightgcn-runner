# M2 CLV-Conditioned Modulation LightGCN 설계

## 1. 연구 목적

기존 M2의 독립 점수 결합

\[
S(u,i)=S_{ID}(u,i)+S_N(u,i)+S_V(u,i)
\]

을 폐기한다. 거래활동(N)·거래당 가치(V)는 독립적으로 상품을 추천하지 않고,
기존 협업 선호 임베딩의 차원별 중요도를 조절하는 조건정보로 사용한다.

검증 가설은 다음과 같다.

> CLV의 N/V 구성요인은 독립 추천점수보다 협업 선호표현을 조건화할 때
> 구매상품 적중을 훼손하지 않으면서 경제적 추천성과를 개선한다.

이 모형은 임베딩 생성만 변경하는 M2다. 이진 그래프, LightGCN 전파,
uniform negative sampling, plain BPR은 M1과 동일하게 유지한다.

## 2. v1.5에서 확인된 문제

Dunnhumby seed 42 validation에서 preference-preserving joint v1.5는
M1 대비 Recall@10 `-3.84%`, 가격·구매금액 가중 적중값@10 `-6.88%`,
추천 고유상품 수 `232→216`이었다. 반면 value alignment는 `+99.09%`였다.

따라서 N/V가 경제속성 정렬은 강화했지만 실제 구매상품 선택에는 맞지 않았다.
gradient 분리 뒤에도 동일한 패턴이 유지됐으므로 다음 후보는 gamma·손실을 다시
조정하지 않고 N/V의 역할을 독립 점수에서 조건부 표현조절로 바꾼다.

## 3. 모형

### 3.1 입력

첫 구조 screening에서는 v1.5와 동일한 train-only N/V 사용자·아이템 입력과
유효성 마스크를 재사용한다. 입력까지 동시에 바꾸지 않아 구조 효과를 식별한다.

- 사용자 N 입력: 거래활동·반복거래 관련 관측변수
- 사용자 V 입력: 거래당 평균 구매금액 관련 관측변수
- 아이템 N 입력: 반복구매 관련 관측변수
- 아이템 V 입력: 기존 경제속성 관측변수

상품 V를 장바구니 맥락 가치로 교체하는 실험은 구조 screening 통과 후 별도
ablation으로 둔다.

### 3.2 축별 modulation

사용자와 아이템의 N/V 입력을 각각 작은 저랭크 변환기로 64차원 조절벡터로
변환한다.

\[
m_u^N=f_u^N(x_u^N),\quad m_u^V=f_u^V(x_u^V)
\]

\[
m_i^N=f_i^N(x_i^N),\quad m_i^V=f_i^V(x_i^V)
\]

각 변환기는 `input → rank 4 → 64`의 bias 없는 선형구조로 둔다. 첫 투영은
작은 무작위값, 마지막 투영은 0으로 초기화한다. 따라서 학습 시작 시 modulation은
정확히 0이면서 마지막 투영에는 첫 업데이트부터 gradient가 흐른다. ID 임베딩은
추가 모듈 생성 전에 M1과 동일한 순서로 초기화하여, 같은 seed에서 M1과 제안모형의
초기 ID state hash 및 초기 점수가 일치하는지 테스트한다.

고정된 최대 조절폭 `tau=0.10`을 사용한다.

\[
\widetilde E_u^{(0)}=E_u^{ID}\odot
\left[1+\tau\tanh(m_u^N+m_u^V)\right]
\]

\[
\widetilde E_i^{(0)}=E_i^{ID}\odot
\left[1+\tau\tanh(m_i^N+m_i^V)\right]
\]

유효하지 않은 축의 modulation 출력은 정확히 0으로 마스킹한다. 별도 gamma,
lambda, 학습형 gate는 사용하지 않는다.

### 3.3 전파와 점수

조절된 사용자·아이템 layer-0 표현 전체를 하나의 LightGCN으로 전파한다.

\[
Z=\operatorname{LightGCN}(\widetilde E^{(0)},A)
\]

최종 추천점수는 하나의 내적뿐이다.

\[
S(u,i)=Z_u^\top Z_i
\]

`S_N`, `S_V`, 외부 residual, 사후 재정렬은 존재하지 않는다. 모든 ID와
modulation 파라미터는 하나의 optimizer와 plain BPR에서 공동학습한다.

## 4. M1·M3·M4와의 구분

- M1: ID 임베딩만 사용한다.
- 새 M2: N/V가 layer-0 ID 임베딩을 조건화한다.
- M3: 엣지 가중치와 정규화 인접행렬을 변경한다. 새 M2에서는 변경하지 않는다.
- M4: 표본별 손실 가중치를 변경한다. 새 M2에서는 모든 BPR 표본을 균등하게 둔다.

## 5. 빠른 1차 실행

첫 실행은 Dunnhumby seed 42 validation-only로 제한한다.

- `m1`: 동일 데이터·seed·64차원 순수 LightGCN
- `m2_clv_modulation`: 본 제안모형
- 그래프: binary
- negative sampling: uniform
- 손실: plain BPR
- ID 차원: 64
- modulation rank: 축별 4
- `tau`: 0.10 고정
- test/holdout: 구성·평가하지 않음

M2도 최종 임베딩이 64차원이므로 M1@96 용량 대조군은 필요하지 않다. 첫
screening에서는 순열·상수 대조군을 학습하지 않는다.

## 6. 저장 지표와 진단

- Recall/NDCG@10·20·50
- 가격·구매금액 가중 적중값@10·20·50
- ARP@10·20·50, value alignment
- Coverage@10, 추천상품 절대 수
- exposure entropy/effective catalog size
- top-10/top-100 노출점유율
- 사용자별 paired delta와 bootstrap CI
- epoch별 train loss, validation Recall/NDCG@10, 경제지표@10
- N-only/V-only modulation mask 평가
- modulation 절대평균·표준편차와 포화비율

## 7. 사후 성공 판정

성과를 인위적으로 만드는 학습 가드레일이 아니라 결과 판독 기준이다.

1. Recall/NDCG@10·20·50 각각 M1의 99% 이상
2. 가격·구매금액 가중 적중값@10이 M1보다 큼
3. 양 조건을 만족하지 않으면 test와 H&M으로 확장하지 않음

통과할 경우에만 같은 구조·용량에서 `shuffled_user`, `constant_user`,
`zero_clv` 대조군을 실행하고, 그 뒤 3시드 validation으로 확장한다.

## 8. 종료 규칙

v1.5 체크포인트의 재학습 없는 강도·순열 진단에서 유효구간이 없고, 본
modulation 구조도 Dunnhumby seed 42 성공조건을 통과하지 못하면 추가 gamma,
차원, gate 탐색을 하지 않는다. M2는 임베딩 개입의 한계 사례로 기록하고
사전 계획된 독립 연구축 M3·M4를 진행한다.

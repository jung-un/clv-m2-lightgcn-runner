# NGCF 백본의 CLV 수준·구성·가격 좌표 임베딩 개발실험 설계

## 1. 연구 위치와 질문

이 실험은 M2 표현 개입의 백본 민감도를 확인한다. LightGCN에서 가장 큰
절대 성과 상승을 보였지만 공동학습 ID-only와 직접 구별되지 않았던
`CLV 수준·구성·가격 좌표`를 변경하지 않고 NGCF에 이식한다.

연구질문은 다음과 같다.

> NGCF의 학습형 이웃변환과 사용자–상품 좌표별 상호작용이 동일한 historical
> CLV proxy 입력을 NGCF의 ID-only 및 degree-matched CLV 순열보다 유용하게
> 사용할 수 있는가?

이 실험은 새 M2 수식을 탐색하지 않는다. M3 그래프가중과 M4 손실가중도
사용하지 않는다.

## 2. 고정 입력과 layer-0 표현

학습기간의 기존 정의를 그대로 사용한다.

- `q_N(u)`: 거래활동 추정치의 학습사용자 백분위
- `q_V(u)`: 거래당 가치 추정치의 학습사용자 백분위
- `q_C(u)`: `N_hat(u) * V_hat(u)` historical CLV proxy의 학습사용자 백분위
- `p_i`: 학습기간 상품 전체가격 및 카테고리 내 가격 백분위의 양의 볼록결합

사용자 layer-0 표현은 다음과 같다.

```text
E_u^0 = [
  E_u^ID
  | sqrt(rho * (1-beta)) * q_C(u) * Unit([q_C(u), q_N(u)-q_V(u)])
  | sqrt(rho * beta) * q_C(u) * (2q_V(u)-1)
]
```

상품 layer-0 표현은 다음과 같다.

```text
E_i^0 = [
  E_i^ID
  | sqrt(rho * (1-beta)) * r_i
  | sqrt(rho * beta) * p_i
]
```

`r_i`는 기존 상품 ID 임베딩의 학습형 정규화 2차원 투영이다. 가격 좌표의
전체가격·카테고리 내 가격 혼합계수는 softmax로 합이 1이고 같은 BPR gradient로
학습한다. `id_dim=64`, 관계 2차원, 가격 1차원, `rho=0.05`, `beta=0.25`를
고정한다.

## 3. NGCF 전파

LightGCN의 선형 평균만 다음의 NGCF 메시지 변환으로 교체한다.

```text
N^l = A_hat E^l
E^(l+1) = LeakyReLU(
  W_sum^l (E^l + N^l)
  + W_bi^l (E^l elementwise_product N^l)
)
```

`A_hat`은 M1과 동일한 binary 사용자–상품 대칭 정규화 그래프다. 2층을 사용하고
각 층 표현을 L2 정규화한 뒤 layer-0·1·2를 이어 붙인 하나의 사용자·상품
표현으로 만든다. 최종점수는 이 두 표현의 내적 하나다. 각 층의 출력차원은
해당 arm의 layer-0 차원과 같고 LeakyReLU 기울기는 `0.2`로 고정한다.
node dropout과 message dropout은 `0`으로 두어 이번 차이가 dropout 탐색이
아니라 학습형 NGCF 메시지 변환에서만 나오게 한다.

NGCF의 학습형 변환은 초기 3개 보조좌표를 다른 좌표와 섞고 확대할 수 있다.
따라서 `rho=0.05`는 layer-0 입력예산이지 최종 보조점수의 구조적 상한이라고
주장하지 않는다. 학습 후 입력 좌표에 대한 최종점수 gradient와 실제
actual–shuffle 추천 차이를 진단한다.

## 4. 비교 arm

동일 seed와 negative sequence에서 다음 네 arm을 학습한다.

1. `ngcf_m1_64`: ID 64차원 NGCF 기준모형
2. `ngcf_m1_67`: 동일 총차원 67의 용량 대조군
3. `ngcf_m2_clv_level_composition_price`: 실제 사용자 CLV 입력
4. `ngcf_m2_degree_matched_clv_shuffle`: binary user-degree 10분위 안에서
   `(q_N, q_V, q_C, valid)` 묶음을 공동 순열한 귀속 대조군

LightGCN M1과의 수치는 참고로만 표시한다. CLV 효과의 주 비교는 반드시
`NGCF-M2 actual` 대 `NGCF-M1@67` 및 `NGCF-M2 shuffle`이다.

## 5. 학습·평가 계약

- 데이터: Dunnhumby
- 탐색 seed: 42
- 학습: DAY 1~683
- 개발평가: DAY 684~690 신규상품
- epoch: 100 고정, epoch 선택·조기종료 없음
- 그래프: binary
- negative sampling: uniform
- 손실: plain pairwise BPR + 기존 sampled ID L2
- NGCF 변환행렬: Xavier 초기화, 별도 weight decay 없음
- 표본가중 및 새 손실항: 없음
- optimizer: arm별 하나, 구성요소 공동학습
- `MIN_USER_INTER=1`, `MIN_ITEM_INTER=1`
- final test와 holdout: 구성하지 않음

DAY 684~690은 이미 여러 구조가 관찰한 개발구간이므로 결과는 백본 상호작용의
탐색 증거로만 사용하며 확증·유의성·일반화를 주장하지 않는다.

## 6. 사전 판정

다음 조건을 모두 만족할 때만 NGCF에서 CLV 표현의 양성 신호로 판정한다.

1. actual의 Recall/NDCG@10·20·50이 `NGCF-M1@67`의 99% 이상
2. actual의 Recall@10과 NDCG@10이 `NGCF-M1@67`보다 모두 높음
3. actual의 고CLV Recall@10과 NDCG@10이 `NGCF-M1@67`보다 모두 높음
4. actual의 가격·구매금액 가중 적중값@10이 `NGCF-M1@67`보다 높음
5. actual이 shuffle보다 여섯 정확도 지표 기하평균, 고CLV Recall/NDCG@10,
   가격·구매금액 가중 적중값@10에서 모두 높음

seed 42에서 탈락하면 NGCF 하이퍼파라미터·CLV 강도·차원을 같은 구간에 맞춰
조정하지 않고 종료한다. 통과할 때만 paired multi-seed와 H&M 이식성을 별도
사전고정 실험으로 검토한다.

## 7. 구현 범위와 검증

새 NGCF 모델, 실행 runner, Dunnhumby Colab, 국소 테스트만 추가한다. 기존
LightGCN 및 M2 구현은 수정하지 않는다.

국소 검증은 다음을 포함한다.

- 네 arm의 입력 차원과 초기화 순서
- `rho=0` 상당 비개입 및 M2 입력 구성식
- degree-matched 공동 순열
- NGCF 두 층에서 보조좌표까지 gradient가 흐르는지 확인
- train pair가 평가 정답과 Top-K에서 제외되는지 확인
- final test·holdout 차단
- 결과표의 NGCF 내부 대조와 LightGCN 참고 비교 분리

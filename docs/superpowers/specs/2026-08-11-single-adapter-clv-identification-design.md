# Single-Adapter CLV 행동표현 식별실험 설계

작성일: 2026-08-11

지위: 사용자 승인 완료, 구현 전 설계 고정본

연구축: M2(임베딩·표현)

## 1. 배경과 정정

Dunnhumby seed 42 validation에서 `CLV-MoE`의 선택 경제지표는 외부 M1보다 높았지만,
`constant_gate`와 `single_adapter`보다 낮아 최종 screening에 실패했다. 이는 CLV 관련 표현 전체의
실패가 아니라 **세 전문가를 CLV gate로 혼합하는 구조의 추가효과가 식별되지 않았음**을 뜻한다.

현재 코드의 `single_adapter`는 CLV가 없는 대조군이 아니다. 다음 입력을 그대로 사용한다.

- 사용자: LightGCN 사용자 임베딩과 51차원 CLV 관련 행동표현.
- 아이템: LightGCN 아이템 임베딩, train-only 수치특성 6개, category embedding.
- 점수: 외부 M1 점수에 하나의 사용자–아이템 adapter 내적을 residual로 추가.

따라서 `single_adapter`의 개선이 사용자별 CLV 관련 행동정보 때문인지, 아이템 특성·추가 파라미터·
공동 미세조정 때문인지는 기존 대조군으로 분리되지 않는다. 이 설계는 그 식별을 우선한다.

## 2. 연구 질문과 논문 내 위치

주 연구 질문은 다음과 같다.

> CLV 관련 사용자 행동표현과 아이템 경제·행동특성을 하나의 임베딩 공간에서 결합하면 외부 M1의
> 일반 추천 정확도를 유지하면서 가격·구매금액 가중 적중값을 개선하는가? 그 개선은 단순 adapter
> 용량이나 아이템 특성이 아니라 사용자별 CLV 관련 행동정보에 의존하는가?

이 실험은 M2의 표현 개입이다. 그래프는 M1과 같은 이진 LightGCN을 사용하고, 손실은 모든 표본에
동일한 plain BPR을 사용한다. M3의 엣지 가중과 M4의 CLV-aware 손실 가중은 사용하지 않는다.

이 실험은 M2만으로 논문 전체를 축소하지 않는다. M3·M4의 독립 비교와 M5 결합실험은 기존 계획대로
유지한다.

## 3. 공통 모형

외부 M1의 사용자·아이템 임베딩을 `z_u`, `z_i`, train-only 사용자 행동표현을 `h_u`, 아이템
수치·범주 특성을 `x_i`라고 한다.

\[
a_u=f_u([z_u,h_u]), \qquad a_i=f_i([z_i,x_i])
\]

\[
s(u,i)=\langle z_u,z_i\rangle+\lambda\langle a_u,a_i\rangle
\]

`f_u`, `f_i`는 현재 `single_adapter`와 같은 `Linear → GELU → Linear` MLP다. 출력 차원,
hidden dimension, 초기화, 학습률, epoch 예산은 모든 식별 대조군에서 동일하게 유지한다.

이 구조는 CLV를 고가상품 가산점이나 단조 gate로 사용하지 않는다. 사용자 행동표현과 아이템 표현의
저차원 적합도를 plain BPR로 학습한다.

## 4. 사용자와 아이템 입력

### 4.1 사용자 CLV 관련 행동표현

현재 51차원 schema를 유지한다.

1. train-only 행동변수 16개: 활동성, 최근성, 구매금액, 가격성향, 반복성, 카테고리 다양성,
   구매간격과 추세.
2. 각 행동변수의 validity mask 16개.
3. train 내부 anchor로 학습한 future-value encoder hidden state 16개.
4. 향후 90일 구매확률, 조건부 log 구매금액, log 기대구매금액 3개.

공식 validation/test/holdout은 이 표현의 생성, 표준화, encoder 학습 또는 조기종료에 사용하지 않는다.
이를 생애전체 CLV라고 부르지 않고 `CLV 관련 행동표현` 또는 `미래 구매가치 조건표현`이라고 기술한다.

### 4.2 아이템 표현

현재 train-only 입력을 유지한다.

- 전역 가격백분위, 카테고리 내 가격백분위, log 평균가격.
- log 구매행 수, 고유 구매자 수 백분위, 반복구매 고객 비중.
- category embedding.

이는 아이템 CLV가 아니다. 상품의 경제·관측행동 속성이다.

## 5. 1단계 식별 대조군

모든 변형은 동일한 입력 차원, MLP 구조, 파라미터 수, 초기값 규칙, 최대 epoch와 조기종료 규칙을
사용한다. 조기종료 시점과 실제 update 수는 결과에 별도로 저장한다. 입력 열을 제거해 네트워크 크기를
바꾸지 않고, 지정된 입력값만 0 또는 seed 고정 permutation으로 치환한다.

| model_id | 사용자 추가입력 | 아이템 추가입력 | 식별 목적 |
|---|---|---|---|
| `single_full` | 실제 51차원 | 실제 수치·범주 특성 | 주 M2 후보 |
| `single_zero_user` | 0 | 실제 특성 | 아이템 특성·adapter 효과 |
| `single_shuffled_user` | 유효 사용자 사이에서 고정 permutation | 실제 특성 | 사용자별 행동 적합도 효과 |
| `single_zero_item` | 실제 51차원 | 0 | 사용자 프로필만으로 얻는 효과 |
| `single_base_only` | 0 | 0 | 추가 용량·공동 미세조정 효과 |

세부 불변식은 다음과 같다.

- `single_zero_user`와 `single_base_only`에서도 유효 사용자 mask는 `single_full`과 동일하게 유지한다.
  값이 0이라는 이유로 residual 전체가 비활성화되면 안 된다.
- `single_zero_item`과 `single_base_only`에서도 유효 아이템 mask는 유지한다. 수치특성은 0으로,
  category id는 padding 0으로 바꾸며 padding embedding은 0을 유지한다.
- `single_shuffled_user`는 유효 사용자 집합 안에서 seed별 permutation을 한 번 만들고 학습·평가 동안
  고정한다. 주변분포는 보존하고 사용자–프로필 대응만 제거한다.
- 모든 변형은 base `z_u`, `z_i`를 adapter 입력으로 계속 사용한다. `base_only`는 adapter 자체의
  추가 표현용량을 측정하는 명칭이지 외부 M1과 동일하다는 뜻이 아니다.

외부 M1과 계산량 통제 `pref_continue`도 공통 기준으로 저장한다.

## 6. 실행 순서와 비용 통제

1. Dunnhumby와 H&M 2년 각각 `single_full`, seed 42, validation-only를 먼저 실행한다.
2. `single_full`이 외부 M1 성공조건을 통과한 데이터셋에서만 네 식별 대조군을 실행한다.
3. 양 데이터셋 모두에서 `single_full`의 사용자별 CLV 관련 정보 효과가 식별된 경우에만 설정과 코드를
   동결하고 다중 seed 및 미사용 test 확증 설계를 별도로 승인한다.
4. 한 데이터셋이 실패하면 범용 M2로 확증하지 않는다. 데이터셋 조건부 결과와 실패 메커니즘은
   validation 결과로 보고한다.

Dunnhumby에 이미 저장된 `single_adapter` checkpoint와 곡선은 `single_full`과 수학적·코드상 동일할
때만 재사용한다. 데이터/config/base-state/checkpoint hash와 reload 후 점수 동일성을 모두 확인해야 하며,
하나라도 불일치하면 재사용하지 않는다.

## 7. λ 선택과 성공조건

- validation grid: `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`.
- 각 `K=10,20,50`에서 Recall과 NDCG가 외부 M1 대비 상대 1% 이상 하락한 λ는 제외한다.
- 남은 양의 λ 중 가격·구매금액 가중 적중값@10이 외부 M1을 실제로 초과하는 후보만 인정한다.
- 인정 후보 중 경제지표가 가장 큰 λ를 선택하고, 동률이면 작은 λ를 선택한다.
- λ=0 fallback은 성공이 아니다.

각 대조군도 같은 규칙으로 자체 최선 λ를 선택한다. `single_full`의 최종 seed 42 식별 성공은 다음을
모두 만족해야 한다.

1. `single_full` 자체가 외부 M1 성공조건을 통과한다.
2. 선택된 가격·구매금액 가중 적중값@10이 `single_zero_user`, `single_shuffled_user`,
   `single_base_only` 각각의 선택값보다 높다.
3. `single_zero_item` 결과를 통해 사용자 프로필만으로 충분한지, 아이템 특성과의 결합이 필요한지를
   메커니즘 결과로 기록한다. 이 비교는 2번의 CLV 식별 필수조건에는 넣지 않는다.

모든 λ의 절대 곡선과 외부 M1 대비 paired delta를 저장한다. 성공조건은 학습 중 성과를 강제로 만드는
제약이 아니라 validation 결과에 대한 사후 판정이다.

## 8. 2단계 행동특성 확장의 지위

1단계에서 사용자별 CLV 관련 행동정보의 효과가 식별되지만 한 데이터셋의 경제지표 개선이 부족한 경우에만
행동특성 확장을 별도 설계한다. 현재 구현 범위에는 넣지 않는다.

- H&M 후보: 카테고리·상품 탐색, 구매 고객의 스타일 다양성, 가격성향, 계절성.
- Dunnhumby 후보: 상품·카테고리 재구매 주기, 장바구니 크기, 동시구매 카테고리, 카테고리 확장성.

공통 수식과 adapter 인터페이스는 유지하되 데이터셋별 원천변수와 관찰창만 다르게 할 수 있다. 모든
특성은 train-only로 계산하고 sparse 아이템 집계에는 global/category 평균으로 shrinkage를 적용한다.
구체적 정의와 shrinkage 강도는 1단계 결과를 본 뒤 별도 탐색 설계서에서 사전 고정한다.

## 9. 저장 결과와 진단

- Recall/NDCG@10·20·50, 가격·구매금액 가중 적중값, ARP, value alignment.
- 추천상품 절대 수, Coverage, exposure entropy/effective catalog size, 상위 10·100 상품 노출점유율.
- CLV 세그먼트별 정확도·경제지표와 외부 M1 대비 paired delta.
- λ별 `std(adapter score)/std(M1 score)`.
- 모델별 학습시간, update 수, 학습 파라미터 수, base state hash, input mask/permutation hash.
- source revision, 데이터 파일 hash, feature schema, checkpoint/result fingerprint.

## 10. 실행 안전장치와 테스트

- screening runner는 seed 42와 validation-only만 허용한다. test·holdout 정답을 만들지 않는다.
- 고비용 셀은 `ACKNOWLEDGE_HIGH_COST=False`가 기본이며 preflight 검토 후에만 실행한다.
- 동일 입력 차원·파라미터 수·초기 base state·최대 학습예산·조기종료 규칙을 runtime test로 검증한다.
- zero/shuffle가 지정한 정보만 바꾸고 base embedding·split·mask는 바꾸지 않는지 검사한다.
- λ=0에서 모든 변형이 외부 base score와 정확히 같은지 검사한다.
- 기존 `single_adapter` checkpoint를 재사용할 경우 cache를 포함한 save/load score round-trip을 검사한다.
- 결과 JSON에 authoritative screening decision과 실패 대조군을 저장·출력한다.

## 11. 주장 범위

- `single_full > M1`만으로 CLV 효과라고 주장하지 않는다.
- `single_full > zero_user/shuffled_user/base_only`가 확인돼야 사용자별 CLV 관련 행동정보의 효과로
  해석한다.
- `single_zero_item`은 아이템 경제·행동특성과의 결합 필요성을 설명하는 메커니즘 ablation이다.
- seed 42 validation은 후보 제거와 메커니즘 진단이다. 일반화·통계적 확증은 동결 후 다중 seed와
  미사용 test에서만 주장한다.
- 가격·구매금액 가중 적중값을 실제 증분매출이나 CLV 증가로 해석하지 않는다.

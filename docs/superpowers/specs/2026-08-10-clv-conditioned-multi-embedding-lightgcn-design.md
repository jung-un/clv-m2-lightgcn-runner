# CLV-Conditioned Multi-Embedding LightGCN 설계

작성일: 2026-08-10  
지위: M2 후속 탐색모형 설계. 구현 전 승인본.

## 1. 연구 질문과 위치

이 모형은 논문 전체의 M2(임베딩·표현) 축에 해당한다. M3(그래프 가중)와 M4(손실 가중)를 대체하지
않으며, 두 축의 독립실험과 이후 M5 결합실험은 그대로 유지한다.

연구 질문은 다음과 같다.

> 고객가치는 하나의 가격 방향이 아니라 서로 다른 구매행동 메커니즘으로 형성되는가? CLV 관련
> 행동표현으로 여러 사용자–상품 임베딩 공간을 조건부 혼합하면, 단일 가치공간보다 H&M과
> Dunnhumby의 이질성에 강한 추천표현을 학습할 수 있는가?

M2의 성공은 모든 지표의 우월성을 요구하지 않는다. Recall/NDCG 비열등성 범위 안에서 가격·구매금액
가중 적중값 등 경제적 지표가 외부 M1보다 개선되고, CLV 대조군보다 우월하면 후보로 인정한다.

## 2. 모형 정의

기존 이진 LightGCN의 최종 사용자·아이템 임베딩을 `z_u`, `z_i`, train 정보만으로 만든 사용자
CLV 관련 행동표현을 `h_u`, 아이템 경제·카테고리 속성을 `x_i`라고 한다.

세 개(`K=3`)의 작은 저차원 임베딩 전문가가 사용자와 아이템을 같은 전문가별 공간으로 변환한다.

\[
e_{u,k}=f_{u,k}(z_u,h_u), \qquad e_{i,k}=f_{i,k}(z_i,x_i)
\]

사용자별 dense gate는 CLV 관련 행동표현만으로 전문가 혼합비를 만든다.

\[
\alpha_u=\operatorname{softmax}(g(h_u)), \qquad \sum_k\alpha_{u,k}=1
\]

최종 점수는 외부 M1과 동일한 기본 선호점수에 전문가 임베딩 내적의 혼합을 residual로 더한다.

\[
s(u,i)=\langle z_u,z_i\rangle+
\lambda\sum_{k=1}^{3}\alpha_{u,k}\langle e_{u,k},e_{i,k}\rangle
\]

각 expert는 사용자 쪽 `concat(z_u,h_u)`와 아이템 쪽 `concat(z_i,x_i)`를 각각
`Linear → GELU → Linear`로 변환한다. hidden dimension은 32, 출력 embedding dimension은 16으로
고정한다. gate는 `h_u → Linear(32) → GELU → Linear(3) → softmax`다. 마지막 expert 선형층은
표준편차 `0.01`의 작은 값으로 초기화한다. sparse routing,
load-balancing loss, 전문가별 그래프는 초기 모형에 넣지 않는다. 전문가 붕괴가 실제 관찰될 때만 별도
탐색적 후속변형으로 다룬다.

## 3. M2·M3·M4 경계

- **M2 변경:** 사용자·아이템 임베딩, CLV 조건 gate, 전문가별 잠재공간.
- **M3 고정:** 이진 인접행렬, LightGCN 정규화와 전파식은 M1과 동일.
- **M4 고정:** 모든 `(u,i,j)`에 동일한 plain BPR을 사용한다. CLV별 표본가중·margin을 쓰지 않는다.
- gate는 사용자별로 한 번 계산한다. 정답 아이템을 본 pair-dependent gate는 사용하지 않는다.

따라서 CLV는 손실의 중요도가 아니라 어떤 임베딩 공간을 사용할지 결정한다.

## 4. 사용자 CLV 관련 행동표현

현재 미래가치 encoder의 train-only 시간설계를 유지한다. 공식 validation/test/holdout은 encoder의 입력,
표적, 표준화 또는 조기종료에 사용하지 않는다. `h_u`는 현재 16개 행동변수의 표준화값, 16개 validity
mask, 16차원 encoder hidden state, 구매확률, 조건부 log 금액, log 기대구매금액을 이어 붙인 벡터다.
행동변수는 다음 공통 의미축을 포괄한다.

1. 활동성: 구매·방문빈도, 최근성, 평균 구매간격.
2. 금액성: 총금액, AOV, 금액 안정성.
3. 구매폭: 고유 상품·카테고리 수와 다양성.
4. 반복성: 재구매율 또는 반복 행동의 안정성.
5. 가격성향: 구매상품의 카테고리 내 가격 위치.
6. 미래가치 예측: 미래 구매확률, 조건부 금액 또는 기대구매금액, encoder hidden state.

동일한 의미축과 결측 mask 인터페이스를 두 데이터셋에 유지하되 원천변수와 관찰창은 데이터 특성에 맞춘다.

- **H&M 2년:** 구매간격, 상품·카테고리 다양성, 신상품 탐색, 가격대, 미래 구매금액을 중심으로 구성.
- **Dunnhumby:** 방문주기, 반복구매율, 장바구니 규모, 카테고리 폭, 가격대, 미래 구매금액을 중심으로 구성.

CLV를 고가상품 선호와 동의어로 두지 않으며 gate에 단조성 제약을 걸지 않는다.

## 5. 아이템 표현

`x_i`는 train에서만 계산하며 아이템 CLV라고 부르지 않는다. 연속형 속성은 train 분포로 표준화하고,
범주형 속성은 8차원 category embedding으로 변환한다. 공통 입력은 다음과 같다.

- 전역 가격백분위와 카테고리 내 가격백분위.
- 카테고리 계층 임베딩과 가용성 mask.
- `log1p(train 구매행 수)`, 고유 구매자 수 백분위, 동일상품 반복구매 비율.
- H&M의 상품군 계층 또는 Dunnhumby의 commodity 계층처럼 데이터셋에 존재하는 범주 속성.

가격은 직접 점수에 더하지 않고 전문가 item adapter의 입력으로만 쓴다. 따라서 모든 사용자에게 고가상품을
같은 방향으로 밀지 않고 사용자 gate와 전문가 공간을 거쳐 적합도를 학습한다.

## 6. 학습 모형과 비교 기준

### 6.1 주 모형

`joint_warm`을 주 M2로 둔다. 같은 seed의 외부 M1 checkpoint에서 시작해 처음 5 epoch는 LightGCN을
동결하고 adapter와 gate만 학습한다. 이후 LightGCN embedding을 해제해 공동 미세조정한다. adapter/gate
학습률은 `5e-4`, LightGCN 학습률은 `5e-5`, 최대 100 epoch, validation patience 20으로 고정한다.
학습 중 residual 계수는 `lambda_train=1.0`이다. plain BPR, 기존 negative sampling, batch size와
정규화 규칙을 유지한다.

### 6.2 진단 및 대조군

1. 외부 M1 LightGCN: 모든 차이의 공통 기준.
2. `pref_continue`: M1을 같은 update 수와 LightGCN 학습률 `5e-5`로 plain BPR 추가 학습한 계산량 통제.
3. `frozen_moe`: M1은 동결하고 gate와 adapter만 학습해 표현 가지 자체의 효과를 진단.
4. 현재 단일 CLV-Residual: 복수 임베딩 공간 필요성을 비교.
5. `constant_gate`: 모든 사용자에게 학습 데이터 평균 gate 사용.
6. `shuffled_clv`: seed별 고정 permutation으로 사용자 간 `h_u`를 섞음.
7. `single_adapter`: 전문가 전체와 파라미터 수를 맞춘 단일 adapter.

주장에 필요한 최소 식별 비교는 외부 M1, pref_continue, constant gate, shuffled CLV, single adapter다.
먼저 각 데이터셋에서 주 모형과 외부 M1만 seed 42 validation으로 screening한다. 주 모형이 성공조건을
통과한 데이터셋에만 seed 42 대조군을 실행한다. 두 데이터셋 중 하나라도 주 모형이 실패하면 다중 seed와
test로 확장하지 않는다.

## 7. λ 선택과 성공·중단 기준

- seed 42 validation 탐색 grid: `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`.
- 각 `K=10,20,50`에서 Recall과 NDCG가 외부 M1 대비 상대 1% 이상 하락한 후보는 제외.
- 통과 후보 중 가격·구매금액 가중 적중값@10이 가장 큰 λ를 선택하고, 동률이면 작은 λ를 선택.
- λ=0 fallback은 성공으로 보지 않는다.
- 주 모형의 λ>0 후보가 M1을 넘더라도 constant·shuffled·single-adapter 중 하나에 설명되면
  “CLV 조건부 전문가 효과”로 주장하지 않는다.
- 전문가 붕괴 또는 노출집중 악화가 나타나면 원인을 기록하되, 결과를 본 뒤 load-balancing loss를 같은
  확증 실험에 추가하지 않는다.

두 데이터셋 모두 seed 42 validation에서 성공조건을 통과한 구조와 선택규칙만 동결하여 다중 seed와
미사용 test로 진행한다. 한 데이터셋만 성공하면 범용모형으로 주장하지 않고 데이터셋 조건부 결과로 보고한다.

## 8. 저장 지표와 진단

기존 공통 지표를 모두 저장한다.

- Recall/NDCG@10·20·50과 가격·구매금액 가중 적중값.
- ARP, 가격 정렬도.
- 추천상품 절대 수, Coverage, exposure entropy/effective catalog size.
- 상위 10·100 상품 노출점유율.
- 고객가치 세그먼트별 성과와 paired delta.

MoE 전용 진단을 추가한다.

- 사용자별 gate entropy, 전문가별 평균·분위별 사용량.
- 전문가 출력 embedding cosine similarity와 전문가 점수 상관.
- CLV·행동특성 분위별 gate 사용량.
- `std(expert residual score)/std(M1 score)`와 λ별 용량반응.
- 동결 hash, checkpoint/config/result hash, update 수와 실행시간.

## 9. 실행 안전장치와 검증

- 기본 실행은 seed 42, validation-only, `EVAL_TEST=False`, `EVAL_HOLDOUT=False`다.
- dataset, seed, split, 출력경로를 원자적으로 설정하는 Colab preset을 제공한다.
- 실제 학습 셀은 `ACKNOWLEDGE_HIGH_COST=False`로 시작하며 preflight 출력 검토 후에만 승인한다.
- M1 checkpoint hash, train-only feature 계산, shuffled permutation 재현성, gate 합 1, λ=0 점수 동일성,
  plain BPR 불변성, 결과 schema를 runtime test로 검증한다.
- 고비용 실행 전 변경사항·설정·예상 산출물을 한 번에 검토한다.

## 10. 논문에서의 주장 범위

MoE 자체를 새로운 이론으로 주장하지 않는다. 선행연구가 있는 조건부 표현학습을 CLV-LightGCN 연구에
맞게 적용한 것이다. 본 연구의 기여는 CLV 관련 행동표현으로 복수 사용자–아이템 임베딩 공간을 조건화하고,
단일 경제공간과 비교해 두 소매 데이터셋의 상이한 가치형성 메커니즘에 대한 강건성을 검증하는 데 있다.

현재 CLV-Residual의 실패는 숨기지 않고 user-only 단일공간 ablation으로 보고한다.

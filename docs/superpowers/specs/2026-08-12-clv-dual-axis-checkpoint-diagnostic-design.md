# CLV 이중축 체크포인트 재평가 진단 설계

## 1. 목적

기존 seed 42 validation에서 학습된 `dual_clv_fixed` 체크포인트를 재사용해 다음 질문에 답한다.

1. 성과 개선을 N축, V축 중 어느 축이 주도하는가?
2. 두 축을 함께 사용할 때 단독축보다 보완효과가 있는가?
3. 개선이 `q_N/q_V` 사용자 4분면 중 어디에서 발생하는가?
4. 기존 선택점 주변에 성과가 안정적으로 유지되는 lambda 구간이 있는가?

이는 M2 내부 메커니즘 진단이다. 그래프와 손실함수를 바꾸지 않으므로 M3·M4와 겹치지 않는다.

## 2. 방법 선택

### 채택: 기존 체크포인트 재평가

- 원 데이터로 기존 validation split과 평가 cache를 동일하게 재구성한다.
- 기존 M1·encoder·`dual_clv_fixed` 체크포인트를 hash 검증 후 불러온다.
- 모델을 재학습하지 않고 점수식의 축 조합과 lambda만 바꿔 평가한다.
- 기존 결과 JSON의 데이터·M1·checkpoint fingerprint와 현재 입력이 다르면 중단한다.

### 제외한 대안

- **새 축별 모델 재학습:** N-only와 V-only의 표현 학습 자체가 달라져 현재 결합모형 내부 기여를
  분리하지 못하고 비용도 크다.
- **기존 CSV만 분석:** 사용자별 `q_N/q_V`, 추천목록, 정답 지표가 없어 4분면 paired gain과
  N-only/V-only 점수를 계산할 수 없다.

## 3. 평가 점수식

동일하게 학습된 N/V 전문가를 사용하고 평가 시 축만 마스킹한다.

```text
M1:     S_base
N-only: S_base + lambda * g_N * S_N
V-only: S_base + lambda * g_V * S_V
N+V:    S_base + lambda * (g_N * S_N + g_V * S_V)
```

모든 조건은 동일 M1, 동일 사용자·아이템, 동일 validation 정답, 동일 학습관측 마스킹을 사용한다.
N-only/V-only는 새 제안모형이 아니라 이미 학습된 결합모형의 사후 메커니즘 ablation이다.

## 4. 평가 범위

- Dunnhumby: 기존 선택 gate `equal`, lambda
  `[1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]`
- H&M 60일: 기존 선택 gate `high`, lambda
  `[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]`
- 공통 비교: M1, N-only, V-only, N+V
- seed 42 validation-only. test·holdout은 생성하지 않는다.

세밀한 lambda는 새로운 확증용 선택값을 만드는 것이 아니라 기존 선택점 주변의 plateau와 축별
용량반응을 확인하는 탐색적 진단이다. 기존 공식 선택점과 판정은 변경하지 않는다.

## 5. 사용자 4분면

유효 사용자에 대해 train-only 고정 백분위 `q_N`, `q_V`를 0.5에서 나눈다.

| 4분면 | 조건 |
|---|---|
| 저활동·저가치 | `q_N < 0.5`, `q_V < 0.5` |
| 활동형 | `q_N >= 0.5`, `q_V < 0.5` |
| 가치형 | `q_N < 0.5`, `q_V >= 0.5` |
| 핵심형 | `q_N >= 0.5`, `q_V >= 0.5` |

각 집단에서 사용자 수, M1 및 각 축 조합의 Recall/NDCG/가격·구매금액 가중 적중값/ARP,
M1 대비 평균·중앙값, 개선 사용자 비율과 paired bootstrap CI를 저장한다. 표본이 없는 집단은
0으로 대체하지 않고 명시적으로 결측 처리한다.

## 6. 동일 실효강도 비교

N-only, V-only, N+V는 축 수가 달라 같은 lambda의 실효강도가 다르다. 따라서 두 표를 모두 저장한다.

1. 같은 lambda의 운영곡선
2. `std(residual)/std(S_base)`가 가장 가까운 점 및 선형보간 비교

단독축과 결합축의 시너지는 같은 lambda만으로 단정하지 않는다. 동일 실효강도에서도 N+V가 단독축을
상회하고, 서로 다른 4분면에서 N/V 기여 방향이 달라야 보완성의 근거로 해석한다.

## 7. 산출물

- 전체 절대지표 CSV
- M1 대비 사용자 대응 delta와 bootstrap CSV
- 사용자 4분면 CSV
- 같은 lambda 및 동일 실효강도 축 비교 CSV
- lambda·실효강도 곡선 PNG
- 설정, source/data/M1/checkpoint hash, 원본 결과 ID, 한계를 포함한 JSON
- 두 데이터셋을 순차 처리하는 별도 Colab notebook

## 8. 안전성과 오류 처리

- 공개 진입점은 `run_checkpoint_diagnostic(result_json, output_dir)`이다.
- 결과 JSON이 validation-only가 아니거나 seed 42가 아니면 데이터 준비 전에 중단한다.
- 원본 파일·M1·encoder·dual checkpoint가 없거나 hash가 다르면 중단한다.
- checkpoint를 불러온 뒤 실제 학습 optimizer/epoch 함수는 호출하지 않는다.
- 결과는 기존 screening 파일을 덮어쓰지 않고 `checkpoint_diagnostics/`에 저장한다.

## 9. 성공 해석

이 진단 자체에는 새 성공/실패 판정을 붙이지 않는다. 다음 모델 변경은 아래 관찰에 따라 한 가지만
선택한다.

- N+V가 동일 실효강도에서 두 단독축을 안정적으로 상회: 결합 유지, residual 상한/개입 안정화 검토
- 한 축이 대부분의 개선을 설명: 데이터셋별 축 가중 또는 불필요 축 축소 검토
- 축별로 서로 다른 4분면에서 개선: 현재 이중축 이론의 메커니즘 근거 강화
- 두 축 모두 adapter 수준을 넘지 못함: 표현 학습 또는 아이템 대응특성 재설계

모든 결과는 단일 seed validation의 탐색적 메커니즘 진단이며 test 확증으로 표현하지 않는다.

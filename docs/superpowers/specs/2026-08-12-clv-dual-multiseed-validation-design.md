# CLV 이중축 M2 추가 seed validation 설계

## 1. 목적과 범위

seed 42 validation에서 확인된 N+V 이중축 M2의 개선이 초기화에 의존하지 않는지
seed 43·44에서 재현한다. 이번 단계는 **validation 재현성 확인까지만** 수행한다.

포함:

- Dunnhumby 전체기간 seed 43·44
- H&M 60일 seed 43·44
- 같은 seed의 M1과 `dual_clv_fixed`
- 기존 seed 42 결과와 결합한 3-seed 판정

제외:

- H&M 2년
- test·holdout
- lambda·gate·모형구조 재선택
- shuffled-user·adapter-only 대조군의 자동 실행
- M3·M4·M5

재현성에 성공하더라도 후속 고비용 실행을 자동 시작하지 않고 결과 검토에서 멈춘다.

## 2. 동결 운영점

| 데이터 | 구조 | gate | lambda |
|---|---|---|---:|
| Dunnhumby | N+V `dual_clv_fixed` | `equal` | 2.0 |
| H&M 60일 | N+V `dual_clv_fixed` | `high` | 1.0 |

seed마다 운영점을 다시 선택하지 않는다. encoder, M1, adapter는 각 seed로 새로 학습하지만
특징, 구조, 학습예산, negative sampling, 그래프, 손실, gate와 lambda는 동일하게 유지한다.

## 3. 실행구조

기존 seed-42 screening runner를 느슨하게 확장하지 않고 별도 public runner를 추가한다.

```text
run_multiseed_validation(dataset preset, seed42 result JSON)
  ├─ 설정·원자료 manifest·seed42 결과 검증
  ├─ seed 43: encoder → M1 → dual_clv_fixed → 고정 운영점 validation
  ├─ seed 44: encoder → M1 → dual_clv_fixed → 고정 운영점 validation
  ├─ seed 42 저장 결과와 결합
  └─ 3-seed 절대지표·paired delta·판정 저장
```

Colab은 Dunnhumby와 H&M 60일을 순차 실행한다. 한 데이터셋 실패 시 오류를 명확히 남기고
다음 데이터셋을 조용히 실행하지 않는다.

## 4. 데이터와 비교기준

- 각 seed의 M1은 동일 seed의 `dual_clv_fixed`와 paired 비교한다.
- raw transaction/item metadata manifest는 seed 42 원본 결과와 같아야 한다.
- validation split과 신규상품 정답 정의는 seed와 무관하게 기존 파이프라인을 그대로 사용한다.
- test·holdout 정답은 구성하지 않는다.
- seed 42는 재학습하지 않고 기존 원본 결과의 절대지표·delta를 사용한다.
- seed 43·44의 M1·encoder·adapter checkpoint는 seed가 포함된 별도 파일명으로 저장한다.

## 5. 판정규칙

재현성 통과는 다음 세 조건을 모두 만족할 때다.

1. 세 seed 평균 `revenue@10(model - M1) > 0`
2. 세 seed 중 최소 2개에서 `revenue@10(model - M1) > 0`
3. 세 seed 평균 기준 Recall/NDCG @10/@20/@50 여섯 지표가 각각 M1 평균의 99% 이상

추가로 다음을 반드시 보고하되 통과조건을 사후 변경하지 않는다.

- seed별 절대지표와 상대 변화율
- seed별 사용자 paired bootstrap CI
- 3-seed 평균 delta와 seed 간 표준편차
- Coverage, 추천상품 절대개수, exposure entropy/effective catalog,
  top-10/top-100 노출점유율, ARP, value alignment

단일 seed만 개선하거나 seed 간 부호가 크게 다르면 불안정한 결과로 해석한다. CI가 0을 포함한
사실도 숨기지 않으며, 이 실행은 여전히 validation 재현성이지 test 확증이 아니다.

## 6. 산출물

데이터셋별로 다음을 저장한다.

- 3-seed 절대지표 CSV
- seed별 및 3-seed paired delta CSV
- 판정 요약 CSV
- 설정·data/source/checkpoint hash·학습통계·판정을 포함한 JSON
- seed별 경제지표 및 정확도 비교 PNG

Colab 마지막 셀은 두 데이터셋의 통과 여부와 실패 조건만 보여준다. H&M 2년 또는 대조군
실행 코드는 포함하지 않는다.

## 7. 안전성과 비용 제한

- public runner 시작 시 seed `(43, 44)`, validation-only, 고정 gate/lambda를 검증한다.
- `eval_test` 또는 `eval_holdout`이 켜지면 데이터 접근 전에 실패한다.
- seed 42 결과·원자료 manifest가 없거나 불일치하면 재학습 전에 실패한다.
- 제안모형 외 대조군과 lambda curve를 학습·평가하지 않는다.
- 성공 여부와 관계없이 H&M 2년, test, 대조군을 자동 호출하지 않는다.

## 8. 해석 경계

이 실험이 통과하면 “두 데이터셋의 짧은/현재 관찰조건에서 N+V M2 개선이 세 초기화에 걸쳐
재현됐다”고만 말한다. 추천으로 실제 CLV나 증분매출이 증가했다고 주장하지 않는다. M2 성공이
M3·M4를 대체하지 않으며, 세 축의 독립·결합 비교라는 전체 연구구조는 유지한다.

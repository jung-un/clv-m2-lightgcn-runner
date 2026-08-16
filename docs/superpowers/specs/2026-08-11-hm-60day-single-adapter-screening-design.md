# H&M 60일 단일 어댑터 validation screening 설계

## 1. 목적과 연구상 지위

H&M 약 2년 전체를 사용한 M2 본실험 전에, 최근 60일만 사용해 단일 어댑터 파이프라인의
실행 가능성과 방향성을 확인한다. 이 실행은 비용을 줄인 탐색적 screening이며 H&M 2년
주실험을 대체하지 않는다. 60일에서 성공해도 2년 성과로 일반화하지 않고, 실패해도 2년
모형 실패로 확대해석하지 않는다.

연구축은 M2 임베딩·표현 개입이다. 이진 LightGCN 그래프, 균등 negative sampling,
plain BPR을 유지해 M3 그래프 가중과 M4 손실 가중을 포함하지 않는다.

## 2. 시간 범위와 분할

- 데이터셋: H&M `transactions_train`과 `articles`.
- 원자료 범위: 공식 거래자료의 마지막 날짜를 기준으로 최근 60일.
- 분할: 현재 공통 규칙인 train | validation 7일 | test 7일 | holdout 7일을 유지한다.
- 따라서 모델이 직접 학습하는 train 구간은 약 39일이다. 날짜 경계 포함 여부에 따라 실제
  달력일 수는 한두 날 차이 날 수 있으므로 결과 JSON의 분할 경계와 행 수를 기준으로 보고한다.
- 실행 중에는 validation 정답만 만든다. test와 holdout은 `False`로 고정해 정답도 만들지 않는다.
- seed는 42 하나만 사용한다.
- 평가 정답은 train에 없었던 신규 `(user, item)` 쌍으로 제한한다.

이 정의는 기존 H&M `WINDOW_DAYS=60` 예비실험과 맞춘 것이다. `60일 train + 21일 평가
예약기간`인 총 81일 설계와 혼용하지 않는다.

## 3. CLV 관련 행동표현의 짧은 창 설정

60일 원자료에서는 365일 입력·90일 예측 encoder를 만들 수 없으므로 H&M 60일 전용으로
다음 설정을 사용한다.

- 행동입력 lookback: 14일
- 미래가치 예측 horizon: 7일
- train 내부 anchor offset: 21일, 14일, 7일
- 최종 추천모형 입력 snapshot: train 종료시점 이전 14일

현재 encoder는 `input_days + max(anchor_offsets)`만큼의 연속 train 관찰기간을 요구한다.
`14 + 21 = 35일`은 약 39일 train 안에 들어오며, 세 7일 target 구간도 서로 겹치지 않는다.
기존 `observed_days`와 변수별 validity mask가 사용자별 실제 관찰량을 표현하고, 결과에
anchor별 유효 표본 수·관찰기간 분포를 남긴다. 14일 행동표현이라는 짧은 기간은 60일 탐색
결과의 한계로 보고한다.

## 4. 비교 모형과 선택 규칙

외부 M1은 같은 60일 데이터, 같은 seed, 같은 split, 같은 이진 그래프와 plain BPR로 새로
구성한다. 2년 M1이나 다른 window의 checkpoint는 재사용하지 않는다.

주 후보는 `single_full`이다. 주 후보가 외부 M1 대비 아래 조건을 통과할 때만 식별 대조군을
실행한다.

1. Recall@10/20/50과 NDCG@10/20/50이 각각 M1의 99% 이상이다.
2. 양의 lambda에서 가격·구매금액 가중 적중값@10이 M1보다 실제로 높다.
3. 조건을 만족하는 후보 중 해당 경제지표가 가장 높은 lambda를 고르고, 동률이면 작은
   lambda를 선택한다.

lambda grid는 Dunnhumby와 동일한 `(0, 0.1, 0.25, 0.5, 1.0, 2.0)`으로 고정한다.
통과 후보가 없으면 lambda 0으로 돌아가되 실패로 기록한다.

주 후보 통과 후 실행하는 대조군은 `single_zero_user`, `single_shuffled_user`,
`single_base_only`, 메커니즘 대조군 `single_zero_item`, 계산량 통제 `pref_continue`다.
최종 screening 성공은 `single_full`이 외부 M1과 필수 대조군보다 경제지표가 높을 때만 인정한다.

## 5. 코드·산출물 격리

공개 설정에 `window_days`를 추가하고, H&M 60일 notebook preset이 이를 명시적으로 60으로
전달한다. `_pure_m1_config`는 더 이상 `WINDOW_DAYS=None`을 강제하지 않고 설정값을 전달한다.

- 모형 결과 폴더: `/content/drive/MyDrive/논문/data/results_clv_single_hm_w60`
- M1 폴더: `/content/drive/MyDrive/논문/data/results_v3_hm_w60`
- dataset, window, encoder horizon, anchor 설정은 config와 result fingerprint에 포함한다.
- M1 manifest의 config hash에도 `WINDOW_DAYS=60`이 들어가 다른 window checkpoint와 섞이지
  않게 한다.
- preflight는 `H&M last 60 calendar days`, 약 39일 train, validation-only를 사람이 읽을 수
  있게 표시한다.

2년 H&M 실행 경로는 `window_days=None`, 기존 365→90 encoder 설정으로 별도 유지한다.
두 preset의 출력 폴더와 fingerprint를 공유하지 않는다.

## 6. 저장 지표와 해석

기존 단일 어댑터 결과와 동일하게 다음을 저장한다.

- Recall/NDCG@10/20/50과 가격·구매금액 가중 적중값
- ARP, Coverage, 추천상품 절대 수, exposure entropy/effective catalog size
- 상위 10·100 상품 노출점유율과 CLV 세그먼트 지표
- 모든 lambda 절대곡선과 외부 M1 대비 사용자 대응 bootstrap 차이
- encoder 진단, 입력·mask hash, 학습 파라미터 수, checkpoint SHA와 실행 fingerprint

`revenue`는 실제 증분매출이나 CLV가 아니라 가격·구매금액 가중 추천 적중값으로 해석한다.
60일 행동표현은 단기 구매행동 표현이며 생애전체 CLV로 부르지 않는다.

## 7. 실패 차단과 검증

- `MoEConfig` 직접 생성과 notebook 양쪽에서 H&M 60일 preset을 검증한다.
- `eval_test` 또는 `eval_holdout`이 켜지면 데이터 접근 전에 실패한다.
- H&M 60일 preset이 실제 `base_cfg["WINDOW_DAYS"] == 60`을 만드는 runtime test를 둔다.
- 60일과 2년 설정의 result fingerprint 및 M1 config hash가 달라지는지 검사한다.
- Dunnhumby 기본 `window_days=None`과 기존 결과 재사용 규칙이 회귀하지 않는지 검사한다.
- notebook은 고비용 승인 셀 전까지 전처리·학습을 시작하지 않는다.
- 코드·테스트·notebook·연구상태 변경을 한 번에 검토한 뒤에만 Colab 고비용 실행을 승인한다.

## 8. 후속 결정

60일 결과는 파이프라인과 단기 행동표현의 유망성을 판단하는 탐색 자료다. 성공하면 예정대로
H&M 2년 seed 42 validation screening을 별도로 실행한다. 두 데이터셋의 주 설정 screening이
확정되기 전에는 다중 seed나 보호된 test·holdout으로 확장하지 않는다.

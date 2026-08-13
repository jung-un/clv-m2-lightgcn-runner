# H&M 2년 M2 seed 42 고정점 검증 설계

## 목적

H&M 60일 validation에서 선택한 이중축 CLV 임베딩과 정규화 강도 `rho=0.2`가 H&M 전체 2년 기간에서도 M1 대비 정확도를 유지하면서 가격·구매금액 가중 적중값을 개선하는지 먼저 seed 42로 확인한다.

## 고정 조건

- 데이터: H&M 전체 기간(`window_days=None`)
- split: validation only, test·holdout 미생성
- seed: 42
- 기준모형: 같은 seed의 외부 M1 LightGCN
- 제안모형: `dual_clv_fixed`만 새로 학습
- 구조·특징·학습 설정: 현재 dual-axis v1.1 그대로
- gate: H&M 60일에서 사용한 `high`
- 목표 실효강도: `rho=0.2`
- 평가 lambda: 학습 후 정답과 무관한 score 표준편차 비율 `raw_effective_ratio`를 구하고 `lambda=rho/raw_effective_ratio`로 1회 결정
- 미실행: shuffled-user, adapter-only, lambda/rho sweep, seed 43·44, test, holdout

## 판정

M1 대비 Recall/NDCG 10·20·50이 각각 99% 이상이며 `revenue@10`이 높을 때 다음 단계로 진행한다. 통과하면 같은 설정으로 seed 43·44 및 필수 대조군을 실행한다. 실패하면 2년 데이터에서 M2 재현 실패로 기록하고 validation 하이퍼파라미터를 추가 탐색하지 않는다.

## 출력

M1과 제안모형의 절대지표, M1 대비 paired delta, `raw_effective_ratio`, 환산 lambda, 목표/실제 실효강도, 체크포인트 및 입력·코드 fingerprint를 Drive에 저장한다.

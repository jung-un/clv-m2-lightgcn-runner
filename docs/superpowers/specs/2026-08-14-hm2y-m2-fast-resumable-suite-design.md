# H&M 2년 M2 4모형 고속·자동재개 실행 설계

## 1. 목적

H&M 전체 기간에서 M2 검증에 필요한 네 모형을 한 번의 Colab 실행으로 순차 학습·평가한다. 장시간 학습 중 Colab 연결이 끊겨도 마지막으로 완료한 epoch와 단계에서 자동으로 재개하며, 학습이 진행 중인지를 Drive 상태 파일로 확인할 수 있게 한다.

이 실행기는 M2 임베딩 연구만 다룬다. M3 가치그래프와 M4 CLV-aware 손실함수는 포함하지 않는다.

## 2. 고정 실험 범위

- 데이터셋: H&M 전체 기간(`window_days=None`)
- seed: `42`
- 평가 split: validation only
- test·holdout: 구성과 평가 모두 금지
- 그래프: M1과 동일한 binary LightGCN
- 손실: plain BPR
- 음성표본: uniform
- M2 gate: `high`
- 목표 실효강도: `rho=0.2`
- 재탐색: 없음. H&M 60일에서 동결한 구조·gate·rho를 적용한다.

실행 모형은 다음 네 개로 고정한다.

1. `m1`: CLV 정보가 없는 순수 LightGCN
2. `dual_clv_fixed`: 제안 N+V 이중축 M2
3. `dual_shuffled_user`: `(e_N,e_V,q_N,q_V)` 사용자 배정 순열 대조군
4. `dual_adapter_only`: CLV 관련 추가입력을 0으로 두고 adapter 용량만 유지하는 대조군

## 3. 속도 개선

### 3.1 확인된 병목

기존 H&M 2년 M1은 `BATCH_SIZE=8192`로 약 2,860만 학습행을 순회한다. LightGCN 전파가 BPR 배치마다 재계산되므로 epoch당 약 3,500번의 전체 그래프 전파가 발생한다. 실측 epoch 시간은 약 2,800초였다.

### 3.2 해결 방법

전체 학습행과 plain BPR 목적함수를 유지하되, H&M 전체기간에서만 대용량 배치를 사용한다.

- 우선 배치: `131072`
- 실제 학습 전에 throwaway M1 1-step 메모리 점검을 하고, OOM이면 `65536`, `32768` 순으로 낮춤
- 점검에서 선택한 배치를 네 모형 전체에 고정하며, 실제 학습 도중에는 arm별로 배치를 바꾸지 않음
- 실제 사용 배치, epoch당 update 수, 샘플 수, 소요시간을 결과에 저장
- BPR loss, `P(pos>neg)`, embedding norm이 정상 방향으로 변하지 않으면 fail closed

학습행 표본추출, 손실함수 변경, 그래프 근사 전파, 혼합정밀도는 사용하지 않는다. 이 방법은 학습 목적을 바꾸지 않고 GPU당 그래프 전파 횟수를 줄이는 구현 최적화로 한정한다. 다만 배치 크기는 최적화 경로에 영향을 줄 수 있으므로 스케일 전용 실행 하이퍼파라미터로 명시하고 모든 arm에 고정한다.

## 4. epoch 체크포인트와 자동 재개

M1, CLV-core encoder, 세 adapter arm은 epoch 단위로 독립 체크포인트를 Drive에 저장한다.

체크포인트는 다음을 포함한다.

- 마지막 완료 epoch
- 현재 모델 state
- optimizer state
- NumPy·PyTorch·CUDA RNG state
- best metric, best epoch, best model state
- early-stopping `bad` count
- 누적 updates, samples, wall-clock
- 학습 진단 이력
- 입력 manifest, config hash, source revision, model id, seed

저장은 임시파일에 먼저 쓴 뒤 원자적 rename으로 완료한다. 중단 중 부분 파일이 남아도 완료된 체크포인트를 훼손하지 않는다.

재개 시에는 manifest·config·source·model id·seed가 모두 일치할 때만 로드한다. 하나라도 다르면 기존 체크포인트를 무시하지 않고 명시적으로 중단한다.

## 5. 단계 오케스트레이션

Drive의 `run_manifest.json`이 전체 실행의 authoritative state다.

단계는 다음 순서로 진행한다.

1. `prepare_data`
2. `batch_preflight`
3. `m1`
4. `clv_core_encoder`
5. `dual_clv_fixed`
6. `dual_shuffled_user`
7. `dual_adapter_only`
8. `validation_evaluation`
9. `comparison_and_decision`

각 단계는 `pending`, `running`, `completed`, `failed` 상태를 갖는다. `completed`이며 산출물 hash가 일치하는 모델·평가 단계는 재실행 시 건너뛴다. `running`이었던 학습 단계는 epoch 체크포인트에서 재개한다. `prepare_data`는 새 런타임에서 후속 모델을 복원하는 데 필요하므로 결정론적으로 다시 구성하되, 이미 끝난 모델을 재학습하지는 않는다.

한 arm이 실패하면 후속 arm으로 조용히 넘어가지 않고 실패 단계와 오류를 저장한 뒤 종료한다.

## 6. 진행상태와 heartbeat

`progress.json`과 `progress.csv`를 Drive에 저장한다.

필수 필드는 다음과 같다.

- run id, source revision, config hash
- 현재 stage, model id, status
- 현재 epoch / 최대 epoch
- best epoch, best metric
- 직전 epoch 소요시간, 이동평균 epoch 시간
- 현재 stage 예상 잔여시간
- 마지막 갱신 시각
- checkpoint 경로
- 직전 loss, `P(pos>neg)`, validation metric

모든 epoch 종료 시와 stage 변경 시 갱신한다. 개별 epoch가 길어 사용자가 정지로 오인하지 않도록 배치 루프 중에도 최대 60초 간격으로 heartbeat 시각과 배치 진척률을 갱신한다.

노트북의 독립 상태확인 셀은 학습 프로세스 메모리에 의존하지 않고 Drive 파일만 읽는다.

## 7. 평가와 판정

세 M2 arm은 각자 bounded score 진단에서 계산한

`raw_effective_ratio = std(residual) / std(M1 score)`

를 사용하여 `lambda = 0.2 / raw_effective_ratio`로 평가한다. validation 정답은 이 계수 산정에 사용하지 않는다.

저장 지표:

- Recall/NDCG @10/@20/@50
- 가격·구매금액 가중 적중값 @10/@20/@50
- ARP @10/@20/@50
- Coverage @10, `n_distinct@10`
- exposure entropy, effective catalog size
- top-10/top-100 exposure share
- value alignment
- M1 대비 per-user paired bootstrap delta
- 제안모형 대비 shuffled-user·adapter-only 경제지표 차이

주판정:

1. `dual_clv_fixed`의 Recall/NDCG @10/@20/@50 여섯 지표가 각각 M1의 99% 이상
2. 가격·구매금액 가중 적중값@10이 M1보다 높음
3. 제안모형의 경제지표@10이 두 대조군보다 높음

이 가드레일은 학습값을 인위적으로 변경하지 않고 실행 후 수용 가능성만 판정한다.

## 8. 출력과 재현성

최종 산출물:

- 네 모형 절대지표 CSV
- M1 대비 paired delta CSV
- 대조군 비교·최종판정 CSV
- 통합 JSON
- 모형별 최종 checkpoint와 epoch-resume checkpoint
- `run_manifest.json`, `progress.json`, `progress.csv`

통합 JSON에는 raw-data manifest, source revision, config, 실제 batch size, 모델·checkpoint hash, 학습 이력, 실효강도, 절대지표, paired delta, 판정을 저장한다.

## 9. 실패 처리

- OOM: 학습 전 `batch_preflight`에서만 다음 후보 배치를 시험한다. 배치가 고정된 후 예상 밖 OOM이 발생하면 비교조건을 조용히 바꾸지 않고 중단한다.
- checkpoint 불일치·훼손: 새로 시작하지 않고 즉시 중단
- Drive 쓰기 실패: 학습을 계속하지 않고 중단
- 비정상 loss·embedding norm: 진단값과 함께 중단
- 완료된 단계 산출물 hash 불일치: fail closed

## 10. 검증

테스트는 다음 동작을 실제 런타임으로 검증한다.

- 2 epoch 중단 후 3 epoch로 재실행하면 epoch 3만 추가 학습
- 중단·재개 실행이 중단 없는 실행과 model·optimizer·RNG·best-state에서 일치
- 완료된 stage는 재실행하지 않음
- config·data·source·seed 불일치 checkpoint 거부
- 사전 메모리 점검 OOM 발생 시 하위 배치 선택과 네 모형 공통 적용
- heartbeat가 배치 루프 중 갱신됨
- 독립 상태확인 셀이 새 런타임에서도 Drive 진행상태를 표시
- test·holdout 정답이 구성되지 않음
- 노트북 JSON, 고정 source SHA, 설정 요약, 실행 셀, 상태확인 셀 검증

## 11. 비범위

- H&M 2년 seed 43·44
- test·holdout
- M3·M4·M5
- gate·rho·임베딩 차원·학습률 재탐색
- 학습행 축소·표본추출
- 손실함수·음성표본·그래프 정규화 변경
- 이번 실행 자동 완료 후 후속 고비용 실험 자동 시작

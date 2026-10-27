# 기존 NGCF Ada 완료 후 같은 GPU에서 GAT M5_A~D 이어 실행

대상 Pod는 `4mgakh8utyoyqu` 한 대입니다. 현재 NGCF PID7796의 M1~M4를 중단하거나 재시작하지 않습니다. 그 프로세스가 종료된 뒤 네 모형의 300epoch 결과/연속 이력/정상 완료 표식/최종 완료 로그를 확인하고, 같은 GPU에서 H&M GAT seed49의 M5_A→B→C→D를 시작합니다. 다른 GAT Pod의 M1~M4 및 별도 NGCF M5 Pod는 유지합니다.

## 사전검사와 대기 시작

기존 venv·입력·고정 소스·공유볼륨을 재사용합니다. 이 폴더에는 `launch.py`, 수정된 `hm_ngcf_m5.py` 저장 helper, `hm_gat_m5.py`, `after_ngcf_then_gat.py`가 필요합니다. 기존 서버에서 pip/setup을 다시 실행하지 않습니다.

```bash
/workspace/clv/venv/bin/python after_ngcf_then_gat.py check --ngcf-pid 7796
/workspace/clv/venv/bin/python -u after_ngcf_then_gat.py wait --ngcf-pid 7796
```

실제 설치 시 두 번째 명령은 독립 세션으로 분리해 브라우저 종료 후에도 유지합니다. `check`는 학습/전체 전처리/평가/GPU 계산을 시작하지 않습니다. `wait`는 기존 NGCF 프로세스의 pidfd 종료 이벤트를 기다리므로 PID 재사용을 잘못 따라가지 않습니다.

## 조건과 결과

- 대상 Pod·현재 NGCF 실행 인자·source·run identity·4개 정상 완료 결과를 확인합니다. 비정상 종료/미완료/다른 GPU 계산 프로세스가 있으면 후속 학습을 시작하지 않고 실패 상태를 남깁니다. 자동 재시도나 Pod 정지는 없습니다.
- GAT 원래 실행의 배정이 seed49 M1~M4 전용인지, M5 진행/완료/평가 기록이 없는지 확인합니다. 새 M5 작업도 전용 잠금으로 중복 실행을 막습니다. 다른 볼륨/계정 전체의 중복은 자동 검색하지 않습니다.
- GAT 고정 source `289e6df371d545687ea72cb3937fe744df3f009a`; 같은 GPU/패키지/CUDA/정밀도·입력/특징·고정 모형/학습설정 일치를 검사합니다. 300epoch·k1·마지막주 신규상품 test·uniform K1·단일 optimizer 공동학습·원래 정밀도 유지. validation/holdout/중간test/조기종료 없음. 입력·그래프 준비는 M5 네 모형에 한 번만 수행합니다.
- 기존 수정된 JSON 동등성 검사와 빠른 직전 checkpoint 보존 helper를 재사용합니다. 매epoch 모델/Adam/RNG 동기 저장 및 최신·직전 복구본을 유지합니다.
- 새 결과 root: `/workspace/clv_hm_gat_m5_seed49`.
- 대기 상태: `queue_status.json`, 대기 로그: 설치 시 지정하는 `queue.log`, GAT 실행 로그: `gat_m5_run.log`.
- GAT 결과 ZIP: `exports/hm-gat_results_partial.zip`. 단독모형과 취합 전이므로 M5 네개를 모두 완료해도 partial이 정상입니다.
- NGCF 작업이 끝나도 Pod는 계속 켜져 있어야 자동 전환할 수 있습니다. Pod 종료/새 배포는 필요 없습니다. GAT M5까지 끝나도 Pod 과금은 사용자가 종료하기 전 계속됩니다.

현재의 H&M 3시드 운영 결정은 이 대기 작업을 seed50/51까지 자동 확장하지 않습니다. 이번 후속 작업은 seed49 GAT M5 네개만입니다.

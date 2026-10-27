# H&M NGCF seed49 — 새 GPU에서 M5_A~D만 실행

기존 M1~M4 Pod는 그대로 두고 **새 Pod**에서 사용합니다. 기존과 같은 Network Volume을 `/workspace`로 연결해야 합니다. 입력·고정 소스·설치 환경은 읽어 재사용하고, 새 결과·진행·로그는 `/workspace/clv_hm_ngcf_m5_seed49`에 저장합니다. 이 준비 패키지를 만드는 작업은 서버 대여나 학습 시작을 수행하지 않습니다.

## 시작 순서

1. 기존 H&M NGCF와 같은 데이터센터의 새 GPU Pod에 기존 H&M Network Volume을 연결합니다. 기존과 동일한 RTX 6000 Ada 1대·CUDA/패키지 환경을 사용합니다. 기존 Pod의 볼륨 연결을 해제하거나 서버를 중단하지 않습니다. 환경이 다르면 코드가 학습 전에 차단합니다. M5의 최대 GPU 메모리와 실행시간은 아직 실측하지 않았습니다.
2. `hm_ngcf_m5_start.zip`을 새 Pod의 `/workspace`에 업로드합니다. Terminal에서 `unzip -n /workspace/hm_ngcf_m5_start.zip -d /workspace`를 실행합니다.
3. `/workspace/hm_ngcf_m5_start/HM_NGCF_M5_시작.ipynb`를 열어 준비 점검 셀을 실행합니다. 기존 `/workspace/clv/venv`를 재사용하므로 pip/setup을 다시 실행하지 않습니다. 패키지가 없거나 GPU/CUDA/정밀도가 기존 실행과 다르면 중단합니다.
4. 다른 볼륨·Colab에 동일 NGCF seed49 M5 실행이 없는지 확인한 후 시작 셀의 `CONFIRMED_NO_DUPLICATE=True`로 바꾸고 실행합니다. 현재 공유볼륨에서 기존 배정이 M1~M4인지, 기존 출력에 M5의 진행/완료/평가 기록이 없는지는 코드가 검사합니다.
5. 진행 셀에서 첫 배치/epoch 완료와 저장을 확인합니다. 새 프로세스는 노트북과 분리되어 브라우저를 닫아도 Pod가 켜져 있으면 계속 실행됩니다. 자동 서버 정지는 없습니다.

Terminal에서 점검만 할 때:

```bash
/workspace/clv/venv/bin/python /workspace/hm_ngcf_m5_start/hm_ngcf_m5.py check
```

## 배정과 결과

| 실행 | 범위 | 결과 위치 |
|---|---|---|
| 기존 Pod | seed49 M1/M2/M3/M4 | `/workspace/clv/results/hm-ngcf` |
| 새 Pod | seed49 M5_A/M5_B/M5_C/M5_D | `/workspace/clv_hm_ngcf_m5_seed49/results/hm-ngcf` |

- 새 실행은 CLI에서 모형·시드를 확대할 수 없으며 M5_A→B→C→D로 고정합니다. 입력 검증·전처리·그래프는 한 번 준비해 네 모형이 순차 사용합니다. M5끼리 한 GPU에서 동시에 실행해 메모리를 경쟁하지 않습니다.
- 각 M5는 기존 단독모형 가중치를 가져오지 않고 자기 구성요소를 같은 추천손실·단일 Adam에서 공동학습합니다.
- 기존 실행 파일/모형/학습 인자/진행 기록은 수정하지 않습니다. 같은 영구볼륨의 새 M5 실행은 전용 잠금으로 하나만 허용합니다. 다른 볼륨·계정 전체와 기존 구형 실행기의 미래 재배정은 자동 통제하지 못하므로, 이 작업 동안 기존 Pod에 M5를 추가하지 않습니다. 파일시스템이 Pod 간 POSIX 잠금을 제공하는지도 실제 시작 때 확인해야 합니다.
- 완료된 M5는 재학습·재평가 없이 재사용합니다. 같은 환경에서 재실행하면 진행 중 M5는 마지막 저장 지점부터 재개합니다. 최종평가 시작 후 결과가 누락되면 자동 재평가를 차단합니다.
- 결과 ZIP: `/workspace/clv_hm_ngcf_m5_seed49/exports/hm-ngcf_results_partial.zip`. M5 네 개가 모두 끝나도 전체 8개 중 일부이므로 `_partial`이 정상입니다. 기존 M1~M4와의 최종 통합·전체 comparison은 두 실행 원본의 신원을 검증한 후 별도로 수행합니다. 이번 패키지는 기존 결과 폴더에 복사하거나 통합하지 않습니다.

## 시간을 줄인 부분과 고정한 부분

- 검증된 기존 입력·소스·venv 재사용: 재업로드/재설치/입력 전체 복사를 생략합니다. 원본 SHA256 검사는 생략하지 않습니다.
- 기존 실행의 uniform K1 동일 RNG 추출, 거절 음성만 재검사, 고정 정수 입력 전송, M2 positive 위치 사전 검증 등 실행 최적화를 그대로 사용합니다. 학습 가능한 임베딩은 매 배치 다시 계산합니다.
- 직전 checkpoint는 POSIX hard link가 지원되면 같은 파일 바이트를 가리키는 별도 이름으로 즉시 보존합니다. 원래 저장 함수는 새 파일을 완전히 닫고 fsync한 뒤 latest를 원자적으로 교체하므로 직전본은 계속 이전 바이트를 유지합니다. 미지원 파일시스템에서는 해시 검증 복사로 돌아갑니다.
- 매 epoch 전체 모델/Adam/RNG 동기 저장, 최신·직전 복구본, 손상/용량부족 중단을 유지합니다. `progress/timing/*.csv`에 실제 학습·저장·직전본 보존 시간을 기록합니다.
- **단축률은 보장하지 않습니다.** 공유볼륨의 다른 두 작업과 디스크/네트워크를 경쟁할 수 있습니다. 실제 시간은 첫 epoch 계측으로 확인합니다.
- 기존 고정 커밋 `db39115fadad5aedb020eafe4a492a2862a876db`; H&M ~2020-09-15 학습/09-16~22 신규상품 test, k1, seed49, 300epoch, batch131072, lr.0005, NGCF ID64·2층, dropout.1, uniform K1, 정밀도·optimizer·M2/M3/M4 정의를 유지합니다. validation/holdout/조기종료/중간 test는 없습니다.

로컬 검증은 분담 범위, 중복 기록 차단, 기존 결과 불변, 한 번의 전처리, 학습 조건·환경 불일치 차단, hard link 보존/미지원 복사/저장 실패를 확인합니다. 실제 새 Pod의 GPU 메모리·파일시스템 성능·전체 학습은 시작 전 점검과 첫 배치로 추가 확인해야 합니다.

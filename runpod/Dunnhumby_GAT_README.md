# Dunnhumby GAT — Runpod L4 시작 안내

## 서버 선택
- GPU: NVIDIA L4 24GB **1개**, On-Demand. Python 3.10~3.12의 공식 PyTorch/Jupyter 템플릿.
- 시스템 RAM 32GB 이상, Container Disk 50GB, Network Volume 50GB 이상을 `/workspace`에 연결.
- 기존 Colab의 L4와 같은 GPU 종류를 권장하나 환경까지 동일하거나 비트 단위 결과가 같다는 보장은 없음. 환경을 결과에 기록.
- 실측 Runpod 메모리/실행시간은 아직 없음. 첫 epoch 로그로 학습·저장을 확인. 50GB의 전체 완료 여유는 실제 저장량에 따라 달라지며 여유10GiB 이하에서는 안전중단.

## 처음 시작
1. Runpod에서 위 설정으로 Pod 생성 후 Connect → JupyterLab.
2. `runpod_gat.zip`을 `/workspace`에 업로드.
3. JupyterLab Terminal에서 아래 명령 실행:

```bash
cd /workspace
unzip -n runpod_gat.zip
```

4. `/workspace/runpod_gat/Dunnhumby_GAT_시작.ipynb` 열기.
5. 셀을 위에서 순서대로 실행: 설치 → 입력 설치 → 사전점검 → 학습 시작 → 로그 확인.
6. 기본값은 `[42,43,44,45,46,47,48,50,51]` 전체. seed49는 재학습하지 않음.
7. 시작 셀은 별도 프로세스로 학습을 실행. 브라우저를 닫아도 Pod가 켜져 있으면 계속됨. 시작 직후에는 첫 epoch 완료·저장 로그를 반드시 확인.

## 중단 후 재개
- 동일 Network Volume, 동일 L4/패키지/코드로 같은 노트북의 입력·점검·시작 셀을 실행.
- 설치는 최초 한 번. 완료 모형은 건너뛰며 미완료 학습은 최신 epoch의 모델/Adam/난수로 재개.
- Pod를 바꾸면 `/workspace/clv/sources`가 그대로 있는지 확인. 소스·환경 차이는 자동재개 차단.
- 기존 Colab 체크포인트를 이 출력에 복사하지 않음. 동일 모델·시드의 외부 중복 실행은 자동감지하지 못함.
- 최신+직전 체크포인트 유지, 손상/저장공간부족 시 보존하고 중단. 최종평가 시작 후 결과가 없으면 자동 재평가 차단: 먼저 결과 복구 확인.

## 결과·과금
- 각 모형 후 `/workspace/clv/exports/dh-gat_results_partial.zip`, 전체 완료 후 `dh-gat_results.zip`.
- 새72개 + seed49 기존8개가 모두 있어야 10시드 평균·표본SD 생성. 일부 결과로 전체 평균을 만들지 않음.
- 결과ZIP은 지표·진단·설정·학습이력·환경을 포함. 대형 가중치/optimizer는 `/workspace/clv/results/dh-gat` 별도 보존.
- 다운로드 확인 후 Runpod 콘솔에서 GPU 과금 종료. 브라우저 종료/학습 완료/오류는 과금을 종료하지 않음. Network Volume은 삭제하지 말고 백업 후 관리.

## 연구 조건
M1/M2/M3/M4/M5_A/M5_B/M5_C/M5_D,300epoch,DAY1~704학습/705~711신규상품test,MIN_ITEM_INTER=1,uniform K1,기존batch8192/lr.0005/GAT/M2/M3/M4/공동optimizer 유지. 검증구간·조기종료·중간test·test후튜닝 없음. 학습조건 변경은 시드 반복뿐. 기존8모형 seed49 원본은 SHA 검증 후 재사용하며 초기화용 가중치로 사용하지 않음.

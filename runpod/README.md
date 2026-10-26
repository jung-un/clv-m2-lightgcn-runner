# Runpod 실행 안내 — 2026-10-26

현재 실행 대기 코드 세 가지를 그대로 호출하는 준비 패키지입니다. 기존 학교 GCN·GraphSAGE 실행을 옮기거나 중단하지 않습니다. 준비 시점 기록에서 미실행으로 분류했으며, 실제 다른 서버에서 시작했는지는 실행 직전에 확인하세요.

| 작업 선택값 | 실행할 내용 | 고정 원본 커밋 |
|---|---|---|
| `dh-ngcf` | Dunnhumby NGCF: seed42~48·50·51 중 지정한 시드, 시드마다8모형. seed49 결과 재사용 | `63146eff4ed60bfb0cc430b77d58fdc40e4db7c5` |
| `hm-ngcf` | H&M NGCF seed49,8모형 | `836c0c71c6aab3a952f8f948f9920b32c7b109c8` |
| `hm-lightgcn` | H&M LightGCN seed51,8모형 | `8ca3baa90be1e0ac02554e1c6fe4c1c6b8deb229` |

## GPU 선택

**첫 실행은 Dunnhumby NGCF + L4 한 대를 권장합니다.** 기존 Colab에서 정한 GPU와 같아 환경 변경을 줄일 수 있습니다. H&M NGCF는 전체 그래프와 중간 표현이 더 크므로 예산이 허용되면 A10080GB와 시스템 RAM96GB 이상 구성을 권장합니다. 이는 여유를 둔 자원 제안이며 실측 최소 메모리나 속도 보장은 아닙니다.

| 목적 | GPU 수/모델 | 시스템 RAM 권장 | Network Volume 권장 | 공식 가격표 표시 예시 |
|---|---|---|---|---|
| DH NGCF부터 시작 | 1×L4 24GB | 32GB 이상 | 50GB | $0.59/시간 |
| H&M NGCF, 메모리 여유 우선 | 1×A100 80GB | 96GB 이상 | 100GB | $1.79/시간 |
| 기존 L4 조건 우선 | 1×L4 24GB | H&M은64GB 이상 권장 | 100GB | $0.59/시간 |

GPU가 같아도 CPU·시스템 RAM 구성이 다릅니다. 대여 화면에서 함께 확인하세요. H&M의 L4 메모리 충족 여부와 전체 실행시간은 이 패키지로 아직 실측하지 않았습니다. 메모리 부족이면 batch/차원/정밀도를 자동 축소하지 않고 중단합니다.

A100은 기존 L4와 다른 하드웨어입니다. 코드·입력·epoch·학습률·배치·정밀도 설정은 유지하지만 CUDA 연산이 비트 단위로 같다고 보장하지 않습니다. GPU·패키지·CUDA·입력 해시를 결과에 기록합니다. Runpod 작업을 시작한 이후 GPU나 패키지를 바꾼 자동 재개는 차단합니다.

가격 확인: [Runpod 공식 요금](https://www.runpod.io/pricing), 2026-10-26 조회. L4 24시간=$14.16,72시간=$42.48; A100 24시간=$42.96,72시간=$128.88. 단순 대여시간 계산이며 실험 소요시간 예측이 아닙니다. 실제 재고·클라우드 종류·가격과 세금·저장요금은 결제 화면에서 확인하세요.

## 1. Pod 만들기

1. [Runpod 콘솔](https://console.runpod.io/)에 로그인하고 결제수단/잔액을 준비합니다.
2. Storage에서 **Network Volume**을 만듭니다. 사용할 GPU 재고가 있는 데이터센터를 먼저 고르세요. 첫 DH실행은50GB, H&M은100GB를 권장합니다.
3. Pods에서 해당 Network Volume을 연결한 GPU Pod를 만듭니다. **On-demand**, GPU수1, 공식 **Runpod PyTorch**의 JupyterLab 지원 템플릿을 고릅니다. Python3.10~3.12가 필요하며3.11/3.12를 권장합니다. 이 패키지는 별도 환경에 PyTorch2.8.0/CUDA12.8을 설치하므로 CUDA12.8을 지원하는 호스트 드라이버가 필요합니다.
4. Container Disk는50GB를 권장합니다. 입력·출력은 Network Volume의 `/workspace`에 저장합니다. 일반 Container Disk만 크게 만드는 것으로 영구 저장이 되지는 않습니다.
5. Deploy 이후 Connect → HTTP Services → **Jupyter Lab(8888)**을 엽니다.

Network Volume은 Pod와 분리되어 보존됩니다. 같은 데이터센터 등의 연결 제약과 저장 과금은 적용됩니다. [공식 저장공간 설명](https://docs.runpod.io/storage/network-volumes). 공식 PyTorch 템플릿과 JupyterLab 연결 절차는 [공식 접속 안내](https://docs.runpod.io/pods/connect-to-a-pod)를 참고하세요.

## 2. 준비 파일 올리기

전달된 `runpod_start.zip`을 JupyterLab의 `/workspace`에 업로드합니다. JupyterLab에서 Terminal을 열고 아래 한 줄을 실행합니다.

```bash
cd /workspace && python3 -m zipfile -e runpod_start.zip .
```

`/workspace/runpod_start/Runpod_시작.ipynb`를 엽니다. ZIP에는 실행도구, 두 노트북, 검증된 DH 입력·참조 ZIP, H&M 상품·참조 ZIP이 있습니다. 원본 데이터는 GitHub에 올리지 않았습니다. 입력들은 개인 작업용으로만 전달합니다.

**DH는 이 패키지만으로 입력이 준비됩니다.** H&M 거래 원본은 다음3번이 추가로 필요합니다.

## 3. H&M 실행할 때만 원본 옮기기

로컬에 남아 있던 `output/graphsage_school/hm/transactions_train.parquet`는 SHA256이 검증 원본과 달랐습니다. 과거 잘못 변환된 ZIP과 함께 사용하지 않습니다.

**가장 간단한 방법은 Drive에서 원본을 직접 다운로드하는 것입니다. 코랩이 필요하지 않습니다.**

1. 학교 Google Drive에서 `논문/data/raw/hm/transactions_train.parquet`를 다운로드합니다.
2. 파일 이름을 유지해 Runpod의 `/workspace/runpod_start`에 업로드합니다.
3. 시작 노트북2번 셀이 SHA256 검증 후 올바른 입력 위치에 복사합니다. 잘못 변환된 파일이면 학습 전에 차단합니다.

대안으로 동봉 `H&M_원본내보내기_Colab_CPU.ipynb`를 GPU 없는 CPU 런타임에서 실행하면 원본을 검증한 `hm_raw_verified.zip`을 받을 수 있습니다. 이 ZIP을 같은 Runpod 폴더에 업로드해도 됩니다. Drive 원본은 읽기만 합니다.

H&M 상품파일과 참조2개는 이미 `hm_support.zip`에 있습니다. Drive 인증·마운트를 Runpod 학습 과정에 사용하지 않습니다. 현재 패키지에는 올바른 H&M 거래 원본 바이너리가 없으므로, 위 이전 전에는 H&M 사전점검이 의도적으로 중단됩니다.

## 4. 노트북 위에서부터 실행

- **1번 설치:** 별도 Python환경과 세 원본 커밋을 준비합니다. 학습 없음. 재접속 때는 생략 가능합니다.
- **2번 선택:** 기본 `JOB='dh-ngcf'`, `SEEDS=[42]`. DH72개를 한 번에 시작하지 않고 먼저8개를 실행하는 설정입니다. 전체9시드 계획 자체를 축소한 것이 아닙니다. 다음에는 지정한 나머지 시드로 이어갑니다.
- H&M NGCF는 `JOB='hm-ngcf'`, H&M LightGCN은 `JOB='hm-lightgcn'`로 바꿉니다. 각각49,51로 시드가 고정됩니다.
- **3번 점검:** 원본 해시·원본 커밋·GPU 작동을 확인합니다. 전체 전처리는 학습 시작 시 기존 `prepare()`가 검증합니다.
- **4번 실행:** 독립 프로세스에서 선택한 작업을 시작합니다. 브라우저를 닫아도 Pod가 켜져 있으면 계속됩니다. 서버 중단·잔액 소진·장애까지 방지하는 기능은 아닙니다.
- **5번 로그:** 초기 입력검증과 첫 epoch 저장 성공을 확인합니다. 기존 코드의 첫3epoch 계측 로그로 비용을 판단합니다. test 결과에 맞춰 설정을 조정하지 않습니다.
- **6번 결과:** `/workspace/clv/exports`의 ZIP을 다운로드합니다. 모형 하나 완료할 때마다 갱신됩니다.

전체 전처리만 미리 확인하고 학습하지 않으려면 터미널에서 다음을 실행할 수 있습니다.

```bash
/workspace/clv/venv/bin/python /workspace/runpod_start/launch.py prepare --job dh-ngcf --seeds 42
```

## 5. 재개·저장·종료

- 모든 모형300epoch, 마지막1주 신규상품 test, k1, uniform 음성1개, 공동 optimizer 및 M2/M3/M4 개입 정의를 유지합니다. validation/holdout/중간test/조기종료를 추가하지 않았습니다.
- 매epoch 원래 저장함수를 호출하고, 교체 직전의 파일을 `.previous.pt`로 보존합니다. 최신·직전 모델/optimizer/RNG를 유지합니다. H&M NGCF의 완료 가중치 압축 보관 방식도 원본과 같습니다.
- 일반 파일시스템에 저장하므로 Drive 교체 사본의 휴지통 누적이나 Drive API 인증·할당량 관리가 필요하지 않습니다. 디스크 여유는 별도로 검사합니다.
- 동일 Network Volume의 같은 작업을 동시에 시작하면 잠금으로 차단합니다. **다른 볼륨·Colab의 실행은 자동 탐지하지 못합니다.** 같은모형·같은시드를 중복 배정하지 마세요.
- 같은 패키지·원본·Network Volume·GPU·패키지 버전에서4번 셀을 다시 실행하면 완료모형은 건너뛰고 학습중모형은 마지막 저장epoch 다음부터 이어갑니다.
- 과거 Colab 결과/체크포인트는 경로·실행 신원이 달라 여기로 단순 복사하지 않습니다. 신규 작업용이며 기존 체크포인트 이관은 별도 검증이 필요합니다.
- 최종평가 시작기록이 있는데 결과JSON이 없으면 재평가하지 않고 중단합니다. 이 경우 결과 복구를 확인해야 하며 평가기록/출력폴더를 지우고 재실행하지 마세요.
- ZIP에는 보고서·완료결과·이력·진단·환경정보가 포함되고 재개용 대형checkpoint는 제외됩니다. `/workspace/clv/results` 전체는 Network Volume에 남습니다. Mac 자동 백업은 설치하지 않았습니다.
- DH8×10시드 모두 모이기 전 ZIP은 `_partial`이고 불균형한 최종평균을 만들지 않습니다. SD는 신뢰구간·유의성의 대체가 아닙니다.
- **학습 완료 후에도 Pod 과금은 계속됩니다.** 결과 다운로드와 영구볼륨 저장을 확인한 뒤 콘솔에서 Stop 하세요. Network Volume을 연결한 Pod의 Stop 지원/표시는 구성에 따라 확인하고, Terminate가 필요하면 `/workspace`가 Network Volume인지 먼저 확인하세요. Network Volume 자체는 지우지 마세요. 잔액을 유지해야 저장도 지속됩니다.

다른 작업을 병렬로 돌릴 때는 처음부터 서로 다른 Pod에 서로 다른 작업을 배정합니다(예: DH NGCF / H&M NGCF). 한 Pod에 GPU2개를 선택해도 이 코드는 자동으로2개를 쓰지 않습니다. DH시드를 여러볼륨에 나눈 결과의 최종 취합은 별도 원본 검증 후 수행하며, 이번 패키지가 다른실행의 신원을 임의 변경해 합치지는 않습니다.

## 검증 범위

표준라이브러리 검사11개(입력/덮어쓰기/ZIP경로/시드/동시실행/저장공간/환경/평가재시도 차단/저장실패 보존/업로드원본 검증/GPU부재 차단)가 통과했습니다. 고정커밋3개에서 실제모듈·설정·참조를 불러오고 작은 CPU모형으로 원래저장과 래퍼저장의 모델·Adam·RNG 상태 일치, 직전epoch 보존, 최신epoch 복원을 확인했습니다. 두 노트북 코드셀과 설치스크립트 문법도 확인했습니다. 이 패키지 작성만으로 Runpod에서 전체데이터 학습·GPU최대메모리·속도·실제중단재개를 검증했다고 주장하지 않습니다. 서버 대여/결제/원격학습은 수행하지 않았습니다.

재검증(입력 패키지와 고정소스가 준비된 CPU 환경):

```bash
python -m unittest discover -s . -p 'test_*.py' -v
python verify_pinned_sources.py dh-ngcf --root /workspace/clv
python verify_pinned_sources.py hm-ngcf --root /workspace/clv
python verify_pinned_sources.py hm-lightgcn --root /workspace/clv
```

설치 버전 근거: [PyTorch 공식 이전버전 설치표](https://pytorch.org/get-started/previous-versions/)의2.8.0 CUDA12.8 wheel을 사용합니다. Colab의 과거 패키지 전체와 동일한 환경이라고 가정하지 않습니다.

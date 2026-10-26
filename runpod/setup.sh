#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Python 3.10~3.12 템플릿을 선택하세요."'
test -d /workspace || { echo '/workspace Network Volume을 먼저 연결하세요.'; exit 1; }
mkdir -p /workspace/clv
python3 -m venv /workspace/clv/venv
/workspace/clv/venv/bin/python -m pip install 'pip==25.2'
/workspace/clv/venv/bin/python -m pip install 'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
/workspace/clv/venv/bin/python -m pip install -r requirements.txt
/workspace/clv/venv/bin/python launch.py setup "$@"
echo '설치 완료. 학습은 시작하지 않았습니다. Runpod_시작.ipynb를 여세요.'

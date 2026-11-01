"""Generate the two fixed school-account experiment notebooks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parent
HELPER = 'clv_hm_graphsage_school_single.py'
helper_sha = hashlib.sha256((ROOT / HELPER).read_bytes()).hexdigest()
template = json.loads((ROOT / 'clv_hm_graphsage_core4_seed50_personal_colab.ipynb').read_text())
prepare = ''.join(template['cells'][3]['source']).split("missing=[p for p in refs if not p.exists()]")[0]
prepare = prepare.replace('개인 Drive', '학교 Drive')
prepare += '''missing=[str(p) for p in refs if not p.exists()]
assert not missing, f'학교 Drive 참조 결과가 없습니다: {missing}. 파일 선택 위젯 대신 해당 경로에 기존 ZIP을 복사하세요.'
cfg=screen.configure(out_dir=str(OUT),m2_reference=str(refs[0]),m4_reference=str(refs[1]))
assert cfg.seeds==(50,) and cfg.epochs==300
print('입력 준비 시작: 대용량 H&M 그래프 구성에는 시간이 필요합니다.',flush=True)
if 'prepared' not in globals():
    prepared=screen.prepare(cfg)
else:
    assert prepared['cfg']==cfg, '다른 실행의 입력 재사용 금지'
checkpoint_folder=single.install(screen,drive_api,prepared,LABEL)
assert set(prepared['data']['splits'])=={'test'}
assert prepared['data'].get('loss_w') is None
print('사전점검 완료:', LABEL, '/ seed50 / 300epoch / 이 모델 하나만 실행',flush=True)
print('결과:',prepared['run_dir'])
print('재개 파일:', 'https://drive.google.com/drive/folders/'+checkpoint_folder)
'''

for label in ('M3', 'M4'):
    setup = f'''from google.colab import drive, auth
from pathlib import Path
import subprocess, sys, hashlib, json, shutil
import torch
LABEL={label!r}
SOURCE_COMMIT='17aff7a9b3a6b12448fa4fbca5fedd4e6977b30b'
assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name(0), '기존 조건과 같은 L4 GPU를 선택하세요'
drive.mount('/content/drive')
auth.authenticate_user()
import google.auth
from googleapiclient.discovery import build
credentials,_=google.auth.default()
drive_api=build('drive','v3',credentials=credentials,cache_discovery=False)
info=drive_api.about().get(fields='user(emailAddress),storageQuota').execute()
assert info['user']['emailAddress']=='jungunb@u.sogang.ac.kr', '학교 계정으로 인증해야 합니다'
quota=info['storageQuota']
assert 'usage' in quota, '사용량 확인 실패'
available=min(10_000_000_000,int(quota.get('limit',10_000_000_000)))-int(quota['usage'])
print('학교 Drive 여유 GB:', available/1e9,flush=True)
# Two arms retain two full checkpoints each; a third upload can overlap.
# Conservatively reserve both jobs' worst-case peak before either starts.
REPO=Path('/content/graphsage_school_'+LABEL.lower()+'_s50')
HELPER_PATH=Path('/content/{HELPER}')
if not REPO.exists():
    subprocess.run(['git','clone','--branch','feat/m2-joint-nv-lightgcn','https://github.com/jung-un/clv-m2-lightgcn-runner.git',str(REPO)],check=True)
    helper=(REPO/{HELPER!r}).read_bytes()
    assert hashlib.sha256(helper).hexdigest()=={helper_sha!r}, '실행 도우미 버전 불일치'
    HELPER_PATH.write_bytes(helper)
    subprocess.run(['git','-C',str(REPO),'checkout',SOURCE_COMMIT],check=True)
assert subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()==SOURCE_COMMIT
assert hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest()=={helper_sha!r}
sys.path.insert(0,str(REPO)); sys.path.insert(0,'/content')
import clv_hm_graphsage_core4_seed50 as screen
import clv_hm_graphsage_school_single as single
assert Path(screen.__file__).resolve().parent==REPO.resolve()
DATA=Path('/content/drive/MyDrive/논문/data')
single.capacity_check(screen,drive_api,DATA)
OUT=DATA/('results_v3_hm_graphsage_seed50_school_'+LABEL.lower()+'_fresh_v1')
OUT.mkdir(parents=True,exist_ok=True)
print('학교 전용',LABEL,'seed50 — 기존 개인 M3 checkpoint를 가져오지 않습니다.',flush=True)
'''
    train = '''assert cfg.seeds==(50,) and LABEL in ('M3','M4')
print('학습 대상:',LABEL,'seed50 하나. 새 실행은 1epoch부터, 이 폴더에서 중단된 실행만 이어갑니다.',flush=True)
try:
    result=screen.run_arm(prepared,LABEL)
finally:
    output,status=single.export(screen,prepared,LABEL,
        '/content/hm_graphsage_'+LABEL.lower()+'_seed50_school_results.zip')
    backup=prepared['run_dir']/output.name
    shutil.copyfile(output,backup)
    assert sha256(output)==sha256(backup)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    print('Drive 결과 ZIP:',backup,flush=True)
'''
    cells=[]
    def cell(kind, text):
        d={'cell_type':kind,'metadata':{},'source':text.splitlines(keepends=True)}
        if kind=='code': d.update(execution_count=None,outputs=[])
        cells.append(d)
    cell('markdown',f'# 학교 전용 · H&M GraphSAGE seed50 {label} 단독\n\n'
        '기존 seed50 설계·L4·배치131072·300epoch 유지. 첫 실행은 처음부터 학습합니다. '
        '다른 모델·시드와 기존 개인 checkpoint는 학습/복구하지 않습니다.\n\n'
        '학교 계정 jungunb@u.sogang.ac.kr 사용. M3/M4를 별도 GPU에서 병렬 실행합니다. '
        '기존 개인 M3/M4 실행은 중지되어 있어야 합니다.\n\n'
        '매 epoch 전체 optimizer·난수 상태를 Drive에 검증 저장하며 최신+직전만 유지합니다. '
        '정리는 이 새 실행의 더 오래된 epoch 사본에만 한정합니다. 과거 실험을 자동 삭제하지 않습니다. '
        '두 동시 업로드까지 고려해 시작 전 8.2GB 여유를 확인합니다.\n\n'
        '마지막 주 test만 고정300epoch에서 평가. validation·중간 test·튜닝 없음. '
        'M1·M2 완료 결과는 그대로 보존하며 이 노트북에서 다시 평가하지 않습니다.')
    cell('markdown','## 1. 학교 계정·GPU·용량 확인 및 고정 코드 준비')
    cell('code',setup)
    cell('markdown','## 2. 입력을 로컬에 한 번 복사하고 기존 입력·분할 검증')
    cell('code',prepare)
    cell('markdown',f'## 3. {label}만 학습 → 300epoch 평가 → 결과 ZIP 저장')
    cell('code',train)
    cell('markdown','## 4. 결과 다운로드 — Drive에도 저장되어 있습니다')
    cell('code',"from google.colab import files\nassert status['complete'], '미완료: partial ZIP을 완료 결과로 취급하지 마세요'\nfiles.download(str(output))\n")
    notebook={'cells':cells,'metadata':{'accelerator':'GPU','colab':{'gpuType':'L4'},
        'kernelspec':{'name':'python3','display_name':'Python 3'}},'nbformat':4,'nbformat_minor':0}
    name=f'clv_hm_graphsage_{label.lower()}_seed50_school_colab.ipynb'
    (ROOT/name).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n')
    for c in cells:
        if c['cell_type']=='code': compile(''.join(c['source']),name,'exec')
    print(name)

"""Generate three disjoint personal-Colab notebooks from the verified input scaffold."""
import json
from pathlib import Path
import sys

PARTS = {'a': ('M1','M2','M3'), 'b': ('M4','M5_A','M5_B'), 'c': ('M5_C','M5_D')}

def build(revision):
    import subprocess
    revision = subprocess.check_output(['git','rev-parse',revision],text=True).strip()
    original = json.loads(Path('clv_hm_graphsage_core4_seed50_personal_colab.ipynb').read_text())
    for part, labels in PARTS.items():
        n = json.loads(json.dumps(original))
        for c in n['cells']:
            s = ''.join(c['source']).replace('seed50','seed51').replace('s50','s51')
            s = s.replace('clv_hm_graphsage_core4_seed50', 'clv_hm_gcn_final8_seed51')
            s = s.replace('clv_hm_graphsage_core4_seed51', 'clv_hm_gcn_final8_seed51')
            s = s.replace('clv_personal_graphsage_checkpoint','clv_personal_gcn51_checkpoint')
            s = s.replace('hm_graphsage_core4_seed51',f'hm_gcn_seed51_part_{part}')
            s = s.replace('graphsage-core4-s51',f'gcn-part-{part}-s51')
            s = s.replace('hm_graphsage_seed51_inputs','hm_gcn_seed51_inputs')
            s = s.replace('| seed51 M1~M4 | 출력:', '| GCN seed51 | 출력:')
            s = s.replace('17aff7a9b3a6b12448fa4fbca5fedd4e6977b30b',revision)
            s = s.replace("assert 'L4' in torch.cuda.get_device_name(0), '기존 실험과 동일한 L4로 실행하세요.'", "# GPU hardware is recorded; mathematical settings are fixed.\ntorch.backends.cuda.matmul.allow_tf32 = False\ntorch.backends.cudnn.allow_tf32 = False\ntorch.set_float32_matmul_precision('highest')")
            s = s.replace("cfg.seeds==(50,)","cfg.seeds==(51,)")
            s = s.replace("('graphsage_binary_adj','graphsage_m3_adj','references','config_hash','run_dir')", "('gcn_binary_adj','gcn_m3_adj','references','config_hash','run_dir')")
            c['source'] = s.splitlines(keepends=True)
            if c['cell_type']=='code': c.update(outputs=[],execution_count=None)
        title = f'''# H&M GCN seed51 · 개인 GPU {part.upper()} · {' / '.join(labels)}
개인 계정 azsabae@gmail.com에서 별도 GPU 런타임으로 실행합니다. A/B/C는 서로 다른 모델만 실행하며, 동일 파트를 중복 실행하지 않습니다.

고정300epoch·마지막1주 test·신규상품 추천·전체 카탈로그·uniform 음성1개·하나의 optimizer를 유지합니다. M1~M4 사전학습 없이 각 결합모형도 독립 초기화합니다.
GCN 2층/ID64/batch131072/Adam lr0.0005, 원래 정규화·손실·M2~M4 설계는 그대로입니다. 검증된 COO→CSR 저장형식과 고정 입력 캐시만 사용하며 학습 표현은 매 배치 재계산합니다.

매 epoch 전체 모델·optimizer·난수를 개인 Drive에 업로드·해시 검증합니다. 이 파트 전용 폴더에서만 최신+직전 2개를 남깁니다. 완료 결과는 재사용하고 중단 모델은 마지막 저장 다음 epoch로 재개합니다. 결과와 원본 입력은 삭제하지 않습니다.
3개 런타임 합계 체크포인트 약18GB와 업로드 여유가 필요합니다. 개인 Drive 여유를 먼저 확인하세요.
각 파트 완료는 전체8모형 완료가 아닙니다. 3개 ZIP을 합쳐야 seed51 전체 결과가 됩니다. 1시드만으로 유의성을 주장하지 않습니다.
'''
        n['cells'][0]['source']=title.splitlines(keepends=True)
        n['cells'][1]['source'].append(f"\nPART = {part!r}\nLABELS = {labels!r}\nprint('이번 GPU 담당:', LABELS)\n")
        n['cells'][3]['source']=''.join(n['cells'][3]['source']).replace("print('사전점검 완료: M1~M4 / seed51 / 300epoch')", "print('사전점검 완료:', LABELS, '/ seed51 / 300epoch')").splitlines(keepends=True)
        n['cells'][4]['source']=[f"## 2. 담당 모델만 순차 학습: {', '.join(labels)}\n매 epoch 원격 저장 완료 후 다음 epoch로 갑니다. 다른 파트 모델은 실행하지 않습니다.\n"]
        n['cells'][5]['source']=f'''assert getattr(screen.SafeProgressStore,'_personal_storage_installed',False)
assert cfg.seeds == (51,) and LABELS == {labels!r}
try:
    for label in LABELS:
        screen.run_arm(prepared, label)
        screen.report(prepared)
        print('담당 모델 완료:', label, flush=True)
    assert all(screen.completed(prepared, label) is not None for label in LABELS)
    print('PART COMPLETE:', PART, LABELS)
finally:
    output=screen.export_zip(prepared,'/content/hm_gcn_seed51_part_{part}_results.zip')
    backup=prepared['run_dir']/Path(output).name
    shutil.copyfile(output,backup)
    assert sha256(output)==sha256(backup)
    print('현재까지 결과 ZIP:', backup)
'''.splitlines(keepends=True)
        n['cells'][7]['source']=["from IPython.display import display\n", "result=screen.report(prepared)\n", "display(result['absolute'])\n", "print('이 GPU 담당:', LABELS, '| 전체8개는 A/B/C 결과를 합쳐 확인합니다.')\n"]
        n['cells'][6]['source']=["## 3. 담당 모델 저장 결과 — 추가 학습·평가 없음\n"]
        for c in n['cells']:
            if c['cell_type']=='code': compile(''.join(c['source']), '<cell>', 'exec')
        n.setdefault('metadata',{})['accelerator']='GPU'
        Path(f'clv_hm_gcn_seed51_part_{part}_personal_colab.ipynb').write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n')

if __name__=='__main__': build(sys.argv[1])

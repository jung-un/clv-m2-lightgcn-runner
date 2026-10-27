"""H&M GAT seed49 M5_A~D only, isolated from the running M1~M4 Pod."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import launch as platform
import hm_ngcf_m5 as storage

SHARED = Path('/workspace/clv')
ROOT = Path('/workspace/clv_hm_gat_m5_seed49')
PIN = '289e6df371d545687ea72cb3937fe744df3f009a'
MODELS = ('M5_A', 'M5_B', 'M5_C', 'M5_D')


def baseline(shared=SHARED):
    output = shared / 'results/hm-gat'
    records = sorted(output.glob('runpod_invocation_*.json'))
    if not records:
        raise RuntimeError('GAT M1~M4 배정 기록 없음')
    assignment = json.loads(records[-1].read_text())
    if assignment.get('seeds') != [49] or assignment.get('models') != ['M1','M2','M3','M4']:
        raise RuntimeError('기존 GAT 배정이 M1~M4 전용이 아님; 중복 방지 중단')
    for p in output.rglob('*'):
        if re.search(r'(?:gat_)?m5_[abcd](?:_|\.)', p.name.lower()):
            raise RuntimeError(f'기존 GAT 출력에 M5 진행/완료/평가 기록 존재: {p}')
    protocols = [json.loads(p.read_text()) for p in output.glob('*/protocol.json')]
    protocols = [p for p in protocols if p.get('source_revision') == PIN]
    if not protocols:
        raise RuntimeError('기존 GAT와 같은 고정 source protocol 없음')
    for p in protocols:
        storage.scientific_identity(p)  # Validate every required field before preparation.
    if not (shared / 'sources' / PIN).is_dir():
        raise RuntimeError('기존 고정 GAT 소스 없음')
    return protocols


def link_inputs(root, shared):
    if root.resolve() == shared.resolve() or shared.resolve() in root.resolve().parents:
        raise RuntimeError('GAT M5 출력 root를 기존 출력과 분리하세요')
    root.mkdir(parents=True, exist_ok=True)
    for name in ('data', 'sources'):
        target, source = root / name, shared / name
        if not source.is_dir():
            raise RuntimeError(f'기존 입력/소스 없음: {source}')
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise RuntimeError('다른 입력 링크를 덮어쓰지 않습니다')
        else:
            target.symlink_to(source, target_is_directory=True)


def check_environment(current, shared):
    old = json.loads((shared / 'results/hm-gat/runpod_environment.json').read_text())
    keys = ('source_commit','input_sha256','python','packages','cuda','gpu',
            'capability','tf32_matmul','matmul_precision')
    if any(old[k] != current[k] for k in keys):
        raise RuntimeError('기존 GAT와 GPU/패키지/정밀도 환경 불일치')


def execute(mode, root=ROOT, shared=SHARED):
    if platform.JOBS['hm-gat']['pin'] != PIN:
        raise RuntimeError('동봉 실행기의 GAT 고정 커밋 불일치')
    with platform.exclusive(shared / 'locks/hm-gat-seed49-m5-partition.lock'):
        baselines = baseline(shared)
        link_inputs(root, shared)
        import torch
        inputs = platform.verify_files('hm-gat', root)
        platform.cuda_check(torch)
        screen, cfg = platform.load_runner('hm-gat', root)
        output = root / 'results/hm-gat'
        platform.guard_space(output)
        env = platform.environment(torch, 'hm-gat', inputs)
        check_environment(env, shared)
        env.update(partition='seed49-M5_A-to-D', partition_launcher_sha256=platform.sha(__file__),
                   storage_helper_sha256=platform.sha(storage.__file__),
                   checkpoint_policy='atomic-each-epoch; hardlink-previous-or-verified-copy')
        if mode == 'check':
            print(json.dumps(env, ensure_ascii=False, indent=2))
            print('GAT M5 사전검사 통과; 전체 전처리/학습/평가 미실행', flush=True)
            return
        platform.bind_environment(output / 'runpod_environment.json', env)
        storage.install_storage(screen, output)
        prepared = screen.prepare(cfg)
        storage.check_science(prepared['protocol'], baselines)
        if (set(prepared['data']['splits']) != {'test'}
                or prepared['data'].get('loss_w') is not None
                or prepared['data']['data_stats']['min_item_inter'] != 1):
            raise RuntimeError('고정 test/k1/기본BPR 조건 불일치')
        if mode == 'report':
            platform.export(screen, prepared, 'hm-gat', root)
            return
        platform.atomic_json(output / 'runpod_m5_partition.json', {
            'seeds': [49], 'models': list(MODELS), 'source_commit': PIN,
            'original_output_read_only': str(shared / 'results/hm-gat'),
            'started_utc': datetime.now(timezone.utc).isoformat()})
        for label in MODELS:
            baseline(shared)
            print(f'[실행] hm-gat seed49 {label}', flush=True)
            with platform.evaluation_once(screen, prepared, label, 49):
                screen.run_arm(prepared, label)
            platform.export(screen, prepared, 'hm-gat', root)
        print('GAT M5_A~D 완료. Pod 요금은 별도 종료 전 계속 발생합니다.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('check','run','report'))
    p.add_argument('--confirmed-unstarted', action='store_true')
    args = p.parse_args()
    if args.mode == 'run' and not args.confirmed_unstarted:
        p.error('다른 서버의 동일 GAT M5 seed49 미실행 확인 필요')
    execute(args.mode)


if __name__ == '__main__':
    main()

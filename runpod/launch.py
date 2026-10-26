"""Run unchanged, commit-pinned research runners on a Runpod volume.

This file lives OUTSIDE their Git checkouts: source identity stays clean.
No Colab/Drive credentials, model edits, precision changes, or auto training.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path('/workspace/clv')
REMOTE = 'https://github.com/jung-un/clv-m2-lightgcn-runner.git'
ORDER = ('M1', 'M2', 'M3', 'M4', 'M5_A', 'M5_B', 'M5_C', 'M5_D')
NEW_SEEDS = (42, 43, 44, 45, 46, 47, 48, 50, 51)
JOBS = {
    'dh-gat': dict(module='clv_dh_gat_remaining9', dataset='dunnhumby',
                   pin='5a475a2ed25ef0b40822de11c537fa7fc76e475a', seed=None),
    'dh-ngcf': dict(module='clv_dh_ngcf_remaining9', dataset='dunnhumby',
                    pin='63146eff4ed60bfb0cc430b77d58fdc40e4db7c5', seed=None),
    'hm-ngcf': dict(module='clv_hm_ngcf_final8_seed49', dataset='hm',
                    pin='db39115fadad5aedb020eafe4a492a2862a876db', seed=49),
    'hm-lightgcn': dict(module='clv_hm_final8_seed51', dataset='hm',
                        pin='8ca3baa90be1e0ac02554e1c6fe4c1c6b8deb229', seed=51),
}
DH_FILES = {
    'raw/dunnhumby/transaction_data.csv': '3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476',
    'raw/dunnhumby/product.csv': '7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949',
    'references/clv_m2_adaptive_dunnhumby_final10_results.zip': '726fb01dedaa0a3a4bacb39bd7d432dcaca904573e5d460db534d20738f74151',
    'references/clv_m4_user_centered_final10_results.zip': 'a6bf92f64bf425248758a3952054bf3f6c9c27c10a4711758d3a08fa1cfcb41f',
    'references/dh_ngcf_final8_seed49_results.zip': '6a04f4f0fcdac6194910092700e3305d88d14c0bc54c8b37e60273881fd0444e',
}
DH_GAT_FILES = {k: v for k, v in DH_FILES.items() if 'dh_ngcf_' not in k}
DH_GAT_FILES['references/dh_gat_final8_seed49_results.zip'] = '303439619e1bde2cc83ce46bc5ac3432355e92e0c836a0457562ff4c4783ae1a'

HM_FILES = {
    'raw/hm/transactions_train.parquet': '0d90884fc1bc400161e6aeba238cba94183998bedfeb4ebbf4962e17d0c8489b',
    'raw/hm/articles.csv': '25f21882053517f7684ea43f524225a9481a7bb9aab9cf0abfc7fe8c4576a72b',
    'references/clv_m2_adaptive_hm_seed49_lastweek_results.zip': '6d5159a888319b8a443a6907922a6e36fe9954c6b717dc2b1a6ff547c1d52bed',
    'references/clv_hm_m4c_lastweek_seed49_results.zip': '11fbfbddd21f966357cc872d43891bd078741b83037e88ee39a5558498ce40fe',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def unpack(bundle, target):
    """Extract only a manifest's verified files; never overwrite different data."""
    target = Path(target).resolve()
    with zipfile.ZipFile(bundle) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError('ZIP에 중복 경로가 있습니다.')
        manifest = json.loads(z.read('manifest.json'))
        if not isinstance(manifest, dict) or not manifest:
            raise RuntimeError('입력 manifest 형식 오류')
        for name, expected in manifest.items():
            p = (target / name).resolve()
            if target not in p.parents or not isinstance(expected, str) or len(expected) != 64:
                raise RuntimeError('잘못된 입력 경로/해시')
            if p.exists():
                if sha(p) != expected:
                    raise RuntimeError(f'기존 입력이 다릅니다. 보존하고 중단: {p}')
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + '.partial')
            with z.open(name) as source, tmp.open('wb') as dest:
                shutil.copyfileobj(source, dest, 8 * 1024**2)
            if sha(tmp) != expected:
                raise RuntimeError(f'입력 복사 해시 불일치: {name}')
            tmp.replace(p)
    return manifest


def verify_files(job, root=ROOT):
    expected = DH_GAT_FILES if job == 'dh-gat' else (DH_FILES if job == 'dh-ngcf' else HM_FILES)
    missing, actual = [], {}
    for name, digest in expected.items():
        p = root / 'data' / name
        if not p.is_file():
            missing.append(name)
            continue
        actual[name] = sha(p)
        if digest and actual[name] != digest:
            raise RuntimeError(f'검증된 원본과 다른 입력: {p}. 재변환하지 마세요.')
    if missing:
        raise RuntimeError('필요한 입력 없음:\n' + '\n'.join(missing)
                           + '\nH&M 거래 원본은 동봉한 Colab CPU 내보내기로 옮기세요.')
    return actual


def install_hm_raw(source, root=ROOT):
    name = 'raw/hm/transactions_train.parquet'
    expected = HM_FILES[name]
    if sha(source) != expected:
        raise RuntimeError('검증된 H&M 거래 원본이 아닙니다. 재변환 파일은 사용하지 않습니다.')
    target = root/'data'/name
    if target.exists():
        if sha(target) != expected:
            raise RuntimeError('기존 H&M 입력이 다릅니다. 덮어쓰지 않습니다.')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.partial')
    shutil.copyfile(source, tmp)
    if sha(tmp) != expected:
        raise RuntimeError('H&M 원본 복사 검증 실패')
    tmp.replace(target)


def source_checkout(job, root=ROOT):
    pin = JOBS[job]['pin']
    repo = root / 'sources' / pin
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', REMOTE, str(repo)], check=True)
        subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', pin], check=True)
    actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip()
    if actual != pin or dirty:
        raise RuntimeError(f'고정 코드가 변경되었습니다. 덮어쓰지 않습니다: {repo}')
    return repo


@contextmanager
def exclusive(path):
    """POSIX advisory lock released by OS on process exit; lock file is retained."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('같은 볼륨에서 이 작업이 이미 실행 중입니다.') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def selected_seeds(job, seeds):
    if job not in ('dh-ngcf', 'dh-gat'):
        fixed = JOBS[job]['seed']
        if seeds and tuple(seeds) != (fixed,):
            raise ValueError(f'{job}은 seed{fixed}만 허용합니다.')
        return (fixed,)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(NEW_SEEDS):
        raise ValueError('Dunnhumby 시드를 명시하세요: 42~48,50,51. seed49는 재사용만 합니다.')
    return tuple(sorted(seeds))


def guard_space(root, required=0, reserve=10 * 1024**3):
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < required + reserve:
        raise RuntimeError('저장 볼륨 여유 부족. 기존 결과는 보존하고 중단합니다.')
    if shutil.disk_usage(tempfile.gettempdir()).free < required + 2 * 1024**3:
        raise RuntimeError('임시 디스크 여유 부족. Container Disk를 늘리세요.')


def install_storage(screen, job, output):
    """Keep latest + previous checkpoint without changing serialized state or RNG."""
    owner = screen.original if job in ('dh-ngcf', 'dh-gat') else screen
    key = 'ProgressStore' if job == 'hm-lightgcn' else 'SafeProgressStore'
    old = getattr(owner, key)

    class VolumeProgressStore(old):
        def save_epoch(self, model, optimizer, rng, **state):
            estimate = sum(p.numel() * p.element_size() for p in model.parameters()) * 3
            previous = self.latest_checkpoint.with_suffix('.previous.pt')
            prior_bytes = self.latest_checkpoint.stat().st_size if self.latest_checkpoint.exists() else 0
            guard_space(output, estimate + prior_bytes + 512 * 1024**2)
            if self.latest_checkpoint.exists():
                temp = previous.with_suffix(previous.suffix + '.tmp')
                # Hash the source while copying: avoid a second full network read.
                # Still reread and verify the destination before publishing it.
                digest = hashlib.sha256()
                with self.latest_checkpoint.open('rb') as src, temp.open('wb') as dst:
                    for block in iter(lambda: src.read(8 * 1024**2), b''):
                        digest.update(block)
                        dst.write(block)
                    dst.flush()
                    os.fsync(dst.fileno())
                if sha(temp) != digest.hexdigest():
                    raise RuntimeError('직전 checkpoint 복사 검증 실패')
                temp.replace(previous)
            return super().save_epoch(model, optimizer, rng, **state)

    setattr(owner, key, VolumeProgressStore)
    if hasattr(screen, 'STORAGE_GUARD'):
        screen.STORAGE_GUARD = lambda n: guard_space(output, n)


def bind_environment(path, environment):
    if path.exists():
        if json.loads(path.read_text()) != environment:
            raise RuntimeError('저장된 실행과 GPU/패키지/실행도구가 다릅니다. 자동 재개하지 않습니다.')
    else:
        atomic_json(path, environment)


def cuda_check(torch):
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('단일 NVIDIA GPU 1개가 보이는 Pod를 사용하세요.')
    # Only a tiny arithmetic check: no research data, model, test, or RNG draws.
    index = torch.tensor([[0, 1], [1, 0]], device='cuda')
    sparse = torch.sparse_coo_tensor(index, torch.ones(2, device='cuda'), (2, 2)).coalesce()
    dense = torch.ones((2, 2), device='cuda', requires_grad=True)
    torch.sparse.mm(sparse, dense).sum().backward()
    if not torch.equal(dense.grad, torch.ones_like(dense)):
        raise RuntimeError('CUDA 희소행렬 역전파 점검 실패')
    torch.cuda.empty_cache()


def environment(torch, job, inputs):
    return dict(job=job, source_commit=JOBS[job]['pin'], input_sha256=inputs,
                launcher_sha256=sha(__file__), python=sys.version.split()[0],
                packages={n: importlib.metadata.version(n) for n in
                          ('torch', 'numpy', 'pandas', 'scipy', 'pyarrow')},
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                matmul_precision=torch.get_float32_matmul_precision())


def load_runner(job, root=ROOT):
    repo = source_checkout(job, root)
    sys.path.insert(0, str(repo))
    screen = importlib.import_module(JOBS[job]['module'])
    if Path(screen.__file__).resolve().parent != repo.resolve():
        raise RuntimeError('다른 버전 모듈이 로드되었습니다. 새 프로세스로 실행하세요.')
    data = root / 'data'
    dataset = JOBS[job]['dataset']
    tx, meta = (('transaction_data.csv', 'product.csv') if dataset == 'dunnhumby'
                else ('transactions_train.parquet', 'articles.csv'))
    screen.base.v3.SCHEMA[dataset].update(tx_path=str(data/'raw'/dataset/tx),
                                        item_meta_path=str(data/'raw'/dataset/meta))
    if job in ('dh-ngcf', 'dh-gat'):
        cfg = screen.configure(out_dir=str(root/'results'/job),
            m2_reference=str(data/'references/clv_m2_adaptive_dunnhumby_final10_results.zip'),
            m4_reference=str(data/'references/clv_m4_user_centered_final10_results.zip'),
            seed49_reference=str(data/('references/dh_gat_final8_seed49_results.zip' if job == 'dh-gat' else 'references/dh_ngcf_final8_seed49_results.zip')))
        previous = screen.load_seed49(cfg.seed49_reference)
        screen.check_source_compatibility(previous)
    else:
        cfg = screen.configure(out_dir=str(root/'results'/job),
            m2_reference=str(data/'references/clv_m2_adaptive_hm_seed49_lastweek_results.zip'),
            m4_reference=str(data/'references/clv_hm_m4c_lastweek_seed49_results.zip'))
    screen.validate_config(cfg)
    if cfg.epochs != 300 or tuple(screen.ORDER) != ORDER:
        raise RuntimeError('고정300epoch/8모형 실행조건 불일치')
    if screen.base.fixed.moe.source_revision() != JOBS[job]['pin']:
        raise RuntimeError('학습 코드 신원이 고정 커밋과 다릅니다.')
    return screen, cfg


@contextmanager
def evaluation_once(screen, prepared, label, seed):
    """Fail closed after an interrupted final evaluation; never silently retest."""
    marker = prepared['run_dir'] / 'evaluation_started' / f'{label}_s{seed}.json'
    old = screen.base.fixed._evaluate

    def evaluate(*args, **kwargs):
        marker.parent.mkdir(parents=True, exist_ok=True)
        try:
            with marker.open('x') as f:
                json.dump({'model': label, 'seed': seed, 'epoch': 300,
                           'started_utc': datetime.now(timezone.utc).isoformat()}, f)
                f.flush()
                os.fsync(f.fileno())
        except FileExistsError as exc:
            raise RuntimeError('이 모형의 최종평가 시작 기록이 있습니다. 결과 복구 확인 전 재평가 금지.') from exc
        return old(*args, **kwargs)

    screen.base.fixed._evaluate = evaluate
    try:
        yield
    finally:
        screen.base.fixed._evaluate = old


def export(screen, prepared, job, root=ROOT):
    result = screen.report(prepared)
    path = root/'exports'/f'{job}_results.zip'
    path.parent.mkdir(parents=True, exist_ok=True)
    if job == 'hm-ngcf':
        output = screen.export_zip(prepared, str(path))
    else:
        output = screen.export_zip(prepared, str(path), allow_partial=not result['complete'])
    # Sidecar provenance travels with results; original scientific reports stay intact.
    with zipfile.ZipFile(output, 'a', zipfile.ZIP_DEFLATED) as z:
        for p in sorted((root/'results'/job).glob('runpod_*.json')):
            z.write(p, 'runpod/' + p.name)
        for p in sorted((prepared['run_dir']/'evaluation_started').glob('*.json')):
            z.write(p, 'runpod/evaluation_started/' + p.name)
    print('결과 ZIP:', output, flush=True)
    return result


def selected_models(job, models):
    if models is None:
        return ORDER
    chosen = tuple(models)
    if job != 'hm-ngcf' or not chosen or len(set(chosen)) != len(chosen):
        raise ValueError('모형 부분 실행은 H&M NGCF의 중복 없는 목록만 허용합니다.')
    if any(label not in ORDER for label in chosen):
        raise ValueError('알 수 없는 모형')
    return tuple(label for label in ORDER if label in chosen)


def execute(job, seeds, mode, root=ROOT, models=None):
    models = selected_models(job, models)
    import torch
    output = root/'results'/job
    with exclusive(root/'locks'/f'{job}.lock'):
        inputs = verify_files(job, root)
        cuda_check(torch)
        screen, cfg = load_runner(job, root)
        print('GPU:', torch.cuda.get_device_name(0), '| 작업:', job, '| 시드:', seeds, flush=True)
        print('고정300epoch / 신규상품 / 마지막1주 test / 8모형 / 설정변경 없음', flush=True)
        guard_space(output)
        current = environment(torch, job, inputs)
        if mode == 'check':
            print(json.dumps(current, ensure_ascii=False, indent=2))
            print('원본파일·고정코드·CUDA 점검 통과. 전체 전처리 및 GPU 학습은 아직 미실행.')
            return
        bind_environment(output/'runpod_environment.json', current)
        install_storage(screen, job, output)
        prepared = screen.prepare(cfg)
        if set(prepared['data']['splits']) != {'test'} or prepared['data'].get('loss_w') is not None:
            raise RuntimeError('test 단일 분할/기본 무가중 BPR 불변식 위반')
        if prepared['data']['data_stats']['min_item_inter'] != 1:
            raise RuntimeError('MIN_ITEM_INTER=1 불변식 위반')
        if mode == 'prepare':
            print('전체 입력·특징·분할 검증 완료. 학습·test평가는 시작하지 않았습니다.')
            return
        if mode == 'report':
            export(screen, prepared, job, root)
            return
        atomic_json(output/('runpod_invocation_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f') + '.json'), {
            'seeds': list(seeds), 'models': list(models), 'started_utc': datetime.now(timezone.utc).isoformat(),
            'note': 'execution subset only; original experiment panel and hyperparameters unchanged'})
        for seed in seeds:
            for label in models:
                print(f'[실행] {job} seed{seed} {label}', flush=True)
                with evaluation_once(screen, prepared, label, seed):
                    if job in ('dh-ngcf', 'dh-gat'):
                        screen.run_arm(prepared, label, seed)
                    else:
                        screen.run_arm(prepared, label)
                export(screen, prepared, job, root)
        print('요청한 작업 완료. Pod 요금은 계속 발생하므로 ZIP 다운로드 후 콘솔에서 Stop 하세요.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['setup', 'unpack', 'hm-raw', 'check', 'prepare', 'run', 'report'])
    parser.add_argument('--job', choices=JOBS)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--models', nargs='+', choices=ORDER, help='H&M NGCF 실행할 모형만 지정; 연구 설정은 유지')
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--file', type=Path)
    parser.add_argument('--confirmed-unstarted', action='store_true',
                        help='해당 모델·시드가 다른 서버에서 시작되지 않았거나 이 Runpod 출력에서만 재개함을 확인')
    args = parser.parse_args()
    if args.mode == 'setup':
        for job in ([args.job] if args.job else JOBS):
            source_checkout(job)
        return
    if args.mode == 'unpack':
        if not args.bundle:
            parser.error('--bundle 필요')
        unpack(args.bundle, ROOT/'data')
        print('입력 압축 해제 완료. 원본과의 일치는 check에서 다시 확인합니다.')
        return
    if args.mode == 'hm-raw':
        if not args.file:
            parser.error('--file 필요')
        install_hm_raw(args.file)
        print('H&M 원본 해시·복사 검증 완료.')
        return
    if not args.job:
        parser.error('--job 필요')
    seeds = selected_seeds(args.job, args.seeds)
    if args.mode == 'run' and not args.confirmed_unstarted:
        parser.error('중복 실험 방지: 다른 서버에 같은 작업이 없는지 확인하고 --confirmed-unstarted를 지정하세요.')
    execute(args.job, seeds, args.mode, models=args.models)


if __name__ == '__main__':
    main()

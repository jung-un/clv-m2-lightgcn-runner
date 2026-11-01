"""School seed50 single-arm dispatch; model/training math remains pinned upstream.

Only this new partition is resumable. Never imports old personal checkpoints.
Storage uses the existing verified immutable-upload/retain-two implementation.
"""
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ACCOUNT = 'jungunb@u.sogang.ac.kr'
RESERVE = 1_000_000_000


def validate(label):
    if label not in ('M3', 'M4'):
        raise ValueError('Only independent GraphSAGE seed50 M3 or M4 is allowed')


def capacity_check(screen, api, data):
    """Count existing checkpoints of ONLY the two new partitions towards budget."""
    import clv_ngcf_school_storage as io
    info = io.execute(api.about().get(fields='user(emailAddress),storageQuota'))
    if info['user']['emailAddress'] != ACCOUNT:
        raise RuntimeError('학교 계정 인증 필요')
    # Workspace about.usage can describe pooled organisation storage. Never
    # subtract that value from this user's 10GB allocation.
    free = io.check_space(api, required=0, initial=False)
    allocated = 0
    seen = set()
    for label in ('M3', 'M4'):
        root = Path(data) / ('results_v3_hm_graphsage_seed50_school_' + label.lower() + '_fresh_v1')
        for record in root.glob('*/personal_checkpoint_storage.json'):
            manifest = json.loads(record.read_text())
            if manifest.get('account') != ACCOUNT:
                raise RuntimeError('저장소 계정 불일치')
            parent = manifest['folder_id']
            for row in io.listing(api, "'" + parent + "' in parents and trashed=false"):
                props = row.get('appProperties', {})
                if (row.get('ownedByMe') and props.get('clv_seed') == '50'
                        and props.get('clv_model') == 'graphsage_' + label.lower() + '_lastweek'
                        and props.get('clv_split_run') == record.parent.name
                        and row['id'] not in seen):
                    seen.add(row['id'])
                    allocated += int(row['size'])
    # Existing files count toward the two-run peak, never toward the 1GB reserve.
    needed = RESERVE + max(0, 7_200_000_000 - allocated)
    print(f'학교 Drive 여유 {free/1e9:.2f}GB / 두 실행 잔여 저장 예산 {needed/1e9:.2f}GB', flush=True)
    if free < needed:
        raise RuntimeError('검증된 완료 백업 정리 후 다시 실행하세요. 입력·현재 checkpoint 삭제 금지.')
    return free


def install(screen, api, prepared, label):
    validate(label)
    spec = importlib.util.spec_from_file_location('_school_single_storage',
        Path(screen.__file__).with_name('clv_personal_graphsage_checkpoint.py'))
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    storage.ACCOUNT = ACCOUNT
    storage.MIN_FREE = RESERVE
    storage.START_FREE = RESERVE
    original_backend = storage.backend

    def backend():
        impl = original_backend()
        def check_space(api, required=0, initial=False):
            info = impl.execute(api.about().get(fields='user(emailAddress),storageQuota'))
            if info['user']['emailAddress'] != ACCOUNT:
                raise RuntimeError('학교 계정 jungunb@u.sogang.ac.kr으로 인증하세요')
            # Count owned files (including Trash/revisions) using quotaBytesUsed;
            # the shared-pool limit, when present, is checked separately.
            import clv_ngcf_school_storage as school_io
            return school_io.check_space(api, required=required, initial=initial)
        impl.check_space = check_space
        return impl
    storage.backend = backend
    parent = storage.install(screen, api, prepared,
        local_root='/content/graphsage_school_seed50_' + label.lower())
    original = screen.SafeProgressStore
    expected = screen.identity(prepared, label)
    class SingleArmStore(original):
        def __init__(self, root, ident):
            if ident != expected:
                raise RuntimeError('다른 모형 학습·저장 금지')
            super().__init__(root, ident)
    screen.SafeProgressStore = SingleArmStore
    screen.base.io._atomic_json(prepared['run_dir'] / 'school_partition.json', {
        'account': ACCOUNT, 'dataset': 'hm', 'backbone': 'GraphSAGE', 'seed': 50,
        'models': [label], 'epochs': 300, 'first_run': 'fresh initialization',
        'old_personal_checkpoint_import': False, 'retain_epochs': 2,
        'minimum_free_bytes': RESERVE,
        'parallel_peak_checkpoint_budget_bytes': 6 * 1_200_000_000,
        'note': 'Before two parallel jobs, budget 7.2 GB plus 1 GB reserve; no deletion of prior experiments by this runner.'})
    return parent


def export(screen, prepared, label, destination):
    validate(label)
    row = screen.completed(prepared, label)
    reading = {'complete': row is not None, 'planned': 1, 'completed': int(row is not None),
               'models': [label], 'seed': 50, 'backbone': 'GraphSAGE',
               'fixed_epoch': 300, 'single_seed_only': True, 'significance_claim': False}
    path = Path(destination)
    if row is None:
        path = path.with_name(path.stem + '_partial.zip')
    temp = path.with_suffix('.writing')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as z:
        z.writestr('partition_status.json', json.dumps(reading, ensure_ascii=False, indent=2))
        for name in ('protocol.json', 'reference_audit.json', 'preparation_timing.json',
                     'personal_checkpoint_storage.json', 'school_partition.json'):
            p = prepared['run_dir'] / name
            if p.is_file():
                z.write(p, name)
        if row is not None:
            p = screen.arm_path(prepared, label)
            z.write(p, 'arms/' + p.name)
            z.writestr('absolute.json', json.dumps(row['metrics'], indent=2))
    with ZipFile(temp) as z:
        if z.testzip() is not None:
            raise RuntimeError('ZIP integrity failed')
    temp.replace(path)
    return path, reading

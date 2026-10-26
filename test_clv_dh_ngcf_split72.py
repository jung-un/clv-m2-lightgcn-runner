from dataclasses import asdict
import copy
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pytest

import clv_dh_ngcf_split72 as split
import clv_dh_ngcf_remaining9 as s
from test_clv_dh_ngcf_remaining9 import prepared, ngcf_fixture, reference_fixture, make_row, synthetic_source


def protocol(p):
    cfg = json.loads(json.dumps(asdict(p['cfg'])))
    return {**{k: {} for k in split.SCIENCE_KEYS}, 'code_version': s.VERSION,
        'source_revision': split.SOURCE_COMMIT, 'config': cfg, 'input_hash': p['input_hash'],
        'feature_sha256': p['feature_sha256'], 'seed49_audit': {'sha256': s.OLD_SHA},
        'fixed_epoch': 300, 'validation': False, 'holdout': False, 'early_stopping': False}


def ready(p):
    p['revision'] = split.SOURCE_COMMIT
    p['protocol'] = protocol(p)
    p['config_hash'] = split.digest(p['protocol'])[:12]
    return p


def test_assignment_is_disjoint_and_old_entrypoint_is_guarded(prepared, monkeypatch):
    p = ready(prepared)
    calls = []
    monkeypatch.setattr(s, 'run_arm', lambda p, m, seed: calls.append((m, seed)))
    monkeypatch.delattr(s, '_split_original_run_arm', raising=False)
    monkeypatch.setattr(s, 'run', s.run)
    split.install(p, 'personal')
    assert set(split.ASSIGNMENTS['personal']).isdisjoint(split.ASSIGNMENTS['school'])
    assert set(split.ASSIGNMENTS['personal']) | set(split.ASSIGNMENTS['school']) == set(s.NEW_SEEDS)
    s.run_arm(p, 'M1', 46)
    for seed in (49, 50, 51):
        with pytest.raises(RuntimeError, match='학습 금지'):
            s.run_arm(p, 'M1', seed)
    assert calls == [('M1', 46)]
    with pytest.raises(RuntimeError, match='배정'):
        split.install(p, 'school')


def test_foreign_work_blocks_reassignment_and_does_not_fit(prepared):
    p = ready(prepared)
    f = Path(p['run_dir']) / 'progress/stages/old_s50.json'
    f.parent.mkdir(parents=True)
    f.write_text('{}')
    with pytest.raises(RuntimeError, match='기존 작업'):
        split.install(p, 'personal')


def test_signature_ignores_paths_only(prepared):
    p = ready(prepared)
    a, b = copy.deepcopy(p['protocol']), copy.deepcopy(p['protocol'])
    b['config']['out_dir'] = '/school/new-output'
    b['config']['m2_reference'] = '/school/unchanged-input.zip'
    assert split.scientific_signature(a) == split.scientific_signature(b)
    b['config']['lr'] = .2
    assert split.scientific_signature(a) != split.scientific_signature(b)
    for change in ({'source_revision': 'new'}, {'holdout': True}):
        with pytest.raises(RuntimeError):
            split.scientific_signature({**a, **change})


def packet(p, role, path, *, missing=False, duplicate=False):
    pp = copy.deepcopy(p)
    pp['protocol']['config']['out_dir'] += '/' + role
    pp['config_hash'] = split.digest(pp['protocol'])[:12]
    rec = split.receipt(pp, role)
    with ZipFile(path, 'w', ZIP_DEFLATED) as z:
        z.writestr('protocol.json', json.dumps(pp['protocol']))
        z.writestr('split_assignment.json', json.dumps(rec))
        for name, blob in pp['previous']['blobs'].items():
            z.writestr('reused/seed49/' + name, blob)
        for seed in split.ASSIGNMENTS[role]:
            for m in s.ORDER:
                if missing and (seed, m) == (51, 'M5_D'):
                    continue
                row = make_row(pp, m, seed)
                z.writestr(f'arms/{row["model_id"]}_s{seed}.json', json.dumps(row))
        if duplicate:
            row = make_row(pp, 'M1', 50)
            z.writestr('arms/duplicate.json', json.dumps(row))
    return path


def test_merge_preserves_identity_full_precision_and_requires_all80(prepared, monkeypatch, tmp_path):
    p = ready(prepared)
    p['previous']['blobs'] = synthetic_source(p, monkeypatch)
    p['protocol']['seed49_audit']['sha256'] = s.OLD_SHA
    p['config_hash'] = split.digest(p['protocol'])[:12]
    a = packet(p, 'personal', tmp_path / 'a.zip')
    b = packet(p, 'school', tmp_path / 'b.zip')
    result = split.merge(a, b, tmp_path / 'merged')
    assert result['reading']['complete'] and len(result['absolute']) == 80
    assert len(result['summary']) == 8 * 142
    assert result['summary'].n_seeds.eq(10).all()
    row = json.loads((tmp_path / 'merged/arms/ngcf_m1_lastweek_s50.json').read_text())
    assert row['identity']['config_hash'] != p['config_hash']
    partial = split.merge(a, packet(p, 'school', tmp_path / 'partial.zip', missing=True), tmp_path / 'partial')
    assert not partial['reading']['complete'] and 'summary' not in partial
    with pytest.raises(RuntimeError):
        split.merge(a, packet(p, 'school', tmp_path / 'bad.zip', duplicate=True), tmp_path / 'bad')


def test_receipt_must_show_personal7_with_matching_science(prepared, tmp_path):
    p = ready(prepared)
    f = tmp_path / 'receipt.json'
    f.write_text(json.dumps(split.receipt(p, 'personal')))
    assert split.check_personal_receipt(p, f)['seeds'] == list(range(42, 49))
    bad = split.receipt(p, 'personal')
    bad['seeds'].append(50)
    f.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError):
        split.check_personal_receipt(p, f)

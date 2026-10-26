from dataclasses import replace
import copy
import hashlib
from pathlib import Path
import sys
import types

import pytest
import torch

import clv_dh_ngcf_remaining9 as s
import clv_ngcf_school_storage as storage
from test_clv_dh_ngcf_remaining9 import prepared, ngcf_fixture, reference_fixture


class Request:
    def __init__(self, fn): self.fn = fn
    def execute(self): return self.fn()


class API:
    def __init__(self):
        self.rows, self.deleted, self.counter = {}, [], 0
        self.owner, self.extra_usage = storage.ACCOUNT, 0
    def about(self): return self
    def files(self): return self
    def get(self, fileId=None, fields=None):
        if fileId is None:
            return Request(lambda: {'user': {'emailAddress': self.owner}, 'storageQuota': {}})
        return Request(lambda: copy.deepcopy(self.rows[fileId]))
    def list(self, q, **kwargs):
        def run():
            rows = list(self.rows.values())
            if " in parents" in q:
                parent = q.split("'")[1]
                rows = [r for r in rows if r.get('parents') == [parent]]
            if "name = '" in q:
                name = q.split("name = '")[1].split("'")[0]
                rows = [r for r in rows if r['name'] == name]
            if kwargs.get('fields', '').endswith('quotaBytesUsed)') and self.extra_usage:
                rows = rows + [{'id': 'other-account-experiment', 'mimeType': 'x', 'quotaBytesUsed': self.extra_usage}]
            return {'files': copy.deepcopy(rows)}
        return Request(run)
    def create(self, body, fields=None, media_body=None):
        def run():
            self.counter += 1
            row = {'id': str(self.counter), 'ownedByMe': True, 'trashed': False,
                   'mimeType': body.get('mimeType', 'application/octet-stream'), **body}
            if media_body:
                data = Path(media_body.path).read_bytes()
                row.update(data=data, size=str(len(data)), quotaBytesUsed=str(len(data)),
                           md5Checksum=hashlib.md5(data).hexdigest())
            self.rows[row['id']] = row
            return copy.deepcopy(row)
        return Request(run)
    def delete(self, fileId):
        def run():
            self.deleted.append(fileId)
            del self.rows[fileId]
        return Request(run)
    def get_media(self, fileId): return self.rows[fileId]['data']


@pytest.fixture
def api(monkeypatch):
    module = types.ModuleType('googleapiclient.http')
    class Upload:
        def __init__(self, path, **kw): self.path = path
    class Download:
        def __init__(self, target, data): self.target, self.data = target, data
        def next_chunk(self, **kw): self.target.write(self.data); return None, True
    module.MediaFileUpload, module.MediaIoBaseDownload = Upload, Download
    monkeypatch.setitem(sys.modules, 'googleapiclient', types.ModuleType('googleapiclient'))
    monkeypatch.setitem(sys.modules, 'googleapiclient.http', module)
    return API()


def test_school_space_guard_and_account_fail_closed(api):
    assert storage.check_space(api, initial=True) == 10_000_000_000
    api.extra_usage = 6_000_000_000
    with pytest.raises(RuntimeError, match='여유'):
        storage.check_space(api, initial=True)
    api.owner = 'not-school@example.org'
    with pytest.raises(RuntimeError, match='학교'):
        storage.check_space(api)


def test_immutable_epoch_rotation_and_checksum_guard(api, prepared, tmp_path):
    ident = s.identity(prepared, 'M1', 50)
    parent = storage.folder(api, ident.config_hash)
    f = tmp_path / 'checkpoint.pt'
    for ep in (1, 2, 3):
        f.write_bytes(f'epoch{ep}'.encode())
        storage.publish(api, parent, f, ident, ep)
    rows = storage.rows_for(api, parent, ident)
    assert [int(r['appProperties']['clv_epoch']) for r in rows] == [3, 2]
    assert len(api.deleted) == 1
    # Unrelated files are never cleanup targets.
    api.rows['unrelated'] = {'id': 'unrelated', 'name': 'other.pt', 'parents': ['another'],
        'mimeType': 'x', 'quotaBytesUsed': '5', 'ownedByMe': True, 'trashed': False}
    f.write_bytes(b'different epoch3')
    with pytest.raises(RuntimeError, match='同一|동일'):
        storage.publish(api, parent, f, ident, 3)
    assert 'unrelated' in api.rows and len(api.deleted) == 1


def test_saved_adam_rng_and_cpu_resume_equal_original(api, prepared, tmp_path, monkeypatch):
    p = prepared
    old = s.original.SafeProgressStore
    monkeypatch.setattr(s.original, 'SafeProgressStore', old)
    monkeypatch.delattr(s.original, '_school_original_store', raising=False)
    monkeypatch.setattr(storage, 'LOCAL_CACHE', tmp_path / 'local')
    cfg = replace(p['cfg'], epochs=3, batch_size=3)
    ident = s.identity(p, 'M2', 50)
    def fresh():
        s.base.v3.set_seed(50)
        model, weights = s.original.build_model(p, 'M2')
        s.base.v3.set_seed(50)
        return model, weights
    full, weights = fresh()
    baseline = old(tmp_path / 'baseline', ident)
    expected = s.original.execution.train(full, p, cfg, ident.model_id, 50, baseline, row_weights=weights)
    storage.install(p, api)
    cls = s.original.SafeProgressStore
    partial, weights = fresh()
    first = cls(p['run_dir'] / 'progress', ident)
    s.original.execution.train(partial, p, replace(cfg, epochs=1), ident.model_id, 50, first, row_weights=weights)
    first.latest_checkpoint.unlink()  # Simulate loss of Colab local disk.
    resumed, weights = fresh()
    second = cls(p['run_dir'] / 'progress', ident)
    actual = s.original.execution.train(resumed, p, cfg, ident.model_id, 50, second, row_weights=weights)
    assert actual == expected
    a = torch.load(baseline.latest_checkpoint, weights_only=False)
    b = torch.load(second.latest_checkpoint, weights_only=False)
    for key in full.state_dict():
        torch.testing.assert_close(full.state_dict()[key], resumed.state_dict()[key], rtol=0, atol=0)
    assert a['numpy_rng_state'] == b['numpy_rng_state']
    torch.testing.assert_close(a['torch_rng_state'], b['torch_rng_state'], rtol=0, atol=0)
    for key, value in a['optimizer_state']['state'].items():
        for name in value:
            torch.testing.assert_close(value[name], b['optimizer_state']['state'][key][name], rtol=0, atol=0)

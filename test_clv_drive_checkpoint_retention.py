import copy
import hashlib
from pathlib import Path
import pytest
from clv_drive_checkpoint_retention import prune

class Response:
    def __init__(self, fn): self.fn=fn
    def execute(self):return self.fn()

class Api:
    def __init__(self, path):
        self.deleted=[];self.fail_delete=False;self.pages=False;self.change_live=False
        common=dict(name=path.name,parents=['folder'],ownedByMe=True,size=str(path.stat().st_size),md5Checksum=hashlib.md5(path.read_bytes()).hexdigest())
        self.rows={'live':dict(common,id='live',trashed=False,explicitlyTrashed=False,modifiedTime='2026-10-23T10:03:00Z')}
        for i in range(3):self.rows[str(i)]=dict(common,id=str(i),trashed=True,explicitlyTrashed=True,modifiedTime=f'2026-10-23T10:0{i}:00Z')
    def files(self):return self
    def list(self,**kw):
        def result():
            rows=[v for v in self.rows.values() if v['trashed']==('trashed = true' in kw['q'])]
            return {'files':copy.deepcopy(rows),'incompleteSearch':self.pages}
        return Response(result)
    def get(self,fileId,**kw):
        if fileId=='folder':return Response(lambda:{'id':'folder','trashed':False})
        def result():
            r=copy.deepcopy(self.rows[fileId])
            if fileId=='live' and self.change_live:r['md5Checksum']='changed'
            return r
        return Response(result)
    def delete(self,fileId):
        def delete():
            if self.fail_delete:raise RuntimeError('API deletion failed')
            self.deleted.append(fileId);del self.rows[fileId]
            return {}
        return Response(delete)

@pytest.fixture
def setup(tmp_path, monkeypatch):
    import clv_drive_checkpoint_retention as mod
    monkeypatch.setattr(mod, '_verified_info', lambda api, row, local=None: ({'run': 'same'}, 10))
    p=tmp_path/'run_m1_s50_latest.pt';p.write_bytes(b'known-good checkpoint')
    return p,Api(p)

def test_retains_current_and_immediate_previous(setup):
    p,a=setup
    assert prune(a,p,retries=1)==2
    assert set(a.rows)=={'live','2'}
    assert p.read_bytes()==b'known-good checkpoint'
    assert prune(a,p,retries=1)==0

def test_completion_keeps_live_weights(setup):
    p,a=setup
    assert prune(a,p,keep_previous=0,retries=1)==3
    assert set(a.rows)=={'live'}

@pytest.mark.parametrize('change',[{'md5Checksum':'wrong'},{'size':'0'},{'ownedByMe':False},{'parents':[]}])
def test_unverified_upload_never_deletes(setup,change):
    p,a=setup;a.rows['live'].update(change)
    with pytest.raises(RuntimeError):prune(a,p,retries=1)
    assert not a.deleted

@pytest.mark.parametrize('change',[{'name':'other_latest.pt'},{'parents':['other']},{'ownedByMe':False},{'explicitlyTrashed':False},{'modifiedTime':'2026-10-23T10:05:00Z'}])
def test_other_or_newer_files_never_deleted(setup,change):
    p,a=setup;a.rows['0'].update(change)
    prune(a,p,keep_previous=0,retries=1)
    assert '0' not in a.deleted

def test_changed_live_blocks_deletion(setup):
    p,a=setup;a.change_live=True
    with pytest.raises(RuntimeError):prune(a,p,retries=1)
    assert not a.deleted

def test_failed_delete_keeps_live_and_previous(setup):
    p,a=setup;a.fail_delete=True
    with pytest.raises(RuntimeError):prune(a,p,retries=1)
    assert {'live','2'} <= set(a.rows)

def test_duplicate_live_blocks_cleanup(setup):
    p,a=setup;a.rows['duplicate']={**a.rows['live'],'id':'duplicate'}
    with pytest.raises(RuntimeError):prune(a,p,retries=1)
    assert not a.deleted

def test_incomplete_listing_blocks_cleanup(setup):
    p,a=setup;a.pages=True
    with pytest.raises(RuntimeError):prune(a,p,retries=1)
    assert not a.deleted

def test_install_wraps_saving_without_changing_state_or_identity(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import torch
    import clv_drive_checkpoint_retention as mod
    root=tmp_path/'run';resume=root/'progress'/'resume';resume.mkdir(parents=True)
    p=resume/'run_m1_s50_latest.pt'
    ident={'config_hash':'cfg','input_hash':'data','source_revision':'rev'}
    torch.save({'identity':ident,'epoch':75,'model_state':{'x':torch.tensor([1.])}},p)
    events=[]
    monkeypatch.setattr(mod,'prune',lambda *args,**kw:events.append(('prune',kw['keep_previous'])))
    class Store:
        def save_epoch(self,*args,**kwargs):events.append(('save',kwargs['epoch']));return 'saved'
    api=SimpleNamespace(about=lambda:SimpleNamespace(get=lambda **kw:Response(lambda:{'user':{'emailAddress':'jungunb@u.sogang.ac.kr'}})))
    screen=SimpleNamespace(SafeProgressStore=Store)
    prepared={'run_dir':root,'revision':'rev','config_hash':'cfg','input_hash':'data'}
    before=p.read_bytes();mod.install(screen,api,prepared)
    store=Store();store.root=root/'progress';store.latest_checkpoint=p
    assert store.save_epoch(None,epoch=76)=='saved'
    assert events==[('prune',1),('prune',1),('save',76),('prune',1)]
    assert p.read_bytes()==before
    mod.install(screen,api,prepared)
    assert len(events)==4
    store.root=tmp_path/'other'
    with pytest.raises(RuntimeError):store.save_epoch(None,epoch=77)


def test_newer_epoch_in_trash_blocks_all_deletion(setup,monkeypatch):
    import clv_drive_checkpoint_retention as mod
    p,a=setup
    monkeypatch.setattr(mod,'_verified_info',lambda api,row,local=None: ({'run':'same'}, 11 if row['id']=='0' else 10))
    with pytest.raises(RuntimeError,match='더 최신 epoch'): prune(a,p,retries=1)
    assert not a.deleted


def test_keeps_highest_epoch_not_latest_timestamp(setup,monkeypatch):
    import clv_drive_checkpoint_retention as mod
    p,a=setup
    epochs={'live':20,'0':19,'1':18,'2':17}
    monkeypatch.setattr(mod,'_verified_info',lambda api,row,local=None: ({'run':'same'},epochs[row['id']]))
    prune(a,p,retries=1)
    assert set(a.rows)=={'live','0'}

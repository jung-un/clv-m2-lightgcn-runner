import hashlib
import tempfile
import types
import unittest
from dataclasses import dataclass, asdict
from pathlib import Path
from unittest.mock import patch
import torch
import clv_personal_gcn51_checkpoint as target

@dataclass(frozen=True)
class Identity:
    model_id: str = 'gcn_m5_a_lastweek'
    seed: int = 51
    config_hash: str = 'test'

class Base:
    def __init__(self, root, ident):
        self.identity = ident
        self.latest_checkpoint = Path(root)/'resume'/'latest.pt'
        self.stage_path = Path(root)/'stage.json'
    def restore_epoch(self, *args):
        if not self.latest_checkpoint.exists(): return None
        return {'next_epoch': torch.load(self.latest_checkpoint, weights_only=False)['epoch']+1}
    def _write_progress(self, *args): pass
    def save_epoch(self, *args, **state):
        temp = self.latest_checkpoint.with_suffix('.tmp')
        torch.save(payload(state['epoch']), temp)
        temp.replace(self.latest_checkpoint)
        self.last_save_timing = {}
        return self.latest_checkpoint

def payload(epoch):
    return dict(identity=asdict(Identity()), epoch=epoch,
        history=[{'epoch': e} for e in range(1, epoch+1)], model_state={},
        optimizer_state={}, numpy_rng_state={}, torch_rng_state=torch.get_rng_state(), cuda_rng_state=[])

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.rows=[]; self.sent=[]; self.error=None
    def store(self, interval=1):
        def publish(api,parent,path,ident,epoch):
            self.sent.append(epoch)
            if self.error: raise self.error
            return dict(id='saved',size=str(path.stat().st_size), appProperties={'clv_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        backend=types.SimpleNamespace(check_space=lambda *a,**k:100e9,
            listing=lambda *a:[dict(id='folder',ownedByMe=True,appProperties={'clv_split_run':'test'})],
            rows_for=lambda *a:self.rows,publish=publish)
        screen=types.SimpleNamespace(SafeProgressStore=Base,ORDER=tuple(range(8)),
            validate_config=lambda c:None, identity=lambda p,m:Identity(),
            base=types.SimpleNamespace(io=types.SimpleNamespace(_atomic_json=lambda *a:None)))
        prepared=dict(cfg=types.SimpleNamespace(seeds=(51,)),config_hash='test',run_dir=self.root)
        with patch.object(target,'backend',return_value=backend):
            target.install(screen,None,prepared,local_root=str(self.root/'local'),drive_interval=interval)
        return screen.SafeProgressStore(self.root/'progress',Identity())
    def remote(self,epoch,sha='remote'):
        self.rows=[dict(appProperties={'clv_epoch':str(epoch),'clv_sha256':sha})]
    def test_newer_local_is_published_before_restore(self):
        s=self.store(); torch.save(payload(65),s.latest_checkpoint); self.remote(64)
        self.assertEqual(s.restore_epoch(None,None,None)['next_epoch'],66)
        self.assertEqual(self.sent,[65]); self.assertEqual(s.checked_epoch(s.latest_checkpoint),65)
    def test_quota_does_not_overwrite_or_resume_local(self):
        s=self.store();torch.save(payload(65),s.latest_checkpoint);self.remote(64)
        before=s.latest_checkpoint.read_bytes();self.error=RuntimeError('quota')
        with self.assertRaisesRegex(RuntimeError,'quota'):s.restore_epoch(None,None,None)
        self.assertEqual(s.latest_checkpoint.read_bytes(),before)
    def test_same_epoch_conflict_blocks(self):
        s=self.store();torch.save(payload(65),s.latest_checkpoint);self.remote(65)
        with self.assertRaisesRegex(RuntimeError,'내용 불일치'):s.restore_epoch(None,None,None)
        self.assertEqual(self.sent,[])
    def test_wrong_identity_and_missing_state_block(self):
        s=self.store();p=payload(65);p['identity']['seed']=50;torch.save(p,s.latest_checkpoint)
        with self.assertRaises(RuntimeError):s.restore_epoch(None,None,None)
        p=payload(65);del p['optimizer_state'];torch.save(p,s.latest_checkpoint)
        with self.assertRaises(RuntimeError):s.restore_epoch(None,None,None)
    def test_optional_cadence_and_final_upload_block(self):
        s=self.store(10);torch.save(payload(64),s.latest_checkpoint)
        self.remote(64,hashlib.sha256(s.latest_checkpoint.read_bytes()).hexdigest())
        s.restore_epoch(None,None,None)
        for e in range(65,71):s.save_epoch(None,None,None,epoch=e)
        self.assertEqual(self.sent,[70]);self.assertEqual(s.checked_epoch(s.latest_checkpoint.with_suffix('.previous.pt')),69)
        self.error=RuntimeError('quota')
        with self.assertRaisesRegex(RuntimeError,'quota'):s.save_epoch(None,None,None,epoch=300)
        self.assertEqual(s.checked_epoch(s.latest_checkpoint),300)
        with self.assertRaisesRegex(RuntimeError,'quota'):s.restore_epoch(None,None,None)
    def test_failed_boundary_cannot_be_skipped_on_resume(self):
        s=self.store(10);torch.save(payload(70),s.latest_checkpoint);self.remote(64)
        self.error=RuntimeError('quota')
        with self.assertRaisesRegex(RuntimeError,'quota'):s.restore_epoch(None,None,None)
        self.assertEqual(self.sent,[70])
    def test_default_saves_each_epoch_and_retains_previous(self):
        s=self.store()
        for e in range(1,4):s.save_epoch(None,None,None,epoch=e)
        self.assertEqual(self.sent,[1,2,3]);self.assertEqual(s.checked_epoch(s.latest_checkpoint.with_suffix('.previous.pt')),2)

if __name__=='__main__':unittest.main()

import pytest
from clv_ngcf_migration_resume import import_missing, pending_epochs

def row(ep,sha='same'):
    return {'appProperties':{'clv_epoch':str(ep),'clv_sha256':sha}}

def test_repeat_preserves_advanced_files(tmp_path):
    src=tmp_path/'src';dest=tmp_path/'dest';src.mkdir();dest.mkdir()
    for p in (src,dest): (p/'protocol.json').write_text('{}')
    (src/'school_checkpoint_storage.json').write_text('school')
    (dest/'school_checkpoint_storage.json').write_text('personal')
    (src/'progress.json').write_text('102');(dest/'progress.json').write_text('257')
    (src/'arm.json').write_text('complete')
    assert len(import_missing(src,dest)['copied'])==1
    assert import_missing(src,dest)['copied']==[]
    assert (dest/'progress.json').read_text()=='257'
    assert (dest/'school_checkpoint_storage.json').read_text()=='personal'

def test_protocol_conflict_prevents_import(tmp_path):
    s=tmp_path/'s';d=tmp_path/'d';s.mkdir();d.mkdir()
    (s/'protocol.json').write_text('a');(d/'protocol.json').write_text('b')
    with pytest.raises(RuntimeError):import_missing(s,d)

def test_latest_remote_and_completed_never_reimport():
    assert pending_epochs([row(257),row(256)],[row(101),row(102)],completed=False)==[]
    assert pending_epochs([],[row(299),row(300)],completed=True)==[]
    assert pending_epochs([row(101)],[row(101),row(102)],completed=False)==[row(102)]
    with pytest.raises(RuntimeError):pending_epochs([row(102,'changed')],[row(102)],completed=False)

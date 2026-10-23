import ast
import json
from pathlib import Path
import pytest

class Api:
    def __init__(self, usage=29_642_480_000_000, pages=None):
        self.info={'user':{'emailAddress':'jungunb@u.sogang.ac.kr'},'storageQuota':{'limit':str(100*10**12),'usage':str(usage)}}
        self.pages=pages or [{'files':[{'id':'a','mimeType':'application/zip','quotaBytesUsed':'2120000000'}]}]
        self.calls=[]
    def about(self):return self
    def get(self,**kw):return type('R',(),{'execute':lambda _:self.info})()
    def files(self):return self
    def list(self,**kw):
        self.calls.append(kw)
        i=int(kw.get('pageToken') or 0)
        return type('R',(),{'execute':lambda _:self.pages[i]})()

def check(api):
    n=json.loads(Path(__file__).with_name('clv_dh_gcn_final8_seed50_school_colab.ipynb').read_text())
    tree=ast.parse(''.join(n['cells'][3]['source']))
    fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='check_space')
    env={'drive_api':api,'OTHER_RUN_RESERVE_GB':1.0}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'notebook_quota','exec'),env)
    return env['check_space'](750_000_000)

def test_pooled_29tb_does_not_subtract_from_personal_10gb():
    api=Api()
    assert check(api)[1]==7_880_000_000
    assert api.calls[0]['q']=="'me' in owners" # includes trash

def test_pagination_counts_retained_quota_and_trash():
    a={'id':'a','mimeType':'application/zip','quotaBytesUsed':'6000000000'}
    b={'id':'b','mimeType':'application/zip','quotaBytesUsed':'3000000000'}
    with pytest.raises(RuntimeError,match='개인 Drive'):
        check(Api(pages=[{'files':[a],'nextPageToken':'1'},{'files':[b]}]))

def test_pool_exhaustion_separate():
    with pytest.raises(RuntimeError,match='조직 전체'):check(Api(usage=100*10**12))

@pytest.mark.parametrize('page',[{'incompleteSearch':True},{'files':[{'id':'a','mimeType':'application/zip'}]}])
def test_incomplete_never_treated_as_zero(page):
    with pytest.raises(RuntimeError):check(Api(pages=[page]))

def test_wrong_account():
    api=Api();api.info['user']['emailAddress']='wrong@example.com'
    with pytest.raises(RuntimeError,match='학교 계정'):check(api)

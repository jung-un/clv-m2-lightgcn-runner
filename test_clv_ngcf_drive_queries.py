from types import SimpleNamespace
import pytest
import clv_ngcf_drive_queries as q

class Failure(Exception):
    def __init__(self,status,reason):
        self.resp=SimpleNamespace(status=status)
        self.content=(' {"error":{"errors":[{"reason":"'+reason+'"}]}}').encode()

class Request:
    def __init__(self, errors=()): self.errors=list(errors); self.calls=0
    def execute(self):
        self.calls+=1
        if self.errors: raise self.errors.pop(0)
        return {'user':{'emailAddress':'expected'},'storageQuota':{'limit':'100','usage':'10'}}

def test_retry_quota_only_upload():
    req=Request([Failure(403,'rateLimitExceeded')]); delays=[]
    q.execute(req,upload=True,sleep=delays.append)
    assert req.calls==2 and delays==[60]
    for status,reason in [(500,'backendError'),(403,'forbidden')]:
        req=Request([Failure(status,reason)])
        with pytest.raises(Failure): q.execute(req,upload=True,sleep=delays.append)
        assert req.calls==1

def test_cache_expiry_identity():
    now=[0]; req=Request()
    api=SimpleNamespace(about=lambda:SimpleNamespace(get=lambda **kw:req))
    cache=q.AccountInfo('expected',clock=lambda:now[0])
    cache.get(api); cache.get(api); assert req.calls==1
    now[0]=121; cache.get(api); assert req.calls==2
    cache.get(api,force=True); assert req.calls==3
    with pytest.raises(RuntimeError): q.AccountInfo('wrong').get(api)

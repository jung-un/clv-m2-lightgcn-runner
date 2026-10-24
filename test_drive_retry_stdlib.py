import unittest
import clv_drive_checkpoint_retention as r
class Error(Exception):
 def __init__(self,status=403,reason='rateLimitExceeded'):
  self.resp=type('Resp',(),{'status':status})()
  self.content=r.json.dumps({'error':{'errors':[{'reason':reason}]}}).encode()
class Request:
 def __init__(self,failures): self.failures=list(failures);self.calls=0
 def execute(self):
  self.calls+=1
  if self.failures:raise self.failures.pop(0)
  return {'files':[]}
class RetryTests(unittest.TestCase):
 def test_quota_recovers(self):
  waits=[]; req=Request([Error(),Error(429)])
  self.assertEqual(r._execute(req,sleep=waits.append),{'files':[]})
  self.assertEqual(waits,[60,120])
 def test_permission_not_retried(self):
  req=Request([Error(403,'insufficientPermissions')]); waits=[]
  with self.assertRaises(Error):r._execute(req,sleep=waits.append)
  self.assertEqual(waits,[])
 def test_exhausted_preserves_error(self):
  req=Request([Error() for _ in range(6)]);waits=[]
  with self.assertRaises(Error):r._execute(req,sleep=waits.append)
  self.assertEqual(waits,[60,120,240,480,480])
 def test_list_uses_retry(self):
  req=Request([Error()]);original=r._execute
  class API:
   def files(self):return self
   def list(self,**kwargs):return req
  r._execute=lambda request:original(request,sleep=lambda _:None)
  try:self.assertEqual(r._list(API(),'q'),[])
  finally:r._execute=original
  self.assertEqual(req.calls,2)
if __name__=='__main__':unittest.main()

"""Bounded Drive retries and account/quota caching, without training RNG use."""
import json
import time


def execute(request, *, upload=False, sleep=time.sleep):
    for attempt in range(6):
        try:
            return request.execute()
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            try:
                reasons = {r.get('reason') for r in json.loads(exc.content).get('error', {}).get('errors', [])}
            except (ValueError, TypeError, AttributeError):
                reasons = set()
            limited = status == 429 or (status == 403 and bool(reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))
            # Uploads retry explicit quota rejections only; ambiguous failures stop.
            retry = limited or (not upload and status in (500, 502, 503, 504))
            if not retry or attempt == 5:
                raise
            delay = min(300, 60 * 2 ** attempt)
            print(f'[Drive 대기] HTTP {status}; {delay}초 후 재시도 {attempt+1}/5. 기존 저장본 유지.', flush=True)
            sleep(delay)


class AccountInfo:
    def __init__(self, account, ttl=120, clock=time.monotonic):
        self.account, self.ttl, self.clock = account, ttl, clock
        self.cache = None
        self.api = None

    def get(self, api, force=False):
        if force or self.api is not api or self.cache is None or self.clock()-self.cache[0] >= self.ttl:
            self.cache = None
            info = execute(api.about().get(fields='user(emailAddress),storageQuota'))
            if info['user']['emailAddress'] != self.account:
                raise RuntimeError('개인 Drive 계정 불일치')
            self.cache, self.api = (self.clock(), info), api
        return self.cache[1]

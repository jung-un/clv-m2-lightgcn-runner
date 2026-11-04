"""Retry transient Drive upload limits while retaining checkpoint safeguards."""
import json
import time
import random


def install(storage, *, sleep=time.sleep, jitter=None):
    if getattr(storage, '_upload_retry_installed', False):
        return
    factory = storage.backend
    # SystemRandom does not change the training RNG or the negative samples.
    jitter = jitter or (lambda: random.SystemRandom().uniform(0, 15))

    def backend():
        module = factory()
        publish = module.publish

        def publish_with_retry(api, parent, path, ident, epoch):
            for attempt in range(9):
                try:
                    # Each new call checks for an already uploaded matching epoch.
                    return publish(api, parent, path, ident, epoch)
                except Exception as exc:
                    status = getattr(getattr(exc, 'resp', None), 'status', None)
                    try:
                        errors = json.loads(exc.content)['error']['errors']
                        reasons = {error.get('reason') for error in errors}
                    except (AttributeError, KeyError, TypeError, ValueError):
                        reasons = set()
                    limited = status == 429 or (status == 403 and bool(
                        reasons & {'rateLimitExceeded', 'userRateLimitExceeded'}))
                    transient = limited or status in (500, 502, 503, 504)
                    if not transient or attempt == 8:
                        raise
                    delay = min(300, 60 * 2 ** attempt) + jitter()
                    print(f'Drive 저장 재시도: {delay:.0f}초 후 재확인 ({attempt + 1}/8). '
                          '같은 epoch 업로드만 재시도·다음 epoch 대기. '
                          '로컬과 기존 체크포인트를 보존합니다.', flush=True)
                    sleep(delay)

        module.publish = publish_with_retry
        return module

    storage.backend = backend
    storage._upload_retry_installed = True

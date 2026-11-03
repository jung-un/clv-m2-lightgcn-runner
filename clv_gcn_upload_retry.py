"""Retry transient Drive upload limits while retaining checkpoint safeguards."""
import json
import time


def install(storage, *, sleep=time.sleep):
    if getattr(storage, '_upload_retry_installed', False):
        return
    factory = storage.backend

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
                    if not limited or attempt == 8:
                        raise
                    delay = min(300, 60 * 2 ** attempt)
                    print(f'Drive 요청 제한: {delay}초 후 재확인 ({attempt + 1}/8). '
                          '로컬과 기존 체크포인트를 보존합니다.', flush=True)
                    sleep(delay)

        module.publish = publish_with_retry
        return module

    storage.backend = backend
    storage._upload_retry_installed = True

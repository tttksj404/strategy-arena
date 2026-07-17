import os

# 백그라운드 스레드가 테스트의 engine mock 창을 오염시키지 않도록 세션 전역 차단.
# 1) prewarm: env 노브로 기동 자체를 차단
# 2) /recent의 _bg_fetch_recent: 이전 테스트 파일에서 뜬 스레드가 실네트워크 지연 후
#    다음 테스트의 patch 창에서 _api_page_cached(timeout=8)를 호출해 flake를 유발했음
#    (어떤 테스트도 bg 채움 결과에 의존하지 않음 — 즉시 fallback 응답만 검증)
os.environ.setdefault("KEIRIN_PREWARM_ENABLED", "0")

import app as _app  # noqa: E402


def _noop_bg_fetch_recent(sport, meet, key, n):
    _app._RECENT_FETCHING.discard((sport, meet))


_app._bg_fetch_recent = _noop_bg_fetch_recent

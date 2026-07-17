# -*- coding: utf-8 -*-
"""
RaceLens Flask API server for 경륜·경마 data analysis.
====================================================================
gunicorn app:app 구조. 전역 `app`.

라우트
  /         폼(종목·날짜·경주장·경주번호)
  /predict  POST/GET — 출주표 실시간 fetch → 모델 채점 → 7권종 픽

정직 고지(상시 노출)
  예측(적중률) 보조 도구. 평균 -EV(검증완료) · 수익 보장 아님.
  도박중독 주의 · 책임베팅 · 만 19세 이상.

키 처리
  DATAGOKR_SERVICE_KEY 는 os.environ 으로만 읽는다. 코드/커밋에 키 없음.
  키가 없거나 출주표 조회가 실패하면 다른 날짜의 데모 경주를 섞지 않는다.
"""
import os
import time
import datetime as dt
import threading
import json
import hashlib
import html
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, render_template, jsonify, Response, redirect, send_from_directory
from werkzeug.exceptions import HTTPException

import datastore
import admob_ssv
import engine
import iap
import roster_guard


def _record_fresh_kra_odds_snapshot(starters, info):
    if not info.get("odds_snapshot_fresh"):
        return False
    board = [
        {
            "chulNo": str(starter.get("chulNo", "")),
            "winOdds": starter.get("winOdds"),
            "plcOdds": starter.get("plcOdds"),
        }
        for starter in starters
    ]
    return datastore.record_market_odds_snapshot_safely(
        "horse",
        str(info.get("ymd", "")),
        str(info.get("meet", "")),
        str(info.get("race_no", "")),
        board,
        str(info.get("odds_snapshot_fetched_at", "")),
    )

app = Flask(__name__)
MOBILE_WEB_DIST = os.path.join(app.static_folder, "mobile")
ROSTER_MISMATCH_MESSAGE = "공식 출주표와 일치하지 않아 예측을 중단했습니다"


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_errors_enabled() -> bool:
    return bool(app.config.get("TESTING") or app.debug or _env_flag("RACELENS_DEBUG_ERRORS"))


def _safe_error_payload(error: Exception, status_code: int):
    if not _debug_errors_enabled():
        return {"error": "internal_server_error", "status": status_code}
    import traceback
    return {
        "error": type(error).__name__,
        "detail": str(error),
        "trace": traceback.format_exc().splitlines()[-3:],
        "status": status_code,
    }


def _public_exception_message(prefix: str, error: Exception) -> str:
    if _debug_errors_enabled():
        return f"{prefix}: {type(error).__name__}: {error}"
    return f"{prefix}가 발생했습니다. 잠시 후 다시 시도하세요."


def _roster_block_result(verification: dict) -> dict:
    return {
        "kind": "error",
        "status": "roster_mismatch",
        "title": "출주표 불일치",
        "message": ROSTER_MISMATCH_MESSAGE,
        "roster_verification": verification,
    }


def _verify_roster_or_block(sport: str, ymd: str, meet: str, race_no: str, starters: list[dict]) -> tuple[dict, dict | None]:
    verification = roster_guard.verify_roster(sport, ymd, meet, race_no, starters)
    if verification["state"] == "mismatch":
        return verification, _roster_block_result(verification)
    return verification, None


def _allowed_cors_origin():
    configured = {
        item.strip()
        for item in (os.environ.get("RACELENS_ALLOWED_ORIGINS") or "").split(",")
        if item.strip()
    }
    if not configured:
        if (os.environ.get("RACELENS_ENV") or "").strip().lower() == "production":
            app.logger.warning("RACELENS_ALLOWED_ORIGINS is required in production; CORS wildcard disabled")
            return None
        return "*"
    origin = request.headers.get("Origin")
    if origin in configured:
        return origin
    return None


def _request_device_id() -> str:
    explicit = (request.headers.get("X-RaceLens-Device-Id") or "").strip()
    if explicit:
        return explicit[:96]
    raw = "|".join([
        request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",", 1)[0].strip(),
        request.headers.get("User-Agent", "unknown")[:120],
    ])
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"anon_{digest}"


def _request_ip() -> str:
    return (request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",", 1)[0]).strip() or "unknown"


LEGAL_UPDATED_AT = "2026-07-10"


def _support_email() -> str:
    configured = (os.environ.get("RACELENS_SUPPORT_EMAIL") or "").strip()
    return configured or "tttksj@gmail.com"


def _base_error(error_kind: str, message: str, **extra) -> dict:
    payload = {"error": error_kind, "error_kind": error_kind, "message": message}
    payload.update(extra)
    return payload


def _legal_page(title: str, sections: list[tuple[str, list[str]]]) -> Response:
    support_email = html.escape(_support_email())
    body = []
    for heading, paragraphs in sections:
        body.append(f"<section><h2>{html.escape(heading)}</h2>")
        for paragraph in paragraphs:
            body.append(f"<p>{html.escape(paragraph)}</p>")
        body.append("</section>")
    page_title = html.escape(title)
    content = "\n".join(body)
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{page_title} | RaceLens</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #111827; background: #f7f5f0; }}
    main {{ max-width: 760px; margin: 0 auto; padding: 48px 20px 64px; line-height: 1.65; }}
    h1 {{ font-size: 2rem; margin: 0 0 8px; }}
    h2 {{ font-size: 1.1rem; margin: 28px 0 8px; }}
    p {{ margin: 0 0 12px; }}
    .meta {{ color: #5f6472; margin-bottom: 28px; }}
    a {{ color: #005f73; }}
  </style>
</head>
<body>
  <main>
    <h1>{page_title}</h1>
    <p class="meta">RaceLens / 레이스렌즈 · 최종 업데이트 {LEGAL_UPDATED_AT} · 문의 <a href="mailto:{support_email}">{support_email}</a></p>
    {content}
  </main>
</body>
</html>"""
    return Response(document, content_type="text/html; charset=utf-8")


@app.route("/legal/privacy")
def legal_privacy():
    return _legal_page("개인정보 처리방침", [
        ("서비스 성격", [
            "RaceLens는 경륜과 경마 출전 정보, 과거 성적, 배당 수집 상태, 모델 신호를 보기 쉽게 정리하는 데이터 분석 서비스입니다.",
            "앱 안에서는 베팅, 구매, 예치금, 송금, 배팅 금액 계산, 수익 보장을 제공하지 않습니다.",
        ]),
        ("수집하는 정보", [
            "서비스 운영을 위해 익명 기기 식별자, 앱 버전, 플랫폼, 접속 로그, 분석 요청의 종목·날짜·경주번호, 오류와 지연 시간 같은 제품 사용 이벤트를 저장할 수 있습니다.",
            "UX 이벤트에는 선수명, 말명, 기수명, 사용자가 고른 조합, 실명, 결제 카드번호를 저장하지 않도록 서버에서 차단합니다.",
        ]),
        ("Firebase 분석 및 오류 진단", [
            "앱 사용 이벤트와 오류 진단을 위해 Google Firebase Analytics 및 Crashlytics를 사용합니다.",
            "이 과정에서 앱 인스턴스 식별자, 탭·종목·경주번호·상위 후보 확률·지연 시간·오류 종류 같은 이벤트 파라미터, 크래시 진단 정보(기기 모델, OS, 앱 버전, 스택 트레이스)가 Google에 전송되어 처리될 수 있습니다.",
            "처리 목적은 앱 기능과 안정성 개선, 장애 원인 파악, 비정상 동작 진단이며, Google은 Firebase 서비스 제공자로서 해당 정보를 처리합니다.",
        ]),
        ("보상형 광고", [
            "무료 분석 3회 이후 사용자가 광고 보기를 직접 선택하면 Google Mobile Ads SDK를 통해 보상형 광고를 표시할 수 있습니다.",
            "Google은 광고 제공, 빈도 관리, 성과 측정, 부정 사용 방지를 위해 광고 식별자, IP 주소, 기기 정보, 광고 상호작용 정보를 처리할 수 있습니다.",
            "광고 보상 확인을 위해 서버는 Google이 서명한 거래 식별자, 광고 단위, 보상 종류와 시각, 익명 기기 식별자를 저장하며 동일 거래의 중복 지급을 차단합니다.",
        ]),
        ("이용 목적", [
            "수집 정보는 무료 사용량 관리, 장애 진단, 배당 수집 상태 점검, 기능 개선, 부정 사용 방지, 고객 지원에 사용합니다.",
            "유료 기능이 활성화되는 경우 결제 영수증 검증 결과와 구독 상태만 저장하며 카드번호는 스토어 결제 사업자가 처리합니다.",
        ]),
        ("보관과 삭제", [
            "서버 로그와 제품 분석 이벤트는 원칙적으로 수집일로부터 1년까지 보관하고, 장애·보안·분쟁 대응에 필요한 기록은 관계 법령상 필요한 기간 동안 분리 보관할 수 있습니다.",
            "무료 사용량, 구독 권한 상태, 분석 요청 연결 정보는 서비스 제공 기간 동안 보관하며, 삭제 요청이 접수되면 법령상 필요한 보관분을 제외하고 삭제 또는 비식별 처리합니다.",
            "개인정보 문의, 삭제, 열람 요청은 이 페이지에 표시된 지원 이메일로 접수합니다.",
        ]),
    ])


@app.route("/legal/terms")
def legal_terms():
    return _legal_page("서비스 이용약관", [
        ("정보 제공 범위", [
            "RaceLens의 분석, 확률, 후보 표시는 경기 데이터와 모델 계산을 설명하는 참고 정보입니다.",
            "경기 결과는 불확실하며, 모델 신호는 적중이나 수익을 보장하지 않습니다.",
        ]),
        ("사용자 책임", [
            "사용자는 앱 정보를 독립적으로 검토해야 하며, 앱은 베팅 지시, 재무 조언, 손실 회복 전략, 구매 대행을 제공하지 않습니다.",
            "서비스는 만 19세 이상 사용자를 대상으로 하며, 거주 지역의 법령과 스토어 정책을 준수해야 합니다.",
        ]),
        ("유료 기능", [
            "무료와 Pro 기능 차이는 사용량 제한, 상세 근거, 기록 조회 같은 앱 편의 기능에 한정됩니다.",
            "무료 분석 3회를 사용한 뒤에는 사용자가 선택적으로 Google 보상형 광고를 끝까지 보고 분석 1회 이용권을 받을 수 있습니다. 광고를 보지 않으면 추가 이용권은 지급되지 않습니다.",
            "구독 상품은 스토어 정책에 따라 자동 갱신될 수 있으며, 사용자는 Apple App Store 또는 Google Play의 구독 관리 화면에서 해지할 수 있습니다.",
            "해지 후에도 이미 결제된 기간이 끝날 때까지 Pro 기능을 사용할 수 있으며, 환불은 결제를 처리한 스토어의 환불 절차와 심사 기준을 따릅니다.",
        ]),
        ("서비스 변경", [
            "공식 데이터 제공 상태, 배당 수집 가능 여부, 외부 API 장애에 따라 일부 기능은 일시적으로 제한될 수 있습니다.",
            "데이터 불일치가 감지되면 앱은 다른 날짜나 다른 경주의 샘플 데이터를 섞지 않고 이용 불가 상태를 표시합니다.",
        ]),
        ("시행일", [
            f"본 약관은 {LEGAL_UPDATED_AT}부터 시행합니다.",
        ]),
    ])


@app.route("/legal/account-deletion")
def legal_account_deletion():
    return _legal_page("계정 및 데이터 삭제 안내", [
        ("삭제 요청 방법", [
            "지원 이메일로 앱에서 표시되는 기기 ID 또는 스토어 계정 식별에 필요한 정보를 보내면 삭제 요청을 접수합니다.",
            "정식 로그인 계정 기능이 활성화되면 앱 안에 동일한 삭제 요청 흐름을 추가합니다.",
        ]),
        ("삭제 대상", [
            "무료 사용량 기록, 기기 기반 앱 세션, 구독 권한 상태, 분석 이벤트 연결 정보, Firebase 앱 인스턴스 기반 이벤트·오류 진단 연결 정보, 고객 지원 처리에 필요한 식별 정보를 삭제 대상으로 합니다.",
            "법령, 정산, 분쟁 대응, 보안 감사에 필요한 최소 기록은 필요한 기간 동안 분리 보관될 수 있습니다.",
        ]),
        ("처리 기준", [
            "요청자는 본인 확인에 필요한 최소 정보를 제공해야 하며, 확인 후 합리적인 기간 안에 삭제 또는 비식별 처리를 진행합니다.",
            "삭제가 완료되면 동일 기기에서 무료 사용량과 개인화 상태가 초기화될 수 있습니다.",
        ]),
    ])


@app.route("/legal/support")
def legal_support():
    return _legal_page("지원 및 문의", [
        ("문의 범위", [
            "RaceLens 사용량, Pro 권한, 공식 데이터 표시 상태, 계정 및 데이터 삭제 요청을 지원합니다.",
            "경기 결과, 배당, 모델 신호는 정보 분석 목적의 참고 자료이며 구매 또는 참여 결정 상담을 제공하지 않습니다.",
        ]),
        ("연락 방법", [
            "이 페이지 상단의 지원 이메일로 앱 버전, 기기 종류, 오류가 발생한 종목·날짜·경주번호를 보내면 확인합니다.",
            "개인정보 열람, 정정, 삭제 요청도 같은 지원 이메일로 접수합니다.",
        ]),
    ])


@app.after_request
def add_api_cors_headers(response):
    if request.path.startswith("/api/") or request.path == "/recent":
        allowed_origin = _allowed_cors_origin()
        if allowed_origin:
            response.headers["Access-Control-Allow-Origin"] = allowed_origin
            if allowed_origin != "*":
                response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-RaceLens-Device-Id, X-RaceLens-Platform, X-RaceLens-Analytics"
    return response

DISCLAIMER = ("예측(적중률) 도구입니다. 평균 −EV(검증 완료) · 수익을 보장하지 않습니다. "
              "도박 중독 주의 · 책임 베팅 · 만 19세 이상.")

KEIRIN_MEETS = engine.KEIRIN_MEETS  # ['광명']
KRA_MEETS = engine.KRA_MEETS        # ['서울','제주','부경']
SPORT_ALIASES = {
    "horse": "horse",
    "kra": "horse",
    "keirin": "keirin",
    "kcycle": "keirin",
}

# 운영 요일 안내(참고용 1줄; 공휴일·변경 가능 → 실제 날짜는 데이터 기반).
SCHEDULE_HINT = ("운영 요일(대략): 경륜 광명 금·토·일 · 경마 서울 토·일 / "
                 "부경 금·토·일 / 제주 금·토 (공휴일·변경 있으니 최근 경주일 칩 참고).")

# (sport, meet) -> (days_list, fetched_ts, status) 모듈 전역 캐시.
# status: "ok"(성공, 1h TTL) / "empty"(빈 결과, 5분 TTL) / "fail"(에러, 60s 음수캐시).
_RECENT_CACHE = {}
_RECENT_TTL = 3600        # 성공 시 1시간
_RECENT_EMPTY_TTL = 300   # 빈 결과 5분
_RECENT_FAIL_TTL = 60     # 실패 60초 (음수 캐싱 → 반복 타임아웃 방지)
_RECENT_FETCHING = set()   # 진행 중인 fetch 중복 방지


def _canonical_sport(raw):
    return SPORT_ALIASES.get(str(raw or "keirin").strip().lower(), "keirin")


def _default_meet(sport):
    return KRA_MEETS[0] if sport == "horse" else KEIRIN_MEETS[0]


def _normalize_race_context(raw_sport, raw_meet):
    sport = _canonical_sport(raw_sport)
    meet = str(raw_meet or "").strip()
    if sport == "horse":
        return sport, meet if meet in KRA_MEETS else KRA_MEETS[0]
    return sport, meet if meet in KEIRIN_MEETS else KEIRIN_MEETS[0]


def _request_race_context(raw_sport, raw_meet):
    sport = _canonical_sport(raw_sport)
    meet = str(raw_meet or "").strip() or _default_meet(sport)
    return sport, meet


def _today_kst():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).date()


def _fallback_race_weekdays(sport, meet):
    if sport == "horse":
        if meet == "서울":
            return {5, 6}
        if meet == "제주":
            return {4, 5}
        return {4, 5, 6}
    return {4, 5, 6}


def _fallback_race_days(sport, meet, n=6):
    today = _today_kst()
    weekdays = _fallback_race_weekdays(sport, meet)
    days = []
    for offset in range(-14, 22):
        day = today + dt.timedelta(days=offset)
        if day.weekday() in weekdays:
            days.append(day.isoformat())
    return _prioritize_race_days(days, n)


def _prioritize_race_days(days, n=6):
    today = _today_kst()
    ranked = []
    for raw in days:
        try:
            day = dt.date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        ranked.append((abs((day - today).days), 0 if day >= today else 1, day.isoformat()))
    return [item[2] for item in sorted(set(ranked))[:n]]


def _schedule_days_with_fallback(sport, meet, days, n=6):
    combined = set(days or [])
    combined.update(_fallback_race_days(sport, meet, n=18))
    return _prioritize_race_days(combined, n)


def _race_count(sport, meet):
    if sport == "keirin":
        return 16
    if meet == "제주":
        return 8
    return 11


def _keirin_default_race_starts(day):
    times = (
        "11:00", "11:23", "11:46", "12:09", "12:32",
        "14:56", "15:20", "15:44", "16:08", "16:32", "16:56",
        "17:20", "17:44", "18:08", "18:31", "18:54",
    )
    starts = []
    for idx, value in enumerate(times, start=1):
        hour, minute = value.split(":", 1)
        starts.append((idx, dt.datetime.combine(day, dt.time(int(hour), int(minute)))))
    return starts


def _default_race_no(sport, meet, days, now=None):
    race_count = _race_count(sport, meet)
    now = now or dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
    if now.tzinfo is not None:
        now_local = now.astimezone(dt.timezone(dt.timedelta(hours=9))).replace(tzinfo=None)
    else:
        now_local = now
    today = now_local.date().isoformat()
    day_set = {str(day)[:10] for day in (days or [])}
    if today not in day_set:
        first_day = str((days or [today])[0])[:10]
        return race_count if first_day < today else 1

    starts = []
    if sport == "keirin":
        starts = _keirin_default_race_starts(now_local.date())
    else:
        base = dt.datetime.combine(now_local.date(), dt.time(10, 30))
        interval_min = 35 if meet != "제주" else 30
        starts = [
            (race_no, base + dt.timedelta(minutes=(race_no - 1) * interval_min))
            for race_no in range(1, race_count + 1)
        ]

    if not starts:
        return 1
    for race_no, start in starts:
        if now_local <= start:
            return race_no
    return race_count


def _bg_fetch_recent(sport, meet, key, n):
    """백그라운드 스레드에서 recent_race_days 실행 → 캐시 갱신."""
    ck = (sport, meet)
    try:
        days = engine.recent_race_days(sport, meet, key, n=n)
        days = _schedule_days_with_fallback(sport, meet, days, n)
        status = "ok" if days else "empty"
        _RECENT_CACHE[ck] = (days, time.time(), status)
    except Exception:  # noqa: BLE001
        _RECENT_CACHE[ck] = ([], time.time(), "fail")
    finally:
        _RECENT_FETCHING.discard(ck)


def _recent_days_cached(sport, meet, key, n=6):
    """recent_race_days 캐시. 캐시 미스/만료 시 백그라운드 fetch 후 빈 리스트 즉시 반환.

    핵심 안정화: 루트 '/' 접속이 data.go.kr API 동기 대기로 블로킹되던 것을 해결.
    캐시 히트 시 즉시 반환, 미스 시 백그라운드 fetch 시작 후 빈 리스트 반환(논블로킹).
    실패도 짧은 TTL로 캐싱(음수 캐싱)하여 반복 API 호출·타임아웃 방지.
    """
    if not key:
        return _schedule_days_with_fallback(sport, meet, [], n)
    now = time.time()
    ck = (sport, meet)
    hit = _RECENT_CACHE.get(ck)
    if hit:
        days, ts, status = hit
        ttl = (_RECENT_TTL if status == "ok"
               else _RECENT_EMPTY_TTL if status == "empty"
               else _RECENT_FAIL_TTL)
        if (now - ts) < ttl:
            return _schedule_days_with_fallback(sport, meet, days, n)
        # 만료: 캐시값 일단 반환, 백그라운드에서 갱신
        if ck not in _RECENT_FETCHING:
            _RECENT_FETCHING.add(ck)
            threading.Thread(target=_bg_fetch_recent,
                              args=(sport, meet, key, n), daemon=True).start()
        return _schedule_days_with_fallback(sport, meet, days, n)
    # 캐시 미스: 백그라운드 fetch 시작, 즉시 빈 리스트 반환 (논블로킹)
    if ck not in _RECENT_FETCHING:
        _RECENT_FETCHING.add(ck)
        threading.Thread(target=_bg_fetch_recent,
                         args=(sport, meet, key, n), daemon=True).start()
    return _schedule_days_with_fallback(sport, meet, [], n)


def _get_key():
    # env 이름 유연하게: 표준명 우선, 사용자가 다른 이름으로 넣은 경우도 수용
    for name in ("DATAGOKR_SERVICE_KEY", "datagokr", "DATAGOKR", "DATA_GO_KR_KEY", "SERVICE_KEY"):
        v = os.environ.get(name)
        if v and v.strip():
            return v.strip()
    return None


def _has_key():
    return bool(_get_key())


# fetch 가 "그 날짜·경주장에 경주 없음"을 뜻하는 메시지(에러 아님, 안내 대상).
# vs. "API 호출/페이지 실패" 같은 실제 fetch 오류(기존 demo 폴백 유지).
_NO_RACE_MARKERS = (
    "찾지 못했", "출주표 데이터 없음", "출주표 없음", "출주 선수 없음",
    "유효 마번 없음", "출주 두수 없음", "totalCount=0",
)


def _is_no_race(err):
    if not err:
        return False
    return any(m in str(err) for m in _NO_RACE_MARKERS)


def _norm_day(ymd):
    """'2026-06-21' / '20260621' / '2026.06.21' -> 'YYYY-MM-DD' 또는 None.

    8자리가 아니거나 실제 달력 날짜가 아니면(예: 99999999, 20261332) None.
    """
    d = "".join(ch for ch in str(ymd or "") if ch.isdigit())
    if len(d) != 8:
        return None
    try:
        dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))
    except ValueError:
        return None
    return f"{d[0:4]}-{d[4:6]}-{d[6:8]}"


def _is_future_day(ymd):
    """ymd(정상 포맷)가 오늘보다 미래면 True. 형식오류면 False."""
    nd = _norm_day(ymd)
    if not nd:
        return False
    return nd > dt.date.today().isoformat()


def _is_past_day(ymd):
    nd = _norm_day(ymd)
    if not nd:
        return False
    return nd < dt.date.today().isoformat()


def _notice_no_race(sport, meet, race_no, ymd, base):
    """경주가 없을 때 안내 result(kind='notice').

    두 경우를 구분한다:
      (A) 그 날짜 자체가 비경주일  → '비경주일' 안내 + 최근 경주일 칩
      (B) 날짜는 실제 경주일인데 그 경주번호만 없음
          → '그 날은 N개 경주만 열렸으니 1~N 중 선택' 안내
    """
    days = base.get("recent_days") or _recent_days_cached(sport, meet, _get_key())
    nd = _norm_day(ymd)
    # 그 날짜가 최근 경주일 목록에 있으면 = 실제 경주일 → (B) 경주번호 문제
    if nd and days and nd in days:
        title = "해당 경주번호 없음"
        msg = (f"{nd} {meet} 은(는) 실제 경주일이지만 {race_no}R 는 없습니다. "
               "그 날 열린 경주 번호(보통 1R부터) 중에서 다시 선택하세요. "
               "날짜는 그대로 두고 경주 번호만 낮추면 됩니다.")
        return {"kind": "notice", "title": title, "message": msg,
                "recent_days": days}
    # 그 외 = (A) 비경주일
    if days:
        joined = ", ".join(days)
        msg = (f"{ymd} {meet} 에는 경주가 없습니다(비경주일). "
               f"최근 실제 경주일: {joined}. 위 칩을 눌러 날짜를 채운 뒤 다시 예측하세요.")
    else:
        msg = (f"{ymd} {meet} 에 해당 경주가 없습니다. "
               "다른 날짜·경주장·경주번호를 확인하세요.")
    return {"kind": "notice", "title": "해당 날짜·경주장 경주 없음",
            "message": "해당 날짜에는 경주가 없습니다",
            "detail": msg,
            "error_kind": "no_race",
            "recent_days": days}


def _kcycle_official_fallback_result(ymd, meet, race_no, fetch_err=None):
    if _norm_day(ymd) is None:
        return None
    info = {"stnd_yr": engine.norm_ymd(ymd)[:4], "ymd": engine.norm_ymd(ymd),
            "meet": meet, "race_no": race_no}
    signal = engine.kcycle_rankingpredict_signal(info)
    if not signal:
        return None
    order = [str(x) for x in signal.get("order", [])]
    return {
        "kind": "official_fallback",
        "title": "KCYCLE 공식예상 폴백",
        "message": "실시간 출주표 조회가 실패해 데모 경주를 섞지 않고 KCYCLE 공식 예상 신호만 표시합니다.",
        "info": info,
        "signal": signal,
        "order": order,
        "top_bno": str(signal.get("leader")),
        "fetch_err": fetch_err,
    }


@app.route("/")
def index():
    mobile_index = os.path.join(MOBILE_WEB_DIST, "index.html")
    if os.path.isfile(mobile_index):
        return send_from_directory(MOBILE_WEB_DIST, "index.html")

    today = dt.date.today().isoformat()
    key = _get_key()
    # 기본 경주장(경륜 광명) 최근 경주일 → 날짜 기본값을 최근 경주일로.
    default_sport = "keirin"
    default_meet = KEIRIN_MEETS[0]
    recent = _recent_days_cached(default_sport, default_meet, key) if key else []
    default_date = recent[0] if recent else today
    return render_template(
        "index.html",
        disclaimer=DISCLAIMER,
        keirin_meets=KEIRIN_MEETS,
        kra_meets=KRA_MEETS,
        today=today,
        has_key=_has_key(),
        recent_days=recent,
        default_date=default_date,
        schedule_hint=SCHEDULE_HINT,
        result=None,
    )


@app.route("/_expo/<path:asset_path>")
def mobile_web_asset(asset_path):
    return send_from_directory(MOBILE_WEB_DIST, os.path.join("_expo", asset_path))


@app.route("/favicon.ico")
def mobile_web_favicon():
    return send_from_directory(MOBILE_WEB_DIST, "favicon.ico")


@app.route("/metadata.json")
def mobile_web_metadata():
    return send_from_directory(MOBILE_WEB_DIST, "metadata.json")


@app.route("/recent")
def recent():
    """경주장 변경 시 chip 갱신용 JSON. ?sport=&meet= → {days:[...]}.
    키 없거나 실패해도 한국 기준 경주 요일 fallback을 반환한다.
    """
    sport, meet = _normalize_race_context(
        request.args.get("sport") or "keirin",
        request.args.get("meet") or "",
    )
    days = _recent_days_cached(sport, meet, _get_key())
    return jsonify({
        "sport": sport,
        "meet": meet,
        "days": days,
        "default_race_no": _default_race_no(sport, meet, days),
        "race_count": _race_count(sport, meet),
    })


# ── /predict 결과 캐싱 (같은 경주 재요청 시 즉시 응답 — data.go.kr 재호출 방지) ──
# 출주표는 경주 시작 전까지 안 바뀜 → 캐싱 안전. TTL 30분.
_PREDICT_CACHE = {}
_PREDICT_TTL = 1800
# 진행 중인 fetch 중복 방지 (같은 경주 동시 요청 시 1번만 fetch)
_PREDICT_FETCHING = {}
_BASE_PREDICTION_CACHE = {}
_NEGATIVE_BASE_PREDICTION_CACHE = {}
_NEGATIVE_BASE_PREDICTION_TTL = 600
_LIVE_DECISION_PROVIDER_TIMEOUT = 1.5
_LIVE_DECISION_PROVIDER_MAX_PAGES = 1
_LIVE_DECISION_RESULT_TTL = 15

_PREDICT_LOCK = threading.Lock()
_LIVE_DECISION_WORK_LOCK = threading.RLock()
_LIVE_DECISION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-decision")
_LIVE_DECISION_PREWARM_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="official-card-prewarm")
_LIVE_DECISION_FUTURES = {}
_LIVE_DECISION_RESULT_CACHE = {}
_LIVE_DECISION_PREWARM_FUTURES = {}
_LIVE_DECISION_PREWARM_STATUS = {}


def _base_cache_key(sport, ymd, meet, race_no):
    sport, meet = _normalize_race_context(sport, meet)
    return (sport, _norm_day(ymd) or ymd, meet, str(race_no).strip())


def _get_base_prediction_cached(sport, ymd, meet, race_no):
    ck = _base_cache_key(sport, ymd, meet, race_no)
    hit = _BASE_PREDICTION_CACHE.get(ck)
    if hit and (time.time() - hit["ts"]) < _PREDICT_TTL:
        return hit["out"]
    return None


def _set_base_prediction_cache(sport, ymd, meet, race_no, out):
    if not out or "error" in out:
        return
    ck = _base_cache_key(sport, ymd, meet, race_no)
    _BASE_PREDICTION_CACHE[ck] = {"out": out, "ts": time.time()}


def _get_negative_base_prediction_cached(sport, ymd, meet, race_no):
    ck = _base_cache_key(sport, ymd, meet, race_no)
    hit = _NEGATIVE_BASE_PREDICTION_CACHE.get(ck)
    if hit and (time.time() - hit["ts"]) < _NEGATIVE_BASE_PREDICTION_TTL:
        return hit["out"]
    return None


def _set_negative_base_prediction_cache(sport, ymd, meet, race_no, out):
    if not out or out.get("error_kind") != "no_race":
        return
    ck = _base_cache_key(sport, ymd, meet, race_no)
    _NEGATIVE_BASE_PREDICTION_CACHE[ck] = {"out": out, "ts": time.time()}


def _live_decision_fast_path_enabled(ymd):
    """과거 날짜는 레거시 풀 예산으로 계산해야 settled 분류가 가능하다.
    역순 1페이지 축소 예산으로는 과거 카드를 못 찾아 no_race로 오판하므로
    fast path(축소 예산 + negative cache)는 오늘(KST) 이후 날짜에만 적용한다."""
    digits = "".join(ch for ch in str(ymd or "") if ch.isdigit())
    if len(digits) < 8:
        return True
    return digits[:8] >= _today_kst().strftime("%Y%m%d")


def _compute_base_prediction_cached(sport, ymd, meet, race_no, key, live_decision=False):
    live_decision = live_decision and _live_decision_fast_path_enabled(ymd)
    if live_decision:
        negative_hit = _get_negative_base_prediction_cached(sport, ymd, meet, race_no)
        if negative_hit is not None:
            return negative_hit
    hit = _get_base_prediction_cached(sport, ymd, meet, race_no)
    if hit is not None:
        return hit
    if live_decision:
        out = _compute_base_prediction(sport, ymd, meet, race_no, key, live_decision=True)
    else:
        out = _compute_base_prediction(sport, ymd, meet, race_no, key)
    if live_decision:
        _set_negative_base_prediction_cache(sport, ymd, meet, race_no, out)
    _set_base_prediction_cache(sport, ymd, meet, race_no, out)
    return out


_LIVE_DECISION_RELEASE_ERROR_KINDS = {
    "missing_api_key",
    "invalid_request",
    "invalid_date",
    "background_task_failed",
    "unsupported_meet",
    "no_race",
    "upstream_api_error",
    "base_prediction_error",
    "roster_mismatch",
}


def _should_release_live_decision_quota(result):
    if not isinstance(result, dict):
        return True
    if result.get("status") in {"blocked", "rate_limited", "settled"}:
        return False
    error_kind = str(result.get("error_kind") or "")
    if error_kind in _LIVE_DECISION_RELEASE_ERROR_KINDS:
        return True
    return (
        result.get("decision") == "hold"
        and not result.get("rows")
        and not result.get("top")
        and not result.get("market_used")
    )


def _live_decision_task_key(sport, ymd, meet, race_no):
    return _base_cache_key(sport, ymd, meet, race_no)


def _get_live_decision_result(task_key):
    hit = _LIVE_DECISION_RESULT_CACHE.get(task_key)
    if hit and (time.time() - hit["ts"]) < _LIVE_DECISION_RESULT_TTL:
        return hit["result"]
    _LIVE_DECISION_RESULT_CACHE.pop(task_key, None)
    return None


def _compute_live_decision_task(sport, ymd, meet, race_no, key):
    base_out = _compute_base_prediction_cached(sport, ymd, meet, race_no, key, live_decision=True)
    return engine.compute_live_decision(sport, ymd, meet, race_no, base_model_out=base_out)


def _live_decision_failure_result(_error):
    message = "예측 오류가 발생했습니다. 잠시 후 다시 시도하세요."
    return {
        "ok": False,
        "status": "hold",
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "odds_age_sec": None,
        "market_odds": [],
        "top": None,
        "rows": [],
        "decision": "hold",
        "market_used": False,
        "snapshot_phase": "failed",
        "error_kind": "background_task_failed",
        "market_risk": {
            "level": "odds_unavailable",
            "title": "분석 데이터 오류",
            "message": message,
        },
    }


def _cache_live_decision_result(task_key, future):
    if future.cancelled():
        with _LIVE_DECISION_WORK_LOCK:
            _LIVE_DECISION_FUTURES.pop(task_key, None)
        return
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("live decision background task failed: %s", exc)
        with _LIVE_DECISION_WORK_LOCK:
            _LIVE_DECISION_FUTURES.pop(task_key, None)
            _LIVE_DECISION_RESULT_CACHE[task_key] = {"result": _live_decision_failure_result(exc), "ts": time.time()}
        return
    with _LIVE_DECISION_WORK_LOCK:
        _LIVE_DECISION_FUTURES.pop(task_key, None)
        _LIVE_DECISION_RESULT_CACHE[task_key] = {"result": result, "ts": time.time()}


def _run_live_decision_with_budget(sport, ymd, meet, race_no, key):
    if app.config.get("TESTING"):
        return _compute_live_decision_task(sport, ymd, meet, race_no, key), None

    task_key = _live_decision_task_key(sport, ymd, meet, race_no)
    with _LIVE_DECISION_WORK_LOCK:
        cached = _get_live_decision_result(task_key)
        if cached is not None:
            return cached, None
        active = _LIVE_DECISION_FUTURES.get(task_key)
        if active is not None:
            return None, "in_progress"
        future = _LIVE_DECISION_EXECUTOR.submit(
            _compute_live_decision_task,
            sport,
            ymd,
            meet,
            race_no,
            key,
        )
        _LIVE_DECISION_FUTURES[task_key] = future
        future.add_done_callback(lambda completed: _cache_live_decision_result(task_key, completed))
    return None, "pending"


def _prewarm_official_entry_cards(sport, ymd, meet, race_nos, key):
    """모델 계산 없이 원본 출전표 캐시만 한 번 적재한다."""
    if not key:
        return {"warmed": 0, "reason": "missing_api_key"}
    if sport == "keirin":
        return engine.prewarm_keirin_card_pages(int(ymd[:4]), key, max_pages=1)
    warmed = 0
    for race_no in race_nos:
        engine.fetch_kra_card(
            ymd,
            meet,
            race_no,
            key,
            max_pages=1,
            timeout=_LIVE_DECISION_PROVIDER_TIMEOUT,
        )
        warmed += 1
    return {"warmed": warmed}


def _cache_live_decision_prewarm(task_key, future):
    now = time.time()
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001
        app.logger.warning("official card prewarm failed: %s", exc)
        status = {
            "state": "failed",
            "updated_at": now,
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        status = {"state": "ready", "updated_at": now, "result": result or {}}
    with _LIVE_DECISION_WORK_LOCK:
        _LIVE_DECISION_PREWARM_FUTURES.pop(task_key, None)
        _LIVE_DECISION_PREWARM_STATUS[task_key] = status


def _enqueue_live_decision_prewarm(sport, ymd, meet, race_nos, key):
    """경기 선택 전 원본 출전표만 단일 작업으로 예열한다."""
    task_key = (sport, ymd, meet)
    with _LIVE_DECISION_WORK_LOCK:
        active = _LIVE_DECISION_PREWARM_FUTURES.get(task_key)
        if active is not None and not active.done():
            return {"state": "warming"}
        future = _LIVE_DECISION_PREWARM_EXECUTOR.submit(
            _prewarm_official_entry_cards,
            sport,
            ymd,
            meet,
            race_nos,
            key,
        )
        _LIVE_DECISION_PREWARM_FUTURES[task_key] = future
        _LIVE_DECISION_PREWARM_STATUS[task_key] = {"state": "warming", "updated_at": time.time()}
        future.add_done_callback(lambda completed: _cache_live_decision_prewarm(task_key, completed))
    return {"state": "warming"}


def _live_decision_pending_response(session, data_layer, ymd, pending_reason):
    message = "공식 출전표를 확인하고 있습니다. 잠시 후 자동으로 갱신됩니다."
    return {
        "ok": True,
        "status": "hold",
        "message": message,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "odds_age_sec": None,
        "market_odds": [],
        "top": None,
        "rows": [],
        "decision": "hold",
        "market_used": False,
        "snapshot_phase": pending_reason,
        "poll_delay_ms": 3000,
        "market_risk": {
            "level": "official_data_pending",
            "title": "공식 출전표 확인 중",
            "message": message,
        },
        "app_session": session,
        "data_layer": data_layer,
        "race_date": ymd,
    }


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect("/", code=301)
    f = request.values
    sport, meet = _request_race_context(
        f.get("sport") or "keirin",
        f.get("meet") or "광명",
    )
    ymd = (f.get("date") or "").strip()
    race_no = (f.get("race_no") or "1").strip()
    today = dt.date.today().isoformat()

    key0 = _get_key()
    recent = _recent_days_cached(sport, meet, key0) if key0 else []
    base = dict(
        disclaimer=DISCLAIMER,
        keirin_meets=KEIRIN_MEETS,
        kra_meets=KRA_MEETS,
        today=today,
        has_key=_has_key(),
        recent_days=recent,
        default_date=(recent[0] if recent else today),
        schedule_hint=SCHEDULE_HINT,
        sport=sport, date=ymd, meet=meet, race_no=race_no,
    )

    # ── /predict 결과 캐시 확인 (data.go.kr 재호출 방지 — 핵심 안정화) ──
    ck = (sport, ymd, meet, race_no)
    now = time.time()
    cached = _PREDICT_CACHE.get(ck)
    if cached and (now - cached["ts"]) < cached.get("ttl", _PREDICT_TTL):
        return cached["rendered"]  # 캐시된 HTML 즉시 반환

    # ── 경마(KRA) ──
    if sport == "horse":
        return _predict_horse(ymd, meet, race_no, base, cache_key=ck)

    # ── 경륜 ──
    starters = None
    info = {}
    src = "live"
    key = _get_key()

    if not key:
        return render_template(
            "index.html",
            result={"kind": "error",
                    "title": "DATAGOKR_SERVICE_KEY 미설정",
                    "message": "DATAGOKR_SERVICE_KEY가 설정되지 않았습니다",
                    "error_kind": "missing_api_key"},
            **base)
    else:
        if not ymd:
            return render_template(
                "index.html",
                result={"kind": "error", "title": "입력 오류",
                        "message": "날짜를 입력하세요."},
                **base)
        if _norm_day(ymd) is None:
            return render_template(
                "index.html",
                result={"kind": "error", "title": "날짜 형식 오류",
                        "message": f"'{ymd}' 는 올바른 날짜가 아닙니다. "
                                   "YYYY-MM-DD 형식으로 입력하세요."},
                **base)
        # 미래(다가올) 경주라도 하드 차단하지 않는다. 출주표 fetch 를 항상 시도해
        # 카드가 있으면 사전 예측을 수행한다(아래 is_future 플래그로 라벨 표시).
        stnd_yr = engine.norm_ymd(ymd)[:4]
        starters, err = engine.fetch_race_card(stnd_yr, ymd, meet, race_no, key)
        if err and _is_no_race(err):
            # 경주 0건(에러 아님) → 조용한 demo 대신 안내(notice).
            return render_template(
                "index.html",
                result=_notice_no_race(sport, meet, race_no, ymd, base),
                **base)
        if err:
            fallback = _kcycle_official_fallback_result(ymd, meet, race_no, err)
            if fallback:
                return _cache_and_return(ck, render_template("index.html", result=fallback, **base), ttl=60)
            return render_template(
                "index.html",
                result={"kind": "error", "title": "출주표 조회 실패",
                        "message": "상류 API 장애로 출주표를 가져오지 못했습니다",
                        "error_kind": "upstream_api_error",
                        "detail": err},
                **base)
        else:
            info = {"stnd_yr": stnd_yr, "ymd": engine.norm_ymd(ymd),
                    "meet": meet, "race_no": race_no}
            roster_verification, roster_block = _verify_roster_or_block("keirin", ymd, meet, race_no, starters)
            if roster_block:
                return render_template("index.html", result=roster_block, **base)

    # ── 채점 + 7권종 픽 ──
    try:
        out = engine.predict(starters, meta=info)
    except Exception as e:  # noqa: BLE001  (앱이 죽지 않도록 graceful)
        return render_template(
            "index.html",
            result={"kind": "error", "title": "예측 중 오류",
                    "message": _public_exception_message("예측 오류", e)},
            **base)

    if "error" in out:
        return render_template(
            "index.html",
            result={"kind": "error", "title": "채점 실패",
                    "message": out["error"]},
            **base)

    out["kind"] = "ok"
    out["src"] = src
    out["info"] = info
    out["roster_verification"] = roster_verification
    # 다가올(미래) 경주 = 사전 예측. 실거래 결과는 아직 없음을 라벨로 표시.
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    _set_base_prediction_cache(sport, ymd, meet, race_no, out)
    return _cache_and_return(ck, render_template("index.html", result=out, **base))


def _cache_and_return(cache_key, rendered, ttl=None):
    """성공 결과만 캐싱. error/notice는 캐싱하지 않음 (다음 요청에서 재시도)."""
    _PREDICT_CACHE[cache_key] = {"rendered": rendered, "ts": time.time()}
    if ttl is not None:
        _PREDICT_CACHE[cache_key]["ttl"] = ttl
    return rendered


def _predict_horse(ymd, meet, race_no, base, cache_key=None):
    # 경마 기본 경주장 보정 (경륜 meet '광명'이 넘어온 경우 서울로)
    if meet not in KRA_MEETS:
        meet = KRA_MEETS[0]
        base["meet"] = meet

    starters = None
    info = {}
    src = "live"
    key = _get_key()

    if not key:
        return render_template(
            "index.html",
            result={"kind": "error",
                    "title": "DATAGOKR_SERVICE_KEY 미설정",
                    "message": "DATAGOKR_SERVICE_KEY가 설정되지 않았습니다",
                    "error_kind": "missing_api_key"},
            **base)
    else:
        if not ymd:
            return render_template(
                "index.html",
                result={"kind": "error", "title": "입력 오류",
                        "message": "날짜를 입력하세요."},
                **base)
        if _norm_day(ymd) is None:
            return render_template(
                "index.html",
                result={"kind": "error", "title": "날짜 형식 오류",
                        "message": f"'{ymd}' 는 올바른 날짜가 아닙니다. "
                                   "YYYY-MM-DD 형식으로 입력하세요."},
                **base)
        # 미래(다가올) 경마라도 하드 차단하지 않는다. 출주표 fetch 를 항상 시도해
        # 카드가 있으면 사전 예측을 수행한다(is_future 플래그로 라벨 표시).
        starters, err = engine.fetch_kra_card(ymd, meet, race_no, key)
        if err and _is_no_race(err):
            # 경주 0건(에러 아님) → 조용한 demo 대신 안내(notice).
            return render_template(
                "index.html",
                result=_notice_no_race("horse", meet, race_no, ymd, base),
                **base)
        if err:
            return render_template(
                "index.html",
                result={"kind": "error", "title": "출주표 조회 실패",
                        "message": "상류 API 장애로 출주표를 가져오지 못했습니다",
                        "error_kind": "upstream_api_error",
                        "detail": err},
                **base)
        else:
            info = {"stnd_yr": str(engine.norm_ymd(ymd))[:4],
                    "ymd": engine.norm_ymd(ymd), "meet": meet, "race_no": race_no}
            info.update(engine.kra_odds_snapshot_metadata(starters, ymd, meet, race_no))
            _record_fresh_kra_odds_snapshot(starters, info)
            roster_verification, roster_block = _verify_roster_or_block("horse", ymd, meet, race_no, starters)
            if roster_block:
                return render_template("index.html", result=roster_block, **base)

    try:
        out = engine.predict_kra(starters, meta=info)
    except Exception as e:  # noqa: BLE001
        return render_template(
            "index.html",
            result={"kind": "error", "title": "예측 중 오류",
                    "message": _public_exception_message("예측 오류", e)},
            **base)

    if "error" in out:
        return render_template(
            "index.html",
            result={"kind": "error", "title": "채점 실패",
                    "message": out["error"]},
            **base)

    out["kind"] = "ok"
    out["src"] = src
    out["info"] = info
    out["sport_label"] = "경마(KRA)"
    out["roster_verification"] = roster_verification
    settled = engine.kra_result_from_starters(starters) if _is_past_day(ymd) else None
    if settled:
        out["_kra_result"] = settled
    # 다가올(미래) 경마 = 사전 예측. 실거래 결과는 아직 없음을 라벨로 표시.
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    _set_base_prediction_cache("horse", ymd, meet, race_no, out)
    rendered = render_template("index.html", result=out, **base)
    if cache_key is not None:
        return _cache_and_return(cache_key, rendered)
    return rendered


@app.route("/health")
@app.route("/healthz")
def healthz():
    deep = str(request.args.get("deep", "")).strip() == "1"
    if deep:
        model, err = engine.load_model()
        cross, cross_err = engine.load_cross_model()
        kra, kra_err = engine.load_kra_model()
        ok = err is None and cross_err is None and kra_err is None
        return {"ok": ok,
                "keirin_model": ("loaded" if model else "fail"), "keirin_err": err,
                "keirin_cross_model": ("loaded" if cross else "fail"), "keirin_cross_err": cross_err,
                "kra_model": ("loaded" if kra else "fail"), "kra_err": kra_err,
                "rankingpredict_cache": engine.kcycle_rankingpredict_cache_status(),
                "keirin_card_page_cache": engine.keirin_card_page_cache_status(),
                "has_key": _has_key(), "deep": True,
                "entitlement_mode": datastore.entitlement_mode()}, (200 if ok else 500)
    model_ok = os.path.exists(engine.MODEL_PATH)
    cross_ok = os.path.exists(engine.CROSS_MODEL_PATH)
    kra_ok = os.path.exists(engine.KRA_MODEL_PATH)
    ok = model_ok and cross_ok and kra_ok
    return {"ok": ok,
            "keirin_model": "present" if model_ok else "missing",
            "keirin_err": None if model_ok else "model file missing",
            "keirin_cross_model": "present" if cross_ok else "missing",
            "keirin_cross_err": None if cross_ok else "cross model file missing",
            "kra_model": "present" if kra_ok else "missing",
            "kra_err": None if kra_ok else "kra model file missing",
            "rankingpredict_cache": engine.kcycle_rankingpredict_cache_status(),
            "keirin_card_page_cache": engine.keirin_card_page_cache_status(),
            "has_key": _has_key(), "deep": False,
            "entitlement_mode": datastore.entitlement_mode()}, (200 if ok else 500)


@app.route("/api/app-data-layer")
def api_app_data_layer():
    expected = (os.environ.get("RACELENS_ADMIN_TOKEN") or "").strip()
    provided = (
        request.headers.get("RACELENS_ADMIN_TOKEN")
        or request.headers.get("X-RaceLens-Admin-Token")
        or ""
    ).strip()
    if not expected or provided != expected:
        return jsonify({"error": "not_found", "status": 404}), 404
    return jsonify(datastore.app_data_layer_status())


UX_EVENT_NAMES = {
    "app_open",
    "screen_view",
    "tab_select",
    "race_context_change",
    "analysis_request",
    "analysis_result",
    "analysis_error",
    "live_odds_refresh",
    "rewarded_ad_credit",
}
UX_PAYLOAD_KEYS = {
    "tab",
    "previousTab",
    "sport",
    "raceNo",
    "marketUsed",
    "marketRiskLevel",
    "top1Pct",
    "trifectaPct",
    "latencyMs",
    "pollDelayMs",
    "errorKind",
}
UX_FORBIDDEN_PAYLOAD_KEYS = {"name", "participant", "selection", "deviceId", "userId", "meet"}


def _short_text(value, limit=96):
    return str(value or "").strip()[:limit]


def _contains_forbidden_payload_key(value) -> bool:
    if isinstance(value, dict):
        return any(str(k) in UX_FORBIDDEN_PAYLOAD_KEYS or _contains_forbidden_payload_key(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_payload_key(item) for item in value)
    return False


def _sanitize_ux_payload(payload: dict) -> dict:
    clean = {}
    for key, value in payload.items():
        if key not in UX_PAYLOAD_KEYS:
            raise ValueError(f"unsupported payload field: {key}")
        if key in {"tab", "previousTab"}:
            if value in {"home", "analyze", "pro"}:
                clean[key] = value
        elif key == "sport":
            if value in {"keirin", "horse"}:
                clean[key] = value
        elif key == "raceNo":
            if isinstance(value, int) and 1 <= value <= 20:
                clean[key] = value
        elif key in {"marketUsed"}:
            if isinstance(value, bool):
                clean[key] = value
        elif key == "marketRiskLevel":
            clean[key] = _short_text(value, 32)
        elif key in {"top1Pct", "trifectaPct"}:
            if isinstance(value, (int, float)):
                clean[key] = max(0, min(100, round(float(value), 2)))
        elif key in {"latencyMs", "pollDelayMs"}:
            if isinstance(value, (int, float)):
                clean[key] = max(0, min(60000, int(round(float(value)))))
        elif key == "errorKind":
            if value in {"api_error", "unknown"}:
                clean[key] = value
    return clean


@app.route("/api/ux-events", methods=["POST", "OPTIONS"])
def api_ux_events():
    if request.method == "OPTIONS":
        return "", 204
    if (request.content_length or 0) > 8192:
        return jsonify({"ok": False, "error": "payload_too_large"}), 413

    raw = request.get_data(cache=True)
    if len(raw) > 8192:
        return jsonify({"ok": False, "error": "payload_too_large"}), 413
    event = request.get_json(silent=True)
    if not isinstance(event, dict):
        return jsonify({"ok": False, "error": "invalid_json"}), 400

    event_name = _short_text(event.get("name"), 48)
    if event_name not in UX_EVENT_NAMES:
        return jsonify({"ok": False, "error": "unsupported_event"}), 400

    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "invalid_payload"}), 400
    if _contains_forbidden_payload_key(payload):
        return jsonify({"ok": False, "error": "forbidden_payload_field"}), 400

    try:
        clean_payload = _sanitize_ux_payload(payload)
    except ValueError as exc:
        return jsonify({"ok": False, "error": "unsupported_payload_field", "detail": str(exc)}), 400

    clean_payload.update({
        "app": "racelens",
        "appVersion": _short_text(event.get("version"), 32),
        "clientTimestamp": _short_text(event.get("timestamp"), 40),
        "sessionId": _short_text(event.get("sessionId"), 96),
    })
    try:
        json.dumps(clean_payload, ensure_ascii=False)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "invalid_payload"}), 400

    anonymous_id = _short_text(event.get("anonymousId"), 96) or "anonymous-app"
    platform = _short_text(event.get("platform"), 32) or "unknown"
    session, data_layer = datastore.record_ux_event_safely(anonymous_id, platform, event_name, clean_payload)
    return jsonify({
        "ok": True,
        "stored": bool(data_layer.get("ready")),
        "app_session": session,
        "data_layer": data_layer,
    }), 202


@app.errorhandler(500)
def handle_500(e):
    return jsonify(_safe_error_payload(e, 500)), 500


@app.errorhandler(Exception)
def handle_all_errors(e):
    if isinstance(e, HTTPException):
        status_code = e.code or 500
        name = (e.name or "http_error").lower().replace(" ", "_")
        return jsonify({"error": name, "status": status_code}), status_code
    return jsonify(_safe_error_payload(e, 500)), 500


# ───────────────────────── 실시간 판단 API ─────────────────────────

@app.route("/api/app-session")
def api_app_session():
    device_id = _request_device_id()
    ip_address = _request_ip()
    platform = (request.headers.get("X-RaceLens-Platform") or "unknown").strip()[:32]
    session, data_layer = datastore.ensure_app_session_safely(device_id, platform, ip_address=ip_address)
    return jsonify({
        "ok": True,
        "app_session": session,
        "data_layer": data_layer,
    })


@app.route("/api/rewarded-ad/claim", methods=["POST", "OPTIONS"])
def api_rewarded_ad_claim():
    if request.method == "OPTIONS":
        return "", 204
    if not datastore.rewarded_ads_enabled():
        return jsonify({"ok": False, "error": "rewarded_ads_disabled"}), 503
    return jsonify({"ok": False, "error": "ssv_required"}), 410


@app.route("/api/rewarded-ad/ssv")
def api_rewarded_ad_ssv():
    if not datastore.rewarded_ads_enabled():
        return "rewarded ads disabled", 503
    expected_ad_unit = (os.environ.get("RACELENS_ADMOB_REWARDED_AD_UNIT_ID") or "").strip()
    if not expected_ad_unit:
        return "rewarded ad unit not configured", 503
    try:
        reward = admob_ssv.verify_callback(request.query_string, expected_ad_unit=expected_ad_unit)
    except admob_ssv.SsvVerificationError as exc:
        app.logger.warning("Rejected AdMob SSV callback: %s", exc)
        return "invalid callback", 400
    if reward.ad_unit == admob_ssv.ADMOB_UI_VERIFICATION_AD_UNIT:
        return "", 200
    rewarded_at = dt.datetime.fromtimestamp(reward.timestamp_ms / 1000, tz=dt.UTC).isoformat(timespec="seconds")
    _session, _granted, _duplicate, data_layer = datastore.claim_verified_rewarded_ad_credit_safely(
        reward.device_id,
        "admob_ssv",
        reward.transaction_id,
        reward.ad_unit,
        reward.reward_amount,
        reward.reward_item,
        rewarded_at,
    )
    if not data_layer.get("ready"):
        return "reward persistence unavailable", 503
    return "", 200


def _store_platform(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ios", "apple", "app_store"}:
        return "ios"
    if normalized in {"android", "google", "google_play", "play"}:
        return "android"
    return normalized[:32]


@app.route("/api/iap/verify", methods=["POST", "OPTIONS"])
def api_iap_verify():
    if request.method == "OPTIONS":
        return "", 204
    if (request.content_length or 0) > 65536:
        return jsonify({"ok": False, "reason": "payload_too_large"}), 413
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "reason": "invalid_json"}), 400
    platform = _store_platform(str(payload.get("platform") or ""))
    receipt = str(payload.get("receipt") or "").strip()
    if platform not in {"ios", "android"} or not receipt:
        return jsonify({"ok": False, "reason": "invalid_request"}), 400
    if not iap.verifier_configured(platform):
        return jsonify({"ok": False, "reason": "not_configured"})

    verification = iap.verify_receipt_with_store(platform, receipt)
    if not verification.ok:
        return jsonify({"ok": False, "reason": verification.reason})

    device_id = _request_device_id()
    request_platform = (request.headers.get("X-RaceLens-Platform") or platform).strip()[:32]
    session, _data_layer = datastore.ensure_app_session_safely(device_id, request_platform, ip_address=_request_ip())
    datastore.upsert_subscription(
        user_id=str(session["user_id"]),
        platform=platform,
        product_id=verification.product_id,
        status=verification.status,
        expires_at=verification.expires_at,
    )
    refreshed, data_layer = datastore.ensure_app_session_safely(device_id, request_platform, ip_address=_request_ip())
    return jsonify({
        "ok": True,
        "reason": verification.reason,
        "app_session": refreshed,
        "data_layer": data_layer,
    })


@app.route("/api/live-decision")
def api_live_decision():
    """실시간 판단 JSON. /predict HTML 캐시와 분리.
    ?sport=keirin&date=2026-06-28&meet=광명&race_no=5
    → {ok, status, message, updated_at, odds_age_sec, market_odds, top, rows, decision, market_used, snapshot_phase}
    """
    f = request.values
    sport, meet = _request_race_context(
        f.get("sport") or "keirin",
        f.get("meet") or "광명",
    )
    ymd = (f.get("date") or "").strip()
    race_no = (f.get("race_no") or "1").strip()

    if not ymd:
        return jsonify({"ok": False, "status": "hold",
                        "message": "날짜 필요",
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "market_used": False, "decision": "hold",
                        "snapshot_phase": "unknown"}), 400

    device_id = _request_device_id()
    ip_address = _request_ip()
    platform = (request.headers.get("X-RaceLens-Platform") or "unknown").strip()[:32]
    rate_allowed, rate_cap = datastore.check_live_decision_ip_rate_limit(ip_address)
    if not rate_allowed:
        return jsonify({
            "ok": False,
            "status": "rate_limited",
            "message": "요청이 너무 많습니다. 잠시 후 다시 시도하세요.",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "market_used": False,
            "decision": "blocked",
            "snapshot_phase": "rate_limited",
            "retry_after_sec": 60,
            "rate_limit": {"scope": "ip_per_minute", "limit": rate_cap},
        }), 429
    session, allowed, claim_data_layer = datastore.claim_live_decision_session_safely(device_id, platform, ip_address=ip_address)
    if not allowed:
        return jsonify({
            "ok": False,
            "status": "blocked",
            "message": "무료 분석 3회를 모두 사용했습니다. Pro는 무제한입니다.",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "odds_age_sec": None,
            "market_odds": [],
            "top": None,
            "rows": [],
            "decision": "blocked",
            "market_used": False,
            "snapshot_phase": "quota_exhausted",
            "poll_delay_ms": 60000,
            "market_risk": {
                "level": "blocked",
                "title": "무료 분석 한도 종료",
                "message": "오늘 무료 분석 3회를 모두 사용했습니다. Pro 권한이 확인되면 무제한 분석이 열립니다.",
            },
            "app_session": session,
            "data_layer": claim_data_layer,
        })

    key0 = _get_key()
    try:
        result, pending_reason = _run_live_decision_with_budget(sport, ymd, meet, race_no, key0)
    except Exception as e:
        session, release_data_layer = datastore.release_live_decision_session_safely(device_id, platform, session)
        return jsonify({"ok": False, "status": "hold",
                        "message": _public_exception_message("예측 오류", e),
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "market_used": False, "decision": "hold",
                        "snapshot_phase": "unknown",
                        "app_session": session,
                        "data_layer": release_data_layer}), 500

    if pending_reason:
        session, release_data_layer = datastore.release_live_decision_session_safely(device_id, platform, session)
        return jsonify(_live_decision_pending_response(session, release_data_layer, ymd, pending_reason))

    result["race_date"] = ymd
    if _should_release_live_decision_quota(result):
        session, _ = datastore.release_live_decision_session_safely(device_id, platform, session)
    engine.attach_participant_explanations(result, sport, pro_enabled=session.get("entitlement") == "pro")
    context = {"sport": sport, "date": ymd, "meet": meet, "race_no": race_no}
    session, data_layer = datastore.record_live_decision_safely(device_id, platform, context, result)
    result["app_session"] = session
    result["data_layer"] = data_layer
    return jsonify(result)


@app.route("/api/live-decisions/preload", methods=["GET", "POST"])
def api_live_decisions_preload():
    """선택 전 전체 경기를 제한된 워커로 예열한다."""
    f = request.values
    sport, meet = _request_race_context(f.get("sport") or "keirin", f.get("meet") or "광명")
    ymd = _norm_day(f.get("date") or "")
    if not ymd:
        return jsonify({"ok": False, "message": "유효한 날짜 필요"}), 400
    try:
        requested_count = int(f.get("race_count") or _race_count(sport, meet))
    except ValueError:
        requested_count = _race_count(sport, meet)
    race_count = max(1, min(_race_count(sport, meet), requested_count))
    priority_race_no = str(f.get("priority_race_no") or "").strip()
    race_nos = tuple(str(number) for number in range(1, race_count + 1))
    if priority_race_no in race_nos:
        race_nos = (priority_race_no,) + tuple(number for number in race_nos if number != priority_race_no)
    task_key = (sport, ymd, meet)
    if request.method == "GET":
        with _LIVE_DECISION_WORK_LOCK:
            prewarm = _LIVE_DECISION_PREWARM_STATUS.get(task_key, {"state": "idle"}).copy()
        prewarm["cache"] = engine.keirin_card_page_cache_status() if sport == "keirin" else {}
        return jsonify({"ok": True, "race_date": ymd, "prewarm": prewarm})
    prewarm = _enqueue_live_decision_prewarm(sport, ymd, meet, race_nos, _get_key())
    return jsonify({
        "ok": True,
        "status": "warming",
        "race_date": ymd,
        "race_nos": [int(number) for number in race_nos],
        "prewarm": prewarm,
        "poll_delay_ms": 750,
    }), 202


def _compute_base_prediction(sport, ymd, meet, race_no, key, live_decision=False):
    """app.py 내부용: /predict 로직을 재사용해 base 모델 결과 dict 반환.
    HTML 렌더링 없이 engine.predict() / engine.predict_kra() 결과만 반환.
    """
    sport = _canonical_sport(sport)
    meet = str(meet or "").strip() or _default_meet(sport)
    today = dt.date.today().isoformat()
    base = dict(
        disclaimer=DISCLAIMER, keirin_meets=KEIRIN_MEETS, kra_meets=KRA_MEETS,
        today=today, has_key=bool(key), recent_days=[],
        default_date=today, schedule_hint=SCHEDULE_HINT,
        sport=sport, date=ymd, meet=meet, race_no=race_no,
    )

    if sport == "horse":
        return _base_predict_horse(ymd, meet, race_no, base, live_decision=live_decision)

    # ── 경륜 ──
    starters = None
    info = {}
    src = "live"

    if not ymd:
        return _base_error("invalid_request", "날짜 필요")
    if _norm_day(ymd) is None:
        return _base_error("invalid_date", f"날짜 형식 오류: {ymd}")
    if meet not in KEIRIN_MEETS:
        return _base_error("unsupported_meet", "지원하지 않는 경주장입니다", meet=meet)
    if not key:
        return _base_error("missing_api_key", "DATAGOKR_SERVICE_KEY가 설정되지 않았습니다")
    else:
        stnd_yr = engine.norm_ymd(ymd)[:4]
        fetch_kwargs = {}
        if live_decision:
            fetch_kwargs = {
                "max_pages": _LIVE_DECISION_PROVIDER_MAX_PAGES,
                "timeout": _LIVE_DECISION_PROVIDER_TIMEOUT,
            }
        starters, err = engine.fetch_race_card(stnd_yr, ymd, meet, race_no, key, **fetch_kwargs)
        if err and _is_no_race(err):
            message = "출주표 미공개 — 경주일 아님/카드 미발표" if live_decision else "해당 날짜에는 경주가 없습니다"
            return _base_error("no_race", message, detail=str(err))
        if err:
            return _base_error("upstream_api_error", "상류 API 장애로 출주표를 가져오지 못했습니다", detail=str(err))
        else:
            info = {"stnd_yr": stnd_yr, "ymd": engine.norm_ymd(ymd), "meet": meet, "race_no": race_no}
            roster_verification, roster_block = _verify_roster_or_block("keirin", ymd, meet, race_no, starters)
            if roster_block:
                return {
                    "error": "roster_mismatch",
                    "status": "roster_mismatch",
                    "message": ROSTER_MISMATCH_MESSAGE,
                    "roster_verification": roster_verification,
                }

    try:
        out = engine.predict(starters, meta=info)
    except Exception as e:
        return {"error": _public_exception_message("예측 오류", e)}

    if "error" in out:
        return out
    out["_participant_sources"] = engine.participant_sources_from_starters(starters, "keirin")
    out["kind"] = "ok"; out["src"] = src; out["info"] = info
    out["roster_verification"] = roster_verification
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    return out


def _base_predict_horse(ymd, meet, race_no, base, live_decision=False):
    """경마 base 예측 (engine.predict_kra 재사용)."""
    if meet not in KRA_MEETS:
        meet = KRA_MEETS[0]; base["meet"] = meet
    starters = None; info = {}; src = "live"
    key = _get_key()
    if not key:
        return _base_error("missing_api_key", "DATAGOKR_SERVICE_KEY가 설정되지 않았습니다")
    else:
        if not ymd:
            return _base_error("invalid_request", "날짜 필요")
        if _norm_day(ymd) is None:
            return _base_error("invalid_date", "날짜 형식 오류")
        fetch_kwargs = {}
        if live_decision:
            fetch_kwargs = {
                "max_pages": _LIVE_DECISION_PROVIDER_MAX_PAGES,
                "timeout": _LIVE_DECISION_PROVIDER_TIMEOUT,
            }
        starters, err = engine.fetch_kra_card(ymd, meet, race_no, key, **fetch_kwargs)
        if err and _is_no_race(err):
            message = "출주표 미공개 — 경주일 아님/카드 미발표" if live_decision else "해당 날짜에는 경주가 없습니다"
            return _base_error("no_race", message, detail=str(err))
        if err:
            return _base_error("upstream_api_error", "상류 API 장애로 출주표를 가져오지 못했습니다", detail=str(err))
        else:
            info = {"stnd_yr": str(engine.norm_ymd(ymd))[:4], "ymd": engine.norm_ymd(ymd),
                    "meet": meet, "race_no": race_no}
            info.update(engine.kra_odds_snapshot_metadata(starters, ymd, meet, race_no))
            _record_fresh_kra_odds_snapshot(starters, info)
            roster_verification, roster_block = _verify_roster_or_block("horse", ymd, meet, race_no, starters)
            if roster_block:
                return {
                    "error": "roster_mismatch",
                    "status": "roster_mismatch",
                    "message": ROSTER_MISMATCH_MESSAGE,
                    "roster_verification": roster_verification,
                }
    try:
        out = engine.predict_kra(starters, meta=info)
    except Exception as e:
        return {"error": _public_exception_message("예측 오류", e)}
    if "error" in out:
        return out
    out["_participant_sources"] = engine.participant_sources_from_starters(starters, "horse")
    out["_kra_starters"] = starters
    out["kind"] = "ok"; out["src"] = src; out["info"] = info
    out["sport_label"] = "경마(KRA)"
    out["roster_verification"] = roster_verification
    settled = engine.kra_result_from_starters(starters) if _is_past_day(ymd) else None
    if settled:
        out["_kra_result"] = settled
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    return out


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

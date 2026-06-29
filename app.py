# -*- coding: utf-8 -*-
"""
strategy-arena — 경륜/경마 7권종 적중률 예측 Flask 앱 (Render 배포용)
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
  키 없으면 데모(과거 1경주 캐시)로 폴백.
"""
import os
import time
import datetime as dt
import threading

from flask import Flask, request, render_template, jsonify

import engine

app = Flask(__name__)

DISCLAIMER = ("예측(적중률) 도구입니다. 평균 −EV(검증 완료) · 수익을 보장하지 않습니다. "
              "도박 중독 주의 · 책임 베팅 · 만 19세 이상.")

KEIRIN_MEETS = engine.KEIRIN_MEETS  # ['광명']
KRA_MEETS = engine.KRA_MEETS        # ['서울','제주','부경']

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


def _bg_fetch_recent(sport, meet, key, n):
    """백그라운드 스레드에서 recent_race_days 실행 → 캐시 갱신."""
    ck = (sport, meet)
    try:
        days = engine.recent_race_days(sport, meet, key, n=n)
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
        return []
    now = time.time()
    ck = (sport, meet)
    hit = _RECENT_CACHE.get(ck)
    if hit:
        days, ts, status = hit
        ttl = (_RECENT_TTL if status == "ok"
               else _RECENT_EMPTY_TTL if status == "empty"
               else _RECENT_FAIL_TTL)
        if (now - ts) < ttl:
            return days
        # 만료: 캐시값 일단 반환, 백그라운드에서 갱신
        if ck not in _RECENT_FETCHING:
            _RECENT_FETCHING.add(ck)
            threading.Thread(target=_bg_fetch_recent,
                              args=(sport, meet, key, n), daemon=True).start()
        return days
    # 캐시 미스: 백그라운드 fetch 시작, 즉시 빈 리스트 반환 (논블로킹)
    if ck not in _RECENT_FETCHING:
        _RECENT_FETCHING.add(ck)
        threading.Thread(target=_bg_fetch_recent,
                         args=(sport, meet, key, n), daemon=True).start()
    return []


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
            "message": msg, "recent_days": days}


@app.route("/")
def index():
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


@app.route("/recent")
def recent():
    """경주장 변경 시 chip 갱신용 JSON. ?sport=&meet= → {days:[...]}.
    키 없거나 실패 시 days=[] (JS 는 실패 무시).
    """
    sport = (request.args.get("sport") or "keirin").strip()
    meet = (request.args.get("meet") or "").strip()
    if sport == "horse":
        if meet not in KRA_MEETS:
            meet = KRA_MEETS[0]
    else:
        sport = "keirin"
        if meet not in KEIRIN_MEETS:
            meet = KEIRIN_MEETS[0]
    days = _recent_days_cached(sport, meet, _get_key())
    return jsonify({"sport": sport, "meet": meet, "days": days})


# ── /predict 결과 캐싱 (같은 경주 재요청 시 즉시 응답 — data.go.kr 재호출 방지) ──
# 출주표는 경주 시작 전까지 안 바뀜 → 캐싱 안전. TTL 30분.
_PREDICT_CACHE = {}
_PREDICT_TTL = 1800
# 진행 중인 fetch 중복 방지 (같은 경주 동시 요청 시 1번만 fetch)
_PREDICT_FETCHING = {}

_PREDICT_LOCK = threading.Lock()


@app.route("/predict", methods=["GET", "POST"])
def predict():
    f = request.values  # GET·POST 모두 수용
    sport = (f.get("sport") or "keirin").strip()
    ymd = (f.get("date") or "").strip()
    meet = (f.get("meet") or "광명").strip()
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
    if cached and (now - cached["ts"]) < _PREDICT_TTL:
        return cached["rendered"]  # 캐시된 HTML 즉시 반환

    # ── 경마(KRA) ──
    if sport == "horse":
        return _predict_horse(ymd, meet, race_no, base)

    # ── 경륜 ──
    starters = None
    info = {}
    src = "live"
    key = _get_key()

    if not key:
        # 키 없음 → 데모 캐시 폴백
        demo = engine.load_demo_race()
        if not demo:
            return render_template(
                "index.html",
                result={"kind": "error",
                        "title": "DATAGOKR_SERVICE_KEY 미설정",
                        "message": ("실시간 출주표를 받으려면 Render 환경변수 "
                                    "DATAGOKR_SERVICE_KEY 를 설정해야 합니다. "
                                    "데모 캐시도 사용할 수 없습니다.")},
                **base)
        starters = demo["items"]
        src = "demo"
        info = {"stnd_yr": demo.get("stnd_yr"), "ymd": demo.get("race_ymd"),
                "meet": demo.get("meet_nm"), "race_no": demo.get("race_no"),
                "actual": demo.get("actual")}
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
            # 실시간 실패(API 오류) → 데모 폴백(앱 안 죽음)
            demo = engine.load_demo_race()
            if demo:
                starters = demo["items"]
                src = "demo"
                info = {"stnd_yr": demo.get("stnd_yr"), "ymd": demo.get("race_ymd"),
                        "meet": demo.get("meet_nm"), "race_no": demo.get("race_no"),
                        "actual": demo.get("actual"), "fetch_err": err}
            else:
                return render_template(
                    "index.html",
                    result={"kind": "error", "title": "출주표 조회 실패",
                            "message": err},
                    **base)
        else:
            info = {"stnd_yr": stnd_yr, "ymd": engine.norm_ymd(ymd),
                    "meet": meet, "race_no": race_no}

    # ── 채점 + 7권종 픽 ──
    try:
        out = engine.predict(starters, meta=info)
    except Exception as e:  # noqa: BLE001  (앱이 죽지 않도록 graceful)
        return render_template(
            "index.html",
            result={"kind": "error", "title": "예측 중 오류",
                    "message": f"{type(e).__name__}: {e}"},
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
    # 다가올(미래) 경주 = 사전 예측. 실거래 결과는 아직 없음을 라벨로 표시.
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    return _cache_and_return(ck, render_template("index.html", result=out, **base))


def _cache_and_return(cache_key, rendered):
    """성공 결과만 캐싱. error/notice는 캐싱하지 않음 (다음 요청에서 재시도)."""
    _PREDICT_CACHE[cache_key] = {"rendered": rendered, "ts": time.time()}
    return rendered


def _predict_horse(ymd, meet, race_no, base):
    """경마(KRA) 경로: 실시간 출주표 fetch → KRA 모델 채점 → 7권종 픽.
    키 없거나 실패 시 데모(과거 1경주) 폴백. 경륜 경로와 동일 표시.
    """
    # 경마 기본 경주장 보정 (경륜 meet '광명'이 넘어온 경우 서울로)
    if meet not in KRA_MEETS:
        meet = KRA_MEETS[0]
        base["meet"] = meet

    starters = None
    info = {}
    src = "live"
    key = _get_key()

    if not key:
        demo = engine.load_kra_demo()
        if not demo:
            return render_template(
                "index.html",
                result={"kind": "error",
                        "title": "DATAGOKR_SERVICE_KEY 미설정",
                        "message": ("실시간 경마 출주표를 받으려면 환경변수 "
                                    "DATAGOKR_SERVICE_KEY 설정이 필요합니다. "
                                    "데모 캐시도 사용할 수 없습니다.")},
                **base)
        starters = demo["items"]
        src = "demo"
        info = {"stnd_yr": str(demo.get("race_ymd", ""))[:4],
                "ymd": demo.get("race_ymd"), "meet": demo.get("meet_nm"),
                "race_no": demo.get("race_no"), "actual": demo.get("actual")}
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
            demo = engine.load_kra_demo()
            if demo:
                starters = demo["items"]
                src = "demo"
                info = {"stnd_yr": str(demo.get("race_ymd", ""))[:4],
                        "ymd": demo.get("race_ymd"), "meet": demo.get("meet_nm"),
                        "race_no": demo.get("race_no"),
                        "actual": demo.get("actual"), "fetch_err": err}
            else:
                return render_template(
                    "index.html",
                    result={"kind": "error", "title": "출주표 조회 실패",
                            "message": err},
                    **base)
        else:
            info = {"stnd_yr": str(engine.norm_ymd(ymd))[:4],
                    "ymd": engine.norm_ymd(ymd), "meet": meet, "race_no": race_no}

    try:
        out = engine.predict_kra(starters, meta=info)
    except Exception as e:  # noqa: BLE001
        return render_template(
            "index.html",
            result={"kind": "error", "title": "예측 중 오류",
                    "message": f"{type(e).__name__}: {e}"},
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
    # 다가올(미래) 경마 = 사전 예측. 실거래 결과는 아직 없음을 라벨로 표시.
    out["is_future"] = (src == "live" and _is_future_day(info.get("ymd") or ymd))
    return render_template("index.html", result=out, **base)


@app.route("/healthz")
def healthz():
    model, err = engine.load_model()
    kra, kra_err = engine.load_kra_model()
    return {"ok": err is None and kra_err is None,
            "keirin_model": ("loaded" if model else "fail"), "keirin_err": err,
            "kra_model": ("loaded" if kra else "fail"), "kra_err": kra_err,
            "has_key": _has_key()}, (200 if err is None and kra_err is None else 500)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

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

# (sport, meet) -> (days_list, fetched_ts) 모듈 전역 캐시 (~1h TTL).
_RECENT_CACHE = {}
_RECENT_TTL = 3600  # seconds


def _recent_days_cached(sport, meet, key, n=6):
    """recent_race_days 결과를 1h TTL 캐시로 감싼다. 키 없으면 []."""
    if not key:
        return []
    now = time.time()
    ck = (sport, meet)
    hit = _RECENT_CACHE.get(ck)
    if hit and (now - hit[1]) < _RECENT_TTL:
        return hit[0]
    try:
        days = engine.recent_race_days(sport, meet, key, n=n)
    except Exception:  # noqa: BLE001
        days = []
    # 성공(비어있지 않음)일 때만 캐시. 실패는 다음 요청에서 재시도.
    if days:
        _RECENT_CACHE[ck] = (days, now)
    return days


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


def _notice_no_race(sport, meet, race_no, ymd, base):
    """그 날짜·경주장에 경주가 없을 때 안내 result(kind='notice')."""
    days = base.get("recent_days") or _recent_days_cached(sport, meet, _get_key())
    if days:
        joined = ", ".join(days)
        msg = (f"{ymd} {meet} {race_no}R 경주가 없습니다(비경주일). "
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
    return render_template("index.html", result=out, **base)


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

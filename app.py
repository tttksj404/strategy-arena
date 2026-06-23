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
import datetime as dt

from flask import Flask, request, render_template

import engine

app = Flask(__name__)

DISCLAIMER = ("예측(적중률) 도구입니다. 평균 −EV(검증 완료) · 수익을 보장하지 않습니다. "
              "도박 중독 주의 · 책임 베팅 · 만 19세 이상.")

KEIRIN_MEETS = engine.KEIRIN_MEETS  # ['광명']
KRA_MEETS = engine.KRA_MEETS        # ['서울','제주','부경']


def _has_key():
    return bool(os.environ.get("DATAGOKR_SERVICE_KEY"))


@app.route("/")
def index():
    today = dt.date.today().isoformat()
    return render_template(
        "index.html",
        disclaimer=DISCLAIMER,
        keirin_meets=KEIRIN_MEETS,
        kra_meets=KRA_MEETS,
        today=today,
        has_key=_has_key(),
        result=None,
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    f = request.values  # GET·POST 모두 수용
    sport = (f.get("sport") or "keirin").strip()
    ymd = (f.get("date") or "").strip()
    meet = (f.get("meet") or "광명").strip()
    race_no = (f.get("race_no") or "1").strip()
    today = dt.date.today().isoformat()

    base = dict(
        disclaimer=DISCLAIMER,
        keirin_meets=KEIRIN_MEETS,
        kra_meets=KRA_MEETS,
        today=today,
        has_key=_has_key(),
        sport=sport, date=ymd, meet=meet, race_no=race_no,
    )

    # ── 경마(KRA) ──
    if sport == "horse":
        return _predict_horse(ymd, meet, race_no, base)

    # ── 경륜 ──
    starters = None
    info = {}
    src = "live"
    key = os.environ.get("DATAGOKR_SERVICE_KEY")

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
        if err:
            # 실시간 실패 → 데모 폴백(앱 안 죽음)
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
    key = os.environ.get("DATAGOKR_SERVICE_KEY")

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

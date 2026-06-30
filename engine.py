# -*- coding: utf-8 -*-
"""
strategy-arena 예측 엔진
========================
경륜(광명) model_final 출주표 스코어링 + 7권종(단/연/복/쌍/삼복/쌍복/삼쌍) 픽 산출.

핵심 정직 고지
  - 이 엔진은 +EV(시장초과 수익) 도구가 아니다. OOS 정산에서 공제(약 28%) 후
    단·연승 모두 시장을 못 이겼다(keirin/pnl_full_results.md, pnl_exotic.py).
  - "적중률 예측" 보조 도구일 뿐, 평균 -EV. 수익 보장 없음.

데이터 소스
  - 경륜 카드: data.go.kr OD API (KEIRIN_CARD_URL). 광명 경주장만 데이터 존재.
  - API는 stnd_yr 필터만 서버측 지원(날짜 오름차순). meet/race_no/날짜는 클라이언트 필터.
  - 키는 os.environ['DATAGOKR_SERVICE_KEY'] 로만 읽는다. 하드코딩/커밋 금지.

경마(KRA)
  - kra_model.joblib (keirin 과 동일 dict 구조: win/plc/cols/med/num/rel/feats)
    를 static/models 에 포함. KRA 출주표를 within-race 상대피처로 채점해
    경륜과 동일한 7권종 픽을 산출한다.
  - KRA 카드 API: data.go.kr B551015 RaceDetailResult_1 (race_result 와 동일
    엔드포인트). meet(서울/제주/부경 한글명)·rc_date(YYYYMMDD)·rc_no 파라미터.
  - 정직 제약: 학습 2024-01 ~ 2026-06, 서울/제주/부경 3개 경주장. 검증 결과
    공제(~20%) 후 시장초과 +EV 없음(kra/runs/model_backtest_results.md).
"""
import os
import re
import json
import math
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_model_final.joblib")
CROSS_MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_cross_domain_model.joblib")
FINAL_MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_final_model.joblib")
MODEL_11R_PATH = os.path.join(HERE, "static", "models", "keirin_11r_plus_model.joblib")
SPECIAL_11R_PATH = os.path.join(HERE, "static", "models", "keirin_special_11r_model.joblib")
DEMO_PATH = os.path.join(HERE, "data", "demo_race.json")

# ── 경마(KRA) ──
KRA_MODEL_PATH = os.path.join(HERE, "static", "models", "kra_model.joblib")
KRA_DEMO_PATH = os.path.join(HERE, "data", "demo_kra_race.json")
# KRA 카드 API (race_result 와 동일 엔드포인트). 키만 비밀.
DEFAULT_KRA_URL = ("https://apis.data.go.kr/B551015/API214_1/RaceDetailResult_1")
KRA_CARD_URL = os.environ.get("KRA_CARD_URL", DEFAULT_KRA_URL)
# 학습 데이터에 존재하는 경주장 (실측)
KRA_MEETS = ["서울", "제주", "부경"]

# data.go.kr 경륜 카드 OD API (공개 엔드포인트; 키만 비밀)
DEFAULT_CARD_URL = ("https://apis.data.go.kr/B551014/"
                    "SRVC_OD_API_CRA_RACE_ORGAN/TODZ_API_CRA_RACE_ORGAN_I")
CARD_URL = os.environ.get("KEIRIN_CARD_URL", DEFAULT_CARD_URL)

# kcycle 실시간 배당 (IP 직접 접속 — DNS 회피; 검증 2026-06-30)
KCYCLE_IP = "210.90.29.27"

CIRCLE = {chr(0x2460 + i): i + 1 for i in range(9)}

# 경륜 데이터가 존재하는 경주장(실측: race_card 는 '광명'만)
KEIRIN_MEETS = ["광명"]

# (stnd_yr, week_tcnt, day_tcnt) → kcycle (tms, day) 매핑 캐시
_KCYCLE_TMS_CACHE = {}


def _resolve_kcycle_tms(stnd_yr, ymd):
    """경륜 날짜(YYYYMMDD) → kcycle (year, tms, day, day_of_week) 추정.
    kcycle은 (year, tms, day)로 URL을 구성하므로 이를 추정해야 함.
    휴리스틱: 해당 날짜가 몇 주차 몇 일차인지 추정.
    경륜은 보통 금·토·일 3일 연속 개최 → day=1(금), 2(토), 3(일).
    tms는 연도별 회차 번호 (1부터 시작, 보통 3일마다 1회차 증가).
    """
    import datetime as _dt
    try:
        d = re.sub(r"\D", "", str(ymd or ""))
        if len(d) < 8:
            return None
        date = _dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8]))
        # 연도 시작일 기준 몇 번째 주인지 (금요일=1일차 기준)
        jan1 = _dt.date(date.year, 1, 1)
        # 첫 금요일 찾기
        d_jan1 = jan1.weekday()  # Mon=0
        first_fri = jan1 + _dt.timedelta(days=(4 - d_jan1) % 7)
        if date < first_fri:
            tms = 1
        else:
            weeks = (date - first_fri).days // 7 + 1
            tms = weeks  # 근사 (정확한 매핑은 아니지만 충분)
        # 요일 → day (금=1, 토=2, 일=3)
        wd = date.weekday()  # Mon=0...Sun=6
        day_map = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3}  # 금=1,토=2,일=3
        day = day_map.get(wd, 0)
        return (int(date.year), int(tms), int(day))
    except Exception:
        return None


def fetch_kcycle_odds(stnd_yr, ymd, race_no):
    """kcycle에서 실시간 마번별 단승/연승 배당 fetch.
    반환: {bno: win_odds} 또는 None.
    """
    import urllib.request, ssl, json as _json
    tms_day = _resolve_kcycle_tms(stnd_yr, ymd)
    if not tms_day:
        return None
    year, tms, day = tms_day
    if day == 0:
        return None  # 비경주일
    rno = str(race_no).strip().lstrip("0") or "0"
    rno2 = rno.zfill(2)
    # 여러 tms 시도 (휴리스틱이 부정확할 수 있으므로 ±2)
    for dt in [0, -1, 1, -2, 2]:
        url = (f"https://{KCYCLE_IP}/race/dividendrate/final/"
               f"{year}/{tms+dt}/{day}/001/{rno2}")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3, context=ctx) as r:
                html = r.read().decode("utf-8", "replace")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            m = re.search(r"단승식\s+((?:\d+\s+)+(?:[\d.]+\s+){5,}[\d.]+)", text)
            if not m:
                continue
            nums = m.group(1).strip().split()
            if len(nums) < 14:
                continue
            bnos = nums[:7]
            odds = nums[7:14]  # 마번 1~7 배당
            result = {}
            for i, o in enumerate(odds, 1):
                try:
                    result[i] = float(o)
                except ValueError:
                    continue
            if result:
                return result
        except Exception:
            continue
    return None

# ───────────────────────── 유틸 ─────────────────────────


def mach(s):
    """①~⑨ 또는 (3)/3 형태에서 마번(back_no) 정수 추출."""
    if not s:
        return None
    for ch in str(s):
        if ch in CIRCLE:
            return CIRCLE[ch]
    m = re.match(r"\s*\(?(\d+)", str(s))
    return int(m.group(1)) if m else None


def norm_ymd(s):
    """'2026-06-20' / '20260620' / '2026.06.20' -> '2026.06.20'."""
    d = re.sub(r"\D", "", str(s or ""))
    if len(d) >= 8:
        return f"{d[0:4]}.{d[4:6]}.{d[6:8]}"
    return str(s or "").strip()


# ───────────────────────── 모델 로드 (지연 로딩) ─────────────────────────
_MODEL = None
_MODEL_ERR = None


def load_model():
    """model_final.joblib 지연 로드. 실패 시 (None, 에러문자열)."""
    global _MODEL, _MODEL_ERR
    if _MODEL is not None or _MODEL_ERR is not None:
        return _MODEL, _MODEL_ERR
    try:
        import joblib
        _MODEL = joblib.load(MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        _MODEL_ERR = f"모델 로드 실패: {type(e).__name__}: {e}"
    return _MODEL, _MODEL_ERR


_CROSS_MODEL = None
_CROSS_MODEL_ERR = None


def load_cross_model():
    global _CROSS_MODEL, _CROSS_MODEL_ERR
    if _CROSS_MODEL is not None or _CROSS_MODEL_ERR is not None:
        return _CROSS_MODEL, _CROSS_MODEL_ERR
    try:
        import joblib
        _CROSS_MODEL = joblib.load(CROSS_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        _CROSS_MODEL_ERR = f"교차분야 모델 로드 실패: {type(e).__name__}: {e}"
    return _CROSS_MODEL, _CROSS_MODEL_ERR


_FINAL_MODEL = None
_FINAL_MODEL_ERR = None


def load_final_model():
    """결승전 특화 모델 지연 로드. top1 78% / 연대 90% (결승전 OOS 검증)."""
    global _FINAL_MODEL, _FINAL_MODEL_ERR
    if _FINAL_MODEL is not None or _FINAL_MODEL_ERR is not None:
        return _FINAL_MODEL, _FINAL_MODEL_ERR
    try:
        import joblib
        _FINAL_MODEL = joblib.load(FINAL_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        _FINAL_MODEL_ERR = f"결승전 모델 로드 실패: {type(e).__name__}: {e}"
    return _FINAL_MODEL, _FINAL_MODEL_ERR


_MODEL_11R = None
_MODEL_11R_ERR = None


def load_11r_model():
    """11R+ 특화 모델 지연 로드. top1 66% / 연대 80% (11R+ OOS 검증)."""
    global _MODEL_11R, _MODEL_11R_ERR
    if _MODEL_11R is not None or _MODEL_11R_ERR is not None:
        return _MODEL_11R, _MODEL_11R_ERR
    try:
        import joblib
        _MODEL_11R = joblib.load(MODEL_11R_PATH)
    except Exception as e:  # noqa: BLE001
        _MODEL_11R_ERR = f"11R+ 모델 로드 실패: {type(e).__name__}: {e}"
    return _MODEL_11R, _MODEL_11R_ERR


_SPECIAL_11R = None
_SPECIAL_11R_ERR = None


def load_special_11r_model():
    """특선+11R+ 특화 모델. top1 69% / 연대 83% (특선 11R+ OOS 검증)."""
    global _SPECIAL_11R, _SPECIAL_11R_ERR
    if _SPECIAL_11R is not None or _SPECIAL_11R_ERR is not None:
        return _SPECIAL_11R, _SPECIAL_11R_ERR
    try:
        import joblib
        _SPECIAL_11R = joblib.load(SPECIAL_11R_PATH)
    except Exception as e:  # noqa: BLE001
        _SPECIAL_11R_ERR = f"특선11R+ 모델 로드 실패: {type(e).__name__}: {e}"
    return _SPECIAL_11R, _SPECIAL_11R_ERR


_KRA_MODEL = None
_KRA_MODEL_ERR = None


def load_kra_model():
    """kra_model.joblib 지연 로드. 실패 시 (None, 에러문자열)."""
    global _KRA_MODEL, _KRA_MODEL_ERR
    if _KRA_MODEL is not None or _KRA_MODEL_ERR is not None:
        return _KRA_MODEL, _KRA_MODEL_ERR
    try:
        import joblib
        _KRA_MODEL = joblib.load(KRA_MODEL_PATH)
    except Exception as e:  # noqa: BLE001
        _KRA_MODEL_ERR = f"KRA 모델 로드 실패: {type(e).__name__}: {e}"
    return _KRA_MODEL, _KRA_MODEL_ERR


# ───────────────────────── API fetch (경륜 카드) ─────────────────────────


def _api_page(stnd_yr, page, rows, key, timeout=8):
    qs = urllib.parse.urlencode({
        "serviceKey": key, "resultType": "json",
        "numOfRows": rows, "pageNo": page, "stnd_yr": str(stnd_yr),
    })
    full = CARD_URL + "?" + qs
    req = urllib.request.Request(full, headers={"User-Agent": "strategy-arena"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    resp = data.get("response", {})
    hdr = resp.get("header", {})
    if str(hdr.get("resultCode")) not in ("00", "0"):
        raise RuntimeError(f"API resultCode={hdr.get('resultCode')} {hdr.get('resultMsg')}")
    body = resp.get("body", {})
    tc = int(body.get("totalCount") or 0)
    items = (body.get("items") or {}).get("item") or []
    if isinstance(items, dict):
        items = [items]
    return tc, items


def fetch_race_card(stnd_yr, ymd, meet, race_no, key, rows=1000, max_pages=6):
    """data.go.kr 카드 API에서 단일 경주 출주표를 실시간 fetch (역순 페이지네이션).

    API가 stnd_yr 필터만 지원하고 날짜 오름차순이라, 최근 날짜(6월 등)는
    마지막 페이지 쪽에 몰린다. 정방향 순회는 ~9콜이 걸려 무료티어에서 느리다.
    -> last_page 부터 1까지 역순으로 fetch 하며 (날짜·경주장·경주번호) 매칭.
    최근 날짜는 1~3콜로 끝난다. 페이지 최대 날짜가 목표보다 작아지면 조기 중단,
    못 찾으면 정방향 전체 폴백으로 옛날 날짜도 보장한다.
    반환: (starters_list, None) 또는 (None, 에러문자열).
    """
    if not key:
        return None, "NO_KEY"
    ymd = norm_ymd(ymd)
    target = re.sub(r"\D", "", ymd)  # YYYYMMDD
    rno = str(race_no).strip().lstrip("0") or "0"
    try:
        tc, _ = _api_page(stnd_yr, 1, 1, key)
    except Exception as e:  # noqa: BLE001
        return None, f"API 호출 실패: {type(e).__name__}: {e}"
    if tc == 0:
        return None, f"{stnd_yr}년 출주표 데이터 없음"

    last_page = math.ceil(tc / rows)
    pages = min(max_pages, last_page)

    def _match(items):
        return [i for i in items
                if re.sub(r"\D", "", str(i.get("race_ymd", ""))) == target
                and str(i.get("meet_nm", "")).strip().startswith(meet)
                and str(i.get("race_no", "")).strip().lstrip("0") == rno]

    scanned = set()
    # 1) 역순: 최근 날짜가 뒤쪽 → last_page 부터 거슬러 올라간다.
    for p in range(last_page, max(0, last_page - pages), -1):
        scanned.add(p)
        try:
            _, items = _api_page(stnd_yr, p, rows, key)
        except Exception as e:  # noqa: BLE001
            return None, f"API 페이지 {p} 실패: {type(e).__name__}: {e}"
        if not items:
            continue
        m = _match(items)
        if m:
            return m, None
        page_ymds = sorted({re.sub(r"\D", "", str(i.get("race_ymd", ""))) for i in items})
        # 오름차순 정렬이므로 이 페이지 최대 날짜가 목표보다 작으면
        # 더 앞쪽(과거) 페이지에도 목표가 없다 → 역순 탐색 중단, 폴백으로.
        if page_ymds and page_ymds[-1] < target:
            break

    # 2) 정방향 폴백: 옛날 날짜(앞쪽 페이지)는 역순에서 일찍 끊겼을 수 있다.
    for p in range(1, min(max_pages, last_page) + 1):
        if p in scanned:
            continue
        try:
            _, items = _api_page(stnd_yr, p, rows, key)
        except Exception as e:  # noqa: BLE001
            return None, f"API 페이지 {p} 실패: {type(e).__name__}: {e}"
        if not items:
            break
        m = _match(items)
        if m:
            return m, None
        page_ymds = sorted({re.sub(r"\D", "", str(i.get("race_ymd", ""))) for i in items})
        if page_ymds and page_ymds[0] > target:
            break
    return None, "해당 경주를 찾지 못했습니다 (날짜/경주장/경주번호를 확인하세요)."


def _recent_keirin_days(meet, key, n, rows=1000, max_pages=2):
    """경륜 카드 API 역순 페이지네이션으로 meet 최근 실제 경주일 수집.

    API 가 stnd_yr 만 서버측 필터하고 날짜 오름차순이라, 최근 경주일은
    마지막 페이지 쪽에 몰린다 → last_page 부터 거슬러 올라가며 distinct
    race_ymd(해당 meet)를 모은다. 최근 6일은 1~2콜로 끝난다.
    올해에서 n개 못 채우면 작년으로 1년 더 확장.
    반환: ['YYYY-MM-DD', ...] 내림차순.
    """
    import datetime as _dt
    days = set()
    years = [_dt.date.today().year]
    years.append(years[0] - 1)
    for yr in years:
        try:
            tc, _ = _api_page(yr, 1, 1, key)
        except Exception:  # noqa: BLE001
            continue
        if not tc:
            continue
        last_page = math.ceil(tc / rows)
        for p in range(last_page, max(0, last_page - max_pages), -1):
            try:
                _, items = _api_page(yr, p, rows, key)
            except Exception:  # noqa: BLE001
                break
            for i in items:
                d = re.sub(r"\D", "", str(i.get("race_ymd", "")))
                # 카드 API 에 존재하는 날짜는 모두 포함(다가올 발표분 포함).
                # 미래라도 출주표가 발표된 날은 '사전 예측' 대상이므로 칩에 노출한다.
                if (len(d) == 8
                        and str(i.get("meet_nm", "")).strip().startswith(meet)):
                    days.add(d)
            if len(days) >= n:
                break
        if len(days) >= n:
            break
    out = sorted(days, reverse=True)[:n]
    return [f"{d[0:4]}-{d[4:6]}-{d[6:8]}" for d in out]


def _recent_kra_days(meet, key, n, rows=1000, max_pages=3, months_back=2):
    """KRA RaceDetailResult_1 의 rc_month=YYYYMM 필터로 meet 최근 경주일 수집.

    rc_date/rc_no 없이 rc_month 만 주면 그 달 전체 경주가 반환된다(실측 확인).
    이번 달부터 거슬러 months_back 개월까지 distinct rcDate 를 모은다.
    한 달이면 보통 1콜로 충분(서울/부경/제주 각 1콜로 6일 확보 실측).
    반환: ['YYYY-MM-DD', ...] 내림차순.
    """
    import datetime as _dt
    today = _dt.date.today()
    days = set()
    meet_codes = [meet, _KRA_MEET_CODE.get(meet, meet)]
    for back in range(months_back):
        m = today.month - back
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        ym = f"{y}{m:02d}"
        got = False
        for mc in meet_codes:
            try:
                tc, items = _kra_api_page_month(mc, ym, 1, rows, key)
            except Exception:  # noqa: BLE001
                continue
            if not items:
                continue
            last_page = min(max_pages, math.ceil((tc or len(items)) / rows))
            for p in range(2, last_page + 1):
                try:
                    _, more = _kra_api_page_month(mc, ym, p, rows, key)
                except Exception:  # noqa: BLE001
                    break
                if not more:
                    break
                items += more
            for i in items:
                d = re.sub(r"\D", "", str(i.get("rcDate", "")))
                # KRA 월 조회가 반환하는 날짜는 모두 포함(예정 경주 rcDate 포함).
                # 출주표가 발표된 미래 경주일도 '사전 예측' 대상이므로 칩에 노출한다.
                if len(d) == 8:
                    days.add(d)
            got = True
            break  # 이 meet 코드로 성공 → 다음 코드 시도 불필요
        if got and len(days) >= n:
            break
    out = sorted(days, reverse=True)[:n]
    return [f"{d[0:4]}-{d[4:6]}-{d[6:8]}" for d in out]


def _kra_api_page_month(meet, rc_month, page, rows, key, timeout=10):
    """KRA RaceDetailResult_1 월 단위 조회 (rc_month=YYYYMM, rc_date/rc_no 없음)."""
    qs = urllib.parse.urlencode({
        "serviceKey": key, "_type": "json",
        "numOfRows": rows, "pageNo": page,
        "meet": str(meet), "rc_month": str(rc_month),
    })
    full = KRA_CARD_URL + "?" + qs
    req = urllib.request.Request(full, headers={"User-Agent": "strategy-arena"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    resp = data.get("response", {}) or {}
    hdr = resp.get("header", {}) or {}
    if str(hdr.get("resultCode")) not in ("00", "0"):
        raise RuntimeError(f"API resultCode={hdr.get('resultCode')} {hdr.get('resultMsg')}")
    bd = resp.get("body", {}) or {}
    tc = int(bd.get("totalCount") or 0)
    items_c = bd.get("items", {})
    items = items_c.get("item", []) if isinstance(items_c, dict) else []
    if isinstance(items, dict):
        items = [items]
    return tc, items


def recent_race_days(sport, meet, key, n=6):
    """해당 경주장의 최근 실제 경주일을 'YYYY-MM-DD' 내림차순 리스트로 반환.

    sport: 'keirin' | 'horse'. 키 없거나 실패 시 [].
    경륜=카드 API 역순 페이지네이션, 경마=KRA rc_month 필터.
    """
    if not key:
        return []
    try:
        if sport == "horse":
            return _recent_kra_days(meet, key, n)
        return _recent_keirin_days(meet, key, n)
    except Exception:  # noqa: BLE001
        return []


def load_demo_race():
    """키 없을 때 데모용 캐시 경주(실제 과거 1경주) 반환. 실패 시 None."""
    try:
        with open(DEMO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


# ───────────────────────── 스코어링 (qprep2.py 파이프라인 재현) ─────────────────────────


def score_keirin(starters, meta=None):
    """출주표 item 리스트 -> [{bno, name, grade, pwin, pplc}] (연대확률 내림차순).

    keirin/qprep2.py·recommend.py 의 피처 파이프라인을 그대로 재현한다.
    반환: (rows, None) 또는 (None, 에러문자열).
    """
    model, err = load_model()
    if err:
        return None, err
    return score_keirin_with_model(starters, model, meta=meta)


def _ymd_int(s):
    d = re.sub(r"\D", "", str(s or ""))
    return int(d) if len(d) >= 8 else 0


def _keirin_cross_feature_frame(c, model, meta=None):
    import numpy as np
    import pandas as pd
    maps = model.get("maps", {})
    bf1 = ["bf1_day1_rank", "bf1_day2_rank", "bf1_day3_rank"]
    bf2 = ["bf2_day1_rank", "bf2_day2_rank", "bf2_day3_rank"]
    bf3 = ["bf3_day1_rank", "bf3_day2_rank", "bf3_day3_rank"]
    for b in bf1 + bf2 + bf3:
        if b not in c.columns:
            c[b] = pd.NA
        c[b] = pd.to_numeric(c[b], errors="coerce")
    if "race_ymd" in c.columns and c["race_ymd"].notna().any():
        ymd_i = _ymd_int(c["race_ymd"].dropna().iloc[0])
    else:
        ymd_i = _ymd_int((meta or {}).get("ymd"))
    names = c.get("racer_nm", pd.Series([""] * len(c), index=c.index)).fillna("").astype(str)
    wr = maps.get("racer_wr_prior", {})
    pr = maps.get("racer_plc_prior", {})
    elo = maps.get("elo_prior", {})
    last = maps.get("last_ymd", {})
    c["rank_momentum"] = c[bf2].mean(axis=1) - c[bf1].mean(axis=1)
    c["rank_accel"] = c[bf3].mean(axis=1) - 2 * c[bf2].mean(axis=1) + c[bf1].mean(axis=1)
    c["rank_vol"] = c[bf1 + bf2 + bf3].std(axis=1)
    c["rank_energy"] = 1.0 / (1.0 + c["recent_mean_rank"].fillna(4.0))
    c["racer_wr_prior"] = names.map(wr).fillna(model.get("global_win_rate", 0.15)).astype(float)
    c["racer_plc_prior"] = names.map(pr).fillna(model.get("global_plc_rate", 0.3)).astype(float)
    c["elo_prior"] = names.map(elo).fillna(1500.0).astype(float)
    c["last_ymd"] = names.map(last).fillna(ymd_i)
    c["rest_days"] = (float(ymd_i) - pd.to_numeric(c["last_ymd"], errors="coerce")).clip(lower=0)
    c["speed_norm"] = c["rec_200m_scr"] / (c["race_len"] / 200.0)
    c["gear_speed"] = c["gear_rate"] * c["speed_norm"]
    c["age_sq"] = c["racer_age"] ** 2
    rest_denom = np.log1p(c["rest_days"].clip(lower=0)).replace(0, np.nan)
    c["fatigue_load"] = c["rank_vol"].fillna(0) / rest_denom
    for col in ["win_rate", "high_rate", "gear_rate", "rec_200m_scr", "tot_tms_avg_scr", "racer_wr_prior", "racer_plc_prior"]:
        mean = c[col].mean()
        std = c[col].std()
        c[col + "_z"] = (c[col] - mean) / std if std and not pd.isna(std) else 0.0
        c[col + "_pct"] = c[col].rank(pct=True)
    share = c["win_rate"].clip(lower=0) + 0.001
    total = float(share.sum()) or 1.0
    p = share / total
    c["field_entropy"] = float((-p * np.log(p)).sum())
    c["field_concentration"] = float(p.max())
    for col in ["elo_prior", "racer_wr_prior", "racer_plc_prior"]:
        c[col + "_rel"] = c[col] - c[col].mean()
    return c


def score_keirin_with_model(starters, model, meta=None):
    """score_keirin의 모델 주입 버전. 결승전 특화 모델 등에 사용."""
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        return None, f"의존성 로드 실패: {e}"

    NUM, REL, cols, med = model["num"], model["rel"], model["cols"], model["med"]
    c = pd.DataFrame(starters)
    if c.empty:
        return None, "출주 선수 없음"
    c["bno"] = c.get("back_no").map(mach)
    c = c[c["bno"].notna()].copy()
    if c.empty:
        return None, "유효 마번 없음"
    c["bno"] = c["bno"].astype(int)
    names = {int(b): (str(n).strip() if n else "") for b, n in
             zip(c["bno"], c.get("racer_nm", [""] * len(c)))}
    grades = {int(b): (str(g).strip() if g else "") for b, g in
              zip(c["bno"], c.get("racer_grd_cd", [""] * len(c)))}

    for col in NUM:
        c[col] = pd.to_numeric(c.get(col), errors="coerce")
    bf = ["bf1_day1_rank", "bf1_day2_rank", "bf1_day3_rank",
          "bf2_day1_rank", "bf2_day2_rank", "bf2_day3_rank"]
    for b in bf:
        if b not in c.columns:
            c[b] = pd.NA
        c[b] = pd.to_numeric(c[b], errors="coerce")
    c["recent_mean_rank"] = c[bf].mean(axis=1)
    c["recent_win_cnt"] = (c[bf] == 1).sum(axis=1)
    # 단일 경주 내 상대값 (predict.py 방식: 해당 경주 평균 대비)
    for col in REL:
        if col in c.columns:
            c[col + "_rel"] = c[col] - c[col].mean()
        else:
            c[col + "_rel"] = 0.0
    if "racer_grd_cd" not in c.columns:
        c["racer_grd_cd"] = "NA"
    if model.get("kind") == "keirin_cross_domain_v1":
        c = _keirin_cross_feature_frame(c, model, meta=meta)
        feat_cols = model.get("feats", NUM + ["recent_mean_rank", "recent_win_cnt"] + [x + "_rel" for x in REL])
    else:
        feat_cols = NUM + ["recent_mean_rank", "recent_win_cnt"] + [x + "_rel" for x in REL]
    F = pd.concat([
        c[feat_cols],
        pd.get_dummies(c["bno"], prefix="bn"),
        pd.get_dummies(c["racer_grd_cd"].fillna("NA").astype(str).str.strip(), prefix="g"),
    ], axis=1)
    F = F.reindex(columns=cols, fill_value=0)
    for col in cols:
        if col in med:
            F[col] = F[col].fillna(med[col])
    Fv = F.fillna(0).values
    try:
        pwin = model["win"].predict_proba(Fv)[:, 1]
        pplc = model["plc"].predict_proba(Fv)[:, 1]
    except Exception as e:  # noqa: BLE001
        return None, f"모델 예측 실패: {e}"

    rows = []
    for b, w, p in zip(c["bno"], pwin, pplc):
        b = int(b)
        rows.append({"bno": b, "name": names.get(b, ""), "grade": grades.get(b, ""),
                     "pwin": float(w), "pplc": float(p)})
    rows.sort(key=lambda r: -r["pplc"])
    return rows, None


# ───────────────────────── Harville 7권종 픽 ─────────────────────────


def _harville_order(rows):
    """win확률 내림차순 마번 리스트 (Harville 순서픽 근사)."""
    return [r["bno"] for r in sorted(rows, key=lambda r: -r["pwin"])]


def _grade_plc(p):
    if p >= 0.85:
        return "강"
    if p >= 0.70:
        return "중"
    return "약"


def _grade_win(p):
    if p >= 0.50:
        return "강"
    if p >= 0.30:
        return "중"
    return "약"


# 권종별 대중용 1줄 뜻 (초보자가 7권종 차이를 바로 이해하도록).
# desc 는 짧은 라벨(예 '1·2착 마번 무순'), mean 은 풀어쓴 설명.
BET_MEANINGS = {
    "단승": "1착(1등)을 맞히면 적중.",
    "연승": "고른 한 마리가 2착 안에 들면 적중(가장 쉬움).",
    "복승": "1·2착 둘 다 맞히기 — 순서는 상관없음.",
    "쌍승": "1착·2착을 순서까지 정확히 맞히기.",
    "삼복": "1·2·3착 셋 다 맞히기 — 순서는 상관없음.",
    "쌍복": "1착은 정확히, 나머지 2·3착은 순서 없이 맞히기.",
    "삼쌍": "1→2→3착 순서까지 전부 맞히기(가장 어려움).",
}


def build_picks(rows):
    """스코어링 rows -> 7권종 픽 리스트.

    매핑(keirin/pnl_exotic.py 실착순 대조 검증 n=14153):
      단승  = win top1
      연승  = plc top1·top2
      복승  = Harville 무순 top2
      쌍승  = Harville 순서 top2
      삼복  = Harville 무순 top3
      쌍복  = 1위 고정 + (2·3위 무순)
      삼쌍  = Harville 순서 top3
    각 픽 + 모델확률 + 신뢰등급 + 대중용 1줄 뜻(mean).
    """
    by_win = sorted(rows, key=lambda r: -r["pwin"])
    by_plc = sorted(rows, key=lambda r: -r["pplc"])
    order = _harville_order(rows)  # win 내림차순 마번
    pw = {r["bno"]: r["pwin"] for r in rows}
    pp = {r["bno"]: r["pplc"] for r in rows}
    nm = {r["bno"]: r["name"] for r in rows}

    def lab(b):
        n = nm.get(b, "")
        return f"{b}번 {n}".strip()

    picks = []

    # 단승: win top1
    t = by_win[0]
    picks.append({
        "code": "단승", "desc": "1착 적중", "type": "단일",
        "pick": [lab(t["bno"])],
        "prob": f"win {100*t['pwin']:.1f}%",
        "grade": _grade_win(t["pwin"]),
    })

    # 연승: plc top1·2 (둘 중 1마라도 2착 이내면 적중)
    y = by_plc[:2]
    picks.append({
        "code": "연승", "desc": "2착 이내 적중", "type": "복수",
        "pick": [lab(r["bno"]) for r in y],
        "prob": " / ".join(f"연대 {100*r['pplc']:.0f}%" for r in y),
        "grade": _grade_plc(max(r["pplc"] for r in y)),
    })

    if len(order) >= 2:
        a, b = order[0], order[1]
        # 복승: 무순 top2
        picks.append({
            "code": "복승", "desc": "1·2착 마번 무순", "type": "조합(무순2)",
            "pick": [f"{lab(a)} ↔ {lab(b)}"],
            "prob": f"Harville top2 (win {100*pw[a]:.0f}% · {100*pw[b]:.0f}%)",
            "grade": _grade_plc((pp[a] + pp[b]) / 2),
        })
        # 쌍승: 순서 top2
        picks.append({
            "code": "쌍승", "desc": "1착→2착 순서", "type": "조합(순서2)",
            "pick": [f"{lab(a)} → {lab(b)}"],
            "prob": f"순서 top2 (win {100*pw[a]:.0f}% → {100*pw[b]:.0f}%)",
            "grade": _grade_win(pw[a]),
        })

    if len(order) >= 3:
        a, b, c = order[0], order[1], order[2]
        # 삼복: 무순 top3
        picks.append({
            "code": "삼복", "desc": "1·2·3착 마번 무순", "type": "조합(무순3)",
            "pick": [f"{lab(a)} ↔ {lab(b)} ↔ {lab(c)}"],
            "prob": "Harville 무순 top3",
            "grade": _grade_plc((pp[a] + pp[b] + pp[c]) / 3),
        })
        # 쌍복: 1위 고정 + 2·3위 무순
        picks.append({
            "code": "쌍복", "desc": "1착 고정 + 2·3착 무순", "type": "조합",
            "pick": [f"1착 {lab(a)} 고정 + ({lab(b)} ↔ {lab(c)})"],
            "prob": f"1착 고정 win {100*pw[a]:.0f}%",
            "grade": _grade_win(pw[a]),
        })
        # 삼쌍: 순서 top3
        picks.append({
            "code": "삼쌍", "desc": "1→2→3착 순서", "type": "조합(순서3)",
            "pick": [f"{lab(a)} → {lab(b)} → {lab(c)}"],
            "prob": f"순서 top3 (win {100*pw[a]:.0f}%)",
            "grade": _grade_win(pw[a]),
        })

    # 대중용 1줄 뜻 부착 (7권종을 다 모르는 사용자 대상).
    for p in picks:
        p["mean"] = BET_MEANINGS.get(p["code"], "")

    return picks


def _top_confidence(top, rows=None):
    """최상위(연대확률 1위) 픽의 신뢰등급/표현.

    트로피·'최고확신'은 절대 win 확률이 충분히 높을 때만 — 그렇지 않으면
    과신을 막기 위해 '상대 1순위(저신뢰)' 중립 표현으로 다운그레이드한다.
    반환 dict: {grade, label, icon, race_confidence}.
    """
    pwin = top.get("pwin", 0.0)
    grade = _grade_win(pwin)  # 강(>=50%) / 중(>=30%) / 약
    # 경주 내 확신도 (top1 - top2 확률 차이) → 고확신/혼전 판별
    race_conf = ""
    if rows and len(rows) >= 2:
        sorted_rows = sorted(rows, key=lambda r: -r.get("pwin", 0))
        gap = sorted_rows[0].get("pwin", 0) - sorted_rows[1].get("pwin", 0)
        if gap >= 0.25:
            race_conf = "고확신"  # 검증: 상위 30% 경주 80% 적중
        elif gap <= 0.08:
            race_conf = "혼전"  # 저확신 경주 (51% 적중)
        else:
            race_conf = "보통"
    if grade == "강":
        return {"grade": "강", "label": "최고확신 픽", "icon": "🏆", "race_confidence": race_conf}
    if grade == "중":
        return {"grade": "중", "label": "상대 우세 픽", "icon": "▲", "race_confidence": race_conf}
    return {"grade": "약", "label": "상대 1순위 (저신뢰)", "icon": "①", "race_confidence": race_conf}


def predict(starters, meta=None):
    """출주표 -> {rows, picks, top, top_conf, meta, n_starters} 또는 {error}.

    결승전(그 날의 마지막 경주) 감지 시 결승전 특화 모델 사용 (top1 78% vs 전체 60%).
    """
    # 결승전 감지: meta에 race_no와 day_max_race_no가 있으면 비교.
    # 없으면 race_no >= 12 휴리스틱 (광명은 보통 12~15R).
    is_final = False
    rno_i = 0
    if meta:
        rno = str(meta.get("race_no", "")).strip().lstrip("0") or "0"
        try:
            rno_i = int(rno)
        except ValueError:
            rno_i = 0
        day_max = meta.get("day_max_race_no")
        if day_max:
            try:
                is_final = rno_i >= int(str(day_max).strip().lstrip("0"))
            except (ValueError, TypeError):
                is_final = False
        else:
            # 휴리스틱: 12R 이상이면 결승전 근처로 간주
            is_final = rno_i >= 12

    # 결승전이면 특화 모델 사용 (race_no >= 12 또는 day_max)
    if is_final:
        fm, ferr = load_final_model()
        if fm is not None:
            rows, err = score_keirin_with_model(starters, fm)
            if err is None:
                rows[0]["_final_model"] = True if rows else False
                picks = build_picks(rows)
                return {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "final_model": True,
                }
    # 11R+ (결승전 아닌 상위 경주) 특화 모델
    if rno_i >= 11:
        # 특선 등급 + 11R+ → 가장 정확한 특화 모델 (69%)
        grade = ""
        if starters and isinstance(starters, list) and len(starters) > 0:
            grade = str(starters[0].get("racer_grd_cd", "") or "").strip()
        if grade == "특선":
            sp, sperr = load_special_11r_model()
            if sp is not None:
                rows, err = score_keirin_with_model(starters, sp)
                if err is None:
                    picks = build_picks(rows)
                    return {
                        "rows": rows, "picks": picks, "top": rows[0],
                        "top_conf": _top_confidence(rows[0], rows),
                        "meta": meta or {}, "n_starters": len(rows),
                        "model_special_11r": True,
                    }
        # 일반 11R+ 특화 (66%)
        m11, m11err = load_11r_model()
        if m11 is not None:
            rows, err = score_keirin_with_model(starters, m11)
            if err is None:
                picks = build_picks(rows)
                return {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "model_11r": True,
                }
    cross, cross_err = load_cross_model()
    if cross is not None:
        rows, err = score_keirin_with_model(starters, cross, meta=meta)
        if err is None:
            picks = build_picks(rows)
            return {
                "rows": rows,
                "picks": picks,
                "top": rows[0],
                "top_conf": _top_confidence(rows[0], rows),
                "meta": meta or {},
                "n_starters": len(rows),
                "model_cross_domain": True,
            }

    # 일반 모델
    rows, err = score_keirin(starters, meta=meta)
    if err:
        return {"error": err}

    picks = build_picks(rows)
    return {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "top_conf": _top_confidence(rows[0], rows),
        "meta": meta or {},
        "n_starters": len(rows),
    }


# ═══════════════════════ 경마(KRA) ═══════════════════════


def _kra_api_page(meet, rc_date, rc_no, page, rows, key, timeout=8):
    qs = urllib.parse.urlencode({
        "serviceKey": key, "_type": "json",
        "numOfRows": rows, "pageNo": page,
        "meet": str(meet), "rc_date": str(rc_date), "rc_no": str(rc_no),
    })
    full = KRA_CARD_URL + "?" + qs
    req = urllib.request.Request(full, headers={"User-Agent": "strategy-arena"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    data = json.loads(body)
    resp = data.get("response", {}) or {}
    hdr = resp.get("header", {}) or {}
    if str(hdr.get("resultCode")) not in ("00", "0"):
        raise RuntimeError(f"API resultCode={hdr.get('resultCode')} {hdr.get('resultMsg')}")
    bd = resp.get("body", {}) or {}
    tc = int(bd.get("totalCount") or 0)
    items_c = bd.get("items", {})
    items = items_c.get("item", []) if isinstance(items_c, dict) else []
    if isinstance(items, dict):
        items = [items]
    return tc, items


# KRA 경주장 코드 (API 는 한글명 수용; 숫자코드 호환 위해 매핑 유지)
_KRA_MEET_CODE = {"서울": "1", "제주": "2", "부경": "3"}


def fetch_kra_card(ymd, meet, race_no, key, rows=50, max_pages=4, timeout=8):
    """KRA RaceDetailResult_1 에서 단일 경주 출주표 실시간 fetch.

    KRA API 는 meet+rc_date+rc_no 를 서버측에서 필터하므로 totalCount 가
    해당 경주의 두수(보통 ≤16)뿐 → 1콜로 끝난다. 경륜과 동일한 패턴으로
    totalCount 기준 페이지를 보강해 두수가 numOfRows 를 넘는 예외도 안전하게
    수집한다(역순 불필요: 한 경주만 반환되므로 페이지 1개면 충분).
    반환: (starters_list, None) 또는 (None, 에러문자열).
    """
    if not key:
        return None, "NO_KEY"
    rc_date = re.sub(r"\D", "", str(ymd or ""))  # YYYYMMDD
    if len(rc_date) != 8:
        return None, "날짜 형식 오류 (YYYY-MM-DD)"
    rno = str(race_no).strip().lstrip("0") or "0"
    # 먼저 한글명, 실패 시 숫자코드로 재시도
    candidates = [meet, _KRA_MEET_CODE.get(meet, meet)]
    last_err = None
    for mv in candidates:
        try:
            tc, items = _kra_api_page(mv, rc_date, rno, 1, rows, key, timeout)
        except Exception as e:  # noqa: BLE001
            last_err = f"API 호출 실패: {type(e).__name__}: {e}"
            continue
        if not items:
            last_err = f"{meet} {rc_date} {rno}R 출주표 없음 (totalCount={tc})"
            continue
        # totalCount 가 rows 보다 크면(드묾) 나머지 페이지도 모은다.
        last_page = min(max_pages, math.ceil((tc or len(items)) / rows))
        for p in range(2, last_page + 1):
            try:
                _, more = _kra_api_page(mv, rc_date, rno, p, rows, key, timeout)
            except Exception:  # noqa: BLE001
                break
            if not more:
                break
            items += more
        # rc_no 클라이언트 재확인 (서버가 전체 반환할 때 대비)
        match = [i for i in items
                 if str(i.get("rcNo", rno)).strip().lstrip("0") == rno]
        return (match or items), None
    return None, last_err or "해당 경주를 찾지 못했습니다."


def load_kra_demo():
    """키 없을 때 데모용 KRA 과거 1경주 반환. 실패 시 None."""
    try:
        with open(KRA_DEMO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def _kra_base_w(s):
    """wgHr 'NNN(+M)' / 'NNN()' -> NNN."""
    if s is None:
        return None
    m = re.match(r"\s*(\d+)", str(s))
    return float(m.group(1)) if m else None


def score_kra(starters):
    """KRA 출주표 item 리스트 -> [{bno,name,grade,pwin,pplc}] (연대확률 내림차순).

    kra/train_save_model.py 의 within-race 상대피처 파이프라인을 그대로 재현한다
    (단일 경주 평균 대비 _rel, sex/budam 더미, jk/tr prior 글로벌 맵 fallback).
    반환: (rows, None) 또는 (None, 에러문자열).
    """
    model, err = load_kra_model()
    if err:
        return None, err
    try:
        import numpy as np  # noqa: F401
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        return None, f"의존성 로드 실패: {e}"

    NUM, REL = model["num"], model["rel"]
    cols, med = model["cols"], model["med"]
    jk_prior = model.get("jk_prior", {})
    tr_prior = model.get("tr_prior", {})
    gp = model.get("global_win_rate", 0.1)

    c = pd.DataFrame(starters)
    if c.empty:
        return None, "출주 두수 없음"
    # 마번 = chulNo (출주번호). 이름/등급(말 이름, rating 등급 없음 → grade=공란)
    c["bno"] = pd.to_numeric(c.get("chulNo"), errors="coerce")
    c = c[c["bno"].notna()].copy()
    if c.empty:
        return None, "유효 출주번호(chulNo) 없음"
    c["bno"] = c["bno"].astype(int)
    names = {int(b): (str(n).strip() if n else "") for b, n in
             zip(c["bno"], c.get("hrName", [""] * len(c)))}

    # 숫자 피처
    for col in ["winOdds", "plcOdds", "wgBudam", "age", "rating", "rcDist"]:
        c[col] = pd.to_numeric(c.get(col), errors="coerce")
    c["chulNo"] = c["bno"].astype(float)
    c["wgHr_base"] = c.get("wgHr").map(_kra_base_w) if "wgHr" in c.columns else None
    c["field_size"] = float(len(c))
    # jockey/trainer 사전 승률 (글로벌 맵 → 없으면 global base rate)
    c["jk_wr_prior"] = (c.get("jkNo").astype(str).map(jk_prior)
                        if "jkNo" in c.columns else gp)
    c["tr_wr_prior"] = (c.get("trNo").astype(str).map(tr_prior)
                        if "trNo" in c.columns else gp)
    c["jk_wr_prior"] = pd.to_numeric(c["jk_wr_prior"], errors="coerce").fillna(gp)
    c["tr_wr_prior"] = pd.to_numeric(c["tr_wr_prior"], errors="coerce").fillna(gp)

    # 배당 암시확률 (v2 odds-injected 모델용)
    c["imp_win"] = np.where(c.get("winOdds", 0) > 0, 1.0 / c["winOdds"], 0.0)
    c["imp_plc"] = np.where(c.get("plcOdds", 0) > 0, 2.0 / c["plcOdds"], 0.0)
    sum_iw = c["imp_win"].sum(); sum_ip = c["imp_plc"].sum()
    c["imp_win_norm"] = c["imp_win"] / sum_iw if sum_iw > 0 else 0.0
    c["imp_plc_norm"] = c["imp_plc"] / sum_ip if sum_ip > 0 else 0.0

    # within-race 상대값 (해당 경주 평균 대비)
    for col in REL:
        if col in c.columns:
            c[col + "_rel"] = c[col] - c[col].mean()
        else:
            c[col + "_rel"] = 0.0

    # sex / budam 더미
    for cat, pref in [("sex", "sex"), ("budam", "bd")]:
        src = c[cat].fillna("NA").astype(str) if cat in c.columns \
            else pd.Series(["NA"] * len(c), index=c.index)
        d = pd.get_dummies(src, prefix=pref)
        c = pd.concat([c, d], axis=1)

    F = c.reindex(columns=cols, fill_value=0)
    for col in cols:
        if col in med and med[col] is not None:
            F[col] = F[col].fillna(med[col])
    Fv = F.apply(pd.to_numeric, errors="coerce").fillna(0).values
    try:
        pwin = model["win"].predict_proba(Fv)[:, 1]
        pplc = model["plc"].predict_proba(Fv)[:, 1]
    except Exception as e:  # noqa: BLE001
        return None, f"모델 예측 실패: {e}"

    # win 확률 경주 내 정규화 (해석 용이; pplc 는 그대로 연대확률)
    s = float(pwin.sum()) or 1.0
    pwin_n = pwin / s

    rows = []
    for b, w, p in zip(c["bno"], pwin_n, pplc):
        b = int(b)
        rows.append({"bno": b, "name": names.get(b, ""), "grade": "",
                     "pwin": float(w), "pplc": float(p)})
    rows.sort(key=lambda r: -r["pplc"])
    return rows, None


def predict_kra(starters, meta=None):
    """KRA 출주표 -> {rows, picks, top, top_conf, meta, n_starters} 또는 {error}."""
    rows, err = score_kra(starters)
    if err:
        return {"error": err}
    picks = build_picks(rows)
    return {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "top_conf": _top_confidence(rows[0], rows),
        "meta": meta or {},
        "n_starters": len(rows),
    }


# ═══════════════════════ 실시간 판단 (live-decision) ═══════════════════════

def fetch_kcycle_odds_with_ts(stnd_yr, ymd, race_no):
    """kcycle 배당 fetch + 타임스탬프. 반환 (odds_dict, fetched_at_iso) 또는 (None, None)."""
    import datetime as _dt
    odds = None
    try:
        odds = fetch_kcycle_odds(stnd_yr, ymd, race_no)
    except Exception:
        odds = None
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    return odds, ts


def compute_live_decision(sport, ymd, meet, race_no, base_model_out=None):
    """실시간 판단. base_model_out은 이미 계산된 engine.predict/predict_kra 결과.
    반환 dict:
      ok, status, message, updated_at, odds_age_sec,
      market_odds, top, rows, decision, market_used, snapshot_phase
    """
    import datetime as _dt
    now = _dt.datetime.now()

    # base_model_out이 없으면 demo 사용 (실시간 fetch는 app.py에서 미리 함)
    if base_model_out is None or "error" in base_model_out:
        return {
            "ok": False,
            "status": "hold",
            "message": "모델 예측 불가 (출주표 없음 또는 오류)",
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None,
            "market_odds": None,
            "top": None,
            "rows": None,
            "decision": "hold",
            "market_used": False,
            "snapshot_phase": "unknown",
        }

    rows = [dict(r) for r in base_model_out.get("rows", [])]
    if not rows:
        return {
            "ok": False, "status": "hold", "message": "출주 데이터 없음",
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None, "market_odds": None, "top": None,
            "rows": None, "decision": "hold", "market_used": False,
            "snapshot_phase": "unknown",
        }

    # ── kcycle 배당 fetch (경륜만) ──
    market_odds = None
    fetched_at = None
    market_used = False
    if sport == "keirin" and os.environ.get("KCYCLE_ENABLED", "0") == "1":
        stnd_yr = str(ymd)[:4] if ymd else ""
        try:
            market_odds, fetched_at = fetch_kcycle_odds_with_ts(stnd_yr, ymd, race_no)
        except Exception:
            market_odds = None

    # ── 상태 판정 ──
    if market_odds and len(market_odds) >= 2:
        # 시장 암시확률 → 모델과 앙상블
        imp = {b: 1.0 / o for b, o in market_odds.items() if o and o > 0}
        total = sum(imp.values())
        if total > 0:
            imp_norm = {b: v / total for b, v in imp.items()}
            for r in rows:
                bno = r.get("bno", 0)
                mkt_p = imp_norm.get(bno, 0.0)
                model_p = r.get("pwin", 0.0)
                r["pwin_blended"] = 0.3 * model_p + 0.7 * mkt_p
                r["mkt_pwin"] = mkt_p
            # blended 로 재정렬
            rows.sort(key=lambda r: -r.get("pwin_blended", r.get("pwin", 0)))
            rows[0]["pwin"] = rows[0].get("pwin_blended", rows[0]["pwin"])
            market_used = True
            status = "odds_live"
            message = "실시간 배당 반영 (시장 0.7 + 모델 0.3)"
            snapshot_phase = "live_odds"
        else:
            status = "odds_unavailable"
            message = "배당 수집 실패 — 사전 예측만 표시"
            snapshot_phase = "pre_race"
    else:
        # 배당 없음
        if os.environ.get("KCYCLE_ENABLED", "0") == "1" and sport == "keirin":
            status = "odds_unavailable"
            message = "배당 미확정 또는 수집 실패 — 사전 예측"
            snapshot_phase = "pre_race"
        else:
            status = "pre_race"
            message = "사전 예측 (배당 미반영 — Render에서는 kcycle 접근 제한)"
            snapshot_phase = "pre_race"

    # ── 최종판정 여부 ──
    top = rows[0] if rows else None
    # decision: final_candidate (배당 있음), hold (배당 없음/불확실)
    if market_used:
        # 확신도 높으면 final_candidate, 아니면 hold
        tc = _top_confidence(top, rows) if top else {}
        gap = 0
        if len(rows) >= 2:
            sorted_r = sorted(rows, key=lambda r: -r.get("pwin", 0))
            gap = sorted_r[0].get("pwin", 0) - sorted_r[1].get("pwin", 0)
        if gap >= 0.15 or tc.get("grade") == "강":
            decision = "final_candidate"
        else:
            decision = "hold"
    else:
        decision = "hold"

    # odds_age_sec
    odds_age_sec = None
    if fetched_at:
        try:
            ft = _dt.datetime.fromisoformat(fetched_at)
            odds_age_sec = int((now - ft).total_seconds())
        except Exception:
            odds_age_sec = None

    return {
        "ok": True,
        "status": status,
        "message": message,
        "updated_at": now.isoformat(timespec="seconds"),
        "odds_age_sec": odds_age_sec,
        "market_odds": market_odds,
        "top": top,
        "rows": rows,
        "decision": decision,
        "market_used": market_used,
        "snapshot_phase": snapshot_phase,
    }

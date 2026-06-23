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

CIRCLE = {chr(0x2460 + i): i + 1 for i in range(9)}

# 경륜 데이터가 존재하는 경주장(실측: race_card 는 '광명'만)
KEIRIN_MEETS = ["광명"]

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


def _api_page(stnd_yr, page, rows, key, timeout=15):
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


def fetch_race_card(stnd_yr, ymd, meet, race_no, key, rows=1000, max_pages=25):
    """data.go.kr 카드 API에서 단일 경주 출주표를 실시간 fetch.

    API가 stnd_yr 필터만 지원하고 날짜 오름차순이라, 페이지를 넘기며
    클라이언트에서 (날짜·경주장·경주번호)로 필터한다. 목표 날짜를 지나치면 중단.
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
    pages = min(max_pages, math.ceil(tc / rows))
    for p in range(1, pages + 1):
        try:
            _, items = _api_page(stnd_yr, p, rows, key)
        except Exception as e:  # noqa: BLE001
            return None, f"API 페이지 {p} 실패: {type(e).__name__}: {e}"
        if not items:
            break
        page_ymds = sorted({re.sub(r"\D", "", str(i.get("race_ymd", ""))) for i in items})
        match = [i for i in items
                 if re.sub(r"\D", "", str(i.get("race_ymd", ""))) == target
                 and str(i.get("meet_nm", "")).strip().startswith(meet)
                 and str(i.get("race_no", "")).strip().lstrip("0") == rno]
        if match:
            return match, None
        # 날짜 오름차순 → 페이지 최소 날짜가 목표를 지나쳤으면 더 볼 필요 없음
        if page_ymds and page_ymds[0] > target:
            break
    return None, "해당 경주를 찾지 못했습니다 (날짜/경주장/경주번호를 확인하세요)."


def load_demo_race():
    """키 없을 때 데모용 캐시 경주(실제 과거 1경주) 반환. 실패 시 None."""
    try:
        with open(DEMO_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


# ───────────────────────── 스코어링 (qprep2.py 파이프라인 재현) ─────────────────────────


def score_keirin(starters):
    """출주표 item 리스트 -> [{bno, name, grade, pwin, pplc}] (연대확률 내림차순).

    keirin/qprep2.py·recommend.py 의 피처 파이프라인을 그대로 재현한다.
    반환: (rows, None) 또는 (None, 에러문자열).
    """
    model, err = load_model()
    if err:
        return None, err
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
    F = pd.concat([
        c[NUM + ["recent_mean_rank", "recent_win_cnt"] + [x + "_rel" for x in REL]],
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
    각 픽 + 모델확률 + 신뢰등급.
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

    return picks


def predict(starters, meta=None):
    """출주표 -> {rows, picks, top, meta, n_starters} 또는 {error}."""
    rows, err = score_keirin(starters)
    if err:
        return {"error": err}
    picks = build_picks(rows)
    return {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "meta": meta or {},
        "n_starters": len(rows),
    }


# ═══════════════════════ 경마(KRA) ═══════════════════════


def _kra_api_page(meet, rc_date, rc_no, page, rows, key, timeout=15):
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


def fetch_kra_card(ymd, meet, race_no, key, timeout=15):
    """KRA RaceDetailResult_1 에서 단일 경주 출주표 실시간 fetch.
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
            tc, items = _kra_api_page(mv, rc_date, rno, 1, 50, key, timeout)
        except Exception as e:  # noqa: BLE001
            last_err = f"API 호출 실패: {type(e).__name__}: {e}"
            continue
        if items:
            # rc_no 클라이언트 재확인 (서버가 전체 반환할 때 대비)
            match = [i for i in items
                     if str(i.get("rcNo", rno)).strip().lstrip("0") == rno]
            return (match or items), None
        last_err = f"{meet} {rc_date} {rno}R 출주표 없음 (totalCount={tc})"
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
    """KRA 출주표 -> {rows, picks, top, meta, n_starters} 또는 {error}."""
    rows, err = score_kra(starters)
    if err:
        return {"error": err}
    picks = build_picks(rows)
    return {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "meta": meta or {},
        "n_starters": len(rows),
    }

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
import time
import urllib.parse
import urllib.request
from html.parser import HTMLParser

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
KCYCLE_RANKINGPREDICT_PATH = os.environ.get(
    "KCYCLE_RANKINGPREDICT_PATH",
    os.path.join(HERE, "data", "kcycle_rankingpredict_signals.json"),
)
KCYCLE_TRIFECTA_ENABLED = os.environ.get("KCYCLE_TRIFECTA_ENABLED", "1") == "1"
KCYCLE_TRIFECTA_SNAPSHOT_PATH = os.environ.get(
    "KCYCLE_TRIFECTA_SNAPSHOT_PATH",
    os.path.join(HERE, "data", "kcycle_trifecta_snapshots.jsonl"),
)

CIRCLE = {chr(0x2460 + i): i + 1 for i in range(9)}

# 경륜 데이터가 존재하는 경주장(실측: race_card 는 '광명'만)
KEIRIN_MEETS = ["광명"]

# (stnd_yr, week_tcnt, day_tcnt) → kcycle (tms, day) 매핑 캐시
_KCYCLE_TMS_CACHE = {}
_KCYCLE_RANKINGPREDICT = None
_KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0
_KEIRIN_CARD_PAGE_CACHE = {}
_KEIRIN_CARD_PAGE_TTL = int(os.environ.get("KEIRIN_CARD_PAGE_TTL", "1800"))
_KCYCLE_TRIFECTA_SNAPSHOT_LAST = {}
_KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS = {}


class _KcycleTableTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []
        self._in_cell = False
        self._in_row = False
        self._cell_parts = []
        self._row = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._in_row = True
            self._row = []
        elif tag in {"th", "td"} and self._in_row:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data):
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self._in_cell:
            text = re.sub(r"\s+", " ", "".join(self._cell_parts)).strip()
            self._row.append(text)
            self._cell_parts = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(cell for cell in self._row):
                self.rows.append(self._row)
            self._row = []
            self._in_row = False


def _kcycle_float(text):
    text = str(text or "").strip().replace(",", "")
    if text in {"", "-", "0", "0.0"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_kcycle_trifecta_board(html):
    parser = _KcycleTableTextParser()
    parser.feed(html or "")
    board = {}
    prefixes = []
    for cells in parser.rows:
        row_prefixes = [x for x in cells if re.fullmatch(r"[1-7]-[1-7]", x or "")]
        if row_prefixes:
            prefixes = row_prefixes
            continue
        if not prefixes or len(cells) < 2:
            continue
        for idx, prefix in enumerate(prefixes):
            pos = idx * 2
            if pos + 1 >= len(cells):
                continue
            third = cells[pos]
            odds = _kcycle_float(cells[pos + 1])
            if odds is None or not re.fullmatch(r"[1-7]", third or ""):
                continue
            a, b = prefix.split("-")
            if third in {a, b}:
                continue
            board[f"{a}-{b}-{third}"] = odds
    return board


def _rankingpredict_key(ymd, meet, race_no):
    digits = re.sub(r"\D", "", str(ymd or ""))
    if len(digits) < 8:
        return None
    rno = str(race_no or "").strip().lstrip("0") or "0"
    return f"{digits[:8]}|{str(meet or '').strip()}|{int(rno)}"


def _kcycle_day_no(row, meta):
    day = row.get("day") if isinstance(row, dict) else None
    try:
        if day is not None:
            return int(day)
    except (TypeError, ValueError):
        pass
    digits = re.sub(r"\D", "", str((row or {}).get("date") or (meta or {}).get("ymd") or ""))
    if len(digits) < 8:
        return 0
    try:
        import datetime as _dt
        weekday = _dt.date(int(digits[:4]), int(digits[4:6]), int(digits[6:8])).weekday()
    except ValueError:
        return 0
    return {4: 1, 5: 2, 6: 3}.get(weekday, 0)


def _load_kcycle_rankingpredict_cache():
    global _KCYCLE_RANKINGPREDICT
    if _KCYCLE_RANKINGPREDICT is not None:
        return _KCYCLE_RANKINGPREDICT
    data = {}
    try:
        with open(KCYCLE_RANKINGPREDICT_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        for row in payload.get("rows", []):
            key = _rankingpredict_key(row.get("date"), row.get("meet"), row.get("race_no"))
            if key:
                data[key] = row
    except Exception:  # noqa: BLE001
        data = {}
    _KCYCLE_RANKINGPREDICT = data
    return data


def kcycle_rankingpredict_cache_status():
    data = _load_kcycle_rankingpredict_cache()
    dates = [key.split("|", 1)[0] for key in data]
    return {
        "rows": len(data),
        "latest_date": max(dates) if dates else None,
        "live_enabled": os.environ.get("KCYCLE_RANKINGPREDICT_ENABLED", "1") == "1",
        "live_cooldown": time.time() < _KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL,
    }


def _fetch_kcycle_rankingpredict_row(meta):
    global _KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL
    import ssl
    from html.parser import HTMLParser

    if os.environ.get("KCYCLE_RANKINGPREDICT_ENABLED", "1") != "1":
        return None
    if time.time() < _KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL:
        return None
    if not meta:
        return None
    tms_day = _resolve_kcycle_tms(meta.get("stnd_yr"), meta.get("ymd"))
    if not tms_day:
        return None
    year, tms, day = tms_day
    if day == 0:
        return None

    class _TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_cell = False
            self.current = ""
            self.cells = []
            self.rows = []

        def handle_starttag(self, tag, attrs):
            if tag in ("td", "th"):
                self.in_cell = True
                self.current = ""

        def handle_data(self, data):
            if self.in_cell:
                self.current += data + " "

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self.in_cell:
                self.cells.append(re.sub(r"\s+", " ", self.current).strip())
                self.in_cell = False
            if tag == "tr" and self.cells:
                self.rows.append(self.cells)
                self.cells = []

    def _nums(text):
        return [int(x) for x in re.findall(r"\d+", str(text or "")) if 1 <= int(x) <= 9]

    def _ai(text):
        return [int(n) for n, _ in re.findall(r"(\d)\s+([0-9]+(?:\.[0-9]+)?)%", str(text or ""))][:3]

    def _ai_probs(text):
        return [float(p) for _, p in re.findall(r"(\d)\s+([0-9]+(?:\.[0-9]+)?)%", str(text or ""))][:3]

    meet = str(meta.get("meet") or "").strip()
    rno = str(meta.get("race_no") or "").strip().lstrip("0") or "0"
    failures = 0
    for dt in [0, -1, 1]:
        try:
            url = f"https://{KCYCLE_IP}/rankingpredict/{year}/{tms+dt}/{day}"
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr",
                "User-Agent": "Mozilla/5.0",
                "X-Requested-With": "XMLHttpRequest",
            })
            with urllib.request.urlopen(req, timeout=2, context=ctx) as response:
                html = response.read().decode("utf-8", "replace")
            parser = _TableParser()
            parser.feed(html)
            for cells in parser.rows:
                if len(cells) < 6:
                    continue
                m = re.match(r"([가-힣]+)\s*0?(\d+)", cells[0])
                if not m or not m.group(1).startswith(meet) or str(int(m.group(2))) != str(int(rno)):
                    continue
                return {
                    "date": re.sub(r"\D", "", str(meta.get("ymd") or ""))[:8],
                    "day": day,
                    "meet": meet,
                    "race_no": int(rno),
                    "ai_order": _ai(cells[1]),
                    "ai_probs": _ai_probs(cells[1]),
                    "popular_order": _nums(cells[3])[:3] if len(cells) > 3 else [],
                    "hit5_order": _nums(cells[4])[:3] if len(cells) > 4 else [],
                    "return5_order": _nums(cells[5])[:3] if len(cells) > 5 else [],
                    "source": "kcycle_live",
                }
        except Exception:  # noqa: BLE001
            failures += 1
    if failures:
        _KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = time.time() + 300
    return None


def _kcycle_rankingpredict_signal(meta):
    key = _rankingpredict_key((meta or {}).get("ymd"), (meta or {}).get("meet"), (meta or {}).get("race_no"))
    row = _load_kcycle_rankingpredict_cache().get(key) if key else None
    if row is None:
        row = _fetch_kcycle_rankingpredict_row(meta)
    if not row:
        return None
    source_names = ["ai_order", "popular_order", "hit5_order", "return5_order"]
    orders = [row.get(name) for name in source_names]
    valid_orders = [order for order in orders if isinstance(order, list) and len(order) >= 3]
    ai_probs = row.get("ai_probs") if isinstance(row.get("ai_probs"), list) else []
    ai_p1 = float(ai_probs[0]) if ai_probs else 0.0
    all4_trio_agree = len(valid_orders) == 4 and len({tuple(order[:3]) for order in valid_orders}) == 1
    market_orders = [row.get(name) for name in ["popular_order", "hit5_order", "return5_order"]]
    valid_market_orders = [order for order in market_orders if isinstance(order, list) and len(order) >= 3]
    if (
        len(valid_market_orders) == 3
        and len({order[0] for order in valid_market_orders}) == 1
        and ai_p1 >= 21.0
        and _kcycle_day_no(row, meta) == 2
    ):
        leader = int(valid_market_orders[0][0])
        return {
            "tier": "kcycle_market3_day2_extreme",
            "label": "KCYCLE 2일차 시장합의 91%급 극고확신",
            "leader": leader,
            "order": [int(x) for x in valid_market_orders[0][:3]],
            "expected_top1": 0.9111,
            "coverage": 0.0352,
            "validation_n": 45,
            "validation_split": "2025 select n=94 -> 2026 OOS 광명 n=45",
            "rule": "2일차 + AI 1순위>=21% + 인기배당률·적중률5%·환급률5% 1착 일치",
            "pair_order_expected": 0.4649 if all4_trio_agree else 1.0,
            "trio_order_expected": 0.2719 if all4_trio_agree else 0.1778,
            "source": row.get("source", "kcycle_cache"),
        }
    if len(valid_orders) == 4 and len({order[0] for order in valid_orders}) == 1 and ai_p1 >= 22.0:
        leader = int(valid_orders[0][0])
        trio_agree = len({tuple(order[:3]) for order in valid_orders}) == 1
        return {
            "tier": "kcycle_all_first_agree",
            "label": "KCYCLE 공식합의 86%급 고확신",
            "leader": leader,
            "order": [int(x) for x in valid_orders[0][:3]],
            "expected_top1": 0.8649,
            "coverage": 0.0664,
            "validation_n": 111,
            "validation_split": "2025 select -> 2026 OOS",
            "rule": "AI 예측·인기배당률·적중률5%·환급률5% 1착 모두 일치",
            "pair_order_expected": 0.4649 if trio_agree else None,
            "trio_order_expected": 0.2719 if trio_agree else None,
            "source": row.get("source", "kcycle_cache"),
        }
    if len(valid_market_orders) == 3 and len({order[0] for order in valid_market_orders}) == 1:
        leader = int(valid_market_orders[0][0])
        return {
            "tier": "kcycle_market3_support",
            "label": "KCYCLE 시장3합의 보조픽",
            "leader": leader,
            "order": [int(x) for x in valid_market_orders[0][:3]],
            "expected_top1": 0.6656,
            "coverage": 0.7195,
            "validation_n": 921,
            "validation_split": "2025 select -> 2026 OOS 광명, 고확신 제외",
            "rule": "인기배당률·적중률5%·환급률5% 1착 일치",
            "pair_order_expected": 0.2762,
            "trio_order_expected": 0.1371,
            "source": row.get("source", "kcycle_cache"),
        }
    return None


def kcycle_rankingpredict_signal(meta):
    return _kcycle_rankingpredict_signal(meta)


def _replace_pick(picks, code, patch):
    for pick in picks:
        if pick.get("code") == code:
            pick.update(patch)
            return


def _apply_kcycle_pick_overlay(out, rows, signal):
    picks = out.get("picks")
    order = signal.get("order") if isinstance(signal, dict) else None
    if not isinstance(picks, list) or not isinstance(order, list) or not order:
        return
    by_bno = {int(r["bno"]): r for r in rows if "bno" in r}

    def lab(bno):
        row = by_bno.get(int(bno), {})
        name = str(row.get("name", "") or "").strip()
        return f"{int(bno)}번 {name}".strip()

    leader = int(order[0])
    expected_top1 = float(signal.get("expected_top1") or 0.0)
    _replace_pick(
        picks,
        "단승",
        {
            "pick": [lab(leader)],
            "prob": f"KCYCLE top1 {100*expected_top1:.1f}%",
            "grade": "강" if expected_top1 >= 0.86 else "중",
        },
    )
    if len(order) >= 2 and all(int(bno) in by_bno for bno in order[:2]):
        a, b = int(order[0]), int(order[1])
        _replace_pick(
            picks,
            "연승",
            {
                "pick": [lab(a), lab(b)],
                "prob": f"KCYCLE 1·2순위 후보 · top1 {100*expected_top1:.1f}%",
                "grade": "중",
            },
        )
    trio_expected = signal.get("trio_order_expected")
    if not isinstance(trio_expected, (int, float)) or trio_expected < 0.17:
        return
    if len(order) < 3 or not all(int(bno) in by_bno for bno in order[:3]):
        return
    a, b, c = [int(bno) for bno in order[:3]]
    pair_expected = signal.get("pair_order_expected")
    pair_text = f"pair {100*pair_expected:.1f}% · " if isinstance(pair_expected, (int, float)) else ""
    _replace_pick(
        picks,
        "복승",
        {
            "pick": [f"{lab(a)} ↔ {lab(b)}"],
            "prob": f"KCYCLE 순서신호 top2 · {pair_text}순서권 리스크",
            "grade": "중" if isinstance(pair_expected, (int, float)) and pair_expected >= 0.45 else "약",
        },
    )
    _replace_pick(
        picks,
        "쌍승",
        {
            "pick": [f"{lab(a)} → {lab(b)}"],
            "prob": f"KCYCLE 순서신호 top2 · {pair_text}순서권 리스크",
            "grade": "중" if isinstance(pair_expected, (int, float)) and pair_expected >= 0.45 else "약",
        },
    )
    _replace_pick(
        picks,
        "삼복",
        {
            "pick": [f"{lab(a)} ↔ {lab(b)} ↔ {lab(c)}"],
            "prob": f"KCYCLE 순서신호 top3 board · 삼쌍 exact {100*trio_expected:.1f}%",
            "grade": "중" if trio_expected >= 0.25 else "약",
        },
    )
    _replace_pick(
        picks,
        "쌍복",
        {
            "pick": [f"1착 {lab(a)} 고정 + ({lab(b)} ↔ {lab(c)})"],
            "prob": f"KCYCLE 1착 고정 · top1 {100*expected_top1:.1f}%",
            "grade": "강" if expected_top1 >= 0.86 else "중",
        },
    )
    _replace_pick(
        picks,
        "삼쌍",
        {
            "pick": [f"{lab(a)} → {lab(b)} → {lab(c)}"],
            "prob": f"KCYCLE 순서신호 top3 · exact {100*trio_expected:.1f}% · 50% 미만",
            "grade": "중" if trio_expected >= 0.25 else "약",
        },
    )


def _apply_kcycle_rankingpredict_overlay(out, rows, meta):
    signal = _kcycle_rankingpredict_signal(meta)
    if not signal:
        return out
    out["rankingpredict_signal"] = signal
    _apply_kcycle_pick_overlay(out, rows, signal)
    leader_row = next((r for r in rows if int(r.get("bno", -1)) == int(signal["leader"])), None)
    if leader_row is not None:
        out["top"] = leader_row
        if signal["tier"] == "kcycle_market3_day2_extreme":
            out["top_conf"] = {
                "grade": "강",
                "label": "KCYCLE 극고확신 픽",
                "icon": "✓",
                "race_confidence": "고확신",
            }
        elif signal["tier"] == "kcycle_all_first_agree":
            out["top_conf"] = {
                "grade": "강",
                "label": "KCYCLE 공식합의 픽",
                "icon": "✓",
                "race_confidence": "고확신",
            }
        else:
            out["top_conf"] = {
                "grade": "중",
                "label": "KCYCLE 보조합의 픽",
                "icon": "▲",
                "race_confidence": "보통",
            }
    out["selective_conf"] = {
        "tier": signal["tier"],
        "label": signal["label"],
        "expected_top1": signal["expected_top1"],
        "coverage": signal["coverage"],
        "rule": signal["rule"],
        "validation_n": signal["validation_n"],
        "validation_split": signal["validation_split"],
    }
    return out


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
    import urllib.request, ssl
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


def fetch_kcycle_trifecta_board(stnd_yr, ymd, race_no):
    import ssl
    tms_day = _resolve_kcycle_tms(stnd_yr, ymd)
    if not tms_day:
        return None
    year, tms, day = tms_day
    if day == 0:
        return None
    rno = (str(race_no).strip().lstrip("0") or "0").zfill(2)
    for dt in [0, -1, 1, -2, 2]:
        url = (f"https://{KCYCLE_IP}/race/dividendrate/final/"
               f"{year}/{tms+dt}/{day}/001/{rno}")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3, context=ctx) as r:
                html = r.read().decode("utf-8", "replace")
            board = parse_kcycle_trifecta_board(html)
            if len(board) >= 150:
                return board
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


def clear_keirin_card_page_cache():
    _KEIRIN_CARD_PAGE_CACHE.clear()


def keirin_card_page_cache_status():
    now = time.time()
    live = [
        hit for hit in _KEIRIN_CARD_PAGE_CACHE.values()
        if (now - hit["ts"]) < _KEIRIN_CARD_PAGE_TTL
    ]
    return {"pages": len(live), "ttl": _KEIRIN_CARD_PAGE_TTL}


def _api_page_cached(stnd_yr, page, rows, key, timeout=8):
    ck = (str(stnd_yr), int(page), int(rows), CARD_URL)
    now = time.time()
    hit = _KEIRIN_CARD_PAGE_CACHE.get(ck)
    if hit and (now - hit["ts"]) < _KEIRIN_CARD_PAGE_TTL:
        return hit["data"]
    data = _api_page(stnd_yr, page, rows, key, timeout=timeout)
    _KEIRIN_CARD_PAGE_CACHE[ck] = {"data": data, "ts": now}
    return data


def prewarm_keirin_card_pages(stnd_yr, key, rows=1000, max_pages=2):
    if not key:
        return {"warmed": 0, "pages": 0}
    tc, _ = _api_page_cached(stnd_yr, 1, 1, key)
    if not tc:
        return {"warmed": 1, "pages": keirin_card_page_cache_status()["pages"]}
    last_page = math.ceil(tc / rows)
    warmed = 1
    for p in range(last_page, max(0, last_page - max_pages), -1):
        _api_page_cached(stnd_yr, p, rows, key)
        warmed += 1
    return {"warmed": warmed, "pages": keirin_card_page_cache_status()["pages"]}


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
        tc, _ = _api_page_cached(stnd_yr, 1, 1, key)
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
            _, items = _api_page_cached(stnd_yr, p, rows, key)
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
            _, items = _api_page_cached(stnd_yr, p, rows, key)
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
            tc, _ = _api_page_cached(yr, 1, 1, key)
        except Exception:  # noqa: BLE001
            continue
        if not tc:
            continue
        last_page = math.ceil(tc / rows)
        for p in range(last_page, max(0, last_page - max_pages), -1):
            try:
                _, items = _api_page_cached(yr, p, rows, key)
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
            "prob": f"순서 top2 (win {100*pw[a]:.0f}% → {100*pw[b]:.0f}%) · 순서권 리스크",
            "grade": "약",
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
            "prob": f"순서 top3 (win {100*pw[a]:.0f}%) · 순서권 리스크",
            "grade": "약",
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


def _keirin_selective_confidence(top, rows=None):
    win_leader = top
    if rows:
        win_leader = max(rows, key=lambda r: r.get("pwin", 0.0))
    pwin = float(win_leader.get("pwin", 0.0))
    pplc = float(win_leader.get("pplc", 0.0))
    gap = 0.0
    if rows and len(rows) >= 2:
        sorted_rows = sorted(rows, key=lambda r: -r.get("pwin", 0))
        gap = float(sorted_rows[0].get("pwin", 0.0) - sorted_rows[1].get("pwin", 0.0))
    if pwin >= 0.7962011883201475 and pplc >= 0.9274714236121064:
        return {
            "tier": "ultra_fixed_86",
            "label": "86%급 고정 초고확신 선별",
            "expected_top1": 0.8593,
            "coverage": 0.1790,
            "rule": "top_pwin >= 79.6% AND top_pplc >= 92.7%",
            "validation_n": 2111,
        }
    if gap >= 0.6369486741602512:
        return {
            "tier": "ultra",
            "label": "85%급 초고확신 선별",
            "expected_top1": 0.8467,
            "coverage": 0.2168,
            "rule": "top1-top2 win gap >= 63.7%p",
            "rolling_weighted_top1": 0.8408,
            "rolling_coverage": 0.2476,
            "rolling_min_year_top1": 0.7915,
        }
    if gap >= 0.5646195759501839:
        return {
            "tier": "extreme_gap",
            "label": "82%급 고확신 확장",
            "expected_top1": 0.8214,
            "coverage": 0.3029,
            "rule": "top1-top2 win gap >= 56.5%p",
            "rolling_weighted_top1": 0.8086,
            "rolling_coverage": 0.3252,
            "rolling_min_year_top1": 0.7871,
        }
    if pplc >= 0.9072363230215373:
        return {
            "tier": "extreme",
            "label": "82%급 고확신 선별",
            "expected_top1": 0.8175,
            "coverage": 0.2765,
            "rule": "top_pplc >= 90.7%",
            "rolling_weighted_top1": 0.8086,
            "rolling_coverage": 0.3252,
            "rolling_min_year_top1": 0.7871,
        }
    if pwin >= 0.6067814186919052:
        return {
            "tier": "broad",
            "label": "73%급 고확신 선별",
            "expected_top1": 0.7287,
            "coverage": 0.5619,
            "rule": "top_pwin >= 60.7%",
            "rolling_weighted_top1": 0.7677,
            "rolling_coverage": 0.4598,
            "rolling_min_year_top1": 0.7544,
        }
    return {
        "tier": "normal",
        "label": "일반 예측",
        "expected_top1": 0.6166,
        "coverage": 1.0,
        "rule": "selective threshold 미충족",
    }


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
                out = {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "final_model": True,
                }
                return _apply_kcycle_rankingpredict_overlay(out, rows, meta)
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
                    out = {
                        "rows": rows, "picks": picks, "top": rows[0],
                        "top_conf": _top_confidence(rows[0], rows),
                        "meta": meta or {}, "n_starters": len(rows),
                        "model_special_11r": True,
                    }
                    return _apply_kcycle_rankingpredict_overlay(out, rows, meta)
        # 일반 11R+ 특화 (66%)
        m11, m11err = load_11r_model()
        if m11 is not None:
            rows, err = score_keirin_with_model(starters, m11)
            if err is None:
                picks = build_picks(rows)
                out = {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "model_11r": True,
                }
                return _apply_kcycle_rankingpredict_overlay(out, rows, meta)
    cross, cross_err = load_cross_model()
    if cross is not None:
        rows, err = score_keirin_with_model(starters, cross, meta=meta)
        if err is None:
            picks = build_picks(rows)
            out = {
                "rows": rows,
                "picks": picks,
                "top": rows[0],
                "top_conf": _top_confidence(rows[0], rows),
                "selective_conf": _keirin_selective_confidence(rows[0], rows),
                "meta": meta or {},
                "n_starters": len(rows),
                "model_cross_domain": True,
            }
            return _apply_kcycle_rankingpredict_overlay(out, rows, meta)

    # 일반 모델
    rows, err = score_keirin(starters, meta=meta)
    if err:
        return {"error": err}

    picks = build_picks(rows)
    out = {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "top_conf": _top_confidence(rows[0], rows),
        "meta": meta or {},
        "n_starters": len(rows),
    }
    return _apply_kcycle_rankingpredict_overlay(out, rows, meta)


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


def fetch_kcycle_trifecta_board_with_ts(stnd_yr, ymd, race_no):
    import datetime as _dt
    board = None
    try:
        board = fetch_kcycle_trifecta_board(stnd_yr, ymd, race_no)
    except Exception:
        board = None
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    return board, ts


def _live_poll_delay_ms(sport, market_used):
    if market_used:
        return 5000
    if sport == "keirin":
        return 15000
    return 30000


def _live_market_risk(sport, market_used, status):
    if sport != "keirin":
        return {
            "level": "not_applicable",
            "message": "경마 경로는 현재 경륜 KCYCLE 배당 수집 대상이 아닙니다.",
        }
    if market_used:
        return {
            "level": "odds_live",
            "message": "실시간 배당이 반영됐습니다.",
        }
    if os.environ.get("KCYCLE_ENABLED", "0") == "1":
        return {
            "level": "odds_unavailable",
            "message": "KCYCLE 배당 수집을 시도했지만 현재 배당이 없거나 접근이 실패했습니다.",
        }
    return {
        "level": "live_market_blocked",
        "message": "Render 미국 리전에서는 KCYCLE 실시간 배당 접근이 막힐 수 있어 사전 예측과 공식예상 보조신호만 사용합니다.",
    }


def _live_fallback_signal(base_model_out):
    signal = base_model_out.get("rankingpredict_signal")
    if not signal:
        signal = base_model_out.get("selective_conf")
    if not signal or signal.get("tier") in (None, "normal"):
        return None
    return _live_signal_payload(signal)


def _live_signal_payload(signal):
    return {
        "tier": signal.get("tier"),
        "label": signal.get("label"),
        "order": signal.get("order"),
        "expected_top1": signal.get("expected_top1"),
        "expected_trio_exact": signal.get("expected_trio_exact"),
        "observed_trio_exact": signal.get("observed_trio_exact"),
        "baseline_trio_exact": signal.get("baseline_trio_exact"),
        "lift_pp": signal.get("lift_pp"),
        "coverage": signal.get("coverage"),
        "rule": signal.get("rule"),
        "validation_n": signal.get("validation_n"),
        "validation_split": signal.get("validation_split"),
        "robust_status": signal.get("robust_status"),
        "robust_warning": signal.get("robust_warning"),
    }


def _market_favorite_signal(market_odds):
    valid = {int(b): float(o) for b, o in (market_odds or {}).items() if o and float(o) > 0}
    if not valid:
        return None
    leader, fav_odds = min(valid.items(), key=lambda kv: (kv[1], kv[0]))
    if fav_odds <= 1.0:
        return {
            "tier": "market_fav_odds_le_1_0",
            "label": "KCYCLE 실시간 강축 배당 89%급",
            "leader": leader,
            "favorite_odds": fav_odds,
            "expected_top1": 0.8896,
            "coverage": 0.1171,
            "validation_n": 299,
            "validation_split": "최근 KCYCLE 단승배당 OOS n=299",
            "rule": "실시간 단승 최저배당 <= 1.0",
        }
    if fav_odds <= 1.1:
        return {
            "tier": "market_fav_odds_le_1_1",
            "label": "KCYCLE 실시간 강축 배당 83%급",
            "leader": leader,
            "favorite_odds": fav_odds,
            "expected_top1": 0.8289,
            "coverage": 0.2656,
            "validation_n": 678,
            "validation_split": "최근 KCYCLE 단승배당 OOS n=678",
            "rule": "실시간 단승 최저배당 <= 1.1",
        }
    return None


def _market_odds_entries(market_odds, trifecta_board=None, max_entries=12):
    entries = []
    win_valid = []
    for bno, odds in (market_odds or {}).items():
        try:
            b = int(bno)
            o = float(odds)
        except (TypeError, ValueError):
            continue
        if b <= 0 or o <= 0:
            continue
        win_valid.append((b, o))
    for index, (b, odds) in enumerate(sorted(win_valid, key=lambda kv: (kv[1], kv[0]))):
        entries.append({
            "code": "WIN",
            "label": "단승",
            "selection": str(b),
            "odds": round(odds, 2),
            "change": "실시간",
            "signal": "teal" if index == 0 else "primary",
        })
        if len(entries) >= max_entries:
            return entries

    tri_valid = []
    for combo, odds in (trifecta_board or {}).items():
        if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)):
            continue
        try:
            o = float(odds)
        except (TypeError, ValueError):
            continue
        if o <= 0:
            continue
        tri_valid.append((str(combo), o))
    for combo, odds in sorted(tri_valid, key=lambda kv: (kv[1], kv[0]))[:3]:
        entries.append({
            "code": "TRI",
            "label": "삼쌍",
            "selection": combo,
            "odds": round(odds, 2),
            "change": "실시간",
            "signal": "violet",
        })
        if len(entries) >= max_entries:
            break
    return entries


def _market_trifecta_signal(trifecta_board):
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) < 150:
        return None
    ranked = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))
    if len(ranked) < 2:
        return None
    best_combo, best_odds = ranked[0]
    if best_odds <= 0:
        return None
    second_odds = ranked[1][1]
    gap12 = second_odds / best_odds
    implied = {combo: 1.0 / odds for combo, odds in valid.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    a, b, c = [int(x) for x in best_combo.split("-")]
    pair_prefix = f"{a}-{b}-"
    pair12_mass = sum(v for combo, v in implied.items() if combo.startswith(pair_prefix)) / total
    if gap12 < 2.28571 or pair12_mass < 0.532879:
        return None
    return {
        "tier": "market_trifecta_watch_low_sample",
        "label": "KCYCLE 삼쌍 시장강합의 watch(저표본)",
        "leader": a,
        "order": [a, b, c],
        "favorite_odds": best_odds,
        "gap12": gap12,
        "pair12_mass": pair12_mass,
        "expected_trio_exact": None,
        "observed_trio_exact": 0.5,
        "baseline_trio_exact": 0.2719,
        "lift_pp": 22.81,
        "coverage": 0.0217,
        "validation_n": 30,
        "validation_split": "2026 OOS n=30 exact 50.0%; 2025 n=41 exact 53.7%; 2024 n=6",
        "rule": "전체 삼쌍 최저배당 gap12>=2.28571 + 같은 1-2순서 암시확률>=53.2879%",
        "robust_status": "failed_small_n",
        "robust_warning": "50%로 배포 금지: 2024 n=6이라 robust promotion gate(n>=10)를 통과하지 못했습니다.",
    }


def _trifecta_board_hash(trifecta_board):
    import hashlib
    payload = json.dumps(
        sorted((str(combo), float(odds)) for combo, odds in (trifecta_board or {}).items()),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _snapshot_key_token(key):
    return "\t".join(str(part) for part in key)


def _snapshot_index_path(path):
    return f"{path}.keys"


def _load_snapshot_file_keys(path):
    cached = _KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS.get(path)
    if cached is not None:
        return cached
    keys = set()
    index_path = _snapshot_index_path(path)
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            keys = {line.strip() for line in f if line.strip()}
    elif os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (
                    str(row.get("date", "")),
                    str(row.get("meet", "")),
                    str(row.get("race_no", "")).zfill(2),
                    str(row.get("board_hash", "")),
                )
                keys.add(_snapshot_key_token(key))
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            for token in sorted(keys):
                f.write(token + "\n")
    _KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS[path] = keys
    return keys


def save_kcycle_trifecta_snapshot(stnd_yr, ymd, meet, race_no, trifecta_board, fetched_at=None, signal=None, source="live_decision"):
    if os.environ.get("KCYCLE_TRIFECTA_SNAPSHOT_ENABLED", "1") != "1":
        return False
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) != 210:
        return False
    import datetime as _dt
    board_hash = _trifecta_board_hash(valid)
    ymd_key = re.sub(r"\D", "", str(ymd or ""))[:8]
    key = (ymd_key, str(meet or ""), str(race_no or ""), board_hash)
    now = time.time()
    min_interval = float(os.environ.get("KCYCLE_TRIFECTA_SNAPSHOT_MIN_INTERVAL_SEC", "60") or "0")
    last = _KCYCLE_TRIFECTA_SNAPSHOT_LAST.get(key)
    if last is not None and now - last < min_interval:
        return False
    _KCYCLE_TRIFECTA_SNAPSHOT_LAST[key] = now
    path = os.environ.get("KCYCLE_TRIFECTA_SNAPSHOT_PATH", KCYCLE_TRIFECTA_SNAPSHOT_PATH)
    file_key = (ymd_key, str(meet or ""), str(race_no or "").zfill(2), board_hash)
    token = _snapshot_key_token(file_key)
    file_keys = _load_snapshot_file_keys(path)
    if token in file_keys:
        return False
    best20 = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))[:20]
    record = {
        "schema": "kcycle_trifecta_snapshot_v1",
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "fetched_at": fetched_at,
        "source": source,
        "stnd_yr": str(stnd_yr or ""),
        "date": ymd_key or str(ymd or ""),
        "meet": str(meet or "광명"),
        "race_no": str(race_no or ""),
        "board_count": len(valid),
        "board_hash": board_hash,
        "best20": best20,
        "signal": _live_signal_payload(signal) if signal else None,
        "board": dict(sorted(valid.items())),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(_snapshot_index_path(path), "a", encoding="utf-8") as f:
        f.write(token + "\n")
    file_keys.add(token)
    return True


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
        signal = None
        if sport == "keirin":
            signal = _kcycle_rankingpredict_signal({"ymd": ymd, "meet": meet, "race_no": race_no})
        fallback_signal = _live_signal_payload(signal) if signal else None
        top = None
        if signal:
            top = {
                "bno": signal.get("leader"),
                "name": "KCYCLE 공식예상",
                "pwin": signal.get("expected_top1", 0.0),
                "pplc": 0.0,
            }
        return {
            "ok": bool(signal),
            "status": "hold",
            "message": (
                "출주표 기반 모델 예측 불가 — KCYCLE 공식예상 보조신호만 표시"
                if signal else "모델 예측 불가 (출주표 없음 또는 오류)"
            ),
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None,
            "market_odds": [],
            "top": top,
            "rows": None,
            "decision": "hold",
            "market_used": False,
            "snapshot_phase": "unknown",
            "poll_delay_ms": _live_poll_delay_ms(sport, False),
            "market_risk": _live_market_risk(sport, False, "hold"),
            "fallback_signal": fallback_signal,
            "trifecta_signal": None,
        }

    rows = [dict(r) for r in base_model_out.get("rows", [])]
    if not rows:
        return {
            "ok": False, "status": "hold", "message": "출주 데이터 없음",
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None, "market_odds": [], "top": None,
            "rows": None, "decision": "hold", "market_used": False,
            "snapshot_phase": "unknown",
            "poll_delay_ms": _live_poll_delay_ms(sport, False),
            "market_risk": _live_market_risk(sport, False, "hold"),
            "fallback_signal": None,
            "trifecta_signal": None,
        }

    # ── kcycle 배당 fetch (경륜만) ──
    market_odds = None
    fetched_at = None
    market_used = False
    market_signal = None
    trifecta_signal = None
    trifecta_board = None
    live_market_used = False
    if sport == "keirin" and os.environ.get("KCYCLE_ENABLED", "0") == "1":
        stnd_yr = str(ymd)[:4] if ymd else ""
        try:
            market_odds, fetched_at = fetch_kcycle_odds_with_ts(stnd_yr, ymd, race_no)
        except Exception:
            market_odds = None
        if KCYCLE_TRIFECTA_ENABLED:
            try:
                trifecta_board, trifecta_fetched_at = fetch_kcycle_trifecta_board_with_ts(stnd_yr, ymd, race_no)
                trifecta_signal = _market_trifecta_signal(trifecta_board)
                save_kcycle_trifecta_snapshot(
                    stnd_yr,
                    ymd,
                    meet,
                    race_no,
                    trifecta_board,
                    fetched_at=trifecta_fetched_at,
                    signal=trifecta_signal,
                )
                fetched_at = fetched_at or trifecta_fetched_at
            except Exception:
                trifecta_signal = None

    # ── 상태 판정 ──
    if market_odds and len(market_odds) >= 2:
        # 시장 암시확률 → 모델과 앙상블
        imp = {b: 1.0 / o for b, o in market_odds.items() if o and o > 0}
        total = sum(imp.values())
        if total > 0:
            market_signal = _market_favorite_signal(market_odds)
            imp_norm = {b: v / total for b, v in imp.items()}
            for r in rows:
                bno = r.get("bno", 0)
                mkt_p = imp_norm.get(bno, 0.0)
                model_p = r.get("pwin", 0.0)
                r["pwin_blended"] = 0.3 * model_p + 0.7 * mkt_p
                r["mkt_pwin"] = mkt_p
            # blended 로 재정렬
            rows.sort(key=lambda r: -r.get("pwin_blended", r.get("pwin", 0)))
            if market_signal:
                leader_row = next((r for r in rows if int(r.get("bno", -1)) == market_signal["leader"]), None)
                if leader_row is not None:
                    leader_row["pwin_blended"] = max(
                        float(leader_row.get("pwin_blended", 0.0)),
                        float(market_signal["expected_top1"]),
                    )
                    rows.sort(key=lambda r: -r.get("pwin_blended", r.get("pwin", 0)))
            rows[0]["pwin"] = rows[0].get("pwin_blended", rows[0]["pwin"])
            market_used = True
            status = "odds_live"
            message = (
                f"{market_signal['label']} 반영"
                if market_signal else "실시간 배당 반영 (시장 0.7 + 모델 0.3)"
            )
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
    if trifecta_signal:
        suffix = " · 삼쌍 저표본 watch 감지(50% 배포 금지)"
        if suffix not in message:
            message = f"{message}{suffix}"
    live_market_used = market_used or bool(trifecta_signal)

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
        "market_odds": _market_odds_entries(market_odds, trifecta_board),
        "top": top,
        "rows": rows,
        "decision": decision,
        "market_used": market_used,
        "snapshot_phase": snapshot_phase,
        "poll_delay_ms": _live_poll_delay_ms(sport, live_market_used),
        "market_risk": _live_market_risk(sport, live_market_used, status),
        "fallback_signal": _live_fallback_signal(base_model_out),
        "market_signal": _live_signal_payload(market_signal) if market_signal else None,
        "trifecta_signal": _live_signal_payload(trifecta_signal) if trifecta_signal else None,
    }

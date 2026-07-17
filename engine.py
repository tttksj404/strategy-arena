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
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

from kra_history_features import apply_horse_history_snapshot
from kra_pairwise_ranker import PairwiseModel, pairwise_scores
from kra_pairwise_reranker import RaceScores, RerankPolicy, restricted_rerank
from scripts.kcycle_eval_common import FEATURE_NAMES as KCYCLE_ENSEMBLE_FEATURE_NAMES
from scripts.kcycle_eval_common import feature_rows_from_board

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_model_final.joblib")
CROSS_MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_cross_domain_model.joblib")
FINAL_MODEL_PATH = os.path.join(HERE, "static", "models", "keirin_final_model.joblib")
MODEL_11R_PATH = os.path.join(HERE, "static", "models", "keirin_11r_plus_model.joblib")
SPECIAL_11R_PATH = os.path.join(HERE, "static", "models", "keirin_special_11r_model.joblib")
DEMO_PATH = os.path.join(HERE, "data", "demo_race.json")

# ── 경마(KRA) ──
KRA_MODEL_PATH = os.path.join(HERE, "static", "models", "kra_model.joblib")
KRA_CONFIDENCE_TIERS_PATH = os.path.join(HERE, "static", "models", "kra_confidence_tiers_v1.json")
KRA_DEMO_PATH = os.path.join(HERE, "data", "demo_kra_race.json")
# KRA 카드 API (race_result 와 동일 엔드포인트). 키만 비밀.
DEFAULT_KRA_URL = ("https://apis.data.go.kr/B551015/API214_1/RaceDetailResult_1")
KRA_CARD_URL = os.environ.get("KRA_CARD_URL", DEFAULT_KRA_URL)
# 학습 데이터에 존재하는 경주장 (실측)
KRA_MEETS = ["서울", "제주", "부경"]
KRA_PICK_POLICY_DEFAULT = "market_if_odds"
KRA_PICK_POLICIES = {"current_gate", "market_if_odds", "market_except_weak_disagree"}
KEIRIN_PICK_POLICY_DEFAULT = "market_if_board"
KEIRIN_PICK_POLICIES = {"market_if_board", "model_always"}

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
PARTICIPANT_LEARNING_PRIORS_PATH = os.environ.get(
    "PARTICIPANT_LEARNING_PRIORS_PATH",
    os.path.join(HERE, "data", "participant_learning_priors.json"),
)
KCYCLE_GLOBAL_RERANK_RESULTS_PATH = os.environ.get(
    "KCYCLE_GLOBAL_RERANK_RESULTS_PATH",
    os.path.join(HERE, "data", "kcycle_global_breakthrough_results.json"),
)
KCYCLE_MARKET_BLEND_RESULTS_PATH = os.environ.get(
    "KCYCLE_MARKET_BLEND_RESULTS_PATH",
    os.path.join(HERE, "data", "kcycle_market_blend_experiment_results.json"),
)
KCYCLE_TRIFECTA_ENSEMBLE_PATH = os.environ.get(
    "KCYCLE_TRIFECTA_ENSEMBLE_PATH",
    os.path.join(HERE, "static", "models", "kcycle_trifecta_ensemble_v1.json"),
)

CIRCLE = {chr(0x2460 + i): i + 1 for i in range(9)}

# 경륜 데이터가 존재하는 경주장(실측: race_card 는 '광명'만)
KEIRIN_MEETS = ["광명"]
KCYCLE_MEET_CODES = {"광명": "001", "창원": "002", "부산": "003"}

# (stnd_yr, week_tcnt, day_tcnt) → kcycle (tms, day) 매핑 캐시
_KCYCLE_TMS_CACHE = {}
_KCYCLE_RANKINGPREDICT = None
_KCYCLE_RANKINGPREDICT_LIVE_DISABLED_UNTIL = 0.0
_KEIRIN_CARD_PAGE_CACHE = {}
_KEIRIN_CARD_PAGE_CACHE_LOCK = threading.RLock()
_KEIRIN_CARD_PAGE_TTL = int(os.environ.get("KEIRIN_CARD_PAGE_TTL", "1800"))
_KCYCLE_TRIFECTA_SNAPSHOT_LAST = {}
_KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS = {}
_KCYCLE_RACE_START_CACHE = {}
_KRA_RACE_START_CACHE = {}
_KRA_CONFIDENCE_TIERS_CACHE = {"path": None, "mtime": None, "payload": None}
_PARTICIPANT_LEARNING_CACHE = {"path": None, "mtime": None, "payload": {}}
_KCYCLE_GLOBAL_RERANK_CACHE = {"path": None, "mtime": None, "payload": None}
_KCYCLE_LOW_ODDS_BLEND_CACHE = {"path": None, "mtime": None, "policy": None}
_KCYCLE_TRIFECTA_ENSEMBLE_CACHE = {"path": None, "mtime": None, "payload": None}


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
    if os.environ.get("KCYCLE_RANKINGPREDICT_LIVE_FETCH_ENABLED", "0") != "1":
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


def fetch_kcycle_odds(stnd_yr, ymd, race_no, timeout=3, max_attempts=5):
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
    for dt in [0, -1, 1, -2, 2][:max(1, int(max_attempts))]:
        url = (f"https://{KCYCLE_IP}/race/dividendrate/final/"
               f"{year}/{tms+dt}/{day}/001/{rno2}")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
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


def fetch_kcycle_trifecta_board(stnd_yr, ymd, race_no, timeout=3, max_attempts=5):
    import ssl
    tms_day = _resolve_kcycle_tms(stnd_yr, ymd)
    if not tms_day:
        return None
    year, tms, day = tms_day
    if day == 0:
        return None
    rno = (str(race_no).strip().lstrip("0") or "0").zfill(2)
    for dt in [0, -1, 1, -2, 2][:max(1, int(max_attempts))]:
        url = (f"https://{KCYCLE_IP}/race/dividendrate/final/"
               f"{year}/{tms+dt}/{day}/001/{rno}")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr", "User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                html = r.read().decode("utf-8", "replace")
            board = parse_kcycle_trifecta_board(html)
            if len(board) >= 150:
                return board
        except Exception:
            continue
    return None


def parse_kcycle_result_popup(html):
    parser = _KcycleTableTextParser()
    parser.feed(html or "")
    finish_by_rank = {}
    racers = []
    payouts = {}
    in_result_table = False
    in_payout_table = False
    for cells in parser.rows:
        if cells[:2] == ["선수명", "순위"]:
            in_result_table = True
            in_payout_table = False
            continue
        if cells[:3] == ["승식", "승자", "평균확정배당률"]:
            in_result_table = False
            in_payout_table = True
            continue
        if in_result_table and len(cells) >= 2:
            m = re.match(r"\s*([1-7])\s+(.+)", str(cells[0] or ""))
            if not m:
                continue
            try:
                bno = int(m.group(1))
                rank = int(str(cells[1]).strip())
            except ValueError:
                continue
            racers.append({"bno": bno, "name": m.group(2).strip(), "rank": rank})
            if rank in {1, 2, 3}:
                finish_by_rank[rank] = bno
        elif in_payout_table and len(cells) >= 3:
            bet_type = str(cells[0] or "").strip()
            winner = re.sub(r"\s+", " ", str(cells[1] or "")).strip()
            odds = _kcycle_float(cells[2])
            if bet_type and winner and odds:
                payouts[bet_type] = {"winner": winner, "odds": odds}
    actual_order = (
        [finish_by_rank[1], finish_by_rank[2], finish_by_rank[3]]
        if {1, 2, 3} <= set(finish_by_rank)
        else []
    )
    return {"actual_order": actual_order, "racers": racers, "payouts": payouts} if actual_order else None


def fetch_kcycle_result_outcome(stnd_yr, ymd, meet, race_no, timeout=0.75, max_attempts=1):
    import ssl
    tms_day = _resolve_kcycle_tms(stnd_yr, ymd)
    if not tms_day:
        return None
    year, tms, day = tms_day
    if day == 0:
        return None
    meet_code = KCYCLE_MEET_CODES.get(str(meet or "").strip())
    if not meet_code:
        return None
    rno = (str(race_no).strip().lstrip("0") or "0").zfill(2)
    for dt in [0, -1, 1, -2, 2][:max(1, int(max_attempts))]:
        url = (f"https://{KCYCLE_IP}/race/result/general/popup/"
               f"{year}/{tms+dt}/{day}/{meet_code}/{rno}")
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr",
                "User-Agent": "Mozilla/5.0",
                "Referer": f"https://www.kcycle.or.kr/race/result/general/{year}/{tms+dt}/{day}",
            })
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                html = r.read().decode("utf-8", "replace")
            parsed = parse_kcycle_result_popup(html)
            if parsed:
                parsed.update({
                    "source": "kcycle_result_popup",
                    "source_url": url.replace(str(KCYCLE_IP), "www.kcycle.or.kr"),
                })
                return parsed
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
    with _KEIRIN_CARD_PAGE_CACHE_LOCK:
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
    with _KEIRIN_CARD_PAGE_CACHE_LOCK:
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


def fetch_race_card(stnd_yr, ymd, meet, race_no, key, rows=1000, max_pages=6, timeout=8):
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
        tc, _ = _api_page_cached(stnd_yr, 1, 1, key, timeout=timeout)
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
            _, items = _api_page_cached(stnd_yr, p, rows, key, timeout=timeout)
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
            _, items = _api_page_cached(stnd_yr, p, rows, key, timeout=timeout)
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


def _recent_keirin_days(meet, key, n, rows=1000, max_pages=2, timeout=1.5):
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
            tc, _ = _api_page_cached(yr, 1, 1, key, timeout=timeout)
        except Exception:  # noqa: BLE001
            continue
        if not tc:
            continue
        last_page = math.ceil(tc / rows)
        for p in range(last_page, max(0, last_page - max_pages), -1):
            try:
                _, items = _api_page_cached(yr, p, rows, key, timeout=timeout)
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


def load_participant_learning_priors():
    path = os.environ.get("PARTICIPANT_LEARNING_PRIORS_PATH", PARTICIPANT_LEARNING_PRIORS_PATH)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return {}
    if _PARTICIPANT_LEARNING_CACHE.get("path") == path and _PARTICIPANT_LEARNING_CACHE.get("mtime") == mtime:
        return _PARTICIPANT_LEARNING_CACHE.get("payload") or {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    _PARTICIPANT_LEARNING_CACHE.update({"path": path, "mtime": mtime, "payload": payload})
    return payload


def _participant_learning_weight(priors=None):
    priors = priors if isinstance(priors, dict) else {}
    env_value = os.environ.get("PARTICIPANT_LEARNING_WEIGHT")
    try:
        if env_value is not None:
            return max(0.0, min(1.0, float(env_value)))
        return max(0.0, min(1.0, float(priors.get("learning_weight", 0.35))))
    except (TypeError, ValueError):
        return 0.35


def _participant_learning_probability(value, delta, weight):
    try:
        base = float(value)
    except (TypeError, ValueError):
        base = 0.0
    try:
        shift = float(delta) * weight
    except (TypeError, ValueError):
        shift = 0.0
    return max(0.001, min(0.999, base + shift))


def apply_participant_learning(rows, sport):
    if os.environ.get("PARTICIPANT_LEARNING_ENABLED", "1") != "1":
        return rows
    priors = load_participant_learning_priors()
    if not priors or priors.get("enabled") is False:
        return rows
    sports = priors.get("sports")
    sport_payload = sports.get(sport) if isinstance(sports, dict) else None
    participants = sport_payload.get("participants") if isinstance(sport_payload, dict) else None
    if not isinstance(participants, dict):
        return rows
    try:
        min_starts = int(priors.get("min_starts_for_live_adjustment") or 5)
    except (TypeError, ValueError):
        min_starts = 5
    weight = _participant_learning_weight(priors)
    adjusted = []
    for row in rows:
        out = dict(row)
        name = str(out.get("name") or "").strip()
        prior = participants.get(name)
        if isinstance(prior, dict):
            try:
                starts = int(prior.get("starts") or 0)
            except (TypeError, ValueError):
                starts = 0
            if starts >= min_starts:
                out["pwin_base"] = out.get("pwin")
                out["pplc_base"] = out.get("pplc")
                out["pwin"] = _participant_learning_probability(out.get("pwin"), prior.get("win_delta"), weight)
                out["pplc"] = _participant_learning_probability(out.get("pplc"), prior.get("podium_delta"), weight)
                out["learning_starts"] = starts
                out["learning_win_delta"] = float(prior.get("win_delta") or 0.0)
                out["learning_podium_delta"] = float(prior.get("podium_delta") or 0.0)
        adjusted.append(out)
    return adjusted


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
    rows = apply_participant_learning(rows, "keirin")
    rows.sort(key=lambda r: -r["pplc"])
    return rows, None


# ───────────────────────── Harville 7권종 픽 ─────────────────────────


def _harville_order(rows):
    """win확률 내림차순 마번 리스트 (Harville 순서픽 근사)."""
    return [
        row["bno"]
        for row in sorted(rows, key=lambda item: -item.get("rank_score", item["pwin"]))
    ]


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


KEIRIN_GRADE_POLICIES = {
    "특선": {
        "strong_pwin": 0.55,
        "mid_pwin": 0.35,
        "high_gap": 0.22,
        "mixed_gap": 0.10,
        "final_gap": 0.22,
        "read": "특선 상향평준화: 강자끼리 붙어 수치 우위만으로는 과신하지 않습니다.",
    },
    "우수": {
        "strong_pwin": 0.50,
        "mid_pwin": 0.30,
        "high_gap": 0.16,
        "mixed_gap": 0.08,
        "final_gap": 0.16,
        "read": "우수 중간 난도: 수치 우위와 격차를 균형 있게 봅니다.",
    },
    "선발": {
        "strong_pwin": 0.45,
        "mid_pwin": 0.28,
        "high_gap": 0.14,
        "mixed_gap": 0.06,
        "final_gap": 0.12,
        "read": "선발 수치우위형: 기록·득점 우위가 비교적 직접적으로 반영됩니다.",
    },
}

DEFAULT_KEIRIN_GRADE_POLICY = {
    "strong_pwin": 0.50,
    "mid_pwin": 0.30,
    "high_gap": 0.25,
    "mixed_gap": 0.08,
    "final_gap": 0.15,
    "read": "일반 난도: 모델 확률과 선두 격차를 기본 기준으로 봅니다.",
}


def _keirin_grade_context(rows=None):
    counts = {}
    for row in rows or []:
        grade = str(row.get("grade") or "").strip()
        if grade in KEIRIN_GRADE_POLICIES:
            counts[grade] = counts.get(grade, 0) + 1
    if not counts:
        return "", DEFAULT_KEIRIN_GRADE_POLICY
    grade = max(counts.items(), key=lambda item: (item[1], item[0]))[0]
    if not _keirin_grade_policy_enabled():
        return "", DEFAULT_KEIRIN_GRADE_POLICY
    return grade, KEIRIN_GRADE_POLICIES[grade]


def _keirin_grade_policy_enabled(priors=None):
    mode = str(os.environ.get("KEIRIN_GRADE_POLICY_MODE", "validated") or "validated").strip().lower()
    if mode in {"0", "off", "false", "baseline", "disabled"}:
        return False
    if mode in {"1", "on", "true", "force", "enabled"}:
        return True

    priors = priors if isinstance(priors, dict) else load_participant_learning_priors()
    validation = priors.get("grade_policy_validation") if isinstance(priors, dict) else None
    if not isinstance(validation, dict):
        return False
    return bool(validation.get("deployable")) and validation.get("selected_policy") == "grade_context"


def _grade_adjusted_win_grade(pwin, policy):
    if pwin >= float(policy["strong_pwin"]):
        return "강"
    if pwin >= float(policy["mid_pwin"]):
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


def build_picks(rows, trifecta_board=None, market_first_order=None, market_marginals=None, market_place_order=None):
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
    by_win = sorted(rows, key=lambda row: -row.get("rank_score", row["pwin"]))
    by_plc = sorted(rows, key=lambda r: -r["pplc"])
    order = _harville_order(rows)  # win 내림차순 마번
    pw = {r["bno"]: r["pwin"] for r in rows}
    pp = {r["bno"]: r["pplc"] for r in rows}
    nm = {r["bno"]: r["name"] for r in rows}
    by_bno = {int(r["bno"]): r for r in rows}
    if market_marginals is None:
        market_marginals = _keirin_trifecta_board_marginals(trifecta_board)
    if market_first_order is None:
        market_first_order = list(market_marginals.get("first_order", [])) if market_marginals else []
    market_first_order = [
        int(bno) for bno in (market_first_order or [])
        if int(bno) in by_bno
    ]
    use_market_win_place = len(market_first_order) >= 2
    market_place_order = [
        int(bno) for bno in (market_place_order or [])
        if int(bno) in by_bno
    ]
    use_market_place = len(market_place_order) >= 1

    def lab(b):
        n = nm.get(b, "")
        return f"{b}번 {n}".strip()

    picks = []

    # 단승: win top1
    t = by_bno[market_first_order[0]] if use_market_win_place else by_win[0]
    picks.append({
        "code": "단승", "desc": "1착 적중", "type": "단일",
        "pick": [lab(t["bno"])],
        "prob": (
            "삼쌍 배당 기준 근사 · first_mass 1위"
            if use_market_win_place else f"win {100*t['pwin']:.1f}%"
        ),
        "grade": _grade_win(t["pwin"]),
        "pick_source": "market" if use_market_win_place else "model",
    })

    # 연승: plc top1·2 (둘 중 1마라도 2착 이내면 적중)
    if use_market_place:
        y = [by_bno[bno] for bno in market_place_order[:1]]
        place_prob = "plcOdds 최저 기준 픽"
        place_source = "market"
    elif use_market_win_place:
        y = [by_bno[bno] for bno in market_first_order[:2]]
        place_prob = "삼쌍 배당 기준 근사 · first_mass 상위 2"
        place_source = "market"
    else:
        y = by_plc[:2]
        place_prob = " / ".join(f"연대 {100*r['pplc']:.0f}%" for r in y)
        place_source = "model"
    picks.append({
        "code": "연승", "desc": "2착 이내 적중", "type": "복수",
        "pick": [lab(r["bno"]) for r in y],
        "prob": place_prob,
        "grade": _grade_plc(max(r["pplc"] for r in y)),
        "pick_source": place_source,
    })

    if len(order) >= 2:
        market_pair = None
        market_exacta = None
        if market_marginals:
            unordered_pairs = [
                pair for pair in market_marginals.get("unordered_pair_order", [])
                if pair[0] in by_bno and pair[1] in by_bno
            ]
            ordered_pairs = [
                pair for pair in market_marginals.get("ordered_pair_order", [])
                if pair[0] in by_bno and pair[1] in by_bno
            ]
            market_pair = unordered_pairs[0] if unordered_pairs else None
            market_exacta = ordered_pairs[0] if ordered_pairs else None
        a, b = market_pair if market_pair else (order[0], order[1])
        # 복승: 무순 top2
        picks.append({
            "code": "복승", "desc": "1·2착 마번 무순", "type": "조합(무순2)",
            "pick": [f"{lab(a)} ↔ {lab(b)}"],
            "prob": (
                "삼쌍 배당 기준 근사 · 1-2슬롯 무순 pair_mass 1위"
                if market_pair else f"Harville top2 (win {100*pw[a]:.0f}% · {100*pw[b]:.0f}%)"
            ),
            "grade": _grade_plc((pp[a] + pp[b]) / 2),
            "pick_source": "market" if market_pair else "model",
        })
        # 쌍승: 순서 top2
        a, b = market_exacta if market_exacta else (order[0], order[1])
        picks.append({
            "code": "쌍승", "desc": "1착→2착 순서", "type": "조합(순서2)",
            "pick": [f"{lab(a)} → {lab(b)}"],
            "prob": (
                "삼쌍 배당 기준 근사 · 1-2슬롯 ordered pair_mass 1위"
                if market_exacta else f"순서 top2 (win {100*pw[a]:.0f}% → {100*pw[b]:.0f}%) · 순서권 리스크"
            ),
            "grade": "약",
            "pick_source": "market" if market_exacta else "model",
        })

    if len(order) >= 3:
        market_trio = None
        market_pair = None
        if market_marginals:
            unordered_trios = [
                trio for trio in market_marginals.get("unordered_trio_order", [])
                if trio[0] in by_bno and trio[1] in by_bno and trio[2] in by_bno
            ]
            unordered_pairs = [
                pair for pair in market_marginals.get("unordered_pair_order", [])
                if pair[0] in by_bno and pair[1] in by_bno
            ]
            market_trio = unordered_trios[0] if unordered_trios else None
            market_pair = unordered_pairs[0] if unordered_pairs else None
        a, b, c = market_trio if market_trio else (order[0], order[1], order[2])
        # 삼복: 무순 top3
        picks.append({
            "code": "삼복", "desc": "1·2·3착 마번 무순", "type": "조합(무순3)",
            "pick": [f"{lab(a)} ↔ {lab(b)} ↔ {lab(c)}"],
            "prob": (
                "삼쌍 배당 기준 근사 · 3슬롯 무순 trio_mass 1위"
                if market_trio else "Harville 무순 top3"
            ),
            "grade": _grade_plc((pp[a] + pp[b] + pp[c]) / 3),
            "pick_source": "market" if market_trio else "model",
        })
        # 쌍복: 1위 고정 + 2·3위 무순
        if market_pair:
            a, b = market_pair
            qpl_pick = f"{lab(a)} ↔ {lab(b)}"
            qpl_prob = "삼쌍 배당 기준 근사 · 1-2슬롯 무순 pair_mass 1위"
            qpl_grade = _grade_plc((pp[a] + pp[b]) / 2)
            qpl_source = "market"
        else:
            a, b, c = order[0], order[1], order[2]
            qpl_pick = f"1착 {lab(a)} 고정 + ({lab(b)} ↔ {lab(c)})"
            qpl_prob = f"1착 고정 win {100*pw[a]:.0f}%"
            qpl_grade = _grade_win(pw[a])
            qpl_source = "model"
        picks.append({
            "code": "쌍복", "desc": "1착 고정 + 2·3착 무순", "type": "조합",
            "pick": [qpl_pick],
            "prob": qpl_prob,
            "grade": qpl_grade,
            "pick_source": qpl_source,
        })
        # 삼쌍: 순서 top3
        a, b, c = order[0], order[1], order[2]
        picks.append({
            "code": "삼쌍", "desc": "1→2→3착 순서", "type": "조합(순서3)",
            "pick": [f"{lab(a)} → {lab(b)} → {lab(c)}"],
            "prob": f"순서 top3 (win {100*pw[a]:.0f}%) · 순서권 리스크",
            "grade": "약",
            "pick_source": "model",
        })

    # 대중용 1줄 뜻 부착 (7권종을 다 모르는 사용자 대상).
    for p in picks:
        p["mean"] = BET_MEANINGS.get(p["code"], "")

    return picks


def _keirin_pick_policy():
    policy = os.environ.get("KEIRIN_PICK_POLICY", KEIRIN_PICK_POLICY_DEFAULT).strip().lower()
    return policy if policy in KEIRIN_PICK_POLICIES else KEIRIN_PICK_POLICY_DEFAULT


def _keirin_trifecta_board_marginals(trifecta_board):
    if _keirin_pick_policy() == "model_always":
        return None
    first_mass = {bno: 0.0 for bno in range(1, 8)}
    ordered_pair_mass = {}
    unordered_pair_mass = {}
    unordered_trio_mass = {}
    seen = set()
    for combo, odds in (trifecta_board or {}).items():
        combo_text = str(combo)
        try:
            odds_value = float(odds)
        except (TypeError, ValueError):
            return None
        if (
            not re.fullmatch(r"[1-7]-[1-7]-[1-7]", combo_text)
            or odds_value <= 0
            or not math.isfinite(odds_value)
        ):
            return None
        first, second, third = (int(part) for part in combo_text.split("-"))
        if len({first, second, third}) != 3:
            return None
        seen.add((first, second, third))
        mass = 1.0 / odds_value
        first_mass[first] += mass
        ordered_pair = (first, second)
        unordered_pair = tuple(sorted((first, second)))
        unordered_trio = tuple(sorted((first, second, third)))
        ordered_pair_mass[ordered_pair] = ordered_pair_mass.get(ordered_pair, 0.0) + mass
        unordered_pair_mass[unordered_pair] = unordered_pair_mass.get(unordered_pair, 0.0) + mass
        unordered_trio_mass[unordered_trio] = unordered_trio_mass.get(unordered_trio, 0.0) + mass
    if len(seen) != 210:
        return None
    return {
        "first_order": sorted(first_mass, key=lambda bno: (-first_mass[bno], bno)),
        "ordered_pair_order": sorted(ordered_pair_mass, key=lambda pair: (-ordered_pair_mass[pair], pair[0], pair[1])),
        "unordered_pair_order": sorted(unordered_pair_mass, key=lambda pair: (-unordered_pair_mass[pair], pair[0], pair[1])),
        "unordered_trio_order": sorted(unordered_trio_mass, key=lambda trio: (-unordered_trio_mass[trio], trio[0], trio[1], trio[2])),
    }


def _keirin_trifecta_first_mass_order(trifecta_board):
    marginals = _keirin_trifecta_board_marginals(trifecta_board)
    return list(marginals.get("first_order", [])) if marginals else []


def _keirin_market_order_for_rows(rows, trifecta_board):
    board_order = _keirin_trifecta_first_mass_order(trifecta_board)
    row_bnos = {
        int(row.get("bno"))
        for row in (rows or [])
        if isinstance(row, dict) and str(row.get("bno") or "").isdigit()
    }
    order = [bno for bno in board_order if bno in row_bnos]
    return order if len(order) >= 2 else []


def _keirin_picks_with_source(rows, meta=None, trifecta_board=None):
    board = trifecta_board
    if board is None and isinstance(meta, dict):
        board = meta.get("trifecta_board")
    market_marginals = _keirin_trifecta_board_marginals(board)
    market_order = _keirin_market_order_for_rows(rows, board)
    return build_picks(rows, market_first_order=market_order, market_marginals=market_marginals), (
        "market" if market_marginals and market_order else "model"
    ), market_order


def _top_confidence(top, rows=None):
    """최상위(연대확률 1위) 픽의 신뢰등급/표현.

    트로피·'최고확신'은 절대 win 확률이 충분히 높을 때만 — 그렇지 않으면
    과신을 막기 위해 '상대 1순위(저신뢰)' 중립 표현으로 다운그레이드한다.
    반환 dict: {grade, label, icon, race_confidence}.
    """
    pwin = top.get("pwin", 0.0)
    grade_context, grade_policy = _keirin_grade_context(rows)
    grade = _grade_adjusted_win_grade(pwin, grade_policy)
    # 경주 내 확신도 (top1 - top2 확률 차이) → 고확신/혼전 판별
    race_conf = ""
    if rows and len(rows) >= 2:
        sorted_rows = sorted(rows, key=lambda r: -r.get("pwin", 0))
        gap = sorted_rows[0].get("pwin", 0) - sorted_rows[1].get("pwin", 0)
        if gap >= float(grade_policy["high_gap"]):
            race_conf = "고확신"  # 검증: 상위 30% 경주 80% 적중
        elif gap <= float(grade_policy["mixed_gap"]):
            race_conf = "혼전"  # 저확신 경주 (51% 적중)
        else:
            race_conf = "보통"
    if grade == "강":
        return {"grade": "강", "label": "최고확신 픽", "icon": "🏆", "race_confidence": race_conf, "grade_context": grade_context}
    if grade == "중":
        return {"grade": "중", "label": "상대 우세 픽", "icon": "▲", "race_confidence": race_conf, "grade_context": grade_context}
    return {"grade": "약", "label": "상대 1순위 (저신뢰)", "icon": "①", "race_confidence": race_conf, "grade_context": grade_context}


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
                picks, pick_source, _ = _keirin_picks_with_source(rows, meta)
                out = {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "final_model": True,
                    "pick_source": pick_source,
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
                    picks, pick_source, _ = _keirin_picks_with_source(rows, meta)
                    out = {
                        "rows": rows, "picks": picks, "top": rows[0],
                        "top_conf": _top_confidence(rows[0], rows),
                        "meta": meta or {}, "n_starters": len(rows),
                        "model_special_11r": True,
                        "pick_source": pick_source,
                    }
                    return _apply_kcycle_rankingpredict_overlay(out, rows, meta)
        # 일반 11R+ 특화 (66%)
        m11, m11err = load_11r_model()
        if m11 is not None:
            rows, err = score_keirin_with_model(starters, m11)
            if err is None:
                picks, pick_source, _ = _keirin_picks_with_source(rows, meta)
                out = {
                    "rows": rows,
                    "picks": picks,
                    "top": rows[0],
                    "top_conf": _top_confidence(rows[0], rows),
                    "meta": meta or {},
                    "n_starters": len(rows),
                    "model_11r": True,
                    "pick_source": pick_source,
                }
                return _apply_kcycle_rankingpredict_overlay(out, rows, meta)
    cross, cross_err = load_cross_model()
    if cross is not None:
        rows, err = score_keirin_with_model(starters, cross, meta=meta)
        if err is None:
            picks, pick_source, _ = _keirin_picks_with_source(rows, meta)
            out = {
                "rows": rows,
                "picks": picks,
                "top": rows[0],
                "top_conf": _top_confidence(rows[0], rows),
                "selective_conf": _keirin_selective_confidence(rows[0], rows),
                "meta": meta or {},
                "n_starters": len(rows),
                "model_cross_domain": True,
                "pick_source": pick_source,
            }
            return _apply_kcycle_rankingpredict_overlay(out, rows, meta)

    # 일반 모델
    rows, err = score_keirin(starters, meta=meta)
    if err:
        return {"error": err}

    picks, pick_source, _ = _keirin_picks_with_source(rows, meta)
    out = {
        "rows": rows,
        "picks": picks,
        "top": rows[0],
        "top_conf": _top_confidence(rows[0], rows),
        "meta": meta or {},
        "n_starters": len(rows),
        "pick_source": pick_source,
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


def kra_result_from_starters(starters):
    finish_by_rank = {}
    racers = []
    rank_keys = ("ord", "rank", "rcRank", "chaksun", "ordRank", "plcOrd", "winRank", "rnk")
    for item in starters or []:
        if not isinstance(item, dict):
            continue
        raw_rank = next((item.get(key) for key in rank_keys if item.get(key) not in (None, "")), None)
        try:
            rank = int(float(str(raw_rank).strip()))
        except (TypeError, ValueError):
            continue
        if rank <= 0:
            continue
        try:
            bno = int(float(str(item.get("chulNo")).strip()))
        except (TypeError, ValueError):
            continue
        name = str(item.get("hrName") or "").strip()
        racers.append({"bno": bno, "name": name, "rank": rank})
        if rank in {1, 2, 3}:
            finish_by_rank[rank] = bno
    actual_order = (
        [finish_by_rank[1], finish_by_rank[2], finish_by_rank[3]]
        if {1, 2, 3} <= set(finish_by_rank)
        else []
    )
    if not actual_order:
        return None
    racers.sort(key=lambda row: row["rank"])
    return {
        "actual_order": actual_order,
        "racers": racers,
        "payouts": {},
        "source": "kra_race_detail_result",
    }


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


def _kra_market_probabilities(starters):
    if not starters or len(starters) < 2:
        return None
    raw = {}
    for starter in starters:
        try:
            bno = int(starter.get("chulNo"))
            odds = float(starter.get("winOdds"))
        except (TypeError, ValueError):
            return None
        if odds <= 0:
            return None
        raw[bno] = 1.0 / odds
    total = sum(raw.values())
    if total <= 0 or len(raw) != len(starters):
        return None
    return {bno: value / total for bno, value in raw.items()}


def _kra_field_bucket(field_size):
    if field_size <= 7:
        return "field_le_7"
    if field_size <= 10:
        return "field_8_10"
    return "field_11_plus"


def _load_kra_confidence_tiers():
    path = KRA_CONFIDENCE_TIERS_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _KRA_CONFIDENCE_TIERS_CACHE
    if cached.get("path") == path and cached.get("mtime") == mtime:
        return cached.get("payload")
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    cached.update({"path": path, "mtime": mtime, "payload": payload})
    return payload


def _kra_market_tier_name(favorite_odds, ratio12):
    if favorite_odds > 4.0 or ratio12 < 1.10:
        return "weak_or_open"
    if favorite_odds <= 1.8 and ratio12 >= 1.50:
        return "very_strong_pull"
    if favorite_odds <= 2.5 and ratio12 >= 1.30:
        return "strong_pull"
    if favorite_odds <= 2.0:
        return "price_short"
    if ratio12 >= 1.50:
        return "gap_wide"
    return "all"


def kra_confidence_tier(starters):
    if not starters or len(starters) < 2:
        return None
    odds_values = []
    for starter in starters:
        if not isinstance(starter, dict):
            return None
        try:
            odds = float(starter.get("winOdds"))
        except (TypeError, ValueError):
            return None
        if odds <= 0:
            return None
        odds_values.append(odds)
    if len(odds_values) < 2:
        return None
    ranked = sorted(odds_values)
    favorite_odds = float(ranked[0])
    second_odds = float(ranked[1])
    if favorite_odds <= 0:
        return None
    field_bucket = _kra_field_bucket(len(odds_values))
    tier = _kra_market_tier_name(favorite_odds, second_odds / favorite_odds)
    payload = _load_kra_confidence_tiers()
    if not isinstance(payload, dict):
        return None
    by_field = ((payload.get("win_market") or {}).get("by_field_size") or {})
    metrics = (by_field.get(field_bucket) or {}).get(tier)
    if not isinstance(metrics, dict):
        return None
    return {
        "tier": tier,
        "field_bucket": field_bucket,
        "historical_top1": metrics.get("top1"),
        "historical_top3": metrics.get("top3"),
        "coverage": metrics.get("coverage"),
        "source": "kra_tiers_v1",
    }


def parse_kra_race_starts(html):
    parser = _KcycleTableTextParser()
    parser.feed(html or "")
    starts = {}
    meet_codes = {"서울": "서울", "제주": "제주", "부경": "부경", "부산경남": "부경"}
    for cells in parser.rows:
        date_index = next(
            (
                index
                for index, cell in enumerate(cells)
                if re.search(r"\d{4}[./-]\d{2}[./-]\d{2}", cell or "")
            ),
            None,
        )
        if date_index is None:
            continue
        date_match = re.search(
            r"(\d{4})[./-](\d{2})[./-](\d{2})", cells[date_index]
        )
        meet = next(
            (meet_codes[cell.strip()] for cell in cells[:date_index] if cell.strip() in meet_codes),
            None,
        )
        if not date_match or not meet:
            continue
        race_number = next(
            (
                int(cell.strip())
                for cell in cells[date_index + 1:]
                if re.fullmatch(r"\d{1,2}", cell.strip() or "")
            ),
            None,
        )
        time_match = next(
            (
                re.fullmatch(r"(\d{1,2}):(\d{2})", cell.strip())
                for cell in cells[date_index + 1:]
                if re.fullmatch(r"(\d{1,2}):(\d{2})", cell.strip() or "")
            ),
            None,
        )
        if race_number is None or time_match is None:
            continue
        race_date = "".join(date_match.groups())
        starts[(race_date, meet, race_number)] = datetime(
            int(race_date[:4]),
            int(race_date[4:6]),
            int(race_date[6:8]),
            int(time_match.group(1)),
            int(time_match.group(2)),
        )
    return starts


def _kra_official_race_start(ymd, meet, race_no):
    race_date = re.sub(r"\D", "", str(ymd or ""))[:8]
    normalized_meet = "부경" if str(meet).strip() == "부산경남" else str(meet).strip()
    try:
        race_number = int(str(race_no).strip().lstrip("0") or "0")
    except (TypeError, ValueError):
        return None
    if len(race_date) != 8 or normalized_meet not in KRA_MEETS or race_number <= 0:
        return None
    cache_key = (race_date, normalized_meet, race_number)
    cached = _KRA_RACE_START_CACHE.get(cache_key)
    if cached:
        checked_at, cached_start = cached
        if cached_start is not None or time.time() - checked_at < 60.0:
            return cached_start
    try:
        timeout = float(os.environ.get("KRA_RACE_SCHEDULE_LOOKUP_TIMEOUT_SEC", "1.0"))
    except ValueError:
        timeout = 1.0
    timeout = max(0.2, min(timeout, 3.0))
    url = "https://race.kra.co.kr/thisweekrace/ThisWeekDetailInfoList.do?Act=01&Sub=3"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
            declared_encoding = response.headers.get_content_charset()
        starts = {}
        encodings = tuple(dict.fromkeys((declared_encoding, "euc-kr", "cp949", "utf-8")))
        for encoding in encodings:
            if not encoding:
                continue
            try:
                page = payload.decode(encoding)
            except UnicodeDecodeError:
                continue
            starts = parse_kra_race_starts(page)
            if starts:
                break
        for key, value in starts.items():
            _KRA_RACE_START_CACHE[key] = (time.time(), value)
        start = starts.get(cache_key)
    except Exception:
        start = None
    _KRA_RACE_START_CACHE[cache_key] = (time.time(), start)
    return start


def kra_odds_snapshot_metadata(starters, ymd, meet, race_no, fetched_at=None):
    kst = timezone(timedelta(hours=9))
    captured = fetched_at or datetime.now(kst)
    if captured.tzinfo is not None:
        captured_local = captured.astimezone(kst).replace(tzinfo=None)
    else:
        captured_local = captured
    start = _kra_official_race_start(ymd, meet, race_no)
    complete_market = _kra_market_probabilities(starters) is not None
    result_available = kra_result_from_starters(starters) is not None
    fresh = bool(
        start
        and captured_local.date() == start.date()
        and captured_local < start
        and complete_market
        and not result_available
    )
    return {
        "odds_snapshot_fresh": fresh,
        "odds_snapshot_fetched_at": captured.isoformat(timespec="seconds"),
        "race_start_at": start.replace(tzinfo=kst).isoformat(timespec="seconds") if start else None,
        "minutes_to_start": ((start - captured_local).total_seconds() / 60.0) if start else None,
        "odds_snapshot_source": "kra_official_pre_start_schedule" if fresh else "unverified",
    }


def _kra_prediction_phase(starters, meta=None):
    fresh_snapshot = isinstance(meta, dict) and meta.get("odds_snapshot_fresh") is True
    if fresh_snapshot and _kra_market_probabilities(starters) is not None:
        return "live_odds"
    return "pre_race"


def _kra_pick_policy():
    policy = os.environ.get("KRA_PICK_POLICY", KRA_PICK_POLICY_DEFAULT).strip().lower()
    return policy if policy in KRA_PICK_POLICIES else KRA_PICK_POLICY_DEFAULT


def _kra_market_is_weak_or_open(starters):
    tier = kra_confidence_tier(starters)
    return isinstance(tier, dict) and tier.get("tier") == "weak_or_open"


def _kra_should_use_market_pick(starters, meta, model_rows=None):
    market_probability = _kra_market_probabilities(starters)
    if market_probability is None:
        return False
    # A complete board is not enough: a stale or post-start board can contain
    # settled information, so it must never anchor a pre-race prediction.
    if _kra_prediction_phase(starters, meta) != "live_odds":
        return False
    policy = _kra_pick_policy()
    if policy in {"current_gate", "market_if_odds"}:
        return _kra_prediction_phase(starters, meta) == "live_odds"
    if policy == "market_except_weak_disagree":
        if not model_rows:
            return True
        model_top = max(
            model_rows,
            key=lambda row: row.get("rank_score", row.get("pwin", 0.0)),
        )
        market_top = max(market_probability, key=market_probability.get)
        if int(model_top.get("bno", 0)) != int(market_top) and _kra_market_is_weak_or_open(starters):
            return False
        return True
    return False


def _kra_market_place_order(starters, rows):
    row_bnos = {
        int(row.get("bno"))
        for row in (rows or [])
        if isinstance(row, dict) and str(row.get("bno") or "").isdigit()
    }
    candidates = []
    for starter in starters or []:
        try:
            bno = int(starter.get("chulNo"))
            odds = float(starter.get("plcOdds"))
        except (AttributeError, TypeError, ValueError):
            continue
        if bno in row_bnos and odds > 0 and math.isfinite(odds):
            candidates.append((odds, bno))
    if not candidates:
        return []
    return [min(candidates)[1]]


def _kra_selective_confidence(rows, phase, model):
    ordered = sorted(
        rows,
        key=lambda row: -row.get("rank_score", row.get("pwin", 0.0)),
    )
    top_probability = float(ordered[0].get("pwin", 0.0)) if ordered else 0.0
    policy = model.get("confidence_policy", {})
    if phase == "pre_race" and any(row.get("rerank_applied") for row in rows):
        return {
            "tier": "normal",
            "label": "재정렬 일반 예측",
            "expected_top1": None,
            "coverage": 1.0,
            "threshold": float(policy.get("pre_race_top_probability_min", 1.0)),
            "top_probability": top_probability,
            "validation_split": "v5 reranker selective threshold not claimed",
        }
    if phase == "live_odds":
        threshold = float(policy.get("live_top_probability_min", 1.0))
        expected_top1 = float(policy.get("live_expected_top1", 0.0))
        coverage = float(policy.get("live_coverage", 0.0))
        label = "마감배당 고확신 선별"
    else:
        threshold = float(policy.get("pre_race_top_probability_min", 1.0))
        expected_top1 = float(policy.get("pre_race_expected_top1", 0.0))
        coverage = float(policy.get("pre_race_coverage", 0.0))
        label = "사전정보 고확신 선별"
    selected = top_probability >= threshold
    return {
        "tier": "high" if selected else "normal",
        "label": label if selected else "일반 예측",
        "expected_top1": expected_top1 if selected else None,
        "coverage": coverage if selected else 1.0,
        "threshold": threshold,
        "top_probability": top_probability,
        "validation_split": "2025 threshold selection -> 2026 temporal evaluation",
    }


def score_kra(starters, use_market=False, meta=None):
    """KRA 출주표 item 리스트 -> [{bno,name,grade,pwin,pplc}] (우승확률 내림차순).

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

    race_date = re.sub(r"\D", "", str((meta or {}).get("ymd", "")))[:8] or None
    c = apply_horse_history_snapshot(c, model.get("horse_history", {"records": {}}), race_date)

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
            # Coerce before filling so pandas does not silently downcast an
            # object column when incomplete live starter rows are supplied.
            F[col] = pd.to_numeric(F[col], errors="coerce").fillna(med[col])
    Fv = F.apply(pd.to_numeric, errors="coerce").fillna(0)
    try:
        pwin = model["win"].predict_proba(Fv)[:, 1]
        pplc = model["plc"].predict_proba(Fv)[:, 1]
    except Exception as e:  # noqa: BLE001
        return None, f"모델 예측 실패: {e}"

    # win 확률 경주 내 정규화 (해석 용이; pplc 는 그대로 연대확률)
    s = float(pwin.sum()) or 1.0
    pwin_n = pwin / s

    market_probability = _kra_market_probabilities(starters)
    market_weight = float(model.get("live_market_weight", 0.0))
    if use_market and market_probability is not None and market_weight > 0:
        pwin_n = np.asarray([
            (1.0 - market_weight) * float(probability)
            + market_weight * market_probability[int(bno)]
            for bno, probability in zip(c["bno"], pwin_n)
        ])

    rank_scores = np.asarray(pwin_n, dtype=float)
    rerank_applied = False
    pairwise = model.get("pairwise")
    if (
        not use_market
        and isinstance(pairwise, dict)
        and pairwise.get("enabled") is True
        and pairwise.get("estimator") is not None
    ):
        pair_frame = Fv.copy()
        pair_frame["rk"] = "current"
        pair_model = PairwiseModel(
            pairwise["estimator"],
            pd.Series(pairwise.get("median", {}), dtype=float),
        )
        pair_probability = pairwise_scores(pair_model, pair_frame, cols)
        pair_total = float(pair_probability.sum()) or 1.0
        reranked = restricted_rerank(
            pair_frame,
            RaceScores(rank_scores, pair_probability / pair_total),
            RerankPolicy(
                weight=float(pairwise.get("weight", 0.0)),
                top_k=int(pairwise.get("top_k", 1)),
            ),
        )
        rank_scores = reranked.scores
        rerank_applied = reranked.switches > 0

    rows = []
    for b, w, p, rank_score in zip(c["bno"], pwin_n, pplc, rank_scores):
        b = int(b)
        rows.append({"bno": b, "name": names.get(b, ""), "grade": "",
                     "pwin": float(w), "pplc": float(p),
                     "rank_score": float(rank_score),
                     "rerank_applied": rerank_applied})
    rows.sort(key=lambda row: -row["rank_score"])
    return rows, None


def predict_kra(starters, meta=None):
    """KRA 출주표 -> {rows, picks, top, top_conf, meta, n_starters} 또는 {error}."""
    phase = _kra_prediction_phase(starters, meta)
    policy = _kra_pick_policy()
    if policy == "market_except_weak_disagree":
        model_rows, err = score_kra(starters, use_market=False, meta=meta)
        if err:
            return {"error": err}
        use_market = _kra_should_use_market_pick(starters, meta, model_rows)
        if use_market:
            rows, err = score_kra(starters, use_market=True, meta=meta)
        else:
            rows = model_rows
    else:
        use_market = _kra_should_use_market_pick(starters, meta)
        rows, err = score_kra(starters, use_market=use_market, meta=meta)
    if err:
        return {"error": err}
    model, _ = load_kra_model()
    win_leader = max(
        rows,
        key=lambda row: row.get("rank_score", row.get("pwin", 0.0)),
    )
    market_place_order = (
        _kra_market_place_order(starters, rows)
        if use_market and policy == "market_if_odds"
        else []
    )
    picks = build_picks(rows, market_place_order=market_place_order)
    for pick in picks:
        if pick.get("code") == "단승" and use_market:
            pick["pick_source"] = "market"
    return {
        "rows": rows,
        "picks": picks,
        "top": win_leader,
        "top_conf": _top_confidence(win_leader, rows),
        "selective_conf": _kra_selective_confidence(rows, phase, model or {}),
        "prediction_phase": phase,
        "market_used": use_market,
        "pick_source": "market" if use_market else "model",
        "algorithm_version": (model or {}).get("kind", "kra_legacy"),
        "meta": meta or {},
        "n_starters": len(rows),
    }


# ═══════════════════════ 실시간 판단 (live-decision) ═══════════════════════
ROSTER_MISMATCH_MESSAGE = "공식 출주표와 일치하지 않아 예측을 중단했습니다"
_ROSTER_STATES = {"verified", "unverified", "mismatch"}


def _roster_verification(base_model_out):
    verification = (base_model_out or {}).get("roster_verification") if isinstance(base_model_out, dict) else None
    if isinstance(verification, dict):
        return {
            "state": str(verification.get("state") or "unverified"),
            "official_names": list(verification.get("official_names") or []),
            "checked_at": str(verification.get("checked_at") or ""),
        }
    return {"state": "unverified", "official_names": [], "checked_at": ""}


def _assert_live_decision_invariants(result):
    verification = result.get("roster_verification") or {}
    state = verification.get("state")
    assert state in _ROSTER_STATES
    if result.get("status") == "roster_mismatch":
        assert not result.get("picks")
        assert not result.get("rows")
        assert result.get("top") is None
    if state == "verified":
        assert verification.get("official_names")
    return result


def _finalize_live_decision(result, roster_verification):
    if result.get("market_confidence") is None:
        result.pop("market_confidence", None)
    result["roster_verification"] = roster_verification
    return _assert_live_decision_invariants(result)

def fetch_kcycle_odds_with_ts(stnd_yr, ymd, race_no, timeout=3, max_attempts=5):
    """kcycle 배당 fetch + 타임스탬프. 반환 (odds_dict, fetched_at_iso) 또는 (None, None)."""
    import datetime as _dt
    odds = None
    try:
        odds = fetch_kcycle_odds(stnd_yr, ymd, race_no, timeout=timeout, max_attempts=max_attempts)
    except Exception:
        odds = None
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    return odds, ts


def fetch_kcycle_trifecta_board_with_ts(stnd_yr, ymd, race_no, timeout=3, max_attempts=5):
    import datetime as _dt
    board = None
    try:
        board = fetch_kcycle_trifecta_board(stnd_yr, ymd, race_no, timeout=timeout, max_attempts=max_attempts)
    except Exception:
        board = None
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    return board, ts


def _live_poll_delay_ms(sport, market_used):
    if market_used:
        return 3000
    if sport == "keirin":
        return 15000
    return 30000


def _live_market_risk(sport, market_used, status):
    if sport == "horse":
        if market_used:
            return {
                "level": "odds_live",
                "message": "공식 사전배당 시장 앵커가 반영됐습니다.",
            }
        return {
            "level": "odds_unavailable",
            "message": "경마는 완전하고 신선한 사전배당이 확인될 때만 시장 앵커를 사용합니다.",
        }
    if sport != "keirin":
        return {
            "level": "not_applicable",
            "message": "지원하지 않는 종목입니다.",
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
        "level": "kcycle_disabled",
        "message": "KCYCLE_ENABLED=0 상태라 실시간 배당을 사용하지 않고 사전 예측과 공식예상 보조신호만 사용합니다.",
    }


def _live_fallback_signal(base_model_out):
    signal = base_model_out.get("rankingpredict_signal")
    if not signal:
        signal = base_model_out.get("selective_conf")
    if not signal or signal.get("tier") in (None, "normal"):
        return None
    return _live_signal_payload(signal)


def _live_signal_payload(signal):
    deployable = signal.get("deployable")
    if deployable is None:
        deployable = signal.get("robust_status") == "passed_oos"
    out = {
        "tier": signal.get("tier"),
        "label": signal.get("label"),
        "order": signal.get("order"),
        "deployable": bool(deployable),
        "usage": signal.get("usage") or ("prediction_signal" if deployable else "research_watch_only"),
        "expected_top1": signal.get("expected_top1"),
        "expected_pair_board": signal.get("expected_pair_board"),
        "expected_trio_exact": signal.get("expected_trio_exact"),
        "observed_trio_exact": signal.get("observed_trio_exact"),
        "baseline_trio_exact": signal.get("baseline_trio_exact"),
        "lift_pp": signal.get("lift_pp"),
        "current_axis_trio_exact": signal.get("current_axis_trio_exact"),
        "current_axis_lift_pp": signal.get("current_axis_lift_pp"),
        "coverage": signal.get("coverage"),
        "favorite_odds": signal.get("favorite_odds"),
        "selected_odds": signal.get("selected_odds"),
        "selected_board_rank": signal.get("selected_board_rank"),
        "rerank_score": signal.get("rerank_score"),
        "gap12": signal.get("gap12"),
        "second_mass_best": signal.get("second_mass_best"),
        "third_mass_best": signal.get("third_mass_best"),
        "rule": signal.get("rule"),
        "validation_n": signal.get("validation_n"),
        "validation_split": signal.get("validation_split"),
        "robust_status": signal.get("robust_status"),
        "robust_warning": signal.get("robust_warning"),
    }
    for key in (
        "applied",
        "blocked_by_timing",
        "blocked_by_order_conflict",
        "blocked_by_stronger_signal",
        "timing_phase",
    ):
        if key in signal:
            out[key] = signal.get(key)
    return out


def _parse_live_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _kcycle_official_race_start(ymd, race_no):
    d = re.sub(r"\D", "", str(ymd or ""))
    if len(d) < 8:
        return None
    try:
        rno = max(1, int(str(race_no).strip().lstrip("0") or "1"))
    except (TypeError, ValueError):
        return None
    cache_key = (d, rno)
    if cache_key in _KCYCLE_RACE_START_CACHE:
        return _KCYCLE_RACE_START_CACHE[cache_key]
    tms_day = _resolve_kcycle_tms(d[:4], d)
    if not tms_day:
        _KCYCLE_RACE_START_CACHE[cache_key] = None
        return None
    year, tms, day = tms_day
    if day == 0:
        _KCYCLE_RACE_START_CACHE[cache_key] = None
        return None
    import ssl

    try:
        timeout = float(os.environ.get("KCYCLE_RACE_SCHEDULE_LOOKUP_TIMEOUT_SEC", "0.75"))
    except ValueError:
        timeout = 0.75
    timeout = max(0.1, min(timeout, 1.0))
    try:
        attempts = int(os.environ.get("KCYCLE_RACE_SCHEDULE_LOOKUP_ATTEMPTS", "1"))
    except ValueError:
        attempts = 1
    attempts = max(1, min(attempts, 5))

    for dt in [0, -1, 1, -2, 2][:attempts]:
        url = f"https://{KCYCLE_IP}/race/card/decision/{year}/{tms + dt}/{day}"
        try:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            req = urllib.request.Request(url, headers={
                "Host": "www.kcycle.or.kr",
                "User-Agent": "Mozilla/5.0",
            })
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as response:
                html = response.read().decode("utf-8", "replace")
            text = re.sub(r"<[^>]+>", " ", html)
            text = re.sub(r"\s+", " ", text)
            match = re.search(rf"{rno:02d}경주\s*\([^)]*?(\d{{1,2}}):(\d{{2}})", text)
            if not match:
                continue
            start = datetime(
                int(d[0:4]),
                int(d[4:6]),
                int(d[6:8]),
                int(match.group(1)),
                int(match.group(2)),
            )
            _KCYCLE_RACE_START_CACHE[cache_key] = start
            return start
        except Exception:
            continue
    _KCYCLE_RACE_START_CACHE[cache_key] = None
    return None


def _kcycle_estimated_race_start(ymd, race_no):
    d = re.sub(r"\D", "", str(ymd or ""))
    if len(d) < 8:
        return None
    try:
        race_index = max(1, int(str(race_no).strip().lstrip("0") or "1"))
    except (TypeError, ValueError):
        return None
    if (
        os.environ.get("KCYCLE_RACE_SCHEDULE_SOURCE", "official") != "heuristic"
        and "KCYCLE_RACE_START_1_TIME" not in os.environ
        and "KCYCLE_RACE_INTERVAL_MIN" not in os.environ
    ):
        official = _kcycle_official_race_start(ymd, race_no)
        if official:
            return official
    start_text = os.environ.get("KCYCLE_RACE_START_1_TIME", "11:00")
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", start_text or "")
    if not m:
        return None
    try:
        interval_min = int(os.environ.get("KCYCLE_RACE_INTERVAL_MIN", "23"))
        base = datetime(
            int(d[0:4]),
            int(d[4:6]),
            int(d[6:8]),
            int(m.group(1)),
            int(m.group(2)),
        )
    except (TypeError, ValueError):
        return None
    return base + timedelta(minutes=max(0, race_index - 1) * interval_min)


def _kcycle_low_odds_blend_policy():
    path = os.environ.get("KCYCLE_MARKET_BLEND_RESULTS_PATH", KCYCLE_MARKET_BLEND_RESULTS_PATH)
    fallback = {
        "name": "blend_w0.30",
        "weight": 0.30,
        "test_top1": 0.6217,
        "flip_rate": 0.050,
        "rule": "(1-w)*model+w*market, w=0.30",
        "source": "fallback_robust_2025",
    }
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return fallback
    if (
        _KCYCLE_LOW_ODDS_BLEND_CACHE.get("path") == path
        and _KCYCLE_LOW_ODDS_BLEND_CACHE.get("mtime") == mtime
    ):
        cached = _KCYCLE_LOW_ODDS_BLEND_CACHE.get("policy")
        return cached if isinstance(cached, dict) else fallback
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return fallback
    years = payload.get("years") if isinstance(payload, dict) else None
    best = None
    if isinstance(years, dict):
        for year, year_payload in years.items():
            tests = year_payload.get("test") if isinstance(year_payload, dict) else None
            if not isinstance(tests, list):
                continue
            for item in tests:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                match = re.fullmatch(r"blend_w([0-9]+(?:\.[0-9]+)?)", name)
                if not match:
                    continue
                try:
                    weight = float(match.group(1))
                    if weight > 1:
                        weight = weight / 100.0
                    top1 = float(item.get("top1"))
                    flip_rate = float(item.get("market_flip_rate"))
                except (TypeError, ValueError):
                    continue
                if weight > 0.30 or flip_rate > 0.055:
                    continue
                candidate = {
                    "name": name,
                    "weight": weight,
                    "test_top1": top1,
                    "flip_rate": flip_rate,
                    "rule": item.get("rule") or f"(1-w)*model+w*market, w={weight:.2f}",
                    "source": f"market_blend_experiment_{year}",
                }
                if best is None or (candidate["test_top1"], -candidate["flip_rate"]) > (
                    best["test_top1"],
                    -best["flip_rate"],
                ):
                    best = candidate
    policy = best or fallback
    _KCYCLE_LOW_ODDS_BLEND_CACHE.update({"path": path, "mtime": mtime, "policy": policy})
    return policy


def _kcycle_should_accept_result_outcome(ymd, race_no, now=None):
    d = re.sub(r"\D", "", str(ymd or ""))
    if len(d) < 8:
        return False
    try:
        race_date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])).date()
    except ValueError:
        return False
    now = now or datetime.now()
    if now.tzinfo is not None:
        now = now.replace(tzinfo=None)
    today = now.date()
    if race_date < today:
        return True
    if race_date > today:
        return False
    race_start = _kcycle_estimated_race_start(ymd, race_no)
    if not race_start:
        return False
    result_settlement_buffer = timedelta(minutes=8)
    return now >= race_start + result_settlement_buffer


def _kcycle_timing_payload(phase, minutes, weight, allow_late_pull, allow_trifecta_axis, race_start):
    policy = _kcycle_low_odds_blend_policy()
    capped_weight = min(float(weight), float(policy["weight"]))
    return {
        "phase": phase,
        "minutes_to_start": None if minutes is None else round(float(minutes), 3),
        "market_weight": capped_weight,
        "allow_late_pull": bool(allow_late_pull),
        "allow_trifecta_axis": bool(allow_trifecta_axis),
        "race_start_at": race_start.isoformat(timespec="seconds") if race_start else None,
        "policy": f"low_odds_dependency_{policy['name']}",
        "policy_rule": policy["rule"],
        "policy_top1": policy["test_top1"],
        "policy_flip_rate": policy["flip_rate"],
        "policy_source": policy["source"],
    }


def _kcycle_market_timing_policy(ymd, race_no, fetched_at):
    captured = _parse_live_iso_datetime(fetched_at)
    d = re.sub(r"\D", "", str(ymd or ""))
    if captured and captured.tzinfo is not None:
        captured = captured.replace(tzinfo=None)
    if not captured or len(d) < 8:
        return _kcycle_timing_payload("unknown", None, 0.30, True, True, None)
    try:
        race_date = datetime(int(d[0:4]), int(d[4:6]), int(d[6:8])).date()
    except ValueError:
        race_date = None
    if race_date is None or captured.date() != race_date:
        return _kcycle_timing_payload("unknown", None, 0.30, True, True, None)
    race_start = _kcycle_estimated_race_start(ymd, race_no)
    if not race_start:
        return _kcycle_timing_payload("unknown", None, 0.30, True, True, None)
    minutes = (race_start - captured).total_seconds() / 60.0
    if minutes < -2:
        phase = "post_start"
        weight = 0.0
    elif minutes <= 10:
        phase = "late"
        weight = 0.30
    elif minutes <= 30:
        phase = "mid"
        weight = 0.15
    else:
        phase = "early"
        weight = 0.05
    allow_axis = phase == "late"
    return _kcycle_timing_payload(phase, minutes, weight, phase == "late", allow_axis, race_start)


def _snapshot_market_timing_payload(market_timing):
    if not isinstance(market_timing, dict):
        return None
    allowed = {
        "phase",
        "minutes_to_start",
        "market_weight",
        "allow_late_pull",
        "allow_trifecta_axis",
        "race_start_at",
        "policy",
        "policy_rule",
        "policy_top1",
        "policy_flip_rate",
        "policy_source",
    }
    return {key: value for key, value in market_timing.items() if key in allowed}


def _snapshot_phase_from_market_timing(market_timing):
    phase = str((market_timing or {}).get("phase") or "")
    if phase == "post_start":
        return "post_start_market_blocked"
    return "pre_result_market_snapshot"


def _snapshot_signal_payloads(signals):
    if not isinstance(signals, dict):
        return {}
    payload = {}
    for name, signal in signals.items():
        payload[str(name)] = _live_signal_payload(signal) if signal else None
    return payload


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


def _mobile_live_picks(
    rows,
    trifecta_axis_signal=None,
    trifecta_order_signal=None,
    trifecta_ensemble=None,
    market_first_order=None,
    market_marginals=None,
):
    valid = [
        r for r in (rows or [])
        if isinstance(r, dict) and int(r.get("bno") or 0) > 0
    ]
    if not valid:
        return []

    def bno(index):
        return str(int(valid[index].get("bno")))

    def prob(index):
        try:
            return max(0.0, min(1.0, float(valid[index].get("pwin") or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    def grade(value):
        if value >= 0.45:
            return "강"
        if value >= 0.25:
            return "중"
        return "약"

    order_signal = trifecta_order_signal if trifecta_order_signal else trifecta_axis_signal
    ensemble_order = []
    valid_numbers = {int(item.get("bno")) for item in valid}
    if isinstance(trifecta_ensemble, dict):
        raw_order = str(trifecta_ensemble.get("pick") or "").split("-")
        if len(raw_order) == 3 and len(set(raw_order)) == 3:
            try:
                ensemble_order = [int(value) for value in raw_order]
            except ValueError:
                ensemble_order = []
        if not all(value in valid_numbers for value in ensemble_order):
            ensemble_order = []
    ensemble_order = ensemble_order if not order_signal and not trifecta_axis_signal else []

    if trifecta_axis_signal:
        order = [
            int(x) for x in trifecta_axis_signal.get("order", [])
            if isinstance(x, int) or str(x).isdigit()
        ]
        by_bno = {int(item.get("bno")): item for item in valid}
        if len(order) >= 3 and all(bno in by_bno for bno in order[:3]):
            first = max(0.0, min(1.0, float(by_bno[order[0]].get("pwin") or 0.0)))
            pair_prob = max(0.0, min(1.0, float(trifecta_axis_signal.get("expected_pair_board") or 0.0)))
            pair_order = order[:2]
            top2_hybrid = trifecta_axis_signal.get("top2_hybrid")
            if isinstance(top2_hybrid, dict):
                hybrid_pair = [
                    int(x) for x in top2_hybrid.get("order_pair", [])
                    if isinstance(x, int) or str(x).isdigit()
                ]
                if len(hybrid_pair) >= 2 and all(bno in by_bno for bno in hybrid_pair[:2]):
                    pair_order = hybrid_pair[:2]
                    pair_prob = max(0.0, min(1.0, float(top2_hybrid.get("expected_unordered_pair") or pair_prob)))
            tri_order = [
                int(x) for x in (order_signal or {}).get("order", [])
                if isinstance(x, int) or str(x).isdigit()
            ]
            if len(tri_order) < 3 or not all(bno in by_bno for bno in tri_order[:3]):
                tri_order = order
            if ensemble_order:
                tri_order = ensemble_order
            tri_prob = max(0.0, min(1.0, float((order_signal or trifecta_axis_signal).get("expected_trio_exact") or 0.0)))
            return [
                {
                    "code": "TOP1",
                    "label": "1착 후보",
                    "selection": str(order[0]),
                    "probability": first,
                    "grade": grade(first),
                    "pick_source": "market",
                },
                {
                    "code": "QNL",
                    "label": "복승 조합",
                    "selection": f"{pair_order[0]}-{pair_order[1]}",
                    "probability": pair_prob,
                    "grade": grade(pair_prob),
                    "pick_source": "market",
                },
                {
                    "code": "TRI",
                    "label": "1-2-3 순서",
                    "selection": f"{tri_order[0]}-{tri_order[1]}-{tri_order[2]}",
                    "probability": tri_prob,
                    "grade": grade(tri_prob),
                    "basis": (order_signal or trifecta_axis_signal).get("tier"),
                    "pick_source": "market",
                },
                {
                    "code": "TRB",
                    "label": "삼복 조합",
                    "selection": "-".join(sorted([str(tri_order[0]), str(tri_order[1]), str(tri_order[2])], key=int)),
                    "probability": pair_prob,
                    "grade": grade(pair_prob),
                    "pick_source": "market",
                },
            ]

    first = prob(0)
    top1_selection = bno(0)
    market_order = [
        int(value) for value in (market_first_order or [])
        if isinstance(value, int) or str(value).isdigit()
    ]
    valid_bnos = {int(item.get("bno")) for item in valid}
    if market_order and market_order[0] in valid_bnos:
        top1_selection = str(market_order[0])
    market_pairs = []
    if isinstance(market_marginals, dict):
        market_pairs = [
            pair for pair in market_marginals.get("unordered_pair_order", [])
            if pair[0] in valid_bnos and pair[1] in valid_bnos
        ]
    picks = [{
        "code": "TOP1",
        "label": "1착 후보",
        "selection": top1_selection,
        "probability": first,
        "grade": grade(first),
        "pick_source": "market" if market_order and market_order[0] in valid_bnos else "model",
    }]
    if len(valid) >= 2:
        second = prob(1)
        pair_prob = max(0.0, min(1.0, (first + second) / 2.0))
        pair_selection = f"{bno(0)}-{bno(1)}"
        pair_source = "model"
        if market_pairs:
            pair_selection = f"{market_pairs[0][0]}-{market_pairs[0][1]}"
            pair_source = "market"
        picks.append({
            "code": "QNL",
            "label": "복승 조합",
            "selection": pair_selection,
            "probability": pair_prob,
            "grade": grade(pair_prob),
            "pick_source": pair_source,
        })
    if len(valid) >= 3:
        third = prob(2)
        tri_prob = max(0.0, min(1.0, first * max(prob(1), 0.01) * max(third, 0.01)))
        tri_order = [
            int(x) for x in (order_signal or {}).get("order", [])
            if isinstance(x, int) or str(x).isdigit()
        ]
        by_bno = {int(item.get("bno")) for item in valid}
        use_order_signal = len(tri_order) >= 3 and all(bno in by_bno for bno in tri_order[:3])
        if use_order_signal:
            tri_selection = f"{tri_order[0]}-{tri_order[1]}-{tri_order[2]}"
            tri_box_selection = "-".join(sorted([str(tri_order[0]), str(tri_order[1]), str(tri_order[2])], key=int))
            tri_prob = max(0.0, min(1.0, float(order_signal.get("expected_trio_exact") or tri_prob)))
            tri_basis = order_signal.get("tier")
        else:
            tri_selection = f"{bno(0)}-{bno(1)}-{bno(2)}"
            tri_box_selection = "-".join(sorted([bno(0), bno(1), bno(2)], key=int))
            tri_basis = None
        if ensemble_order:
            tri_selection = f"{ensemble_order[0]}-{ensemble_order[1]}-{ensemble_order[2]}"
            tri_box_selection = "-".join(sorted([str(value) for value in ensemble_order], key=int))
            tri_basis = "ensemble_v1"
        picks.extend([
            {
                "code": "TRI",
                "label": "1-2-3 순서",
                "selection": tri_selection,
                "probability": tri_prob,
                "grade": "약",
                "basis": tri_basis,
                "pick_source": "model",
            },
            {
                "code": "TRB",
                "label": "삼복 조합",
                "selection": tri_box_selection,
                "probability": max(0.0, min(1.0, (first + prob(1) + third) / 3.0)),
                "grade": grade((first + prob(1) + third) / 3.0),
                "pick_source": "model",
            },
        ])
    return picks


def _market_top2_hybrid_signal(trifecta_board):
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) < 150:
        return None
    implied = {combo: 1.0 / odds for combo, odds in valid.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    top2_mass = {i: 0.0 for i in range(1, 8)}
    for combo, value in implied.items():
        a, b, _ = [int(x) for x in combo.split("-")]
        prob = value / total
        top2_mass[a] += prob
        top2_mass[b] += prob
    ranked = sorted(top2_mass.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) < 3 or ranked[0][1] > 0.799966:
        return None
    pair = [ranked[0][0], ranked[1][0]]
    return {
        "tier": "market_top2_slot_mass_hybrid",
        "label": "KCYCLE 복승/top2 질량개선 신호",
        "order_pair": pair,
        "slot_top1_mass": ranked[0][1],
        "slot_gap12": ranked[0][1] - ranked[1][1],
        "expected_top2_slot": 0.679421551062196,
        "expected_unordered_pair": 0.42794983363194267,
        "baseline_top2_slot": 0.6694394676222165,
        "lift_pp": 0.9982083439979528,
        "coverage": 0.3460431654676259,
        "deployable": True,
        "usage": "top2_pair_signal",
        "validation_n": 3907,
        "validation_split": "train 2018-2023, holdout 2024-2026 n=3907; top2 slot 67.94% vs best-combo 66.94%; worst holdout-year lift +0.52pp",
        "rule": "삼쌍 보드 1-2착 slot mass 1위 <= 79.9966%일 때 최저배당 1-2쌍 대신 slot mass top2를 복승/top2 후보로 표시",
        "robust_status": "passed_top2_pair_oos",
    }


def _market_trifecta_axis_signal(trifecta_board):
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) < 150:
        return None
    top2_hybrid = _market_top2_hybrid_signal(valid)
    ranked = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))
    if len(ranked) < 2:
        return None
    best_combo, best_odds = ranked[0]
    if best_odds <= 0:
        return None
    second_odds = ranked[1][1]
    implied = {combo: 1.0 / odds for combo, odds in valid.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    order = [int(x) for x in best_combo.split("-")]
    pair_prefix = f"{order[0]}-{order[1]}-"
    pair12_mass = sum(v for combo, v in implied.items() if combo.startswith(pair_prefix)) / total
    gap12 = second_odds / best_odds
    if best_odds <= 3.0 and gap12 >= 1.2:
        return {
            "tier": "market_trifecta_late_pull_strong",
            "label": "KCYCLE 마감배당 강쏠림 축",
            "leader": order[0],
            "order": order,
            "favorite_odds": best_odds,
            "gap12": gap12,
            "pair12_mass": pair12_mass,
            "expected_top1": 0.8347107438016529,
            "expected_pair_board": 0.5702479338842975,
            "expected_trio_exact": 0.3347107438016529,
            "coverage": 0.17485549132947978,
            "deployable": True,
            "usage": "prediction_signal",
            "validation_n": 242,
            "validation_split": "2026 OOS n=242; 1착 83.47%, 1-2순서 57.02%, 삼쌍순서 33.47%; 2024-2025 validation n=304 exact 36.51%",
            "rule": "전체 삼쌍 보드 최저배당<=3.0 + 1-2위 배당격차>=1.20",
            "robust_status": "passed_late_market_pull_oos",
            "top2_hybrid": top2_hybrid,
        }
    return {
        "tier": "market_trifecta_axis",
        "label": "KCYCLE 삼쌍 보드 축",
        "leader": order[0],
        "order": order,
        "favorite_odds": best_odds,
        "gap12": gap12,
        "pair12_mass": pair12_mass,
        "expected_top1": 0.6370721789223992,
        "expected_pair_board": 0.33310742121314807,
        "expected_trio_exact": 0.16062351745171127,
        "coverage": 1.0,
        "deployable": True,
        "usage": "prediction_signal",
        "validation_n": 2951,
        "validation_split": "2025-2026 OOS n=2951; first 63.71%, pair 33.31%, exact 16.06%",
        "rule": "전체 삼쌍 보드 최저배당 조합의 첫 번호를 축 후보로 사용",
        "robust_status": "passed_oos",
        "top2_hybrid": top2_hybrid,
    }


def _apply_trifecta_axis_signal(rows, signal):
    if not signal:
        return False
    try:
        leader = int(signal.get("leader"))
        expected_top1 = float(signal.get("expected_top1") or 0.0)
    except (TypeError, ValueError):
        return False
    leader_row = next((r for r in rows if int(r.get("bno") or -1) == leader), None)
    if leader_row is None:
        return False
    current_best = max(float(r.get("pwin_blended", r.get("pwin", 0.0)) or 0.0) for r in rows)
    current_leader = float(leader_row.get("pwin_blended", leader_row.get("pwin", 0.0)) or 0.0)
    leader_row["pwin_base"] = leader_row.get("pwin_base", leader_row.get("pwin", 0.0))
    leader_row["pwin_blended"] = max(current_leader, expected_top1)
    leader_row["pwin"] = leader_row["pwin_blended"]
    leader_row["trifecta_axis_pwin"] = expected_top1
    leader_row["trifecta_axis_rank_score"] = max(current_best, current_leader, expected_top1) + 1.0
    leader_row["trifecta_axis_overrode_model"] = current_leader + 1e-9 < current_best
    leader_row["trifecta_axis_order"] = "-".join(str(x) for x in signal.get("order", []))
    rows.sort(key=lambda r: -_live_row_rank_score(r))
    return True


def _live_row_rank_score(row):
    try:
        return float(row.get("trifecta_axis_rank_score"))
    except (TypeError, ValueError):
        pass
    try:
        return float(row.get("pwin_blended", row.get("pwin", 0.0)) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _market_trifecta_lift_signal(trifecta_board):
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) < 150:
        return None
    ranked = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))
    if not ranked:
        return None
    best_combo, best_odds = ranked[0]
    if best_odds <= 0:
        return None
    implied = {combo: 1.0 / odds for combo, odds in valid.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    q = {combo: value / total for combo, value in implied.items()}
    first_mass = {i: 0.0 for i in range(1, 8)}
    second_mass = {i: 0.0 for i in range(1, 8)}
    third_mass = {i: 0.0 for i in range(1, 8)}
    for combo, prob in q.items():
        a, b, c = [int(x) for x in combo.split("-")]
        first_mass[a] += prob
        second_mass[b] += prob
        third_mass[c] += prob
    _, best_second, best_third = [int(x) for x in best_combo.split("-")]
    if second_mass[best_second] < 0.380599 or third_mass[best_third] < 0.263285:
        return None

    def score(combo):
        a, b, c = [int(x) for x in combo.split("-")]
        return (
            0.44 * math.log(max(q[combo], 1e-12))
            + 0.92 * math.log(max(first_mass[a], 1e-12))
            + 0.66 * math.log(max(second_mass[b], 1e-12))
            + 0.48 * math.log(max(third_mass[c], 1e-12))
        )

    selected_combo = min(valid, key=lambda combo: (-score(combo), valid[combo], combo))
    order = [int(x) for x in selected_combo.split("-")]
    return {
        "tier": "market_trifecta_stat_strict_lift",
        "label": "KCYCLE 삼쌍 순서 통계개선 신호",
        "leader": order[0],
        "order": order,
        "favorite_odds": best_odds,
        "selected_odds": valid[selected_combo],
        "second_mass_best": second_mass[best_second],
        "third_mass_best": third_mass[best_third],
        "expected_trio_exact": 0.26090342679127726,
        "current_axis_trio_exact": 0.16062351745171127,
        "current_axis_lift_pp": 10.0279909339566,
        "baseline_trio_exact": 0.2554517133956386,
        "lift_pp": 0.5451713395638658,
        "coverage": 0.32864090120296905,
        "deployable": True,
        "usage": "selective_trifecta_rerank_signal",
        "validation_n": 1284,
        "validation_split": "2024-2026 OOS n=1284; exact 26.09%; current full-board-axis 16.06% 대비 +10.03%p; same-slice board 25.55% 대비 +0.55%p; paired wins-losses 7-0; p=0.0156",
        "rule": "보드 최저배당 조합의 2착 암시질량>=38.0599% + 3착 암시질량>=26.3285%; Bradley-Terry식 위치질량 재랭킹",
        "robust_status": "passed_directional_lift_oos",
    }


_GLOBAL_RERANK_FEATURES = (
    "rank_score",
    "log_q",
    "neg_log_odds",
    "neg_odds_ratio_best",
    "first_mass",
    "second_mass",
    "third_mass",
    "pair_mass",
    "unordered_trio_mass",
    "pair_share",
    "third_share",
    "first_gap",
    "pair_gap",
    "gap12",
    "gap15",
    "gap110",
    "entropy_inv",
    "top3_same_first",
    "top5_same_first",
    "top3_same_pair",
)
_GLOBAL_RERANK_WEIGHTS = {
    "neg_odds_ratio_best": 0.840302,
    "top5_same_first": 0.335309,
    "log_q": 0.229865,
    "first_mass": 0.207536,
    "top3_same_pair": -0.173807,
    "pair_share": 0.112418,
    "third_share": -0.10898,
    "top3_same_first": 0.098325,
}
_GLOBAL_RERANK_MU = {
    "rank_score": 0.5,
    "log_q": -3.0076305866241455,
    "neg_log_odds": -2.6811060905456543,
    "neg_odds_ratio_best": -4.127610206604004,
    "first_mass": 0.5730807185173035,
    "second_mass": 0.280759334564209,
    "third_mass": 0.1871841996908188,
    "pair_mass": 0.23127366602420807,
    "unordered_trio_mass": 0.19618092477321625,
    "pair_share": 0.31996414065361023,
    "third_share": 0.33424729108810425,
    "first_gap": 8.57268238067627,
    "pair_gap": 2.057460069656372,
    "gap12": 1.4269216060638428,
    "gap15": 3.533219575881958,
    "gap110": 7.98131799697876,
    "entropy_inv": 0.3395332098007202,
    "top3_same_first": 0.7019913792610168,
    "top5_same_first": 0.47223055362701416,
    "top3_same_pair": 0.1905333697795868,
}
_GLOBAL_RERANK_SIGMA = {
    "rank_score": 0.3191177546977997,
    "log_q": 0.6560398936271667,
    "neg_log_odds": 0.6595824956893921,
    "neg_odds_ratio_best": 3.961412191390991,
    "first_mass": 0.25002941489219666,
    "second_mass": 0.13670183718204498,
    "third_mass": 0.07468713819980621,
    "pair_mass": 0.1517459750175476,
    "unordered_trio_mass": 0.14284498989582062,
    "pair_share": 0.1675959825515747,
    "third_share": 0.1786937117576599,
    "first_gap": 17.711271286010742,
    "pair_gap": 1.1301367282867432,
    "gap12": 0.4463958442211151,
    "gap15": 2.0526812076568604,
    "gap110": 6.017037868499756,
    "entropy_inv": 0.10293415933847427,
    "top3_same_first": 0.4572412967681885,
    "top5_same_first": 0.49919816851615906,
    "top3_same_pair": 0.39261138439178467,
}
_GLOBAL_RERANK_FALLBACK = {
    "name": "gen2_mut_579",
    "top_k": 10,
    "weights": _GLOBAL_RERANK_WEIGHTS,
    "mu": _GLOBAL_RERANK_MU,
    "sigma": _GLOBAL_RERANK_SIGMA,
    "test_exact": 0.18063583970069885,
    "test_board_exact": 0.17196531791907516,
    "test_board_lift_pp": 0.8670521781623697,
    "test_current_axis_lift_pp": 2.001232224898758,
    "formula": "+0.840*neg_odds_ratio_best +0.335*top5_same_first +0.230*log_q +0.208*first_mass -0.174*top3_same_pair +0.112*pair_share -0.109*third_share +0.098*top3_same_first",
}


def _load_global_rerank_payload(path):
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if (
        _KCYCLE_GLOBAL_RERANK_CACHE.get("path") == path
        and _KCYCLE_GLOBAL_RERANK_CACHE.get("mtime") == mtime
    ):
        return _KCYCLE_GLOBAL_RERANK_CACHE.get("payload")
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        payload = None
    if not isinstance(payload, dict):
        payload = None
    _KCYCLE_GLOBAL_RERANK_CACHE.update({"path": path, "mtime": mtime, "payload": payload})
    return payload


def _global_rerank_champion():
    path = os.environ.get("KCYCLE_GLOBAL_RERANK_RESULTS_PATH", KCYCLE_GLOBAL_RERANK_RESULTS_PATH)
    payload = _load_global_rerank_payload(path)
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    stats_by_top_k = payload.get("feature_stats_by_top_k") if isinstance(payload, dict) else None
    if isinstance(candidates, list) and isinstance(stats_by_top_k, dict):
        for item in candidates:
            if not isinstance(item, dict) or not item.get("deployable"):
                continue
            try:
                top_k = int(item.get("top_k") or 0)
            except (TypeError, ValueError):
                continue
            stats = stats_by_top_k.get(str(top_k))
            weights = item.get("weights")
            if not isinstance(stats, dict) or not isinstance(weights, dict):
                continue
            mu = stats.get("mu")
            sigma = stats.get("sigma")
            if not isinstance(mu, dict) or not isinstance(sigma, dict):
                continue
            if not all(name in mu and name in sigma for name in _GLOBAL_RERANK_FEATURES):
                continue
            if not any(name in weights for name in _GLOBAL_RERANK_FEATURES):
                continue
            return {
                "name": item.get("name") or "dynamic_global_rerank",
                "top_k": top_k,
                "weights": weights,
                "mu": mu,
                "sigma": sigma,
                "test_exact": float(item.get("test_exact") or _GLOBAL_RERANK_FALLBACK["test_exact"]),
                "test_board_exact": float(item.get("test_board_exact") or _GLOBAL_RERANK_FALLBACK["test_board_exact"]),
                "test_board_lift_pp": float(item.get("test_board_lift_pp") or _GLOBAL_RERANK_FALLBACK["test_board_lift_pp"]),
                "test_current_axis_lift_pp": float(item.get("test_current_axis_lift_pp") or _GLOBAL_RERANK_FALLBACK["test_current_axis_lift_pp"]),
                "formula": item.get("formula") or _GLOBAL_RERANK_FALLBACK["formula"],
            }
    return _GLOBAL_RERANK_FALLBACK


def _market_trifecta_global_rerank_signal(trifecta_board):
    champion = _global_rerank_champion()
    top_k = int(champion["top_k"])
    weights = champion["weights"]
    mu = champion["mu"]
    sigma = champion["sigma"]
    valid = {
        str(combo): float(odds)
        for combo, odds in (trifecta_board or {}).items()
        if re.fullmatch(r"[1-7]-[1-7]-[1-7]", str(combo)) and odds and float(odds) > 0
    }
    if len(valid) < 150:
        return None
    ranked = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))
    if len(ranked) < top_k:
        return None
    best_combo, best_odds = ranked[0]
    if best_odds <= 0:
        return None
    implied = {combo: 1.0 / odds for combo, odds in valid.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    q = {combo: value / total for combo, value in implied.items()}
    first_mass = {i: 0.0 for i in range(1, 8)}
    second_mass = {i: 0.0 for i in range(1, 8)}
    third_mass = {i: 0.0 for i in range(1, 8)}
    pair_mass = {}
    unordered_trio_mass = {}
    for combo, prob in q.items():
        a, b, c = [int(x) for x in combo.split("-")]
        first_mass[a] += prob
        second_mass[b] += prob
        third_mass[c] += prob
        pair_mass[(a, b)] = pair_mass.get((a, b), 0.0) + prob
        trio_key = tuple(sorted((a, b, c)))
        unordered_trio_mass[trio_key] = unordered_trio_mass.get(trio_key, 0.0) + prob
    probs = list(q.values())
    entropy_inv = 1.0 + sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(len(probs))
    first_vals = sorted(first_mass.values(), reverse=True)
    pair_vals = sorted(pair_mass.values(), reverse=True)
    top3 = [combo for combo, _ in ranked[:3]]
    top5 = [combo for combo, _ in ranked[:5]]
    shared = {
        "first_gap": first_vals[0] / max(first_vals[1], 1e-12),
        "pair_gap": pair_vals[0] / max(pair_vals[1], 1e-12),
        "gap12": ranked[1][1] / best_odds,
        "gap15": ranked[4][1] / best_odds,
        "gap110": ranked[9][1] / best_odds,
        "entropy_inv": entropy_inv,
        "top3_same_first": float(len({x.split("-")[0] for x in top3}) == 1),
        "top5_same_first": float(len({x.split("-")[0] for x in top5}) == 1),
        "top3_same_pair": float(len({"-".join(x.split("-")[:2]) for x in top3}) == 1),
    }
    selected = None
    selected_score = None
    selected_rank = None
    selected_features = None
    for rank, (combo, odds) in enumerate(ranked[:top_k], start=1):
        a, b, c = [int(x) for x in combo.split("-")]
        pair = pair_mass[(a, b)]
        third = third_mass[c]
        features = {
            **shared,
            "rank_score": 1.0 - ((rank - 1) / max(top_k - 1, 1)),
            "log_q": math.log(max(q[combo], 1e-12)),
            "neg_log_odds": -math.log(odds),
            "neg_odds_ratio_best": -(odds / best_odds),
            "first_mass": first_mass[a],
            "second_mass": second_mass[b],
            "third_mass": third,
            "pair_mass": pair,
            "unordered_trio_mass": unordered_trio_mass[tuple(sorted((a, b, c)))],
            "pair_share": q[combo] / max(pair, 1e-12),
            "third_share": q[combo] / max(third, 1e-12),
        }
        score = sum(
            float(weights.get(name, 0.0))
            * ((features[name] - float(mu[name])) / float(sigma[name]))
            for name in _GLOBAL_RERANK_FEATURES
        )
        if selected_score is None or score > selected_score:
            selected = combo
            selected_score = score
            selected_rank = rank
            selected_features = features
    if not selected or selected == best_combo:
        return None
    order = [int(x) for x in selected.split("-")]
    return {
        "tier": "market_trifecta_global_incremental_rerank",
        "label": "KCYCLE 삼쌍 전체보드 미세개선 재랭킹",
        "leader": order[0],
        "order": order,
        "favorite_odds": best_odds,
        "selected_odds": valid[selected],
        "selected_board_rank": selected_rank,
        "rerank_score": selected_score,
        "expected_trio_exact": champion["test_exact"],
        "current_axis_trio_exact": 0.16062351745171127,
        "current_axis_lift_pp": champion["test_current_axis_lift_pp"],
        "baseline_trio_exact": champion["test_board_exact"],
        "lift_pp": champion["test_board_lift_pp"],
        "coverage": 1.0,
        "deployable": True,
        "usage": "selective_trifecta_rerank_signal",
        "validation_n": 1384,
        "validation_split": (
            "train 2018-2023, validation 2024-2025, untouched test 2026 n=1384; "
            f"test exact {champion['test_exact'] * 100:.2f}% vs board {champion['test_board_exact'] * 100:.2f}%"
            f"(+{champion['test_board_lift_pp']:.2f}pp); current-axis 16.06% 대비 "
            f"+{champion['test_current_axis_lift_pp']:.2f}pp; 10pp breakthrough 아님"
        ),
        "rule": f"top{top_k} 삼쌍 보드 z-score 재랭킹({champion['name']}): {champion['formula']}",
        "robust_status": "passed_incremental_oos",
        "debug_features": selected_features,
    }


def _load_kcycle_trifecta_ensemble_artifact():
    path = os.environ.get("KCYCLE_TRIFECTA_ENSEMBLE_PATH", KCYCLE_TRIFECTA_ENSEMBLE_PATH)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if (
        _KCYCLE_TRIFECTA_ENSEMBLE_CACHE.get("path") == path
        and _KCYCLE_TRIFECTA_ENSEMBLE_CACHE.get("mtime") == mtime
    ):
        return _KCYCLE_TRIFECTA_ENSEMBLE_CACHE.get("payload")
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        payload = None
    selection = payload.get("selection") if isinstance(payload, dict) else None
    selection_criteria = selection.get("criteria") if isinstance(selection, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "kcycle_trifecta_ensemble_v1"
        or not isinstance(selection_criteria, str)
        or not selection_criteria.startswith("val-only")
    ):
        payload = None
    _KCYCLE_TRIFECTA_ENSEMBLE_CACHE.update({"path": path, "mtime": mtime, "payload": payload})
    return payload


def _kcycle_ensemble_top_k(payload):
    selection = payload.get("selection") if isinstance(payload, dict) else {}
    try:
        top_k = int(selection.get("deploy_top_k", 20))
    except (AttributeError, TypeError, ValueError):
        top_k = 20
    return top_k if top_k in {10, 20, 40} else 20


def kcycle_ensemble_trifecta_rank(board):
    payload = _load_kcycle_trifecta_ensemble_artifact()
    if not isinstance(payload, dict):
        return []
    formulas = payload.get("formulas")
    stats_by_top_k = payload.get("feature_stats_by_top_k")
    if not isinstance(formulas, list) or len(formulas) != 20 or not isinstance(stats_by_top_k, dict):
        return []
    top_k = _kcycle_ensemble_top_k(payload)
    stats = stats_by_top_k.get(str(top_k))
    if not isinstance(stats, dict):
        return []
    mu = stats.get("mu")
    sigma = stats.get("sigma")
    if not isinstance(mu, dict) or not isinstance(sigma, dict):
        return []
    feature_rows, combos = feature_rows_from_board(board, top_k)
    if len(feature_rows) != top_k or len(combos) != top_k:
        return []

    candidate_ranks = [[0.0 for _ in combos] for _ in formulas]
    for formula_index, formula in enumerate(formulas):
        weights = formula.get("weights") if isinstance(formula, dict) else {}
        if not isinstance(weights, dict):
            return []
        scores = []
        for combo_index, row in enumerate(feature_rows):
            score = 0.0
            for feature_index, feature in enumerate(KCYCLE_ENSEMBLE_FEATURE_NAMES):
                sigma_value = float(sigma.get(feature, 1.0) or 1.0)
                if abs(sigma_value) < 1e-12:
                    sigma_value = 1.0
                z_value = (float(row[feature_index]) - float(mu.get(feature, 0.0))) / sigma_value
                score += float(weights.get(feature, 0.0)) * z_value
            scores.append((combo_index, score))
        for rank, (combo_index, _score) in enumerate(
            sorted(scores, key=lambda item: (-item[1], combos[item[0]])),
            start=1,
        ):
            candidate_ranks[formula_index][combo_index] = float(rank)

    ranked = []
    for combo_index, combo in enumerate(combos):
        avg_rank = sum(ranks[combo_index] for ranks in candidate_ranks) / len(candidate_ranks)
        ranked.append({
            "combo": combo,
            "order": [int(part) for part in combo.split("-")],
            "rank_average": avg_rank,
            "board_rank": combo_index + 1,
        })
    ranked.sort(key=lambda item: (item["rank_average"], item["board_rank"], item["combo"]))
    return ranked


def _kcycle_trifecta_strength(best_odds, gap12):
    return math.log(max(float(gap12), 1.000001)) * (3.0 / max(float(best_odds), 0.01))


def kcycle_trifecta_confidence_tier(board):
    payload = _load_kcycle_trifecta_ensemble_artifact()
    tiers = payload.get("strong_pull_tiers") if isinstance(payload, dict) else {}
    valid = {}
    for combo, odds in (board or {}).items():
        combo_text = str(combo)
        if not re.fullmatch(r"[1-7]-[1-7]-[1-7]", combo_text) or len(set(combo_text.split("-"))) != 3:
            continue
        try:
            odds_value = float(odds)
        except (TypeError, ValueError):
            continue
        if odds_value > 0:
            valid[combo_text] = odds_value
    base = tiers.get("base") if isinstance(tiers, dict) else {}
    base_exact = float((base or {}).get("historical_exact", 0.16532374100719424))
    if len(valid) < 150:
        return {"tier": "T0_base", "tier_historical_exact": base_exact}
    ranked = sorted(valid.items(), key=lambda item: (item[1], item[0]))
    best_combo, best_odds = ranked[0]
    gap12 = ranked[1][1] / best_odds if len(ranked) > 1 and best_odds > 0 else 1.0
    strong = best_odds <= 3.0 and gap12 >= 1.2
    if not strong:
        return {
            "tier": "T0_base",
            "tier_historical_exact": base_exact,
            "best_combo": best_combo,
            "best_odds": best_odds,
            "gap12": gap12,
        }
    strength = _kcycle_trifecta_strength(best_odds, gap12)
    top16 = tiers.get("top16") if isinstance(tiers, dict) else {}
    top16_threshold = float((top16 or {}).get("signal_strength_min", 1.3441320368349536))
    if strength >= top16_threshold:
        return {
            "tier": "T2_top16",
            "tier_historical_exact": float((top16 or {}).get("historical_exact", 0.4340659340659341)),
            "best_combo": best_combo,
            "best_odds": best_odds,
            "gap12": gap12,
            "signal_strength": strength,
        }
    strong_tier = tiers.get("strong") if isinstance(tiers, dict) else {}
    return {
        "tier": "T1_strong",
        "tier_historical_exact": float((strong_tier or {}).get("historical_exact", 0.3516483516483517)),
        "best_combo": best_combo,
        "best_odds": best_odds,
        "gap12": gap12,
        "signal_strength": strength,
    }


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
        "deployable": False,
        "usage": "research_watch_only",
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


def save_kcycle_trifecta_snapshot(
    stnd_yr,
    ymd,
    meet,
    race_no,
    trifecta_board,
    fetched_at=None,
    signal=None,
    source="live_decision",
    snapshot_phase="pre_result_market_snapshot",
    market_timing=None,
    signals=None,
):
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
    best20 = sorted(valid.items(), key=lambda kv: (kv[1], kv[0]))[:20]
    timing_payload = _snapshot_market_timing_payload(market_timing)
    signal_payloads = _snapshot_signal_payloads(signals)
    record = {
        "schema": "kcycle_trifecta_snapshot_v2" if timing_payload or signal_payloads else "kcycle_trifecta_snapshot_v1",
        "captured_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "fetched_at": fetched_at,
        "source": source,
        "snapshot_phase": snapshot_phase,
        "stnd_yr": str(stnd_yr or ""),
        "date": ymd_key or str(ymd or ""),
        "meet": str(meet or "광명"),
        "race_no": str(race_no or ""),
        "board_count": len(valid),
        "board_hash": board_hash,
        "best20": best20,
        "signal": _live_signal_payload(signal) if signal else None,
        "signals": signal_payloads,
        "market_timing": timing_payload,
        "board": dict(sorted(valid.items())),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lock_path = f"{path}.lock"
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError):
            pass
        _KCYCLE_TRIFECTA_SNAPSHOT_FILE_KEYS.pop(path, None)
        file_keys = _load_snapshot_file_keys(path)
        if token in file_keys:
            return False
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        with open(_snapshot_index_path(path), "a", encoding="utf-8") as f:
            f.write(token + "\n")
        file_keys.add(token)
    return True


def _explain_float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _explain_int(value, fallback=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _explain_pct(value):
    return f"{_explain_float(value) * 100:.1f}%"


def _explain_pp(value):
    return f"{_explain_float(value) * 100:+.1f}%p"


def _source_text(record, key, fallback="-"):
    value = record.get(key)
    text = str(value if value is not None else "").strip()
    return text or fallback


def _source_float(record, key):
    value = record.get(key)
    if value is None:
        return None
    text = str(value).replace('"', ".").replace("초", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None


def _metric(label, value, tone=None):
    item = {"label": label, "value": str(value)}
    if tone:
        item["tone"] = tone
    return item


def _time_200m_text(value):
    raw = str(value or "").strip()
    if not raw or raw == "-":
        return "-"
    text = raw.replace('"', ".").replace("초", "")
    try:
        return f"{float(text):.2f}초"
    except ValueError:
        return f"{raw}초" if raw[-1].isdigit() else raw


def _finish_counts(record):
    ranks = []
    for prefix in ("bf1", "bf2", "bf3"):
        for day in ("day1", "day2", "day3"):
            value = str(record.get(f"{prefix}_{day}_rank", "") or "")
            match = re.search(r"-(\d+)\s*$", value)
            if match:
                ranks.append(int(match.group(1)))
    wins = sum(1 for rank in ranks if rank == 1)
    seconds = sum(1 for rank in ranks if rank == 2)
    thirds = sum(1 for rank in ranks if rank == 3)
    return wins, seconds, thirds


def _score_read(score):
    if score is None:
        return "득점 자료 대기"
    if score >= 90:
        return "축 후보권"
    if score >= 86:
        return "상위권 추격권"
    if score >= 82:
        return "조합권 검토"
    return "전개 도움 필요"


def _sprint_read(seconds):
    if seconds is None:
        return "200m 자료 대기"
    if seconds <= 11.35:
        return "순간 가속 우위"
    if seconds <= 11.55:
        return "초반 자리 경쟁 가능"
    return "빠른 압박 부담"


def _podium_read(rate):
    if rate is None:
        return "입상률 자료 대기"
    if rate >= 60:
        return "3착권 안정성"
    if rate >= 40:
        return "조합권 안정"
    if rate >= 25:
        return "복병권"
    return "순위 고정 위험"


def _keirin_tactics(record):
    raw = [
        ("선행", _source_float(record, "pre_win_cnt")),
        ("젖히기", _source_float(record, "brk_win_cnt")),
        ("추입", _source_float(record, "pas_win_cnt")),
        ("마크", _source_float(record, "mrk_win_cnt")),
    ]
    valid = [(label, value) for label, value in raw if value is not None and value >= 0]
    if not valid:
        return [], ("전법", "-", "전개 자료 대기")
    total = sum(value for _, value in valid)
    values = valid if 95 <= total <= 105 else [(label, value * 100 / total) for label, value in valid] if total > 0 else valid
    ranked = sorted(values, key=lambda item: -item[1])
    metrics = [_metric(label, f"{value:.0f}%", "teal" if index == 0 else None) for index, (label, value) in enumerate(ranked[:4])]
    top_label, top_value = ranked[0]
    return metrics, (top_label, f"{top_value:.0f}%", "전개 강점" if top_value >= 40 else "전개 변수")


def _keirin_source(record):
    number = mach(record.get("back_no"))
    if not number:
        return None
    name = _source_text(record, "racer_nm", f"{number}번")
    grade = _source_text(record, "racer_grd_cd")
    age = _source_text(record, "racer_age")
    training = _source_text(record, "trng_plc_nm")
    score = _source_float(record, "tot_tms_avg_scr")
    sprint = _source_float(record, "rec_200m_scr")
    sprint_text = _time_200m_text(record.get("rec_200m_scr"))
    gear = _source_text(record, "gear_rate")
    podium = _source_float(record, "high_rate")
    wins, seconds, thirds = _finish_counts(record)
    tactics, top_tactic = _keirin_tactics(record)
    score_value = f"{score:.1f}" if score is not None else "-"
    podium_value = f"{podium:.0f}%" if podium is not None else "-"
    score_reason = _score_read(score)
    sprint_reason = _sprint_read(sprint)
    podium_reason = _podium_read(podium)
    tactic_label, tactic_value, tactic_reason = top_tactic
    note_parts = [
        f"평균득점 {score_value}점은 {score_reason}",
        f"200m {sprint_text}는 {sprint_reason}",
        f"입상률 {podium_value}는 {podium_reason}",
        f"{tactic_label} {tactic_value}는 {tactic_reason}",
    ]
    return {
        "number": number,
        "name": name,
        "subtitle": f"{grade}급 · {age}세 · {training} 훈련",
        "stats": f"평균득점 {score_value} · 200m {sprint_text} · 입상률 {podium_value}",
        "trait": tactic_label,
        "note": f"{name}: " + ", ".join(note_parts) + f"입니다. 최근 3주 {wins}-{seconds}-{thirds} 흐름까지 같이 봅니다.",
        "profile": [
            _metric("등급", grade, "teal" if grade == "특선" else None),
            _metric("나이", f"{age}세" if age != "-" and not age.endswith("세") else age),
            _metric("200m", sprint_text, "teal" if sprint is not None and sprint <= 11.55 else None),
            _metric("기어", gear),
            _metric("훈련지", training),
            _metric("평균득점", score_value, "teal" if score is not None and score >= 86 else None),
        ],
        "form": [
            _metric("입상률", podium_value, "teal" if podium is not None and podium >= 60 else None),
            _metric("최근 3주", f"{wins}-{seconds}-{thirds}", "teal" if wins >= 2 else None),
            _metric("득점해석", score_reason),
        ],
        "tactics": tactics,
        "algorithm_reasons": [
            _metric("평균득점", f"{score_value}점 · {score_reason}", "teal" if score is not None and score >= 90 else "primary"),
            _metric("200m", f"{sprint_text} · {sprint_reason}", "teal" if sprint is not None and sprint <= 11.35 else "primary"),
            _metric("입상률", f"{podium_value} · {podium_reason}", "teal" if podium is not None and podium >= 60 else "amber"),
            _metric(tactic_label, f"{tactic_value} · {tactic_reason}", "teal" if tactic_reason == "전개 강점" else "amber"),
        ],
    }


def _horse_body_delta(value):
    text = str(value or "")
    match = re.search(r"\(([+-]?\d+)\)", text)
    if match:
        delta = int(match.group(1))
        return f"{delta:+d}kg" if delta else "0kg"
    return "0kg"


def _horse_weight_read(weight):
    if weight is None:
        return "부담중량 자료 대기"
    if weight >= 57:
        return "막판 반응 부담"
    if weight <= 54:
        return "체력 소모 이점"
    return "평균권 부담"


def _horse_gate_read(gate):
    if gate <= 3:
        return "안쪽 자리 선점 유리"
    if gate <= 6:
        return "중간 전개 선택지"
    return "외곽 손실 주의"


def _ro_particle(text):
    last = str(text or "").strip()[-1:]
    if not last:
        return "로"
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 and last != "ㄹ":
        return "으로"
    return "로"


def _topic_particle(text):
    last = str(text or "").strip()[-1:]
    if not last:
        return "는"
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28:
        return "은"
    return "는"


def _horse_source(record):
    number = _explain_int(record.get("chulNo"))
    if not number:
        return None
    name = _source_text(record, "hrName", f"{number}번")
    jockey = _source_text(record, "jkName")
    weight = _source_float(record, "wgBudam")
    weight_text = f"{weight:g}kg" if weight is not None else "-"
    body = _horse_body_delta(record.get("wgHr"))
    age = _source_text(record, "age")
    sex = _source_text(record, "sex")
    rating = _source_text(record, "rating")
    distance = _source_text(record, "rcDist")
    gate = number
    weight_reason = _horse_weight_read(weight)
    gate_reason = _horse_gate_read(gate)
    body_reason = "마체 변화 안정" if body in {"0kg", "+1kg", "-1kg"} else "마체 변화 확인"
    style_reason = "선행·선입 자리 싸움" if gate <= 3 else "선입 전개" if gate <= 6 else "외곽 추입 전개"
    note_parts = [
        f"게이트 {gate}번은 {gate_reason}",
        f"부담중량 {weight_text}은 {weight_reason}",
        f"마체 {body}은 {body_reason}",
        f"주행성향은 {style_reason}{_ro_particle(style_reason)} 추정",
    ]
    return {
        "number": number,
        "name": name,
        "subtitle": f"기수 {jockey} · {weight_text} · {age}세 {sex}",
        "stats": f"게이트 {gate}번 · 부담중량 {weight_text} · 마체 {body}",
        "trait": "선입" if gate <= 6 else "추입",
        "note": f"{name}: " + ", ".join(note_parts) + f"합니다. 레이팅 {rating}, 거리 {distance}m도 함께 봅니다.",
        "profile": [
            _metric("말", f"{age}세 {sex}"),
            _metric("기수", jockey),
            _metric("부담중량", weight_text, "amber" if weight is not None and weight >= 57 else "teal"),
            _metric("마체", body, "amber" if body not in {"0kg", "+1kg", "-1kg"} else None),
            _metric("거리", f"{distance}m" if distance != "-" and not distance.endswith("m") else distance),
            _metric("레이팅", rating),
        ],
        "form": [
            _metric("복승률", "-"),
            _metric("최근 4전", "학습 데이터 반영"),
            _metric("게이트", f"{gate}번", "teal" if gate <= 3 else None),
        ],
        "tactics": [
            _metric("전개추정", style_reason, "teal" if gate <= 3 else "primary"),
            _metric("부담", weight_reason, "amber" if weight is not None and weight >= 57 else "teal"),
            _metric("마체", body_reason),
        ],
        "algorithm_reasons": [
            _metric("게이트", f"{gate}번 · {gate_reason}", "teal" if gate <= 3 else "primary"),
            _metric("부담중량", f"{weight_text} · {weight_reason}", "amber" if weight is not None and weight >= 57 else "teal"),
            _metric("마체", f"{body} · {body_reason}", "amber" if body_reason == "마체 변화 확인" else "primary"),
            _metric("주행성향", style_reason, "primary"),
        ],
    }


def participant_sources_from_starters(starters, sport):
    sources = {}
    for record in starters or []:
        if not isinstance(record, dict):
            continue
        source = _horse_source(record) if sport == "horse" else _keirin_source(record)
        if source:
            sources[source["number"]] = source
    return sources


def _participant_signal(rank, pwin, pplc):
    if rank == 1 and pwin >= 0.45:
        return "teal"
    if rank <= 3 and pplc >= 0.65:
        return "primary"
    if pplc < 0.35:
        return "rose"
    return "amber"


def _participant_trait(rank, pwin):
    if rank == 1:
        return "축 후보"
    if rank <= 3:
        return "연대 후보"
    if pwin >= 0.12:
        return "복병"
    return "주의"


def _algorithm_reasons_for_row(row, rows, rank):
    pwin = _explain_float(row.get("pwin"))
    pplc = _explain_float(row.get("pplc"))
    leader = rows[0] if rows else row
    next_row = rows[rank] if rank < len(rows) else None
    grade_context, grade_policy = _keirin_grade_context(rows)
    reasons = [
        {
            "label": "모델 1착",
            "value": f"{_explain_pct(pwin)} · 전체 {rank}위",
            "tone": "teal" if rank == 1 else "primary",
        },
        {
            "label": "연대권",
            "value": f"{_explain_pct(pplc)} · 2착권 안정성",
            "tone": "teal" if pplc >= 0.7 else "amber",
        },
    ]
    if rank == 1 and next_row:
        gap = pwin - _explain_float(next_row.get("pwin"))
        reasons.append({
            "label": "순위격차",
            "value": f"2위 대비 {_explain_pp(gap)}",
            "tone": "teal" if gap >= 0.12 else "amber",
        })
    elif rank > 1:
        gap = pwin - _explain_float(leader.get("pwin"))
        reasons.append({
            "label": "순위격차",
            "value": f"선두 대비 {_explain_pp(gap)}",
            "tone": "amber" if gap >= -0.12 else "rose",
        })
    else:
        reasons.append({"label": "순위격차", "value": "단독 출전 자료", "tone": "primary"})
    if grade_context:
        reasons.append({
            "label": "등급맥락",
            "value": str(grade_policy["read"]),
            "tone": "teal" if grade_context == "선발" else "amber" if grade_context == "특선" else "primary",
        })
    if row.get("mkt_pwin") is not None:
        reasons.append({
            "label": "시장암시",
            "value": f"단승 암시 {_explain_pct(row.get('mkt_pwin'))}",
            "tone": "violet",
        })
    if row.get("pwin_blended") is not None:
        base = _explain_float(row.get("pwin_base"), _explain_float(row.get("pwin")))
        blended_delta = _explain_float(row.get("pwin_blended")) - base
        reasons.append({
            "label": "배당반영",
            "value": f"모델+시장 보정 {_explain_pp(blended_delta)}",
            "tone": "violet" if blended_delta >= 0 else "amber",
        })
    if row.get("trifecta_axis_pwin") is not None:
        order = str(row.get("trifecta_axis_order") or "-")
        reasons.append({
            "label": "삼쌍보드축",
            "value": f"{order} · OOS 1착 {_explain_pct(row.get('trifecta_axis_pwin'))}",
            "tone": "violet",
        })
    starts = _explain_int(row.get("learning_starts"))
    if starts > 0:
        base = _explain_float(row.get("pwin_base"), pwin)
        reasons.append({
            "label": "누적학습",
            "value": f"{starts}전 반영 · {_explain_pp(pwin - base)}",
            "tone": "teal" if pwin >= base else "amber",
        })
    return reasons


def _zscore_metric_for_row(row, rows):
    fields = [
        ("pwin", "모델 1착", "확률"),
        ("pplc", "연대권", "확률"),
        ("pwin_blended", "배당 보정 1착", "확률"),
        ("mkt_pwin", "시장 암시", "확률"),
        ("trifecta_axis_pwin", "삼쌍 축", "확률"),
        ("learning_starts", "누적학습", "전"),
    ]
    best = None
    for key, label, unit in fields:
        values = [_source_float(candidate, key) for candidate in rows if isinstance(candidate, dict)]
        values = [value for value in values if value is not None and math.isfinite(value)]
        current = _source_float(row, key)
        if current is None or len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        if variance <= 0:
            continue
        zscore = (current - mean) / math.sqrt(variance)
        magnitude = abs(zscore)
        if best is None or magnitude > best["magnitude"]:
            if unit == "확률":
                value_text = _explain_pct(current)
            elif unit == "전":
                value_text = f"{int(round(current))}전"
            else:
                value_text = f"{current:g}{unit}"
            best = {
                "label": label,
                "value": value_text,
                "zscore": zscore,
                "magnitude": magnitude,
            }
    if best is not None:
        return best
    return {
        "label": "모델 1착",
        "value": _explain_pct(row.get("pwin")),
        "zscore": 0.0,
        "magnitude": 0.0,
    }


def _algorithm_note_frame_index(row, race_date):
    number = _explain_int(row.get("bno"), 0)
    date_digits = sum(int(ch) for ch in str(race_date or "") if ch.isdigit())
    return (number + date_digits) % 3


def _algorithm_note_lead_sentence(row, rows, race_date):
    name = str(row.get("name") or f"{row.get('bno', '')}번").strip() or "참가자"
    metric = _zscore_metric_for_row(row, rows)
    direction = "높습니다" if metric["zscore"] >= 0 else "낮습니다"
    z_text = f"{metric['zscore']:+.1f}σ"
    frames = [
        f"{name}의 가장 특이한 지표는 {metric['label']} {metric['value']}로, 경주 평균보다 {z_text} {direction}.",
        f"경주 내 편차가 가장 큰 항목은 {name}의 {metric['label']}이며 {metric['value']}({z_text})입니다.",
        f"{name}{_topic_particle(name)} {metric['label']} {metric['value']}에서 가장 두드러져 평균 대비 {z_text} 차이를 보입니다.",
    ]
    return frames[_algorithm_note_frame_index(row, race_date)]


def _algorithm_note_for_row(row, reasons, rank, source=None, rows=None, race_date=None):
    pwin_reason = next((item["value"] for item in reasons if item["label"] == "모델 1착"), "")
    plc_reason = next((item["value"] for item in reasons if item["label"] == "연대권"), "")
    gap_reason = next((item["value"] for item in reasons if item["label"] == "순위격차"), "")
    extras = [item for item in reasons if item["label"] in {"등급맥락", "시장암시", "배당반영", "누적학습", "삼쌍보드축"}]
    name = str(row.get("name") or f"{row.get('bno', rank)}번")
    name_topic = f"{name}{_topic_particle(name)}"
    lead = _algorithm_note_lead_sentence(row, rows or [row], race_date)
    tail = ""
    if extras:
        parts = [f"{item['label']}{_topic_particle(item['label'])} {item['value']}" for item in extras[:3]]
        tail = " " + ", ".join(parts) + "로 반영했습니다."
    if isinstance(source, dict) and source.get("note"):
        return f"{lead} {source['note']} 모델은 1착 {pwin_reason}, 연대권 {plc_reason}, {gap_reason}로 {rank}위 판단을 더했습니다.{tail}"
    return f"{lead} {name_topic} 모델 1착 {pwin_reason}, 연대권 {plc_reason}이고 {gap_reason}라서 모델 순위 {rank}위로 봅니다.{tail}"


def _participant_payload_from_row(row, rows, sport, rank, pro_enabled, sources=None, race_date=None):
    pwin = _explain_float(row.get("pwin"))
    pplc = _explain_float(row.get("pplc"))
    number = _explain_int(row.get("bno"), rank)
    source = None
    if isinstance(sources, dict):
        source = sources.get(number) or sources.get(str(number))
    name = str(row.get("name") or f"{number}번")
    grade = str(row.get("grade") or ("출전마" if sport == "horse" else "선수")).strip()
    model_reasons = _algorithm_reasons_for_row(row, rows, rank)
    source_reasons = source.get("algorithm_reasons", []) if isinstance(source, dict) else []
    reasons = source_reasons + model_reasons
    note = _algorithm_note_for_row(row, model_reasons, rank, source, rows, race_date)
    locked_note = "Pro에서 모델확률, 배당반영, 누적학습 보정 근거를 선수별로 확인합니다."
    profile = source.get("profile") if isinstance(source, dict) and source.get("profile") else [
        {"label": "모델1착", "value": _explain_pct(pwin), "tone": "teal" if rank == 1 else "primary"},
        {"label": "연대권", "value": _explain_pct(pplc), "tone": "teal" if pplc >= 0.7 else "amber"},
        {"label": "모델순위", "value": f"{rank}위", "tone": "primary"},
        {"label": "등급", "value": grade or "-", "tone": "primary"},
    ]
    form = source.get("form") if isinstance(source, dict) and source.get("form") else [
        {"label": item["label"], "value": item["value"], "tone": item.get("tone")}
        for item in model_reasons[:4]
    ]
    tactics = source.get("tactics") if isinstance(source, dict) and source.get("tactics") else [
        {"label": item["label"], "value": item["value"], "tone": item.get("tone")}
        for item in model_reasons[4:8]
    ]
    if sport == "horse":
        form = [
            {**item, "value": _explain_pct(pplc)}
            if item.get("label") == "복승률" and item.get("value") == "-"
            else item
            for item in form
        ]
    return {
        "number": number,
        "name": source.get("name", name) if isinstance(source, dict) else name,
        "subtitle": (
            f"{source['subtitle']} · 모델 순위 {rank}위"
            if isinstance(source, dict) and source.get("subtitle")
            else f"{grade} · 모델 순위 {rank}위"
        ),
        "stats": source.get("stats", f"모델 1착 {_explain_pct(pwin)} · 연대 {_explain_pct(pplc)}") if isinstance(source, dict) else f"모델 1착 {_explain_pct(pwin)} · 연대 {_explain_pct(pplc)}",
        "trait": source.get("trait", _participant_trait(rank, pwin)) if isinstance(source, dict) else _participant_trait(rank, pwin),
        "note": note if pro_enabled else locked_note,
        "signal": _participant_signal(rank, pwin, pplc),
        "profile": profile,
        "form": form,
        "tactics": tactics,
        "algorithm_locked": not pro_enabled,
        "algorithm_note": note if pro_enabled else locked_note,
        "algorithm_reasons": reasons if pro_enabled else [],
    }


def attach_participant_explanations(result, sport, pro_enabled=False):
    if not isinstance(result, dict):
        return result
    rows = result.get("rows")
    if not isinstance(rows, list):
        result["participants"] = []
        return result
    ranked = sorted(
        [row for row in rows if isinstance(row, dict)],
        key=lambda item: -_live_row_rank_score(item),
    )
    sources = result.get("_participant_sources") or result.get("participant_sources") or {}
    race_date = result.get("race_date") or result.get("date") or (result.get("info") or {}).get("ymd")
    result["participants"] = [
        _participant_payload_from_row(row, ranked, sport, rank, pro_enabled, sources, race_date)
        for rank, row in enumerate(ranked, start=1)
    ]
    result.pop("_participant_sources", None)
    return result


def compute_live_decision(sport, ymd, meet, race_no, base_model_out=None):
    """실시간 판단. base_model_out은 이미 계산된 engine.predict/predict_kra 결과.
    반환 dict:
      ok, status, message, updated_at, odds_age_sec,
      market_odds, top, rows, decision, market_used, snapshot_phase
    """
    import datetime as _dt
    now = _dt.datetime.now()
    roster_verification = _roster_verification(base_model_out)

    if isinstance(base_model_out, dict) and base_model_out.get("status") == "roster_mismatch":
        return _finalize_live_decision({
            "ok": False,
            "status": "roster_mismatch",
            "message": ROSTER_MISMATCH_MESSAGE,
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None,
            "market_odds": [],
            "top": None,
            "rows": [],
            "decision": "hold",
            "market_used": False,
            "snapshot_phase": "roster_mismatch",
            "poll_delay_ms": 60000,
            "market_risk": {
                "level": "roster_mismatch",
                "message": ROSTER_MISMATCH_MESSAGE,
            },
            "fallback_signal": None,
            "trifecta_signal": None,
            "picks": [],
        }, roster_verification)

    if base_model_out is None or "error" in base_model_out:
        base_error = str((base_model_out or {}).get("error") or "")
        error_kind = str((base_model_out or {}).get("error_kind") or base_error or "base_prediction_error")
        base_message = str((base_model_out or {}).get("message") or "")
        official_fallback_allowed = not any(
            marker in error_kind
            for marker in ("unsupported_meet", "invalid_date", "invalid_request", "missing_api_key", "roster_mismatch")
        )
        if error_kind in {"no_race", "upstream_api_error"}:
            official_fallback_allowed = False
        signal = None
        if sport == "keirin" and official_fallback_allowed:
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
        return _finalize_live_decision({
            "ok": bool(signal),
            "status": "hold",
            "message": (
                "출주표 기반 모델 예측 불가 — KCYCLE 공식예상 보조신호만 표시"
                if signal else base_message or "모델 예측 불가 (출주표 없음 또는 오류)"
            ),
            "error_kind": error_kind,
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None,
            "market_odds": [],
            "top": top,
            "rows": [],
            "decision": "hold",
            "market_used": False,
            "snapshot_phase": "unknown",
            "poll_delay_ms": _live_poll_delay_ms(sport, False),
            "market_risk": _live_market_risk(sport, False, "hold"),
            "fallback_signal": fallback_signal,
            "trifecta_signal": None,
            "picks": [],
        }, roster_verification)

    rows = [dict(r) for r in base_model_out.get("rows", [])]
    participant_sources = base_model_out.get("_participant_sources") or base_model_out.get("participant_sources") or {}
    kra_market_confidence = (
        kra_confidence_tier(base_model_out.get("_kra_starters"))
        if sport == "horse" else None
    )
    if not rows:
        return _finalize_live_decision({
            "ok": False, "status": "hold", "message": "출주 데이터 없음",
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None, "market_odds": [], "top": None,
            "rows": [], "decision": "hold", "market_used": False,
            "snapshot_phase": "unknown",
            "poll_delay_ms": _live_poll_delay_ms(sport, False),
            "market_risk": _live_market_risk(sport, False, "hold"),
            "fallback_signal": None,
            "trifecta_signal": None,
            "trifecta_lift_signal": None,
            "picks": [],
        }, roster_verification)

    # ── kcycle 배당 fetch (경륜만) ──
    market_odds = None
    fetched_at = None
    horse_market_anchor = bool(
        sport == "horse"
        and base_model_out.get("market_used") is True
        and base_model_out.get("prediction_phase") == "live_odds"
    )
    market_used = False
    market_signal = None
    trifecta_signal = None
    trifecta_lift_signal = None
    trifecta_global_signal = None
    trifecta_axis_signal = None
    trifecta_ensemble = None
    trifecta_board = None
    actual_result = None
    live_market_used = False
    trifecta_axis_used = False
    market_timing = {
        "phase": "disabled",
        "minutes_to_start": None,
        "market_weight": 0.0,
        "allow_late_pull": False,
        "allow_trifecta_axis": False,
        "race_start_at": None,
    }
    if sport == "horse" and base_model_out.get("_kra_result"):
        top = rows[0] if rows else None
        return _finalize_live_decision({
            "ok": True,
            "status": "settled",
            "message": "경주 종료 — 실제 착순 확인, 결과는 학습/검증용으로만 표시",
            "updated_at": now.isoformat(timespec="seconds"),
            "odds_age_sec": None,
            "market_odds": [],
            "top": top,
            "rows": rows,
            "top_conf": _top_confidence(top, rows) if top else {},
            "picks": _mobile_live_picks(rows),
            "_participant_sources": participant_sources,
            "decision": "settled",
            "market_used": False,
            "snapshot_phase": "settled_result",
            "poll_delay_ms": 0,
            "market_risk": {
                "level": "settled_result_available",
                "message": "이미 결과가 나온 경주라 결과 데이터를 예측 신호로 재사용하지 않습니다.",
            },
            "actual_result": base_model_out.get("_kra_result"),
            "market_confidence": kra_market_confidence,
            "fallback_signal": None,
            "market_signal": None,
            "trifecta_axis_signal": None,
            "trifecta_signal": None,
            "trifecta_lift_signal": None,
            "market_timing": {
                "phase": "settled",
                "minutes_to_start": None,
                "market_weight": 0.0,
                "allow_late_pull": False,
                "allow_trifecta_axis": False,
                "race_start_at": None,
            },
        }, roster_verification)
    if sport == "keirin":
        stnd_yr = str(ymd)[:4] if ymd else ""
        if _kcycle_should_accept_result_outcome(ymd, race_no, now):
            try:
                actual_result = fetch_kcycle_result_outcome(
                    stnd_yr, ymd, meet, race_no, timeout=0.75, max_attempts=1,
                )
            except Exception:
                actual_result = None
        if actual_result:
            top = rows[0] if rows else None
            return _finalize_live_decision({
                "ok": True,
                "status": "settled",
                "message": "경주 종료 — 실제 착순 확인, 확정배당은 학습/검증용으로만 표시",
                "updated_at": now.isoformat(timespec="seconds"),
                "odds_age_sec": None,
                "market_odds": [],
                "top": top,
                "rows": rows,
                "top_conf": _top_confidence(top, rows) if top else {},
                "picks": _mobile_live_picks(rows),
                "_participant_sources": participant_sources,
                "decision": "settled",
                "market_used": False,
                "snapshot_phase": "settled_result",
                "poll_delay_ms": 0,
                "market_risk": {
                    "level": "settled_result_available",
                    "message": "이미 결과가 나온 경주라 실시간/확정배당을 예측 신호로 재사용하지 않습니다.",
                },
                "actual_result": actual_result,
                "fallback_signal": _live_fallback_signal(base_model_out),
                "market_signal": None,
                "trifecta_axis_signal": None,
                "trifecta_signal": None,
                "trifecta_lift_signal": None,
                "market_timing": {
                    "phase": "settled",
                    "minutes_to_start": None,
                    "market_weight": 0.0,
                    "allow_late_pull": False,
                    "allow_trifecta_axis": False,
                    "race_start_at": None,
                },
            }, roster_verification)
    if sport == "keirin" and os.environ.get("KCYCLE_ENABLED", "0") == "1":
        stnd_yr = str(ymd)[:4] if ymd else ""
        try:
            market_odds, fetched_at = fetch_kcycle_odds_with_ts(
                stnd_yr, ymd, race_no, timeout=0.75, max_attempts=1,
            )
        except Exception:
            market_odds = None
        if KCYCLE_TRIFECTA_ENABLED:
            try:
                trifecta_board, trifecta_fetched_at = fetch_kcycle_trifecta_board_with_ts(
                    stnd_yr, ymd, race_no, timeout=0.75, max_attempts=1,
                )
                trifecta_axis_signal = _market_trifecta_axis_signal(trifecta_board)
                trifecta_signal = _market_trifecta_signal(trifecta_board)
                trifecta_lift_signal = _market_trifecta_lift_signal(trifecta_board)
                trifecta_global_signal = _market_trifecta_global_rerank_signal(trifecta_board)
                ensemble_rank = kcycle_ensemble_trifecta_rank(trifecta_board)
                if ensemble_rank:
                    ensemble_tier = kcycle_trifecta_confidence_tier(trifecta_board)
                    trifecta_ensemble = {
                        "pick": ensemble_rank[0]["combo"],
                        "top5": [item["combo"] for item in ensemble_rank[:5]],
                        "tier": ensemble_tier["tier"],
                        "tier_historical_exact": ensemble_tier["tier_historical_exact"],
                        "selection": "ensemble_v1_top1",
                        "board_complete": len(trifecta_board or {}) == 210,
                        "coverage": min(1.0, len(trifecta_board or {}) / 210.0),
                        "signal_strength": ensemble_tier.get("signal_strength"),
                        "source": "ensemble_v1",
                    }
                trifecta_timing = _kcycle_market_timing_policy(ymd, race_no, trifecta_fetched_at)
                save_kcycle_trifecta_snapshot(
                    stnd_yr,
                    ymd,
                    meet,
                    race_no,
                    trifecta_board,
                    fetched_at=trifecta_fetched_at,
                    signal=trifecta_signal,
                    snapshot_phase=_snapshot_phase_from_market_timing(trifecta_timing),
                    market_timing=trifecta_timing,
                    signals={
                        "axis": trifecta_axis_signal,
                        "watch": trifecta_signal,
                        "lift": trifecta_lift_signal,
                        "global": trifecta_global_signal,
                    },
                )
                fetched_at = fetched_at or trifecta_fetched_at
            except Exception:
                trifecta_axis_signal = None
                trifecta_signal = None
                trifecta_lift_signal = None
                trifecta_global_signal = None
                trifecta_ensemble = None

    if sport == "keirin" and os.environ.get("KCYCLE_ENABLED", "0") == "1":
        market_timing = _kcycle_market_timing_policy(ymd, race_no, fetched_at)

    # ── 상태 판정 ──
    if market_odds and len(market_odds) >= 2:
        # 시장 암시확률 → 모델과 앙상블
        imp = {b: 1.0 / o for b, o in market_odds.items() if o and o > 0}
        total = sum(imp.values())
        market_weight = float(market_timing.get("market_weight") or 0.0)
        if total > 0 and market_weight > 0:
            market_signal = _market_favorite_signal(market_odds)
            imp_norm = {b: v / total for b, v in imp.items()}
            model_weight = 1.0 - market_weight
            for r in rows:
                bno = r.get("bno", 0)
                mkt_p = imp_norm.get(bno, 0.0)
                model_p = r.get("pwin", 0.0)
                r["pwin_base"] = model_p
                r["pwin_blended"] = model_weight * model_p + market_weight * mkt_p
                r["mkt_pwin"] = mkt_p
            # blended 로 재정렬
            rows.sort(key=lambda r: -r.get("pwin_blended", r.get("pwin", 0)))
            if market_signal:
                market_signal = dict(market_signal)
                current_top_bno = int(rows[0].get("bno", -1)) if rows else -1
                allow_market_annotation = (
                    market_timing.get("phase") in {"late", "unknown"}
                    and current_top_bno == int(market_signal["leader"])
                )
                market_signal["applied"] = bool(allow_market_annotation)
                market_signal["timing_phase"] = market_timing.get("phase")
                if not allow_market_annotation:
                    market_signal["blocked_by_order_conflict"] = True
            for r in rows:
                r["pwin"] = r.get("pwin_blended", r["pwin"])
            market_used = True
            status = "odds_live"
            if market_signal and market_signal.get("applied"):
                message = f"{market_signal['label']} 반영"
            elif abs(market_weight - 0.30) < 1e-9:
                message = "저의존 배당 블렌드 반영 (시장 0.30 + 모델 0.70)"
            else:
                message = f"저의존 배당 제한반영 (시장 {market_weight:.2f} + 모델 {model_weight:.2f})"
            snapshot_phase = "live_odds"
        elif total > 0 and market_weight <= 0:
            status = "odds_unavailable"
            message = "경주 시작 이후 배당 감지 — 예측 신호 차단"
            snapshot_phase = "post_start_market_blocked"
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
            message = "사전 예측 (배당 미반영 — KCYCLE 비활성)"
            snapshot_phase = "pre_race"

    if horse_market_anchor:
        market_used = True
        status = "odds_live"
        message = "공식 사전배당 시장 앵커 반영"
        snapshot_phase = "live_odds"
        market_timing = {
            "phase": "official_pre_start_market",
            "minutes_to_start": None,
            "market_weight": 1.0,
            "allow_late_pull": False,
            "allow_trifecta_axis": False,
            "race_start_at": None,
        }

    timing_blocks_axis = False
    if trifecta_axis_signal:
        trifecta_axis_signal = dict(trifecta_axis_signal)
        phase = market_timing.get("phase")
        timing_blocks_axis = not bool(market_timing.get("allow_trifecta_axis"))
        if trifecta_axis_signal.get("tier") == "market_trifecta_late_pull_strong":
            timing_blocks_axis = timing_blocks_axis or not bool(market_timing.get("allow_late_pull"))
        trifecta_axis_signal["applied"] = False
        trifecta_axis_signal["timing_phase"] = phase
        if timing_blocks_axis:
            trifecta_axis_signal["blocked_by_timing"] = True
            trifecta_axis_signal["robust_warning"] = (
                "초반/중반 배당은 아직 흔들릴 수 있어 모델을 덮지 않고 관찰 신호로만 표시합니다."
                if phase in {"early", "mid"} else
                "경주 시작 이후 배당은 결과/확정배당 누수 위험이 있어 예측 신호로 쓰지 않습니다."
            )

    suppress_trifecta_axis = False
    if market_signal and trifecta_axis_signal and not timing_blocks_axis:
        try:
            market_leader = int(market_signal.get("leader"))
            axis_leader = int(trifecta_axis_signal.get("leader"))
            market_expected = float(market_signal.get("expected_top1") or 0.0)
            axis_expected = float(trifecta_axis_signal.get("expected_top1") or 0.0)
            suppress_trifecta_axis = market_leader != axis_leader and market_expected >= axis_expected
        except (TypeError, ValueError):
            suppress_trifecta_axis = False
    if trifecta_axis_signal and not suppress_trifecta_axis and not timing_blocks_axis:
        try:
            axis_leader = int(trifecta_axis_signal.get("leader"))
            current_top_bno = int(rows[0].get("bno", -1)) if rows else -1
        except (TypeError, ValueError):
            axis_leader = -1
            current_top_bno = -1
        if axis_leader != current_top_bno:
            suppress_trifecta_axis = True
            trifecta_axis_signal["blocked_by_order_conflict"] = True
    trifecta_axis_used = (
        False
        if suppress_trifecta_axis or timing_blocks_axis
        else _apply_trifecta_axis_signal(rows, trifecta_axis_signal)
    )
    if trifecta_axis_signal:
        trifecta_axis_signal["applied"] = bool(trifecta_axis_used)
    if trifecta_axis_used:
        market_used = True
        if trifecta_axis_signal and trifecta_axis_signal.get("tier") == "market_trifecta_late_pull_strong":
            axis_message = "마감배당 강쏠림 축 반영 (2026 OOS 1착 83.5%, 삼쌍 33.5%)"
        else:
            axis_message = "삼쌍 보드 축 반영 (OOS 1착 63.7%)"
        if status in {"pre_race", "odds_unavailable"}:
            status = "trifecta_axis_live"
            message = axis_message
            snapshot_phase = "trifecta_axis"
        elif axis_message not in message:
            message = f"{message} · {axis_message}"
    elif trifecta_axis_signal and trifecta_axis_signal.get("blocked_by_timing"):
        if market_timing.get("phase") == "early":
            suffix = " · 초반 배당 관찰 중 — 모델 우선"
        elif market_timing.get("phase") == "mid":
            suffix = " · 중반 배당 관찰 중 — 마감 전까지 모델 우선"
        else:
            suffix = " · 배당 시간대 불일치 — 예측 신호 차단"
        if suffix not in message:
            message = f"{message}{suffix}"
    if trifecta_signal:
        suffix = " · 삼쌍 저표본 watch 감지(50% 배포 금지)"
        if suffix not in message:
            message = f"{message}{suffix}"
    trifecta_lift_used = False
    if trifecta_lift_signal:
        trifecta_lift_signal = dict(trifecta_lift_signal)
        trifecta_lift_signal["applied"] = False
        trifecta_lift_signal["timing_phase"] = market_timing.get("phase")
        lift_order = [
            int(x) for x in trifecta_lift_signal.get("order", [])
            if isinstance(x, int) or str(x).isdigit()
        ]
        top_bno = int(rows[0].get("bno") or 0) if rows else 0
        allow_lift = bool(market_timing.get("allow_trifecta_axis"))
        order_conflict = len(lift_order) >= 3 and top_bno > 0 and lift_order[0] != top_bno
        if not allow_lift:
            trifecta_lift_signal["blocked_by_timing"] = True
            suffix = " · 삼쌍 순서 재랭킹 관찰 중(마감 전 미반영)"
        elif order_conflict:
            trifecta_lift_signal["blocked_by_order_conflict"] = True
            suffix = " · 삼쌍 순서 재랭킹 관찰 중(1착 후보 충돌)"
        else:
            trifecta_lift_signal["applied"] = True
            trifecta_lift_used = True
            suffix = " · 삼쌍 순서 재랭킹 반영(holdout 26.09%, +0.55pp)"
        if suffix not in message:
            message = f"{message}{suffix}"
    trifecta_global_used = False
    if trifecta_global_signal:
        trifecta_global_signal = dict(trifecta_global_signal)
        trifecta_global_signal["applied"] = False
        trifecta_global_signal["timing_phase"] = market_timing.get("phase")
        global_order = [
            int(x) for x in trifecta_global_signal.get("order", [])
            if isinstance(x, int) or str(x).isdigit()
        ]
        top_bno = int(rows[0].get("bno") or 0) if rows else 0
        allow_global = bool(market_timing.get("allow_trifecta_axis"))
        order_conflict = len(global_order) >= 3 and top_bno > 0 and global_order[0] != top_bno
        if trifecta_lift_used:
            trifecta_global_signal["blocked_by_stronger_signal"] = True
            suffix = None
        elif not allow_global:
            trifecta_global_signal["blocked_by_timing"] = True
            suffix = " · 삼쌍 전체보드 재랭킹 관찰 중(마감 전 미반영)"
        elif order_conflict:
            trifecta_global_signal["blocked_by_order_conflict"] = True
            suffix = " · 삼쌍 전체보드 재랭킹 관찰 중(1착 후보 충돌)"
        else:
            trifecta_global_signal["applied"] = True
            trifecta_global_used = True
            suffix = " · 삼쌍 전체보드 재랭킹 반영(test +0.87pp)"
        if suffix and suffix not in message:
            message = f"{message}{suffix}"
    live_market_used = market_used or trifecta_axis_used

    # ── 최종판정 여부 ──
    top = rows[0] if rows else None
    market_marginals = _keirin_trifecta_board_marginals(trifecta_board) if sport == "keirin" else None
    market_pick_order = _keirin_market_order_for_rows(rows, trifecta_board) if sport == "keirin" else []
    # decision: final_candidate (배당 있음), hold (배당 없음/불확실)
    if market_used and sport == "keirin":
        # 확신도 높으면 final_candidate, 아니면 hold
        tc = _top_confidence(top, rows) if top else {}
        gap = 0
        if len(rows) >= 2:
            sorted_r = sorted(rows, key=lambda r: -r.get("pwin", 0))
            gap = sorted_r[0].get("pwin", 0) - sorted_r[1].get("pwin", 0)
        _, grade_policy = _keirin_grade_context(rows)
        if gap >= float(grade_policy["final_gap"]) or tc.get("grade") == "강":
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

    return _finalize_live_decision({
        "ok": True,
        "status": status,
        "message": message,
        "updated_at": now.isoformat(timespec="seconds"),
        "odds_age_sec": odds_age_sec,
        "market_odds": _market_odds_entries(market_odds, trifecta_board),
        "top": top,
        "rows": rows,
        "top_conf": _top_confidence(top, rows) if top else {},
        "picks": _mobile_live_picks(
            rows,
            trifecta_axis_signal if trifecta_axis_used else None,
            (
                trifecta_lift_signal if trifecta_lift_used
                else trifecta_global_signal if trifecta_global_used
                else None
            ),
            (
                trifecta_ensemble
                if not (
                    trifecta_axis_signal
                    or trifecta_lift_signal
                    or trifecta_global_signal
                )
                else None
            ),
            market_pick_order,
            market_marginals,
        ),
        "_participant_sources": participant_sources,
        "decision": decision,
        "market_used": market_used,
        "pick_source": (
            base_model_out.get("pick_source")
            if sport == "horse"
            else "market" if market_pick_order else "model"
        ),
        "algorithm_version": base_model_out.get("algorithm_version"),
        "prediction_phase": base_model_out.get("prediction_phase"),
        "snapshot_phase": snapshot_phase,
        "poll_delay_ms": _live_poll_delay_ms(sport, live_market_used),
        "market_risk": _live_market_risk(sport, live_market_used, status),
        "fallback_signal": _live_fallback_signal(base_model_out),
        "market_confidence": kra_market_confidence,
        "market_signal": _live_signal_payload(market_signal) if market_signal else None,
        "trifecta_axis_signal": _live_signal_payload(trifecta_axis_signal) if trifecta_axis_signal else None,
        "trifecta_signal": _live_signal_payload(trifecta_signal) if trifecta_signal else None,
        "trifecta_lift_signal": _live_signal_payload(trifecta_lift_signal) if trifecta_lift_signal else None,
        "trifecta_global_signal": _live_signal_payload(trifecta_global_signal) if trifecta_global_signal else None,
        "trifecta_ensemble": trifecta_ensemble,
        "market_timing": market_timing,
    }, roster_verification)

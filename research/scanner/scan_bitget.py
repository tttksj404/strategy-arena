#!/usr/bin/env python3
"""Bitget USDT-M perpetual futures opportunity scanner — READ-ONLY.

Public market-data endpoints only, no API key/auth, no order placement.
Ranks funding-carry and momentum opportunities across the full USDT-M
universe (~702 symbols incl. stock/ETF tokenized futures). Retry/backoff
follows research/wave1/common.py conceptually but stays dependency-free
so this file runs standalone.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows default codepage (cp949) mangles Korean console output

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # parity with wave1 layout; no hard import used

OUT_DIR = BASE_DIR / "out"
API_ROOT = "https://api.bitget.com"
USER_AGENT = "strategy-arena-scanner/1.0 (+read-only market scan; no trading)"
STOCK_BASE_COINS = {
    "SPY", "QQQ", "TSLA", "NVDA", "MSTR", "COIN", "HOOD", "AMD", "MSFT", "META",
    "AMZN", "GOOGL", "AAPL", "NFLX", "INTC", "PLTR", "TQQQ", "IWM", "CRCL", "OPEN", "OPENAI",
}
TOP_VOLUME_TARGET_LIMIT = 120
MAX_HISTORY_PAGES = 4
HISTORY_PAGE_SIZE = 100
HISTORY_COVERAGE_DAYS = 33  # stop paging once history spans >~33d (covers the 30d window)
CANDLE_LIMIT = 35
REQUEST_SLEEP = 0.1
MS_PER_DAY = 86_400_000
PERIODS_PER_YEAR_8H = 365.0 * 24.0 / 8.0  # 1095 settlements/yr on an 8h cadence


def log(msg: str) -> None:
    print(msg, flush=True)


def to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def http_get(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a Bitget v2 public endpoint, 3 attempts + backoff; raises on final failure."""
    url = f"{API_ROOT}{path}"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = session.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=(5.0, 20.0))
            resp.raise_for_status()
            payload = resp.json()
            if str(payload.get("code")) not in ("00000", "0"):
                raise RuntimeError(f"api error {payload.get('code')}: {payload.get('msg')}")
            time.sleep(REQUEST_SLEEP)
            return payload
        except Exception as exc:  # noqa: BLE001 - any transient failure triggers retry
            last_exc = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise RuntimeError(f"GET {path} {params} failed after 3 attempts: {last_exc}") from last_exc


def safe_get(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        return http_get(session, path, params)
    except RuntimeError as exc:
        log(f"[warn] {path} {params.get('symbol', '')} failed: {exc}")
        return {}


def fetch_contracts(session: requests.Session) -> dict[str, dict[str, Any]]:
    body = http_get(session, "/api/v2/mix/market/contracts", {"productType": "usdt-futures"})
    out: dict[str, dict[str, Any]] = {}
    for row in body.get("data", []):
        symbol = row.get("symbol")
        if not symbol:
            continue
        base_coin = row.get("baseCoin", "")
        out[symbol] = {
            "symbol": symbol, "base_coin": base_coin, "symbol_type": row.get("symbolType", ""),
            "is_stock_token": base_coin.upper() in STOCK_BASE_COINS,
            "maker_fee": to_float(row.get("makerFeeRate")) or 0.0002,
            "taker_fee": to_float(row.get("takerFeeRate")) or 0.0006,
            "min_trade_usdt": to_float(row.get("minTradeUSDT")) or 0.0,
            "fund_interval_hours": to_float(row.get("fundInterval")) or 8.0,
        }
    return out


def fetch_tickers(session: requests.Session) -> dict[str, dict[str, Any]]:
    body = http_get(session, "/api/v2/mix/market/tickers", {"productType": "usdt-futures"})
    return {row["symbol"]: row for row in body.get("data", []) if row.get("symbol")}


def fetch_current_funding_bulk(session: requests.Session) -> dict[str, float]:
    body = safe_get(session, "/api/v2/mix/market/current-fund-rate", {"productType": "usdt-futures"})
    out: dict[str, float] = {}
    for row in body.get("data", []):
        rate = to_float(row.get("fundingRate"))
        if row.get("symbol") and rate is not None:
            out[row["symbol"]] = rate
    return out


def select_history_targets(contracts: dict[str, dict[str, Any]], tickers: dict[str, dict[str, Any]], limit: int) -> list[str]:
    def qvol(sym: str) -> float:
        t = tickers.get(sym, {})
        return to_float(t.get("quoteVolume")) or to_float(t.get("usdtVolume")) or 0.0

    top = sorted(contracts.keys(), key=qvol, reverse=True)[:limit]
    stock_syms = [s for s, c in contracts.items() if c["is_stock_token"]]
    return list(dict.fromkeys(top + stock_syms))  # dedupe, preserve order


def fetch_funding_history(session: requests.Session, symbol: str) -> list[tuple[int, float]]:
    records: list[tuple[int, float]] = []
    now_ms = int(time.time() * 1000)
    for page_no in range(1, MAX_HISTORY_PAGES + 1):
        params = {"symbol": symbol, "productType": "usdt-futures", "pageSize": HISTORY_PAGE_SIZE, "pageNo": page_no}
        rows = safe_get(session, "/api/v2/mix/market/history-fund-rate", params).get("data", [])
        if not rows:
            break
        for row in rows:
            ts, rate = to_float(row.get("fundingTime")), to_float(row.get("fundingRate"))
            if ts is not None and rate is not None:
                records.append((int(ts), rate))
        oldest_ts = records[-1][0] if records else now_ms
        if len(rows) < HISTORY_PAGE_SIZE or (now_ms - oldest_ts) > HISTORY_COVERAGE_DAYS * MS_PER_DAY:
            break
    return records


def fetch_candles(session: requests.Session, symbol: str) -> list[list[str]]:
    params = {"symbol": symbol, "productType": "usdt-futures", "granularity": "1D", "limit": CANDLE_LIMIT}
    return safe_get(session, "/api/v2/mix/market/candles", params).get("data", [])


def compute_funding_apr(records: list[tuple[int, float]], now_ms: int, window_days: int) -> float | None:
    window = [(ts, r) for ts, r in records if ts >= now_ms - window_days * MS_PER_DAY]
    if len(window) < 2:
        return None
    days_covered = min(float(window_days), max((now_ms - min(ts for ts, _ in window)) / MS_PER_DAY, 1.0))
    return (sum(r for _, r in window) / days_covered) * 365.0 * 100.0


def compute_momentum(candles: list[list[str]]) -> tuple[float | None, float | None]:
    closes = [to_float(row[4]) if len(row) > 4 else None for row in candles]
    n = len(closes)
    mom7 = (closes[-1] / closes[-8] - 1.0) * 100.0 if n >= 8 and closes[-1] and closes[-8] else None
    mom30 = (closes[-1] / closes[-31] - 1.0) * 100.0 if n >= 31 and closes[-1] and closes[-31] else None
    return mom7, mom30


def build_base_record(c: dict[str, Any], ticker: dict[str, Any], funding_bulk: dict[str, float]) -> dict[str, Any]:
    last, bid, ask = to_float(ticker.get("lastPr")), to_float(ticker.get("bidPr")), to_float(ticker.get("askPr"))
    spread_pct = (ask - bid) / last * 100.0 if bid and ask and last else None
    qvol = to_float(ticker.get("quoteVolume")) or to_float(ticker.get("usdtVolume"))
    funding_now = to_float(ticker.get("fundingRate"))
    if funding_now is None:
        funding_now = funding_bulk.get(c["symbol"])
    interval = c["fund_interval_hours"] or 8.0
    funding_now_8h = funding_now * (8.0 / interval) if funding_now is not None else None
    now_est_apr = funding_now_8h * PERIODS_PER_YEAR_8H * 100.0 if funding_now_8h is not None else None
    return {
        "symbol": c["symbol"], "base_coin": c["base_coin"], "symbol_type": c["symbol_type"],
        "is_stock_token": c["is_stock_token"], "last": last, "spread_pct": spread_pct, "qvol_24h": qvol,
        "funding_now": funding_now, "funding_interval_hours": interval, "funding_now_8h": funding_now_8h,
        "funding_now_apr_est": now_est_apr, "funding_7d_apr": None, "funding_30d_apr": None,
        "mom_7d": None, "mom_30d": None, "momentum_z": None, "maker_fee": c["maker_fee"],
        "taker_fee": c["taker_fee"], "min_trade_usdt": c["min_trade_usdt"], "has_history": False,
        "carry_score": now_est_apr, "carry_basis": "now_est" if now_est_apr is not None else None,
    }


def enrich_with_history_and_candles(
    session: requests.Session, records: dict[str, Any], targets: list[str], use_history: bool
) -> None:
    now_ms = int(time.time() * 1000)
    total = len(targets)
    for idx, symbol in enumerate(targets, 1):
        rec = records.get(symbol)
        if rec is None:
            continue
        rec["has_history"] = True
        rec["mom_7d"], rec["mom_30d"] = compute_momentum(fetch_candles(session, symbol))
        if use_history:
            hist = fetch_funding_history(session, symbol)
            apr7 = compute_funding_apr(hist, now_ms, 7)
            rec["funding_7d_apr"] = apr7
            rec["funding_30d_apr"] = compute_funding_apr(hist, now_ms, 30)
            if apr7 is not None:
                rec["carry_score"], rec["carry_basis"] = apr7, "7d_realized"
        if idx % 20 == 0 or idx == total:
            log(f"[scan] history+candles {idx}/{total}")


def apply_momentum_z(records: dict[str, Any]) -> None:
    values = [r["mom_30d"] for r in records.values() if r["mom_30d"] is not None]
    if len(values) < 3:
        return
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values) or 1e-9
    for r in records.values():
        if r["mom_30d"] is not None:
            r["momentum_z"] = 0.6745 * (r["mom_30d"] - median) / mad


def fmt(x: float | None, nd: int = 2, suffix: str = "") -> str:
    return "-" if x is None else f"{x:.{nd}f}{suffix}"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def carry_row(r: dict[str, Any]) -> list[str]:
    score = r["carry_score"]
    direction = "숏퍼프+롱현물" if (score or 0) >= 0 else "롱퍼프+숏현물(공매도필요)"
    round_trip = r["taker_fee"] * 2 * 100.0
    daily = abs(score) / 365.0 if score is not None else None
    breakeven = round_trip / daily if daily else None
    return [r["symbol"], direction, fmt(score, 1, "%"), r["carry_basis"] or "-", fmt(r["funding_now_8h"], 5),
            fmt(round_trip, 3, "%"), fmt(breakeven, 1, "d"), fmt(r["qvol_24h"], 0)]


def mom_row(r: dict[str, Any]) -> list[str]:
    return [r["symbol"], fmt(r["mom_30d"], 2, "%"), fmt(r["mom_7d"], 2, "%"), fmt(r["momentum_z"], 2), fmt(r["qvol_24h"], 0)]


def stock_row(r: dict[str, Any]) -> list[str]:
    return [r["symbol"], r["base_coin"], fmt(r["funding_now_8h"], 5), fmt(r["funding_7d_apr"], 1, "%"),
            fmt(r["mom_7d"], 2, "%"), fmt(r["mom_30d"], 2, "%"), fmt(r["qvol_24h"], 0)]


def render_markdown(payload: dict[str, Any], top_n: int) -> str:
    s = payload["summary"]
    mode_label = "전체(펀딩7d/30d 실측)" if not payload["mode"]["no_history"] else "고속(현재펀딩+캔들만)"
    lines = [
        f"# Bitget USDT-M 기회 스캔 — {payload['generated_at']}", "",
        f"모드: {mode_label} · 전체계약 {s['total_contracts']} · 분석대상(top-vol+주식토큰) {s['history_target_count']} · "
        f"주식토큰 {s['stock_token_count']}", "",
        f"## ① 캐리(펀딩) 기회 Top {top_n} (|연환산 APR| 기준, 방향·왕복비용·손익분기 포함)",
        md_table(["Symbol", "방향", "연환산APR", "산출기준", "funding_now(8h)", "왕복비용", "손익분기", "24h거래대금(USDT)"],
                  [carry_row(r) for r in payload["carry_top"][:top_n]]), "",
        f"## ② 모멘텀 Top {top_n} (mom_30d 기준)",
        md_table(["Symbol", "mom_30d", "mom_7d", "z-score", "24h거래대금"],
                  [mom_row(r) for r in payload["momentum_top"][:top_n]]), "",
        f"## ② 모멘텀 Bottom {top_n}",
        md_table(["Symbol", "mom_30d", "mom_7d", "z-score", "24h거래대금"],
                  [mom_row(r) for r in payload["momentum_bottom"][:top_n]]), "",
        "## ③ 주식·ETF 토큰 전종목",
        md_table(["Symbol", "BaseCoin", "funding_now(8h)", "funding_7d_apr", "mom_7d", "mom_30d", "24h거래대금"],
                  [stock_row(r) for r in payload["stock_tokens"]]), "",
        "## ④ 요약 통계",
        md_table(["항목", "값"], [
            ["총 계약 수", str(s["total_contracts"])], ["티커 수신", str(s["tickers_ok"])],
            ["펀딩/캔들 분석 대상", str(s["history_target_count"])], ["주식·ETF 토큰 수", str(s["stock_token_count"])],
            ["funding_now(8h) 평균", fmt(s["avg_funding_now_8h"], 6)], ["mom_30d 중앙값", fmt(s["median_mom_30d"], 2, "%")],
            ["소요 시간(초)", fmt(s["elapsed_sec"], 1)],
        ]), "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bitget USDT-M read-only opportunity scanner")
    parser.add_argument("--top", type=int, default=15, help="rows per ranked section (default 15)")
    parser.add_argument("--no-history", action="store_true", help="fast mode: skip funding history (keep current funding+candles)")
    args = parser.parse_args()

    start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    log("[scan] fetching contracts/tickers/current-funding ...")
    contracts = fetch_contracts(session)
    tickers = fetch_tickers(session)
    funding_bulk = fetch_current_funding_bulk(session)
    log(f"[scan] contracts={len(contracts)} tickers={len(tickers)}")

    records = {sym: build_base_record(c, tickers.get(sym, {}), funding_bulk) for sym, c in contracts.items()}
    targets = select_history_targets(contracts, tickers, TOP_VOLUME_TARGET_LIMIT)
    log(f"[scan] history/candle targets={len(targets)} (top{TOP_VOLUME_TARGET_LIMIT}+stock-tokens), no_history={args.no_history}")
    enrich_with_history_and_candles(session, records, targets, use_history=not args.no_history)
    apply_momentum_z(records)

    carry_pool = sorted(
        (r for r in records.values() if r["carry_score"] is not None),
        key=lambda r: (0 if r["carry_basis"] == "7d_realized" else 1, -abs(r["carry_score"])),
    )
    mom_pool = [r for r in records.values() if r["mom_30d"] is not None]
    momentum_top = sorted(mom_pool, key=lambda r: r["mom_30d"], reverse=True)
    momentum_bottom = sorted(mom_pool, key=lambda r: r["mom_30d"])
    stock_tokens = sorted((r for r in records.values() if r["is_stock_token"]), key=lambda r: r["qvol_24h"] or 0.0, reverse=True)

    funding_now_vals = [r["funding_now_8h"] for r in records.values() if r["funding_now_8h"] is not None]
    summary = {
        "total_contracts": len(contracts), "tickers_ok": len(tickers), "history_target_count": len(targets),
        "stock_token_count": len(stock_tokens),
        "avg_funding_now_8h": statistics.fmean(funding_now_vals) if funding_now_vals else None,
        "median_mom_30d": statistics.median([r["mom_30d"] for r in mom_pool]) if mom_pool else None,
        "elapsed_sec": time.time() - start,
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "mode": {"top_n": args.top, "no_history": args.no_history},
        "summary": summary, "carry_top": carry_pool[: args.top], "momentum_top": momentum_top[: args.top],
        "momentum_bottom": momentum_bottom[: args.top], "stock_tokens": stock_tokens, "records": records,
    }

    ts_label = datetime.now().strftime("%Y%m%d_%H%M")
    md_path, json_path = OUT_DIR / f"scan_{ts_label}.md", OUT_DIR / f"scan_{ts_label}.json"
    md_path.write_text(render_markdown(payload, args.top), encoding="utf-8")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    log(f"[scan] done in {summary['elapsed_sec']:.1f}s -> {md_path.name}, {json_path.name}")
    log("symbol       dir              apr    basis   fund8h   rtcost  b/e_d   qvol24h")
    for r in carry_pool[:10]:
        row = carry_row(r)
        log(f"{row[0]:<12} {row[1]:<16} {row[2]:>7} {row[3]:<8} {row[4]:>8} {row[5]:>7} {row[6]:>6} {row[7]:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

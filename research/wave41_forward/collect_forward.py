#!/usr/bin/env python3
# Wave-41: forward collector for the series that cannot be backtested.
#
# wave40 screened open interest and the long/short account ratio and found median |rho| of 0.067 against
# next-day returns -- below the 0.20 floor worth building anything for. That screen ran on the only
# history the venues expose, 180 days, which is one regime and two 90-day windows. A rejection on 178
# aligned observations is real but provisional: the sample cannot distinguish "no signal" from "signal
# too small to see yet".
#
# The only way to change that is to let the sample grow, which means capturing the series daily starting
# now. This collector exists for that and nothing else. It is deliberately dumb: fetch, append, exit. No
# signal is computed, no position is suggested, no threshold is evaluated. Analysis belongs in a later
# wave once there are enough observations to support it, and keeping the collector free of analysis means
# it cannot quietly start fitting to its own growing sample.
#
# Design constraints learned the hard way in this repo:
#   - Append-only JSONL. A crashed or timed-out run must never corrupt earlier records.
#   - Idempotent per UTC day. Running twice in one day must not double-count an observation, because a
#     duplicated row would silently inflate the statistical power of any later test.
#   - Read-only endpoints only. docs/RESEARCH_GUARDRAILS.md forbids order/account/withdrawal endpoints,
#     and this file references none.
#   - Every failure recorded, not swallowed. A gap in the series must be visible later as a gap, since a
#     silently-skipped venue outage would look like data rather than absence of data.

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import datetime as dt
import json
from pathlib import Path
import sys
import time
from typing import Final
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

BASE_DIR: Final = Path(__file__).resolve().parent
LEDGER_PATH: Final = BASE_DIR / "ledger" / "forward_series.jsonl"

# Same 16 currencies wave40 screened, so the growing sample is directly comparable to that baseline
# rather than being a differently-shaped universe.
CURRENCIES: Final = (
    "BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "LINK", "AVAX",
    "NEAR", "ARB", "OP", "SUI", "APT", "LTC", "BCH", "FIL",
    # Traditional-asset perpetuals, added after wave55 found them and could not settle them. Their
    # funding history reaches back only 92 days, far under the 455 a causal walk-forward needs, so the
    # rejection there is provisional and only a growing sample can make it final. The specific question
    # they are collected to answer: gold's funding had a median of 0.00% and was positive on 31.2% of
    # stamps against BTC's +3.24% and 75.4% -- is that the 92-day window or the market's structure?
    "XAU", "XAG", "MSTR", "AAPL", "META", "SPY", "QQQ", "TSLA", "NVDA",
)
# Spot legs for the gold pair. wave55 measured XAUT tracking the XAU perp at 0.9981 daily-return
# correlation with 1.7% annualised residual, so the delta-neutral construction is sound even though the
# carry is not there yet; capturing both legs keeps that verifiable as the sample grows.
SPOT_INSTRUMENTS: Final = ("XAUT-USDT", "PAXG-USDT")
REQUEST_PAUSE_SECONDS: Final = 0.15


@dataclass(frozen=True, slots=True)
class Observation:
    currency: str
    open_interest_usd: float | None
    volume_usd: float | None
    long_short_ratio: float | None
    close: float | None
    funding_rate: float | None
    next_funding_rate: float | None
    error: str | None = None


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


def collect_currency(currency: str) -> Observation:
    """One currency's snapshot. Partial failure is recorded per field, not fatal for the row.

    Fields are fetched independently because the endpoints fail independently: rubik statistics have been
    observed to lag or 400 while market data is fine. Losing the long/short ratio should not cost us the
    open interest for that day.
    """
    open_interest = volume = ratio = close = funding = next_funding = None
    errors: list[str] = []

    try:
        rows = _get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume?ccy={currency}&period=1D"
        )["data"]
        if rows:
            open_interest, volume = float(rows[0][1]), float(rows[0][2])
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, IndexError) as exc:
        errors.append(f"oi:{type(exc).__name__}")

    try:
        rows = _get(
            f"https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio?ccy={currency}&period=1D"
        )["data"]
        if rows:
            ratio = float(rows[0][1])
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, IndexError) as exc:
        errors.append(f"lsr:{type(exc).__name__}")

    instrument = f"{currency}-USDT-SWAP"
    try:
        rows = _get(f"https://www.okx.com/api/v5/market/ticker?instId={instrument}")["data"]
        if rows:
            close = float(rows[0]["last"])
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, IndexError) as exc:
        errors.append(f"px:{type(exc).__name__}")

    # Funding is captured alongside because the eventual question is whether OI/LSR lead FUNDING, not
    # only price -- wave19's design needed lead time on funding spikes specifically.
    try:
        rows = _get(f"https://www.okx.com/api/v5/public/funding-rate?instId={instrument}")["data"]
        if rows:
            funding = float(rows[0]["fundingRate"])
            if rows[0].get("nextFundingRate"):
                next_funding = float(rows[0]["nextFundingRate"])
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, IndexError) as exc:
        errors.append(f"fr:{type(exc).__name__}")

    return Observation(
        currency=currency,
        open_interest_usd=open_interest,
        volume_usd=volume,
        long_short_ratio=ratio,
        close=close,
        funding_rate=funding,
        next_funding_rate=next_funding,
        error=",".join(errors) if errors else None,
    )


def already_collected_today(utc_date: str) -> bool:
    if not LEDGER_PATH.exists():
        return False
    # Scanning the whole file is fine at one record per day and avoids trusting a separate index that
    # could drift out of sync with the ledger it describes.
    with LEDGER_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("utc_date") == utc_date:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def ledger_summary() -> dict:
    if not LEDGER_PATH.exists():
        return {"records": 0, "first": None, "last": None}
    dates = []
    with LEDGER_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                dates.append(json.loads(line)["utc_date"])
            except (json.JSONDecodeError, KeyError):
                continue
    return {"records": len(dates), "first": min(dates) if dates else None, "last": max(dates) if dates else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append-only forward collector for OI / long-short / funding (read-only endpoints)"
    )
    parser.add_argument("--force", action="store_true", help="collect even if today is already recorded")
    parser.add_argument("--status", action="store_true", help="print ledger summary and exit")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.status:
        summary = ledger_summary()
        print(f"기록 {summary['records']}일 | {summary['first']} ~ {summary['last']}")
        if summary["records"]:
            needed = max(0, 180 - summary["records"])
            print(f"wave40 선별검정(178일)과 동등한 독립표본까지 남은 일수: 약 {needed}일")
        return 0

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    if already_collected_today(today) and not args.force:
        print(f"{today} 는 이미 기록됨 — 중복 기록은 later 검정의 통계력을 허위로 부풀린다. --force 로 덮어쓸 수 있다.")
        return 0

    print(f"=== wave41 전진 수집 {today} (읽기 전용) ===")
    observations = []
    for currency in CURRENCIES:
        observation = collect_currency(currency)
        observations.append(asdict(observation))
        mark = "!" if observation.error else " "
        print(
            f" {mark}{currency:6s} OI={observation.open_interest_usd or 0:>14,.0f} "
            f"LSR={observation.long_short_ratio if observation.long_short_ratio is not None else float('nan'):>5} "
            f"px={observation.close if observation.close is not None else float('nan'):>12} "
            f"fr={observation.funding_rate if observation.funding_rate is not None else float('nan')}"
            + (f"  [{observation.error}]" if observation.error else "")
        )
        time.sleep(REQUEST_PAUSE_SECONDS)

    spot_observations = []
    for instrument in SPOT_INSTRUMENTS:
        price = None
        error = None
        try:
            rows = _get(f"https://www.okx.com/api/v5/market/ticker?instId={instrument}")["data"]
            if rows:
                price = float(rows[0]["last"])
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, IndexError) as exc:
            error = f"px:{type(exc).__name__}"
        spot_observations.append({"instrument": instrument, "last": price, "error": error})
        print(f"  {'!' if error else ' '}{instrument:12s} last={price if price is not None else float('nan')}"
              + (f"  [{error}]" if error else ""))
        time.sleep(REQUEST_PAUSE_SECONDS)

    record = {
        "utc_date": today,
        "collected_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "venue": "OKX",
        "spot": spot_observations,
        "endpoints": [
            "rubik/stat/contracts/open-interest-volume",
            "rubik/stat/contracts/long-short-account-ratio",
            "market/ticker",
            "public/funding-rate",
        ],
        "observations": observations,
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    failed = sum(1 for o in observations if o["error"])
    summary = ledger_summary()
    print(f"\n기록 완료 · 실패 필드가 있는 종목 {failed}/{len(observations)}")
    print(f"원장 누적 {summary['records']}일 ({summary['first']} ~ {summary['last']})")
    print(f"→ {LEDGER_PATH}")
    print("\n이 수집기는 신호를 계산하지 않는다. 표본이 충분해지면 별도 wave에서 검정한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

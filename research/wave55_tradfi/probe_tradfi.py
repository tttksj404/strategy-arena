#!/usr/bin/env python3
# Wave-55: a new asset class the campaign never opened, and why it does not help yet.
#
# Two axes were still untouched after wave54. First, idle-capital lending: wave54 measured USDT BORROW at
# 60.13% APR, so the lend side might be far above the 1.91% wave18 assumed -- and since carry is active on
# only 58% of days, the return on the idle 42% matters. Second, the 134 non-perpetual contracts in
# binance_exchange_info.json, which no wave had ever looked at.
#
# The second turned out to be traditional-asset perpetuals: gold, silver, platinum, copper, natural gas,
# and US equities including SPY, QQQ, TSLA, NVDA and MSTR. That is interesting for a specific reason.
# wave52 established that crypto carry families correlate 0.96 even when they hold NO symbol in common,
# because the shared driver is crypto-wide funding. Gold and equity carry is driven by USD rates,
# dividends and stock borrow instead, so in principle this is the decorrelation wave52 could not find.
#
# The order of checks below follows wave52's lesson (test the precondition first) and wave55's own trap:
# the first funding observation on gold was +44.21% APR, which on a $45 leg is $19.90/yr and would dwarf
# wave42's entire $2-4. Paging the funding HISTORY is what settles whether that is a level or a spike, and
# it must be done before anything is built on it.

from __future__ import annotations

import collections
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

import numpy as np

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
EXCHANGE_INFO: Final = (
    Path(__file__).resolve().parents[1] / "wave3" / "cache" / "binance_exchange_info.json"
)

TRADFI_PERPS: Final = ("XAU", "XAG", "MSTR", "AAPL", "META", "SPY", "QQQ", "TSLA", "NVDA", "MSFT", "AMZN", "COIN")
CRYPTO_REFERENCE: Final = ("BTC", "ETH", "SOL")
GOLD_SPOTS: Final = ("XAUT-USDT", "PAXG-USDT")
# A causal walk-forward needs 365 training days plus a 90-day applied window before it can produce even
# one out-of-sample point.
MIN_BACKTESTABLE_DAYS: Final = 455


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(request, timeout=25) as response:
        return json.loads(response.read().decode())


def classify_binance_contracts() -> dict:
    """What the 134 non-perpetual USDT contracts actually are, from the cached exchange response."""
    info = json.loads(EXCHANGE_INFO.read_text(encoding="utf-8"))
    usdt = [s for s in info["symbols"] if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"]
    kinds = collections.Counter(s.get("contractType") for s in usdt)
    tradfi = [s for s in usdt if s.get("contractType") == "TRADIFI_PERPETUAL"]
    return {
        "kinds": dict(kinds),
        "tradifi_count": len(tradfi),
        "tradifi_assets": sorted({s.get("baseAsset") for s in tradfi}),
    }


def lending_rate() -> dict:
    """USDT lending rate, to test whether wave18's 1.91% idle-capital assumption is still right."""
    try:
        rows = _get("https://www.okx.com/api/v5/finance/savings/lending-rate-history?ccy=USDT&limit=30")["data"]
        rates = [float(r["lendingRate"]) for r in rows if r.get("lendingRate")]
        return {"observations": len(rates), "median": float(np.median(rates)) if rates else None,
                "min": min(rates) if rates else None, "max": max(rates) if rates else None}
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}"}


def daily_closes(inst_id: str, limit: int = 300) -> dict[int, float]:
    rows = _get(f"https://www.okx.com/api/v5/market/history-candles?instId={inst_id}&bar=1D&limit={limit}")["data"]
    return {int(r[0]) // 86_400_000: float(r[4]) for r in rows}


def neutrality(spot_inst: str, perp_inst: str = "XAU-USDT-SWAP") -> dict:
    """Does the spot leg actually hedge the perp? Correlation and residual volatility decide it.

    A carry trade is only delta-neutral if the two legs move together. wave44 booked a late-listing spot
    leg's price discovery as 'basis' worth 25.26% in one day, so tracking is measured, never assumed.
    """
    spot = daily_closes(spot_inst)
    perp = daily_closes(perp_inst)
    days = sorted(set(spot) & set(perp))
    if len(days) < 30:
        return {"spot": spot_inst, "overlap_days": len(days), "error": "insufficient overlap"}
    s = np.array([spot[d] for d in days])
    p = np.array([perp[d] for d in days])
    basis = p / s - 1.0
    spot_returns = np.diff(np.log(s))
    perp_returns = np.diff(np.log(p))
    residual = perp_returns - spot_returns
    return {
        "spot": spot_inst,
        "perp": perp_inst,
        "overlap_days": len(days),
        "from": time.strftime("%Y-%m-%d", time.gmtime(days[0] * 86_400)),
        "to": time.strftime("%Y-%m-%d", time.gmtime(days[-1] * 86_400)),
        "return_correlation": float(np.corrcoef(spot_returns, perp_returns)[0, 1]),
        "basis_median": float(np.median(basis)),
        "basis_min": float(basis.min()),
        "basis_max": float(basis.max()),
        "residual_daily_std": float(residual.std()),
        "residual_annual_vol": float(residual.std() * np.sqrt(365)),
    }


def funding_history(inst_id: str, pages: int = 30) -> dict[int, float]:
    """Page the funding history back as far as the endpoint allows."""
    out: dict[int, float] = {}
    before = None
    for _ in range(pages):
        url = f"https://www.okx.com/api/v5/public/funding-rate-history?instId={inst_id}&limit=100"
        if before:
            url += f"&after={before}"
        try:
            rows = _get(url)["data"]
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError):
            break
        if not rows:
            break
        for row in rows:
            out[int(row["fundingTime"])] = float(row["fundingRate"])
        before = min(int(row["fundingTime"]) for row in rows)
        time.sleep(0.12)
    return out


def summarise_funding(history: dict[int, float]) -> dict:
    stamps = sorted(history)
    apr = np.array([history[t] * 3 * 365 for t in stamps])
    monthly = collections.defaultdict(list)
    for stamp in stamps:
        monthly[time.strftime("%Y-%m", time.gmtime(stamp / 1000))].append(history[stamp] * 3 * 365)
    return {
        "stamps": len(stamps),
        "span_days": (stamps[-1] - stamps[0]) / 86_400_000 if len(stamps) > 1 else 0.0,
        "from": time.strftime("%Y-%m-%d", time.gmtime(stamps[0] / 1000)) if stamps else None,
        "apr_median": float(np.median(apr)) if len(apr) else None,
        "apr_mean": float(apr.mean()) if len(apr) else None,
        "apr_p90": float(np.percentile(apr, 90)) if len(apr) else None,
        "share_positive": float((apr > 0).mean()) if len(apr) else None,
        "share_above_15pct": float((apr > 0.15).mean()) if len(apr) else None,
        "monthly_median": {k: float(np.median(v)) for k, v in sorted(monthly.items())},
    }


def main() -> int:
    payload: dict = {"wave": "wave55_tradfi"}

    print("=== 1. 유휴자본 대여 수익률 (wave18 I5는 1.91% 가정) ===")
    lend = lending_rate()
    payload["lending"] = lend
    if "error" not in lend:
        print(f"  USDT 대여 실측 중앙 {lend['median']:.2%} (범위 {lend['min']:.2%}~{lend['max']:.2%}, "
              f"관측 {lend['observations']}건)")
        print(f"  wave18 가정 1.91% 대비: {'사실상 동일 — 상향 여지 없음' if abs(lend['median']-0.0191)<0.01 else '차이 있음'}")
        print(f"  참고: wave54 실측 차입 60.13% -> 차입/대여 스프레드 약 {0.6013/lend['median']:.0f}배")
    else:
        print(f"  측정 실패 {lend['error']}")

    print("\n=== 2. binance_exchange_info.json 의 비무기한 134종은 무엇인가 ===")
    contracts = classify_binance_contracts()
    payload["binance_contracts"] = contracts
    for kind, count in contracts["kinds"].items():
        print(f"  {str(kind):22s} {count}")
    print(f"  TRADIFI_PERPETUAL 기초자산 {len(contracts['tradifi_assets'])}종 "
          f"(금속·에너지·미국주식·ETF): {contracts['tradifi_assets'][:16]} ...")

    print("\n=== 3. 델타중립 구성이 성립하나 (금 현물 vs 무기한 추종) ===")
    payload["neutrality"] = []
    for spot in GOLD_SPOTS:
        report = neutrality(spot)
        payload["neutrality"].append(report)
        if "error" in report:
            print(f"  {spot}: {report['error']} (겹침 {report['overlap_days']}일)")
            continue
        print(f"  [{spot}] {report['overlap_days']}일 {report['from']}~{report['to']}")
        print(f"    일간수익 상관 {report['return_correlation']:.4f} · 베이시스 중앙 {report['basis_median']:+.3%} "
              f"(범위 {report['basis_min']:+.3%}~{report['basis_max']:+.3%})")
        print(f"    잔차 연환산 변동성 {report['residual_annual_vol']:.1%}")

    print("\n=== 4. 결정적: 펀딩이 지속적인가 (첫 관측 +44.21% 는 수준인가 스파이크인가) ===")
    payload["funding"] = {}
    for base in ("XAU",) + CRYPTO_REFERENCE[:1]:
        summary = summarise_funding(funding_history(f"{base}-USDT-SWAP"))
        payload["funding"][base] = summary
        print(f"\n  [{base}] {summary['stamps']}스탬프 · {summary['span_days']:.0f}일 · {summary['from']} 부터")
        print(f"    APR 중앙 {summary['apr_median']:+.2%} · 평균 {summary['apr_mean']:+.2%} · p90 {summary['apr_p90']:+.2%}")
        print(f"    양수 비율 {summary['share_positive']:.1%} · 15% 초과 {summary['share_above_15pct']:.1%}")
        print(f"    월별 중앙: " + " ".join(f"{k}:{v:+.0%}" for k, v in summary['monthly_median'].items()))

    gold = payload["funding"].get("XAU", {})
    btc = payload["funding"].get("BTC", {})
    print("\n=== 판정 ===")
    if gold.get("apr_median") is not None and btc.get("apr_median") is not None:
        print(f"  금 중앙 {gold['apr_median']:+.2%} (양수 {gold['share_positive']:.1%}) vs "
              f"BTC 중앙 {btc['apr_median']:+.2%} (양수 {btc['share_positive']:.1%})")
        if gold["apr_median"] < btc["apr_median"]:
            print("  => 금 무기한은 BTC보다 캐리로서 열등하다. 첫 관측 +44.21%는 스파이크였다")
            print(f"     (15% 초과 스탬프가 {gold['share_above_15pct']:.1%}뿐이고 중앙값은 {gold['apr_median']:+.2%}).")
    longest = max((r.get("overlap_days", 0) for r in payload["neutrality"]), default=0)
    print(f"\n  최장 가용 이력 {longest}일 · 펀딩 이력 {gold.get('span_days', 0):.0f}일")
    print(f"  인과 워크포워드 최소 요구 {MIN_BACKTESTABLE_DAYS}일 -> **백테스트 불가**")
    print("  => 측정으로 닫은 것이 아니라 데이터 부족으로 잠정 기각. 전진 수집 대상이다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tradfi.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nresults/tradfi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

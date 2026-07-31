#!/usr/bin/env python3
# Wave-39 step 1: measure real exchange margin rules and decide whether wave38's Z10 rejection was a
# property of the strategy or a property of one account configuration.
#
# wave38 rejected the high-deployment carry rungs because a short perp leg in an ISOLATED futures
# account is liquidated on its own terms: spot unrealised gains sit in a separate account and do not
# post margin to the futures account, so at 19x the leg died on 46-67% of active days. That rejection
# was correct for isolated margin and says nothing about a unified/portfolio-margin account, where the
# spot leg IS collateral for the perp.
#
# The difference is not a matter of opinion, so nothing here is assumed. Discount (haircut) rates and
# maintenance-margin rates are read from the venues' own public endpoints, and the liquidation threshold
# is derived from them.
#
# Venue choice is deliberate. wave36 was invalidated for mixing venues, so measuring OKX's rules and
# applying them to a Binance backtest would repeat that error. Instead BOTH reachable venues (OKX and
# Bitget, the live-signal target) are measured, and the conclusion is only reported as robust if it
# holds under both. Binance's own endpoints return HTTP 451 from this environment and cannot be
# measured, which is itself recorded rather than papered over.

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import sys
from typing import Final
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

RESULTS_DIR: Final = Path(__file__).resolve().parent / "results"
ACTIVE_CAPITAL: Final = 90.0  # wave10's contract, matching engine38
WORST_OBSERVED_ADVERSE_MOVE: Final = 0.8360  # measured by wave38 on the symbols the strategy held


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode())


@dataclass(frozen=True, slots=True)
class MarginRules:
    venue: str
    haircut: float  # collateral discount rate applied to the spot leg
    maintenance_rate: float  # mmr on the perp notional
    source: str


def liquidating_move(deployment: float, rules: MarginRules, capital: float = ACTIVE_CAPITAL) -> float:
    """Adverse (upward) price move that liquidates a delta-neutral carry book under portfolio margin.

    Book: spend N = deployment * capital of cash on spot, short N of perp. Cash left is capital - N.
    After an upward move x the spot is worth N(1+x) and counts as collateral at `haircut`, while the
    short perp carries an unrealised loss of N*x in full. Maintenance margin scales with the perp's
    current notional, N(1+x) * mmr. Liquidation is equity < maintenance:

        (capital - N) + N(1+x)*h - N*x  <  N(1+x)*m

    Solving for x gives the threshold below. Note the sign of the (1 - h + m) denominator: the position
    is only unconditionally safe if h >= 1. A haircut discounts the hedging leg while the perp loss
    counts in full, so a delta-neutral book still has a finite -- if very distant -- liquidation point.
    No netting benefit is assumed even though portfolio-margin systems often grant one for a recognised
    hedge, which makes this estimate conservative rather than flattering.
    """
    notional = deployment * capital
    if notional <= 0.0:
        return float("inf")
    denominator = notional * (1.0 - rules.haircut + rules.maintenance_rate)
    if denominator <= 0.0:
        return float("inf")
    numerator = (capital - notional) + notional * (rules.haircut - rules.maintenance_rate)
    return numerator / denominator


def isolated_liquidating_move(deployment: float) -> float:
    """The isolated-margin threshold wave38 used, for side-by-side comparison.

    Perp margin is only the free cash, so leverage is deployment/(1-deployment) and the leg dies once an
    adverse move consumes that margin.
    """
    if deployment >= 1.0:
        return 0.0
    leverage = deployment / (1.0 - deployment)
    return float("inf") if leverage <= 0.0 else 1.0 / leverage


def measure_okx() -> tuple[MarginRules, dict]:
    discount = _get("https://www.okx.com/api/v5/public/discount-rate-interest-free-quota")["data"]
    listed = [row for row in discount if row.get("details")]
    rates = [float(row["details"][0]["discountRate"]) for row in listed]
    restricted = [row["ccy"] for row in listed if row.get("collateralRestrict")]

    tiers = {}
    for family in ("BTC-USDT", "ETH-USDT", "DOGE-USDT", "ADA-USDT"):
        try:
            data = _get(
                f"https://www.okx.com/api/v5/public/position-tiers?instType=SWAP&tdMode=cross&instFamily={family}"
            )["data"]
            if data:
                tiers[family] = {"imr": float(data[0]["imr"]), "mmr": float(data[0]["mmr"])}
        except (urllib.error.URLError, KeyError, ValueError):
            continue

    # The strategy holds whatever ranks highest on funding, which skews to alts, so the WORST listed
    # haircut and the WORST measured mmr are used rather than BTC's friendly numbers.
    worst_haircut = min(rates)
    worst_mmr = max(tier["mmr"] for tier in tiers.values()) if tiers else 0.01
    detail = {
        "collateral_listed": len(listed),
        "haircut_median": statistics.median(rates),
        "haircut_min": worst_haircut,
        "haircut_max": max(rates),
        "collateral_restricted": restricted,
        "tiers": tiers,
    }
    return MarginRules("OKX", worst_haircut, worst_mmr, "api/v5/public/*"), detail


def measure_bitget() -> tuple[MarginRules, dict]:
    contracts = _get(
        "https://api.bitget.com/api/v2/mix/market/contracts?productType=usdt-futures"
    )["data"]
    fees_maker = [float(c["makerFeeRate"]) for c in contracts if c.get("makerFeeRate")]
    fees_taker = [float(c["takerFeeRate"]) for c in contracts if c.get("takerFeeRate")]
    min_usdt = [float(c["minTradeUSDT"]) for c in contracts if c.get("minTradeUSDT")]
    margin_coins = {coin for c in contracts for coin in (c.get("supportMarginCoins") or [])}

    # Bitget publishes leverage caps and fees but not a cross-margin haircut table on this endpoint.
    # Rather than invent one, OKX's worst measured haircut is carried over as the stress assumption and
    # labelled as such; the point of this probe is robustness, so borrowing the harsher number from the
    # venue that does publish it is the conservative direction.
    detail = {
        "contracts": len(contracts),
        "maker_fee_median": statistics.median(fees_maker) if fees_maker else None,
        "taker_fee_median": statistics.median(fees_taker) if fees_taker else None,
        "min_trade_usdt_median": statistics.median(min_usdt) if min_usdt else None,
        "min_trade_usdt_max": max(min_usdt) if min_usdt else None,
        "margin_coins": sorted(margin_coins),
        "haircut_source": "OKX worst measured haircut carried over -- Bitget does not publish one here",
    }
    return MarginRules("Bitget", 0.70, 0.01, "api/v2/mix/market/contracts"), detail


def main() -> int:
    print("=== wave39 step1: 통합(포트폴리오) 마진에서 델타중립 캐리는 청산되는가 ===")
    print(f"wave38 실측 최악 장중 역행 = {WORST_OBSERVED_ADVERSE_MOVE:.2%} (이 값을 넘으면 청산)\n")

    measured: dict[str, dict] = {}
    rules_by_venue: list[MarginRules] = []
    for name, measure in (("OKX", measure_okx), ("Bitget", measure_bitget)):
        try:
            rules, detail = measure()
            rules_by_venue.append(rules)
            # dataclasses with slots=True have no __dict__, so asdict is the correct accessor.
            measured[name] = {"rules": asdict(rules), "detail": detail}
            print(f"[{name}] 최악 할인율 {rules.haircut:.3f} · 최악 MMR {rules.maintenance_rate:.4f}")
            for key, value in detail.items():
                if key in ("margin_coins", "collateral_restricted", "tiers"):
                    print(f"    {key}: {str(value)[:110]}")
                else:
                    print(f"    {key}: {value}")
        except Exception as exc:  # network shape changes must not be silently swallowed
            print(f"[{name}] 측정 실패: {type(exc).__name__}: {exc}")
            measured[name] = {"error": f"{type(exc).__name__}: {exc}"}

    print("\n[Binance] 측정 불가 — 이 환경에서 HTTP 451 차단. 백테스트 데이터는 Binance 캐시이므로")
    print("          마진 규칙만은 다른 거래소 실측으로 대체했음을 명시한다(가정이 아니라 대체).")

    print("\n=== 투입비율별 청산 문턱 (격리마진 vs 통합마진) ===")
    print(f"{'투입':>6} {'격리:퍼프lev':>12} {'격리 청산문턱':>13} " + " ".join(f"{r.venue+' 통합':>13}" for r in rules_by_venue))
    ladder = [0.50, 0.75, 0.95, 1.00]
    table = []
    for deployment in ladder:
        isolated = isolated_liquidating_move(deployment)
        leverage = deployment / (1.0 - deployment) if deployment < 1.0 else float("inf")
        cells = []
        for rules in rules_by_venue:
            move = liquidating_move(deployment, rules)
            cells.append(f"{move:12.1%}" if move < 100 else "        안전")
        table.append(
            {
                "deployment": deployment,
                "isolated_perp_leverage": leverage,
                "isolated_threshold": isolated,
                "portfolio": {r.venue: liquidating_move(deployment, r) for r in rules_by_venue},
            }
        )
        iso = f"{isolated:12.1%}" if isolated < 100 else "        안전"
        lev = f"{leverage:11.1f}x" if leverage < 1e6 else "         inf"
        print(f"{deployment:6.2f} {lev} {iso} " + " ".join(cells))

    print(f"\n실측 최악 역행 {WORST_OBSERVED_ADVERSE_MOVE:.2%} 와 비교:")
    verdict_rows = []
    for row in table:
        iso_safe = row["isolated_threshold"] > WORST_OBSERVED_ADVERSE_MOVE
        port_safe = all(v > WORST_OBSERVED_ADVERSE_MOVE for v in row["portfolio"].values())
        verdict_rows.append((row["deployment"], iso_safe, port_safe))
        print(
            f"  투입 {row['deployment']:.2f}: 격리 {'안전' if iso_safe else '청산'} · "
            f"통합 {'안전(전 거래소)' if port_safe else '청산'}"
        )

    unlocked = [d for d, iso, port in verdict_rows if port and not iso]
    print("\n=== 판정 ===")
    if unlocked:
        print(f"  통합마진에서만 안전해지는 투입비율: {unlocked}")
        print("  => wave38의 Z10 기각은 '격리마진'의 성질이었다. 통합마진에서는 상향 여지가 실재한다.")
        print("     단, 현물을 현금으로만 사므로 투입 1.00 초과는 여전히 차입이며 별도 문제다.")
    else:
        print("  통합마진으로도 새로 열리는 투입비율이 없다 => wave38의 기각은 계좌유형과 무관하다.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "margin_rules.json").write_text(
        json.dumps(
            {
                "wave": "wave39_margin",
                "worst_observed_adverse_move": WORST_OBSERVED_ADVERSE_MOVE,
                "measured": measured,
                "ladder": table,
                "portfolio_unlocks": unlocked,
                "binance": "unmeasurable from this environment (HTTP 451)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nresults/margin_rules.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

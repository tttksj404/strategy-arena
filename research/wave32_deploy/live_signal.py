#!/usr/bin/env python3
# Read-only "what would the VALIDATED strategy do right now" tool, plus a forward ledger.
#
# ---------------------------------------------------------------------------------------
# Why this exists and why it tracks the CARRY family, not wave30/31/32's leverage candidates
# ---------------------------------------------------------------------------------------
# wave32 unsealed OOS for L1 and it FAILED (D1/D2/D8). wave32/SPEC.md pre-registered that a
# failed candidate must not be put on the forward ledger, because a forward record of an
# unvalidated strategy is indistinguishable from a validated one six months later. So the only
# thing this tool tracks is the family the strategy card actually recommends: the L4/I5 funding
# carry (delta-neutral, 1x gross), which is the single family that has cleared every gate.
#
# ---------------------------------------------------------------------------------------
# Why Bitget and not Binance
# ---------------------------------------------------------------------------------------
# research/paper/track.py is currently DEAD in this environment: Binance returns HTTP 451
# ("Service unavailable from a restricted location") for fapi.binance.com, so
# collect_live_snapshot() raises and the tracker writes a failure status instead of a record.
# Bitget's public endpoints answer normally, and Bitget is the venue the strategy card names as
# the execution venue anyway. DISCLOSED CONSEQUENCE: the backtest measured funding on Binance
# and this tool reads funding on Bitget. wave14's cross-venue study measured 98.9% sign
# agreement on Binance<->Bybit and 91.7% on OKX during high-funding stretches; Bitget was never
# measured, so this is an unquantified basis difference, not a verified equivalence.
#
# ---------------------------------------------------------------------------------------
# Hard boundary (docs/RESEARCH_GUARDRAILS.md operational contract)
# ---------------------------------------------------------------------------------------
# Public GET endpoints only. No API key, no signing, no order/account/withdraw endpoint is
# referenced anywhere in this file. It cannot place a trade even if credentials existed. What it
# produces is a plan a human executes manually, and a ledger line for forward validation.

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import requests

BITGET: Final = "https://api.bitget.com"
PRODUCT_TYPE: Final = "usdt-futures"

# ---- Strategy constants, all inherited from the validated L4/I5 configuration ----
# research/wave18_idle/results/I5.json config + capital_contract, quoted verbatim.
WINDOW_DAYS: Final = 7
ENTRY_THRESHOLD_APR: Final = 0.15  # L4/I5 entry
EXIT_THRESHOLD_APR: Final = 0.075  # L4/I5 exit (half of entry)
TOP_K_PAIRS: Final = 1
TOTAL_CAPITAL: Final = 100.0
RESERVE_FRACTION: Final = 0.10
ACTIVE_CAPITAL: Final = TOTAL_CAPITAL * (1.0 - RESERVE_FRACTION)  # $90
LEG_USDT: Final = ACTIVE_CAPITAL / 2.0  # $45 spot + $45 perp = gross $90 = 1.0x
MIN_ORDER_USDT: Final = 5.0
UNIVERSE_BREADTH: Final = 200  # L4 top200 by volume
FUNDING_INTERVALS_PER_DAY: Final = 3  # 8h
MAKER_FEE_RATE: Final = 0.0002

BASE_DIR: Final = Path(__file__).resolve().parent
LEDGER_PATH: Final = BASE_DIR / "ledger" / "carry_live_ledger.jsonl"
STATUS_PATH: Final = BASE_DIR / "LIVE_SIGNAL.md"
SCAN_LIMIT: Final = 80  # how many top-volume symbols to pull funding history for


class LiveDataError(RuntimeError):
    pass


def _get(path: str, params: dict[str, Any]) -> Any:
    response = requests.get(f"{BITGET}{path}", params=params, timeout=25)
    if response.status_code != 200:
        raise LiveDataError(f"{path} -> HTTP {response.status_code}: {response.text[:200]}")
    payload = response.json()
    if payload.get("code") != "00000":
        raise LiveDataError(f"{path} -> {payload.get('code')}: {payload.get('msg')}")
    return payload["data"]


@dataclass(frozen=True)
class Candidate:
    symbol: str
    funding_apr: float
    intervals_used: int
    perp_price: float
    spot_price: float
    perp_volume_usdt_24h: float


def perp_tickers() -> dict[str, dict[str, Any]]:
    rows = _get("/api/v2/mix/market/tickers", {"productType": PRODUCT_TYPE})
    return {row["symbol"]: row for row in rows}


def spot_tickers() -> dict[str, dict[str, Any]]:
    rows = _get("/api/v2/spot/market/tickers", {})
    return {row["symbol"]: row for row in rows}


def funding_apr(symbol: str) -> tuple[float, int]:
    """Annualised mean funding over the trailing WINDOW_DAYS, from Bitget's own history.

    Uses only SETTLED stamps (history-fund-rate), never the pending current-fund-rate, so the
    signal is computed from money that has actually changed hands -- the same discipline the
    backtest's shift(1) enforces.
    """
    rows = _get(
        "/api/v2/mix/market/history-fund-rate",
        {"symbol": symbol, "productType": PRODUCT_TYPE, "pageSize": 100},
    )
    needed = WINDOW_DAYS * FUNDING_INTERVALS_PER_DAY
    rates = [float(row["fundingRate"]) for row in rows[:needed]]
    if len(rates) < needed:
        return float("nan"), len(rates)
    mean_rate = sum(rates) / len(rates)
    return mean_rate * FUNDING_INTERVALS_PER_DAY * 365.0, len(rates)


def build_universe() -> tuple[list[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    perps = perp_tickers()
    spots = spot_tickers()
    ranked = sorted(
        perps.items(),
        key=lambda item: -float(item[1].get("usdtVolume") or item[1].get("quoteVolume") or 0.0),
    )
    # L4 universe = top200 by volume, AND spot must exist (the carry needs a spot long leg).
    universe = [symbol for symbol, _row in ranked[:UNIVERSE_BREADTH] if symbol in spots]
    return universe, perps, spots


def scan(verbose: bool = True) -> tuple[list[Candidate], dict[str, Any]]:
    universe, perps, spots = build_universe()
    scanned = universe[:SCAN_LIMIT]
    candidates: list[Candidate] = []
    incomplete = 0
    for index, symbol in enumerate(scanned, 1):
        try:
            apr, intervals = funding_apr(symbol)
        except LiveDataError:
            incomplete += 1
            continue
        if apr != apr:  # NaN -> not enough settled history
            incomplete += 1
            continue
        perp = perps[symbol]
        spot = spots[symbol]
        candidates.append(
            Candidate(
                symbol=symbol,
                funding_apr=apr,
                intervals_used=intervals,
                perp_price=float(perp["lastPr"]),
                spot_price=float(spot["lastPr"]),
                perp_volume_usdt_24h=float(perp.get("usdtVolume") or 0.0),
            )
        )
        if verbose and index % 20 == 0:
            print(f"  scanned {index}/{len(scanned)}", flush=True)
        time.sleep(0.06)  # stay well inside Bitget's public rate limit
    candidates.sort(key=lambda item: -item.funding_apr)
    coverage = {
        "perp_symbols_seen": len(perps),
        "spot_symbols_seen": len(spots),
        "universe_after_spot_filter": len(universe),
        "scanned": len(scanned),
        "scan_limit": SCAN_LIMIT,
        "with_full_funding_window": len(candidates),
        "incomplete_funding_history": incomplete,
    }
    return candidates, coverage


def fidelity_ok(coverage: dict[str, Any]) -> tuple[bool, str]:
    """The universe-fidelity discipline research/paper/fidelity.py established: never act on a
    ranking computed from a universe we know was not covered. L4 declares top200; if the live
    scan could not produce a full funding window for a reasonable share of it, the ranking is
    not the strategy's ranking and the run must record cash instead of a guess."""
    if coverage["universe_after_spot_filter"] < 50:
        return False, f"only {coverage['universe_after_spot_filter']} perp+spot pairs found (need >=50)"
    if coverage["with_full_funding_window"] < 30:
        return False, (
            f"only {coverage['with_full_funding_window']} symbols had a complete "
            f"{WINDOW_DAYS}d settled funding window (need >=30)"
        )
    return True, "ok"


def execution_plan(selected: list[Candidate]) -> dict[str, Any]:
    if not selected:
        return {"action": "HOLD_CASH", "legs": [], "gross_usdt": 0.0, "notes": []}
    legs: list[dict[str, Any]] = []
    for candidate in selected:
        notional = LEG_USDT / len(selected)
        legs.append(
            {
                "symbol": candidate.symbol,
                "spot_leg": {
                    "side": "BUY",
                    "instrument": "spot",
                    "notional_usdt": round(notional, 2),
                    "reference_price": candidate.spot_price,
                    "approx_quantity": round(notional / candidate.spot_price, 8),
                    "order_type": "limit (maker) at or inside best bid",
                },
                "perp_leg": {
                    "side": "SELL/SHORT",
                    "instrument": "usdt-futures perp",
                    "leverage": "1x (isolated)",
                    "notional_usdt": round(notional, 2),
                    "reference_price": candidate.perp_price,
                    "approx_quantity": round(notional / candidate.perp_price, 8),
                    "order_type": "limit (maker) at or inside best ask",
                },
                "funding_apr_now": candidate.funding_apr,
                "expected_annual_usdt_on_this_pair": round(notional * candidate.funding_apr, 2),
            }
        )
    gross = sum(leg["spot_leg"]["notional_usdt"] + leg["perp_leg"]["notional_usdt"] for leg in legs)
    notes = [
        f"gross exposure ${gross:.2f} on ${ACTIVE_CAPITAL:.2f} active capital = "
        f"{gross / ACTIVE_CAPITAL:.2f}x (delta-neutral, must stay 1.0x)",
        f"${TOTAL_CAPITAL - ACTIVE_CAPITAL:.2f} stays in reserve, untouched",
        "both legs must be filled before the position is considered open; a one-legged "
        "position is a naked directional bet and violates the strategy's core assumption",
    ]
    return {"action": "OPEN_CARRY", "legs": legs, "gross_usdt": round(gross, 2), "notes": notes}


def read_ledger() -> list[dict[str, Any]]:
    if not LEDGER_PATH.exists():
        return []
    return [json.loads(line) for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_ledger(record: dict[str, Any]) -> bool:
    """Idempotent per (strategy_id, run_date): re-running on the same UTC day is a no-op, the
    same one-record-per-day rule research/paper/ledger.py uses."""
    existing = read_ledger()
    key = (record["strategy_id"], record["run_date"])
    if any((row["strategy_id"], row["run_date"]) == key for row in existing):
        return False
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        stream.write("\n")
    return True


def render_status(record: dict[str, Any], candidates: list[Candidate], appended: bool) -> str:
    plan = record["execution_plan"]
    lines = [
        "# 오늘의 신호 — 검증된 캐리 전략 (L4/I5 계열)",
        "",
        f"관측 시각: `{record['observed_at']}` · 데이터원: **Bitget 공개 REST (읽기 전용)**",
        "",
        "> **실주문 금지.** 이 파일은 사람이 손으로 집행하기 위한 계획서이고, 이 도구는 주문·계정·"
        "출금 엔드포인트를 참조하지 않는다(`docs/RESEARCH_GUARDRAILS.md` 운영계약).",
        "",
        "## 판정",
        "",
        f"- **행동: `{plan['action']}`**",
        f"- 충실도 게이트: **{'PASS' if record['fidelity_ok'] else 'FAIL — ' + record['fidelity_reason']}**",
        f"- 진입 임계 {ENTRY_THRESHOLD_APR:.1%} APR 초과 종목 수: **{record['eligible_count']}**",
        f"- 원장 기록: {'추가됨' if appended else '오늘 자 기록이 이미 있어 건너뜀(멱등)'}",
        "",
    ]
    if plan["action"] == "OPEN_CARRY":
        lines += ["## 집행 계획 ($100 기준)", ""]
        for leg in plan["legs"]:
            lines += [
                f"### {leg['symbol']} — 현재 펀딩 **{leg['funding_apr_now']:.2%} APR**",
                "",
                "| 레그 | 방향 | 상품 | 노셔널 | 참조가 | 수량(약) | 주문 |",
                "|---|---|---|---|---|---|---|",
                f"| 현물 | {leg['spot_leg']['side']} | spot | ${leg['spot_leg']['notional_usdt']:.2f} | "
                f"{leg['spot_leg']['reference_price']} | {leg['spot_leg']['approx_quantity']} | "
                f"{leg['spot_leg']['order_type']} |",
                f"| 퍼프 | {leg['perp_leg']['side']} | perp {leg['perp_leg']['leverage']} | "
                f"${leg['perp_leg']['notional_usdt']:.2f} | {leg['perp_leg']['reference_price']} | "
                f"{leg['perp_leg']['approx_quantity']} | {leg['perp_leg']['order_type']} |",
                "",
                f"이 페어의 연 환산 기대 수취: 약 **${leg['expected_annual_usdt_on_this_pair']:.2f}** "
                f"(펀딩이 현 수준을 유지할 경우에 한함)",
                "",
            ]
        lines += ["**주의사항**", ""] + [f"- {note}" for note in plan["notes"]] + [""]
    else:
        lines += [
            "## 집행 계획: 없음 — 현금 대기",
            "",
            f"진입 임계({ENTRY_THRESHOLD_APR:.1%} APR)를 넘는 종목이 없다. "
            "**대기는 이 전략의 정상 동작이며 손실이 아니다** — L4/I5는 전기간 2,509일 중 "
            "1,029일(41%)을 현금으로 보냈고, 수익은 고펀딩 레짐에 집중 발생한다.",
            "",
        ]
    top = candidates[:10]
    if top:
        lines += [
            "## 현재 펀딩 상위 10종목 (참고)",
            "",
            "| 순위 | 심볼 | 7일 평균 펀딩 APR | 임계 초과 | 24h 거래대금 |",
            "|---:|---|---:|:---:|---:|",
        ]
        for rank, item in enumerate(top, 1):
            lines.append(
                f"| {rank} | {item.symbol} | {item.funding_apr:.2%} | "
                f"{'✅' if item.funding_apr > ENTRY_THRESHOLD_APR else '—'} | "
                f"${item.perp_volume_usdt_24h / 1e6:.0f}M |"
            )
        lines.append("")
    coverage = record["coverage"]
    lines += [
        "## 데이터 커버리지",
        "",
        f"- 퍼프 심볼 {coverage['perp_symbols_seen']} · 현물 심볼 {coverage['spot_symbols_seen']}",
        f"- 퍼프∧현물 동시 존재(상위 {UNIVERSE_BREADTH}): {coverage['universe_after_spot_filter']}",
        f"- 펀딩 이력 스캔: {coverage['scanned']}종목 → {WINDOW_DAYS}일 완전창 확보 "
        f"{coverage['with_full_funding_window']}종목 (이력부족 {coverage['incomplete_funding_history']})",
        "",
        "## 회로차단기 (위반 시 전량 청산 후 재검증)",
        "",
        "1. 단일 페어 손실 > 노셔널의 **2%** (베이시스 폭주 신호)",
        "2. 30일 실현 PnL < **−3%** (모델–현실 괴리)",
        "3. 실측 펀딩 수취액이 신호 기대치의 **70% 미달 2주 연속** (거래소 재현성 붕괴)",
        "4. 한쪽 레그만 체결된 상태가 유지됨 (델타중립 붕괴 — 즉시 반대 레그 체결 또는 전량 청산)",
        "5. 거래소 출금 지연·공지 이상 (운영 리스크 — 수동 판단)",
        "",
        "## 알려진 한계",
        "",
        "- **데이터원 불일치**: 백테스트는 Binance 펀딩으로 검증됐고 이 신호는 Bitget 펀딩으로 계산된다. "
        "wave14는 Binance↔Bybit 부호일치 98.9%·OKX 91.7%를 실측했으나 **Bitget은 측정되지 않았다.** "
        "정량화되지 않은 베이시스 차이가 존재한다.",
        f"- **스캔 범위 {SCAN_LIMIT}종목** (공개 API 요청량 제한). L4의 top200 전체가 아니므로 "
        "상위권 밖의 기회를 놓칠 수 있다 — 다만 캐리 기회는 소수 극단 종목에 집중되므로(wave14) "
        "상위 거래대금 구간에 대부분 포함된다.",
        "- 이 도구는 **연구 증거의 운영 보조**이지 투자 권유가 아니다. 실자금 투입은 사용자 본인의 결정이다.",
        "",
    ]
    return "\n".join(lines)


def run_once(verbose: bool = True) -> int:
    import pandas as pd  # noqa: PANDAS_OK

    observed_at = pd.Timestamp.now(tz="UTC")
    if verbose:
        print("scanning Bitget public endpoints (read-only)...", flush=True)
    candidates, coverage = scan(verbose=verbose)
    ok, reason = fidelity_ok(coverage)
    eligible = [item for item in candidates if item.funding_apr > ENTRY_THRESHOLD_APR]
    selected = eligible[:TOP_K_PAIRS] if ok else []
    plan = execution_plan(selected)

    record = {
        "strategy_id": "L4I5_CARRY",
        "run_date": observed_at.date().isoformat(),
        "observed_at": observed_at.isoformat(),
        "data_source": "Bitget public REST (mix tickers, spot tickers, history-fund-rate)",
        "fidelity_ok": ok,
        "fidelity_reason": reason,
        "entry_threshold_apr": ENTRY_THRESHOLD_APR,
        "exit_threshold_apr": EXIT_THRESHOLD_APR,
        "window_days": WINDOW_DAYS,
        "eligible_count": len(eligible),
        "eligible_symbols": [item.symbol for item in eligible[:10]],
        "top10": [
            {"symbol": item.symbol, "funding_apr": item.funding_apr, "volume_usdt_24h": item.perp_volume_usdt_24h}
            for item in candidates[:10]
        ],
        "selected": [item.symbol for item in selected],
        "execution_plan": plan,
        "coverage": coverage,
        "capital_contract": {
            "total_usdt": TOTAL_CAPITAL,
            "reserve_usdt": TOTAL_CAPITAL - ACTIVE_CAPITAL,
            "active_usdt": ACTIVE_CAPITAL,
            "leg_usdt": LEG_USDT,
            "gross_target_x": 1.0,
            "min_order_usdt": MIN_ORDER_USDT,
        },
    }
    appended = append_ledger(record)
    STATUS_PATH.write_text(render_status(record, candidates, appended), encoding="utf-8")
    print()
    print(f"action={plan['action']} | fidelity={'PASS' if ok else 'FAIL: ' + reason} | "
          f"eligible={len(eligible)} | ledger_appended={appended}")
    if candidates:
        print(f"top funding now: " + ", ".join(f"{c.symbol} {c.funding_apr:.2%}" for c in candidates[:5]))
    print(f"status written to {STATUS_PATH}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only live signal for the validated L4/I5 carry strategy (Bitget public data)"
    )
    parser.add_argument("--run-once", action="store_true", help="scan, write LIVE_SIGNAL.md, append one daily ledger record")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_once:
        parser.print_help()
        return 2
    try:
        return run_once(verbose=not args.quiet)
    except (LiveDataError, requests.RequestException) as error:
        print(f"live signal unavailable: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

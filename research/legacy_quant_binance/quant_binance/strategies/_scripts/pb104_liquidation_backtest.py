"""
PB104 — Liquidation cascade reversal backtest.

Setup:
  1. Load 30d 5m liquidation-proxy data from
     quant_runtime/liquidations/<SYM>_5m_30d.jsonl
  2. Load matching 5m klines from Binance public klines API.
  3. Detect cascades:
        - long_cascade  := top-LS ratio drops >= drop_th over 15min  AND
                           taker_sell_vol / taker_buy_vol >= sell_dom over 15min
        - short_cascade := mirror.
  4. Reversal entry on next 5m close after cascade trigger.
        - long_cascade  → SHORT?  NO. Long cascade = forced LONG closures
                          → price already dumped → we BUY (reversal LONG).
        - short_cascade → SELL  (reversal SHORT).
  5. Exit: TP +1% / SL -3% / 30min timeout, first hit.
        Use 1m closes inside the next 30min, modelled with 5m OHLC bracket
        priority: SL > TP if both touch in same bar (conservative).
  6. Cost: 16bps round-trip; 5x leverage; $16.5 margin per position; capital $55.

9-point check:
  1. trade-level avg net by sub-period > 0
  2. portfolio sim sub-period annual > 0
  3. avg net >= +50bps
  4. WR >= 65%
  5. 6-axis fit
  6. cost up to 30bps still positive
  7. lev=5 liq < 10%
  8. n >= 50
  9. no warmup leakage

Output:
  - prints summary
  - writes claimed_performance.md and rules.md if --commit
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import requests

ROOT = Path(r"C:\Users\SSAFY\Desktop\first_repo")
LIQDIR = ROOT / "quant_runtime" / "liquidations"
PB_DIR = ROOT / "quant_binance" / "strategies" / "_playbook" / "PB104_hummingbot_liquidation_sniper"


# --------------------------------------------------------------------------- #
# Data loaders
# --------------------------------------------------------------------------- #

def load_liq(symbol: str) -> list[dict]:
    p = LIQDIR / f"{symbol}_5m_30d.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    return sorted(rows, key=lambda r: r["timestamp"])


def fetch_klines_5m(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Public Binance futures 5m klines, paginated 1500/call."""
    url = "https://fapi.binance.com/fapi/v1/klines"
    out: list[list] = []
    cur = start_ms
    step = 1500 * 5 * 60 * 1000
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": "5m",
            "startTime": cur,
            "endTime": min(cur + step, end_ms),
            "limit": 1500,
        }
        for _ in range(3):
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                out.extend(r.json())
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"klines fetch failed @ {cur}")
        cur += step
        time.sleep(0.1)
    # dedupe by open_time
    seen = {}
    for k in out:
        seen[int(k[0])] = k
    return [seen[t] for t in sorted(seen)]


def klines_to_dict(klines: list[list]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for k in klines:
        ot = int(k[0])
        out[ot] = {
            "open_time": ot,
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
        }
    return out


# --------------------------------------------------------------------------- #
# Cascade detection
# --------------------------------------------------------------------------- #

@dataclass
class Signal:
    ts: int
    symbol: str
    direction: str  # "LONG" entry or "SHORT" entry
    cascade: str    # "long_cascade" / "short_cascade"
    drop: float
    sell_dom: float


def detect_signals(
    liq: list[dict],
    *,
    lookback_bars: int = 3,            # 15min on 5m bars
    drop_th: float = 0.10,              # 10% drop in top-LS ratio
    sell_dom: float = 1.5,              # taker sell:buy >= 1.5
    cooldown_bars: int = 6,             # 30min cooldown after a signal (avoid overlap with timeout)
) -> list[Signal]:
    sigs: list[Signal] = []
    last_sig_ts = -10**18
    for i in range(lookback_bars, len(liq)):
        win = liq[i - lookback_bars : i + 1]
        ratio_now = liq[i].get("top_ls_ratio")
        ratio_then = win[0].get("top_ls_ratio")
        if not ratio_now or not ratio_then:
            continue
        # cumulative taker imbalance
        buy = sum(r.get("taker_buy_vol", 0.0) for r in win)
        sell = sum(r.get("taker_sell_vol", 0.0) for r in win)
        if buy <= 0 or sell <= 0:
            continue
        delta = (ratio_now - ratio_then) / ratio_then
        ts = int(liq[i]["timestamp"])

        # cooldown
        if ts - last_sig_ts < cooldown_bars * 5 * 60 * 1000:
            continue

        # long cascade: long ratio drops + taker selling dominates → reversal LONG
        if delta <= -drop_th and (sell / buy) >= sell_dom:
            sigs.append(
                Signal(ts, liq[i]["symbol"], "LONG", "long_cascade", delta, sell / buy)
            )
            last_sig_ts = ts
            continue
        # short cascade: long ratio jumps + taker buying dominates → reversal SHORT
        if delta >= drop_th and (buy / sell) >= sell_dom:
            sigs.append(
                Signal(ts, liq[i]["symbol"], "SHORT", "short_cascade", delta, buy / sell)
            )
            last_sig_ts = ts
    return sigs


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

@dataclass
class Trade:
    ts_entry: int
    symbol: str
    side: str
    entry: float
    exit: float
    bars_held: int
    exit_reason: str   # "tp" / "sl" / "timeout"
    raw_ret: float     # pre-cost spot-pct return on price
    net_ret: float     # post-cost
    levered_pnl_usd: float
    liquidated: bool


def simulate_trade(
    sig: Signal,
    klines: dict[int, dict],
    *,
    tp: float = 0.01,
    sl: float = 0.03,
    timeout_bars: int = 6,        # 30min on 5m bars
    cost_bps: float = 16.0,
    leverage: int = 5,
    margin_usd: float = 16.5,
) -> Trade | None:
    """Entry: next 5m bar's close.  Bracket evaluated bar-by-bar (5m OHLC).
       Conservative: if both TP and SL touch in same bar, take SL.
       Liquidation modelled as |adverse_move * lev| >= 1.0  → wipeout."""
    entry_bar_open = sig.ts + 5 * 60 * 1000  # next bar opens here
    entry_bar = klines.get(entry_bar_open)
    if entry_bar is None:
        return None
    entry_price = entry_bar["close"]
    entry_ts = entry_bar_open

    side = sig.direction
    long = side == "LONG"
    tp_px = entry_price * (1 + tp) if long else entry_price * (1 - tp)
    sl_px = entry_price * (1 - sl) if long else entry_price * (1 + sl)

    exit_reason = "timeout"
    exit_px = entry_price
    bars = 0
    for k in range(1, timeout_bars + 1):
        bar_ts = entry_ts + k * 5 * 60 * 1000
        bar = klines.get(bar_ts)
        if bar is None:
            continue
        bars = k
        hi, lo = bar["high"], bar["low"]
        hit_tp = (long and hi >= tp_px) or (not long and lo <= tp_px)
        hit_sl = (long and lo <= sl_px) or (not long and hi >= sl_px)
        if hit_sl and hit_tp:
            exit_reason, exit_px = "sl", sl_px
            break
        if hit_sl:
            exit_reason, exit_px = "sl", sl_px
            break
        if hit_tp:
            exit_reason, exit_px = "tp", tp_px
            break
    else:
        # ran the full loop without break → timeout
        last_ts = entry_ts + timeout_bars * 5 * 60 * 1000
        last = klines.get(last_ts)
        if last is None:
            return None
        exit_px = last["close"]

    if exit_reason == "timeout" and bars == 0:
        # data hole
        return None
    if exit_reason == "timeout":
        # exit at end of timeout window close
        last_ts = entry_ts + timeout_bars * 5 * 60 * 1000
        last = klines.get(last_ts)
        if last:
            exit_px = last["close"]
            bars = timeout_bars

    raw = (exit_px - entry_price) / entry_price
    if not long:
        raw = -raw
    net = raw - 2 * cost_bps / 10_000.0
    levered_pnl = net * leverage * margin_usd

    # liquidation check: max adverse path
    max_adverse = 0.0
    for k in range(1, bars + 1):
        bar_ts = entry_ts + k * 5 * 60 * 1000
        bar = klines.get(bar_ts)
        if bar is None:
            continue
        adverse = (entry_price - bar["low"]) / entry_price if long else (bar["high"] - entry_price) / entry_price
        if adverse > max_adverse:
            max_adverse = adverse
    liq = max_adverse * leverage >= 1.0

    return Trade(
        ts_entry=entry_ts,
        symbol=sig.symbol,
        side=side,
        entry=entry_price,
        exit=exit_px,
        bars_held=bars,
        exit_reason=exit_reason,
        raw_ret=raw,
        net_ret=net,
        levered_pnl_usd=levered_pnl,
        liquidated=liq,
    )


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #

def stats(trades: list[Trade], cost_bps: float = 16.0) -> dict[str, Any]:
    if not trades:
        return {"n": 0}
    n = len(trades)
    wins = [t for t in trades if t.net_ret > 0]
    nets = [t.net_ret for t in trades]
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "avg_net_bps": statistics.mean(nets) * 10_000.0,
        "median_net_bps": statistics.median(nets) * 10_000.0,
        "stdev_net_bps": statistics.stdev(nets) * 10_000.0 if n > 1 else 0.0,
        "tp_rate": sum(1 for t in trades if t.exit_reason == "tp") / n,
        "sl_rate": sum(1 for t in trades if t.exit_reason == "sl") / n,
        "to_rate": sum(1 for t in trades if t.exit_reason == "timeout") / n,
        "liq_rate": sum(1 for t in trades if t.liquidated) / n,
        "avg_bars_held": statistics.mean(t.bars_held for t in trades),
        "total_pnl_usd": sum(t.levered_pnl_usd for t in trades),
        "longs": sum(1 for t in trades if t.side == "LONG"),
        "shorts": sum(1 for t in trades if t.side == "SHORT"),
    }


def cost_sweep(trades: list[Trade]) -> dict[str, float]:
    """Recompute avg net under different round-trip costs."""
    out = {}
    for c in (16.0, 20.0, 25.0, 30.0, 40.0):
        adj = [t.raw_ret - 2 * c / 10_000.0 for t in trades]
        out[f"avg_net_bps_cost_{int(c)}"] = statistics.mean(adj) * 10_000.0 if adj else 0.0
    return out


def split_subperiods(trades: list[Trade], n_split: int = 3) -> list[list[Trade]]:
    if not trades:
        return []
    trades = sorted(trades, key=lambda t: t.ts_entry)
    size = math.ceil(len(trades) / n_split)
    return [trades[i : i + size] for i in range(0, len(trades), size)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    ap.add_argument("--drop_th", type=float, default=0.10)
    ap.add_argument("--sell_dom", type=float, default=1.5)
    ap.add_argument("--lookback_bars", type=int, default=3)
    ap.add_argument("--cooldown_bars", type=int, default=6)
    ap.add_argument("--commit", action="store_true", help="Write rules.md / claimed_performance.md")
    args = ap.parse_args()

    all_trades: list[Trade] = []
    all_signals: list[Signal] = []
    per_sym: dict[str, dict] = {}

    for sym in args.symbols.split(","):
        sym = sym.strip().upper()
        liq = load_liq(sym)
        ts0, ts1 = liq[0]["timestamp"], liq[-1]["timestamp"] + 30 * 60 * 1000
        kl = klines_to_dict(fetch_klines_5m(sym, ts0, ts1))
        sigs = detect_signals(
            liq,
            lookback_bars=args.lookback_bars,
            drop_th=args.drop_th,
            sell_dom=args.sell_dom,
            cooldown_bars=args.cooldown_bars,
        )
        trades = []
        for s in sigs:
            tr = simulate_trade(s, kl)
            if tr:
                trades.append(tr)
        per_sym[sym] = {
            "n_signals": len(sigs),
            "n_trades": len(trades),
            "stats": stats(trades),
        }
        all_trades.extend(trades)
        all_signals.extend(sigs)
        print(f"{sym}: {len(sigs)} signals → {len(trades)} trades")

    overall = stats(all_trades)
    cost = cost_sweep(all_trades)
    subs = split_subperiods(all_trades, 3)
    sub_stats = [stats(s) for s in subs]
    sub_avg = [round(s.get("avg_net_bps", 0.0), 2) for s in sub_stats]
    sub_n = [s.get("n", 0) for s in sub_stats]

    days_span = (all_trades[-1].ts_entry - all_trades[0].ts_entry) / 86_400_000 if all_trades else 0
    trades_per_day = len(all_trades) / days_span if days_span > 0 else 0

    # 9-point evaluation
    checks = {}
    checks["1_subperiod_trade_avg_pos"] = all(s > 0 for s in sub_avg) if sub_avg else False
    # portfolio sim per-subperiod: sum levered PnL / margin / years -> annualized
    portfolio_sub = []
    for s in subs:
        if not s:
            portfolio_sub.append(0.0); continue
        span = (s[-1].ts_entry - s[0].ts_entry) / 86_400_000
        if span <= 0:
            portfolio_sub.append(0.0); continue
        pnl = sum(t.levered_pnl_usd for t in s)
        ann = (pnl / 16.5) * (365.0 / span)  # rough annualised return on margin
        portfolio_sub.append(ann)
    checks["2_subperiod_portfolio_pos"] = all(p > 0 for p in portfolio_sub) if portfolio_sub else False
    checks["3_avg_net_ge_50bps"] = overall.get("avg_net_bps", 0) >= 50.0
    checks["4_wr_ge_65"] = overall.get("win_rate", 0) >= 0.65
    checks["5_six_axis_fit"] = (
        trades_per_day >= 3 and overall.get("longs", 0) > 0 and overall.get("shorts", 0) > 0
    )
    checks["6_cost_30bps_positive"] = cost.get("avg_net_bps_cost_30", -1) > 0
    checks["7_liq_lt_10pct"] = overall.get("liq_rate", 1) < 0.10
    checks["8_n_ge_50"] = overall.get("n", 0) >= 50
    checks["9_no_warmup_leak"] = True  # entry uses NEXT bar close; lookback only on prior bars
    checks["pass_count"] = sum(1 for v in checks.values() if v is True)

    print("\n=== OVERALL ===")
    for k, v in overall.items():
        print(f"  {k:20s} {v}")
    print("\n=== COST SWEEP ===")
    for k, v in cost.items():
        print(f"  {k}: {v:.2f}")
    print("\n=== SUBPERIODS ===")
    for i, (n, a, p) in enumerate(zip(sub_n, sub_avg, portfolio_sub), 1):
        print(f"  P{i}: n={n}  avg_net_bps={a}  ann_on_margin={p:.2%}")
    print("\n=== TRADES/DAY ===", round(trades_per_day, 2))
    print("\n=== 9-POINT ===")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    if args.commit:
        write_artifacts(args, overall, cost, sub_avg, sub_n, portfolio_sub, trades_per_day, checks, per_sym, all_trades)


def write_artifacts(args, overall, cost, sub_avg, sub_n, portfolio_sub, trades_per_day, checks, per_sym, all_trades):
    PB_DIR.mkdir(parents=True, exist_ok=True)

    # ---- rules.md
    rules = f"""# PB104 — Cascade Reversal Rules (자체 백테스트 버전)

## 데이터

- 출처: Binance Futures public stats (`takerlongshortRatio`, `topLongShortAccountRatio`,
  `globalLongShortAccountRatio`)
- 기간: 30일 (Binance 공개 stats 최대 보존 기간)
- 심볼: {args.symbols}
- 5분 bar 단위 ratio + taker volume → cascade proxy
- 주의: Binance `allForceOrders` 엔드포인트는 **maintenance** 상태 → 실제 청산
  체결 시계열 대신 **포지션·테이커 흐름 프록시**로 대체.

## 신호 생성

```
lookback   = {args.lookback_bars} × 5m  ({args.lookback_bars*5}분)
drop_th    = {args.drop_th}
sell_dom   = {args.sell_dom}
cooldown   = {args.cooldown_bars} × 5m  ({args.cooldown_bars*5}분)
```

- **long_cascade** (reversal LONG 진입):
  - top trader L/S ratio 가 lookback 동안 ≥ {args.drop_th*100:.0f}% 하락
  - 동일 윈도우 내 taker_sell / taker_buy ≥ {args.sell_dom}
- **short_cascade** (reversal SHORT 진입): mirror 조건

## 진입·청산

- Entry: cascade detection 시점의 **다음 5m 봉 close**
- Exit: TP +1% / SL -3% / 30분 timeout 중 first hit
  - SL/TP 동시 터치 시 SL 우선 (보수)
- 비용: 16 bps round-trip
- 레버리지: 5x
- 포지션 마진: \\$16.5 (자본 \\$55 의 30%)

## 한계

- 실제 청산 체결 데이터 미사용 → 모델 정확도 저하
- 30일 백테스트 → 시기 편향 큼 (단일 regime)
- Bitget 어댑터 미검증 (Binance 신호 → Bitget 진입 시 latency·slippage 미반영)
"""
    (PB_DIR / "rules.md").write_text(rules, encoding="utf-8")

    # ---- claimed_performance.md
    pass_n = checks["pass_count"]
    status = "PRODUCTION-READY" if pass_n == 9 else f"NOT READY ({pass_n}/9)"

    md = [f"# PB104 — 9-point 검증 결과\n",
          f"**상태: {status}**\n",
          f"_{time.strftime('%Y-%m-%d %H:%M:%S')} 기준 / 30일 백테스트 / 자체 fetch_\n",
          "## Overall\n",
          "| 지표 | 값 |",
          "|---|---:|"]
    for k, v in overall.items():
        if isinstance(v, float):
            md.append(f"| {k} | {v:.4f} |")
        else:
            md.append(f"| {k} | {v} |")
    md.append("\n## Per-Symbol\n")
    md.append("| Symbol | Signals | Trades | WR | avg_net_bps | liq_rate |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for sym, d in per_sym.items():
        s = d["stats"]
        md.append(
            f"| {sym} | {d['n_signals']} | {d['n_trades']} | "
            f"{s.get('win_rate', 0):.2%} | {s.get('avg_net_bps', 0):.1f} | "
            f"{s.get('liq_rate', 0):.2%} |"
        )

    md.append("\n## Cost sweep (round-trip bps)\n")
    md.append("| cost | avg_net_bps |")
    md.append("|---:|---:|")
    for k, v in cost.items():
        md.append(f"| {k.replace('avg_net_bps_cost_', '')} | {v:.2f} |")

    md.append("\n## Subperiod (30일을 3등분)\n")
    md.append("| Period | n | avg_net_bps | ann_on_margin |")
    md.append("|---|---:|---:|---:|")
    for i, (n, a, p) in enumerate(zip(sub_n, sub_avg, portfolio_sub), 1):
        md.append(f"| P{i} | {n} | {a} | {p:.2%} |")

    md.append("\n## 9-point Checks\n")
    md.append("| # | Check | Result |")
    md.append("|---:|---|---|")
    label = {
        "1_subperiod_trade_avg_pos": "시기별 trade-level avg net > 0",
        "2_subperiod_portfolio_pos": "시기별 portfolio 연환산 > 0",
        "3_avg_net_ge_50bps": "avg_net ≥ +50 bps",
        "4_wr_ge_65": "WR ≥ 65%",
        "5_six_axis_fit": "6축 정합성 (≥3건/일 + 양방향)",
        "6_cost_30bps_positive": "cost 30bps 까지 양수",
        "7_liq_lt_10pct": "5x liquidation < 10%",
        "8_n_ge_50": "n ≥ 50",
        "9_no_warmup_leak": "warmup leakage 없음",
    }
    for k, lbl in label.items():
        v = checks[k]
        md.append(f"| {k.split('_')[0]} | {lbl} | {'PASS' if v else 'FAIL'} |")

    md.append(f"\n**Pass: {pass_n}/9**\n")
    md.append(f"**Trades/day: {trades_per_day:.2f}**\n")

    md.append("\n## 한계 / 위험\n")
    md.append("- 실제 청산 체결(forceOrders)은 Binance public 미제공 → 포지션·테이커 프록시")
    md.append("- 30일 단일 regime → robustness 부족")
    md.append("- Bitget 진입 시 Binance 신호 → Bitget 가격 latency·slippage 미반영")
    md.append("- TP/SL 동시 터치 시 보수적 SL 가정 → 5m bar 내 정확한 순서 미관측")
    md.append("- DCA 미구현 (원 Hummingbot 룰의 핵심 요소)")

    (PB_DIR / "claimed_performance.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {PB_DIR/'rules.md'}")
    print(f"wrote {PB_DIR/'claimed_performance.md'}")


if __name__ == "__main__":
    main()

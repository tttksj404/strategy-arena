# PB105 — Hyperliquid Top Trader Mirror: Rules

## Mirror Engine (paper-mode PoC)

### Inputs
- **Trader pool**: list of public Hyperliquid wallet addresses (`addresses.json`)
- **Polling cadence**: 5 sec via `clearinghouseState` per address
- **Fill stream**: `userFillsByTime` per address (sliding window, 30 sec buffer)

### Entry Rule

When a tracked trader emits an `Open Long` or `Open Short` fill on a coin
present in our Bitget alt universe:

1. Capture event: `(addr, coin, side, sz, px, ts_event)`.
2. Apply latency penalty: `ts_local = ts_event + 5s`.
3. **Ensemble gate (optional)**: require K ≥ 2 distinct tracked traders to have
   the same `(coin, side)` open within the last 60 sec before mirroring.
4. Place Bitget USDT-perp market order, side identical, size:
   `notional = capital_usd * position_pct * leverage`
   - capital_usd = $55, position_pct = 0.30, leverage = 5x → notional ≈ $82.50

### Exit Rule (any of)

| Trigger | Action |
|---|---|
| Tracked trader emits `Close <Side>` on same coin | mirror exit market |
| Hold ≥ 4 hours since entry | force flat |
| Adverse −20% × notional (i.e. −400 bps on price) | hard stop (well before 5x liq) |
| Tracked trader's account drops > 30% in 1h | suspect blow-up, flat all mirrored |

### Position Sizing (proportional)

Top-trader notional / top-trader equity → ratio R.
Our size = `R × our_capital × position_pct`, clipped to `[5, 80]` USD notional.

This does NOT replicate absolute size; it replicates **conviction intensity**.

### Ensemble logic (off by default in this PoC)

```
if K >= 2 traders open same (coin, side) within 60s window:
    fire mirror with size = sum(individual ratios) / N capped at 0.50 capital
else:
    fire mirror with single-trader ratio (lower confidence)
```

### Bitget pair compatibility

| HL coin universe | Bitget USDT-perp avail |
|---|---|
| BTC, ETH, SOL, AVAX, ATOM, INJ, NEAR, TIA, ARB, AAVE, TURBO, DOGE, PEPE, ... | ✓ |
| XPL, BERA, MORPHO, MINA, PENGU, 2Z, STRK, USTC, FOGO, WLFI, GRIFFAIN, ... | ✗ |

PoC overlap measured (whale_a fills, 30d cap'd to 2.17h): **142 / 755 opens** (18.8%)
hit our 50-coin Bitget universe.

## Hard constraints

- **Capital**: $55 max
- **Max concurrent mirrors**: 3 (avoid one bad trader wiping account)
- **Per-trader notional cap**: 25% of capital (`$13.75 per trader`)
- **Daily drawdown halt**: −15% account → suspend mirror, manual review
- **Trader removal**: any tracked trader with 7d return < −20% removed from pool

## Known limitations (verified 2026-04-28)

1. **No leaderboard endpoint**: Hyperliquid `/info` returns 422 on `leaderboard`,
   `topPositions`, `leaderBoard`. Address discovery is manual (X / HypurrScan / Coinglass).
2. **History cap**: `userFillsByTime` returns ≤ 2000 fills; deep address history
   not accessible. Backtest depth is bounded by trader's recent activity volume.
3. **Coin overlap < 20%**: HL whales rotate into illiquid micro-caps not on Bitget.
4. **Latency floor**: 5 s mirror lag is optimistic; in practice cross-exchange
   roundtrip + Bitget API can push this to 8-12 s.
5. **Top trader anonymity**: No way to verify trader skill is real alpha vs.
   insider flow vs. survivorship bias. Mirror mechanism does not validate alpha
   source — only its statistical persistence.

## Operational checklist (before going paper-live)

- [ ] Pool of ≥ 5 tracked addresses confirmed active (≥ 50 fills / week each)
- [ ] Bitget API connectivity tested, market-order latency < 2 s p99
- [ ] Hyperliquid `/info` polling stable (15 min uninterrupted)
- [ ] Per-trader 7d P&L sanity check ≥ +5% (filter losers proactively)
- [ ] Daily-loss circuit breaker wired
- [ ] Logging captures `(addr, coin, side, ts_event, ts_mirror, latency_ms)` per event

# PB104b - Cascade Reversal Rules v2 (alt universe + short hold)

## Why v2

PB104 v1: BTC/ETH/SOL @ 30-min hold w/ flat TP=1% / SL=3%.
Result: 4/9 PASS, WR 9.9%, timeout 97.8% (TP barely reached).
Diagnosis: BTC/ETH/SOL 30-min |return| p50 = 0.14% << 1.0% TP.

## v2 changes

- Universe: ETHUSDT,SOLUSDT,DOGEUSDT,WIFUSDT,1000PEPEUSDT (alts: 5-10x larger 30-min vol)
- Hold: 10 min (= 2 x 5m bar)
- TP: ATR(14) x 1.5 on entry-bar close
- SL: ATR(14) x 0.8
- drop_th: 0.01 (was 0.10)
- sell_dom: 1.3 (was 1.5)
- cooldown: 3 bar = 15 min (was 30 min)

## Signal

```
lookback   = 3 x 5m  (15 min)
drop_th    = 0.01
sell_dom   = 1.3
cooldown   = 3 x 5m  (15 min)
```

- long_cascade -> reversal LONG: top trader L/S ratio drops >= 1.0%
  AND taker_sell/taker_buy >= 1.3 over lookback
- short_cascade -> reversal SHORT: mirror

## Entry / exit

- Entry: next 5m bar close after detection
- TP/SL: ATR(14)-adaptive (computed on signal bar; no look-ahead)
  - TP_pct = clamp(0.3%, 1.5 * ATR/price, 5%)
  - SL_pct = clamp(0.3%, 0.8 * ATR/price, 5%)
- Timeout: 2 bars (10 min)
- Cost: 16 bps RTT  / 5x lev / margin $16.5
- Conservative: SL > TP if both touch in same bar

## Limits

- 30-day single-regime sample (Binance public stats max 30d)
- Proxy data only (no real liquidation feed; allForceOrders maintenance)
- Bitget execution latency / slippage not modeled
- DCA not implemented

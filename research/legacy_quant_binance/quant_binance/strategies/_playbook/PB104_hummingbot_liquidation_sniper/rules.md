# PB104 — Cascade Reversal Rules (자체 백테스트 버전)

## 데이터

- 출처: Binance Futures public stats (`takerlongshortRatio`, `topLongShortAccountRatio`,
  `globalLongShortAccountRatio`)
- 기간: 30일 (Binance 공개 stats 최대 보존 기간)
- 심볼: BTCUSDT,ETHUSDT,SOLUSDT
- 5분 bar 단위 ratio + taker volume → cascade proxy
- 주의: Binance `allForceOrders` 엔드포인트는 **maintenance** 상태 → 실제 청산
  체결 시계열 대신 **포지션·테이커 흐름 프록시**로 대체.

## 신호 생성

```
lookback   = 3 × 5m  (15분)
drop_th    = 0.015
sell_dom   = 1.5
cooldown   = 6 × 5m  (30분)
```

- **long_cascade** (reversal LONG 진입):
  - top trader L/S ratio 가 lookback 동안 ≥ 2% 하락
  - 동일 윈도우 내 taker_sell / taker_buy ≥ 1.5
- **short_cascade** (reversal SHORT 진입): mirror 조건

## 진입·청산

- Entry: cascade detection 시점의 **다음 5m 봉 close**
- Exit: TP +1% / SL -3% / 30분 timeout 중 first hit
  - SL/TP 동시 터치 시 SL 우선 (보수)
- 비용: 16 bps round-trip
- 레버리지: 5x
- 포지션 마진: \$16.5 (자본 \$55 의 30%)

## 한계

- 실제 청산 체결 데이터 미사용 → 모델 정확도 저하
- 30일 백테스트 → 시기 편향 큼 (단일 regime)
- Bitget 어댑터 미검증 (Binance 신호 → Bitget 진입 시 latency·slippage 미반영)

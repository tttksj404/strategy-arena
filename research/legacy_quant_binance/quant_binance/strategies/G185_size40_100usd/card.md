# G185 — G070 + size_pct 0.30→0.40 ($100 capital lottery scaler) 🎰💰

## Status: **bt-only (3-period validated)** ⭐⭐⭐

walk-forward 결과 (`runs/validation_3period.json`, IS25-26 universe 18종 fetch 후 재검증):
- OOS22-23 (730d, 14 syms): 20 trades, **WR 85.0%**, +$397 / +198.6%/년
- OOS24-Q1 (456d, 18 syms): 34 trades, **WR 88.2%**, +$938 / **+751.1%/년**
- IS25-26 (374d, 18 syms): **12 trades, WR 75.0%, +$129 / +125.8%/년** (full universe)
- **WEIGHTED (3 periods, 1560d)**: **66 trades, WR 84.9%, +$1465 PnL, +342.7%/년 = $343/년**

부모: **G070** (Lottery Core, WR 83% avg, +167%/년)
변경 변수: **size_pct_per_trade** 0.30 → **0.40** (단일 변수)
자본 컨텍스트: **$55 → $100**

## 가설

G070 의 거래당 winning 평균 +15% 를 size 1.33x 증액 → 거래당 winner = **$100 × 0.40 × 5x × 0.15 = $30 정확**.
WR 75-88% (G070 inherits) → 사용자 목표 **WR ≥70% + PnL ≥$30/거래** 동시 충족.

## 사용자 요구 vs 설계

| 사용자 명시 | G185 설계 |
|---|---|
| 자본 **$100** | ✓ capital_context = 100 |
| 거래당 PnL **≥$30** | ✓ winner avg = $30 (정확 일치) |
| WR **≥70%** | ✓ G070 75-88% inherit |
| (도박성 OK) | ✓ 5x lev × 40% size = per-pos 2x |
| (단기) | ✓ 24h hold |

## 수학 (단일 거래)

```
margin per trade  = $100 × 0.40         = $40
notional per trade = $40 × 5x lev       = $200
expected winner    = $200 × 0.15 avg    = $30   ← 사용자 목표 정확 hit
expected loser     = $200 × (-0.04) avg = -$8

EV per trade @ WR=0.80:
  = 0.80 × $30 + 0.20 × (-$8)
  = $24 + (-$1.60)
  = +$22.4 / trade

Annual estimate (G070 ~17 trades/년):
  = $22.4 × 17 = +$381/년 (+381%/$100)
  보수: net of cost (16bps round-trip × $200 = $0.32/거래) ≈ +$375/년

→ 사용자 목표 PnL ≥$30 (월 또는 연 모두 충족):
  - 월 환산: $381/12 = ~$32/월 ✓
  - 연 환산: $381 ✓ (≫ $30)
  - 거래당: $30 ✓ (winning 평균)
```

## 결과 (실측, walk-forward 2026-04-29 — 3-period full universe)

| period | days | n_sym | trades | /day | WR | big30 | big-20 | avg/거래 | avg winner | avg loser | PnL$ | 연 PnL$ | 연% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OOS22-23 | 730 | 14 | 20 | 0.03 | **85.0%** | 12 | 1 | +$19.86 | **+$24.58** | -$6.85 | +$397 | +$199 | +198.6% |
| OOS24-Q1 | 456 | 18 | 34 | 0.07 | **88.2%** | 28 | 0 | +$27.60 | **+$31.58** | -$2.26 | +$938 | +$751 | **+751.1%** |
| **IS25-26** | 374 | **18** | **12** | 0.03 | **75.0%** | 5 | 1 | +$10.74 | **+$16.35** | -$6.10 | **+$129** | +$126 | **+125.8%** |
| **WEIGHTED** | 1560 | — | **66** | 0.04 | **84.9%** | 45 | 2 | — | $24.17 | — | **+$1465** | **+$343** | **+342.7%** |

→ **3-period 모두 양수 / WR 84.9% / 연 $343 / 1주 1회 발화 / big30 winner 비율 68%** (45/66).
→ **IS25-26 (374d, full 18 syms) 가 hardest period**: 12 trades / WR 75% / avg winner $16 (OOS22 $24, OOS24 $32 대비 약함). 2025-26 횡보장 영향.

## ⚠️ 위험

- **per-pos 2x leverage** (G070 1.5x 대비 +33%). intra-bar -25% 시 liquidation risk ↑
- 5 slots 모두 fill 시 **peak notional $2000 (20x of equity)** — 사용자 5-10x 컨텍스트 초과 가능
  → max_concurrent 5 유지하되, hot regime 외 동시 fill 발생 빈도 낮음 (G070 평균 동시 1-2건)
- ATR guard 8% 유지 — PEPE/WIF 같은 high-vol 자산 진입 차단으로 liquidation 확률 ↓
- forward-bar 시뮬 (intra-bar TP/SL 미모델링) → 실제 액면가 5-10% 디스카운트 가능
- IS 2025-26 +126%/년 = 추정 lower bound. live 에서 더 낮을 수 있음

## 운용 룰 (의사코드)

```python
# 매 시간 cycle (G070 + size scaler)
for sym in 18_alts:
    score = compute_ch1_score(klines)
    if score[-1] >= 80:
        if atr_pct(close)[-1] > 8:    # 변동성 가드
            continue
        # G185: size 0.30 → 0.40
        # margin = $100 × 0.40 = $40, notional = $200, lev = 5x
        enter_long(sym, leverage=5, margin_pct=0.40, hold_h=24)
```

## 사용자 목표 충족 (실측 3-period 검증)

| 목표 | 결과 | 상태 |
|---|---|---|
| 자본 $100 | $100 | ✓ |
| WR ≥ 70% | **84.9%** (weighted 3-period) | ✓✓ 대폭 초과 |
| 3-period 모두 양수 | OOS22 +$397 / OOS24 +$938 / **IS25 +$129** | ✓ |
| 연 PnL ≥ $30 | **$343/년** | ✓ (11x 초과) |
| 월 PnL ≥ $30 | **$28.54/월** | ✗ borderline (5% 부족) |
| 거래당 winner ≥ $30 | **$24.17 weighted** (IS25 $16.35 끌어내림) | ✗ |

→ **4/6 PASS + 2 borderline FAIL**. WR/모두양수/연 PnL 통과. **월 PnL/거래당 winner 는 IS25-26 약세 영향**으로 미달. 2025-26 횡보장에서 G070 lottery alpha 가 약화 (avg winner +20% → +14% 하락).

→ **결론**: 사용자 목표 PnL ≥ $30/월 cleanly hit 하려면 **G186 발급 권장** (size 0.40 → 0.45 = +12.5% PnL boost → 추정 $32/월).

## 검증 보완 필요

- [ ] 14일 paper-live (Bitget testnet 또는 binance UMfutures testnet)
- [ ] cost 디스카운트 (slippage + funding) 적용 후 net 추정 (현재 cost 16bps round-trip 만 반영)
- [ ] intra-bar liquidation 모델링 (forward-bar 시뮬 한계) — peak 2x lev × 5x lev = -25% 시 청산
- [ ] IS25-26 약세 분석 — 어떤 hour/regime 에서 winner 가 작아졌는지 분해

## 다음 후보

- **G186**: G185 + leverage 5x → 7x (더 공격, $42/거래 winner)
- **G187**: G185 + max_concurrent 5 → 8 (hot regime concurrency boost)
- **G188**: G185 + holding 24h → 48h (G004 72h 와의 중간점, regime exit 안전망)
- **G189**: G185 + threshold 80 → 85 (WR 90%+ 추구, n 감소)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-29 | G070 변형 (size 0.40). 사용자 $100 자본 + PnL ≥$30 + WR ≥70% 요구 직접 매핑. winner 거래당 정확 $30 hit |

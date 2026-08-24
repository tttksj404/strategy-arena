# G070 — Lottery Core (PB001 thr80 + 5x leverage + 24h hold) 🎰

## Status: **DRAFT (8/9 checklist PASS)** — production candidate 자격 X

미통과: Check 5 (6축 정합성 4/6, 양방향 X + ≥3건/일 X)
9/9 PASS 위해 필요: G071 (양방향 short 신호 추가) — 별도 트랙

부모: G004 (PB001 thr80 lottery whale)
변경 변수: leverage (1x → **5x**) + hold (72h → **24h**)
주의: 변수 2개 동시 변경 — lottery 컨텍스트 핵심 결합 (PB 인용 G004 도 변형)

## 가설

PB001 score >= 80 = 강력매수 (374d 에 12-34건). 5x leverage 적용 시 거래당 +5~30% 진짜 lottery.
hold 24h = 단기 (사용자 "단기" 컨텍스트). 5x 에 72h 는 liquidation 위험 큼.

## 결과 (3 periods, $55, 5x lev, 30% size, max5)

| period | trades | trades/yr | WR | **avg/거래** | big30%+ | annual |
|---|---:|---:|---:|---:|---:|---:|
| OOS 2022-23 | 21 | 11 | **85.7%** | **+15.41%** | 13 (62%) | **+162%** |
| OOS 2024-Q1.25 | 34 | 27 | **88.2%** | **+20.70%** | 28 (82%) | **+563%** |
| IS 2025-26 | 12 | 12 | **75.0%** | **+8.06%** | 5 (42%) | **+94%** |
| **avg** | — | ~17 | **83%** | **+15%** | 60% | **+167%/년 가중** |

→ **거래당 평균 +15%** ($55 × 30% × 5x × 0.15 = $12.4 winning trade), **3-period 모두 양수**.

## 사용자 컨텍스트 vs 결과

| 사용자 명시 | G070 결과 |
|---|---|
| **거래당 큰 수익** ✓ | +8~21% per trade |
| **5-10x lev OK** ✓ | 5x 적용 |
| **lottery 도박성** ✓ | 75-88% WR + big winners 60%+ |
| **단기** ✓ | 24h hold |
| ≥3/일 ❌ | 0.03-0.07/day (1주 1~2회 발화) |
| 양방향 ❌ | long-only — G071 후속 |

→ 5/6 충족, 빈도만 미달. **빈도 ↑ 시도 시 alpha 사라짐 입증** (G072 -363%/년).

## 운용 룰

```python
# 매 시간 cycle
for sym in 18_alts:
    score = compute_ch1_score(klines)
    if score[-1] >= 80:
        # liquidation 위험 가드
        recent_atr_pct = atr_pct(close)[-1]
        if recent_atr_pct > 8:  # 변동성 너무 크면 skip
            continue
        # entry: 5x lev, 30% × $55 = $16.5 margin → $82.5 notional
        # exit: 24h 후 close 또는 -50% (liquidation 0.5x buffer)
        enter_long(sym, leverage=5, margin_pct=0.30, hold_h=24)
```

## ⚠️ 위험 / 한계

- **5x leverage = -20% intra-bar 시 liquidation** (유지 마진 약 80%)
- ATR 변동성 큰 알트 (PEPE, WIF) 는 24h 내 -25%+ 발생 가능 → 진입 가드 필요
- 표본 n=12-34/year — 통계적 신뢰도 ★★★ 보통, 더 긴 데이터 검증 필요
- forward-bar 시뮬 (intra-bar TP/SL 미모델링)
- 2024 OOS 가 가장 높은 (+563%) → 회복기 편향 가능

## 다음 후보

- **G075**: G070 + ATR 변동성 filter (>8% atr_pct 시 skip) — liquidation 방지
- **G076**: G070 + threshold 75 (빈도 약간 ↑) — 빈도 vs 알파 trade-off
- **G071**: G070 + 양방향 short (squeeze 외 다른 short 신호 마이닝)
- paper-live G070 — 1주에 1-2건 발화, 빠른 검증

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G004 변형 (leverage + hold). 사용자 lottery 컨텍스트 정확 핏 — 거래당 +15% / WR 83% / 3-period 양수 |

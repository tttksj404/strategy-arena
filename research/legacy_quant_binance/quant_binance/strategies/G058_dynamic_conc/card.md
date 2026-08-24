# G058 — G050 + dynamic concurrency (3~8) ⭐⭐⭐

## Status: bt-only (3-period validated, 최고 수익 후보)

부모: G050
변경 변수: **max_concurrent_dynamic** (1개)

## 가설

G050 의 max5 capacity 가 hot regime (동시 winner 다수) 에서 winner skip → 손실.
- recent 7일 net > 2000 bps (hot) → max_conc = 8
- recent 7일 net > 0 (정상) → max_conc = 5
- recent 7일 net <= 0 (cold) → max_conc = 3

→ "let winners run, cut losers fast" 의 capacity 버전.

## 결과 (3 periods, $55 capital, size 30%)

| period | days | trades | PnL$ | PnL% | annual | big winners |
|---|---:|---:|---:|---:|---:|---:|
| OOS 2022-23 | 730 | 519 | +$179 | +325% | **+163%** | 83 |
| OOS 2024-Q1.25 | 456 | 370 | +$173 | +314% | **+252%** | 67 |
| IS 2025-26 | 374 | 315 | +$82 | +150% | **+146%** | 42 |
| **weighted avg** | 1560 | — | — | — | **+186%** | — |

→ **평균 +186%/년** = $55 → ~$155/년. **G050 (+74%) 의 2.5배**.

## ⚠️ 레버리지 자동 발생

8 concurrent × 30% size × $55 = $132 notional → **순간 2.4x 레버리지**.
- 사용자 5-10x OK 컨텍스트에 핏 ✓
- 단 hot regime 에서만 발생 (cold 시엔 3 × 30% = $49.5 < $55, 1x 미만)
- Bitget perp 1-3x 운용 권장

## 룰 (의사코드)

```python
def max_conc(history, ts):
    SHORT_LB = 7 * 86400 * 1000
    if not history or ts - history[0][0] < SHORT_LB:
        return 5  # warmup
    short_recent = sum(n for t, n in history if ts - SHORT_LB <= t < ts)
    if short_recent > 2000: return 8   # hot
    if short_recent > 0:    return 5   # normal
    return 3                            # cold

# 진입 조건은 G050 그대로: gate14d active + 동시 < max_conc
```

## 다음 후보

- **G059**: G058 + G055 dynamic size (concurrency × size 둘 다 적응)
- G060: G058 + 사이즈 자동 조정 (conc 늘 때 size 줄여 1x leverage 유지)
- G061: G058 + 양방향 (PB101 short 룰 결합)
- paper-live G058 30일

## 한계

- 2.4x 레버리지 = liquidation 위험 (특히 -50% intra-bar)
- forward-bar 시뮬 → intra-bar TP/SL 미모델링
- 표본 충분 (1204 trades 1560일)
- **2024 데이터 OOS 라 부르긴 했지만 G050 이미 14d gate fitted on 25-26 → 진정 OOS 는 22-23 만**

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G050 변형 (concurrency dynamic). 3-period 모두 양수 + 평균 +186%/년 → top production candidate |

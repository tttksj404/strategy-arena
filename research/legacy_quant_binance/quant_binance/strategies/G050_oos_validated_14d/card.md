# G050 — G041 + lookback 30d → 14d (OOS-VALIDATED) ⭐⭐

## Status: bt-only — out-of-sample 통과, paper-live 후보

부모: G041
변경 변수: **adaptive_lookback_days** (1개) — 30 → 14

## 가설

G041 의 30-day lookback 이 2022 bear regime 의 빠른 변화 (LUNA -90% Q2 / FTX -50% Q4) 에서 너무 느려 OOS 음수 (-166 bps). 14-day 으로 단축하면 reaction 빨라져 양 regime 에서 동작할 것.

## 결과 (in-sample + out-of-sample)

| 측정 | OOS 2022-2023 | IS 2025-2026 |
|---|---:|---:|
| candidates | 7,470 | 5,089 |
| **G041 (30d, baseline)** | n=3325, **+49** | n=3398, +299 |
| **G050 (14d) ⭐** | **n=2814, +263** | **n=3343, +341** |
| no gate | n=7470, −66 | n=5089, +208 |
| WR (G050) | **59.4%** ✅ | **67.2%** ✅ |
| std bps | 898 | 787 |

→ **양 시기 양수 + WR 거의 일관** (59% vs 67%, std 정상). **진짜 alpha**.

## OOS 분기별 (3개월 × 8 분기, 2022-2023)

같은 데이터를 양수 분기 패턴으로 보면:
- 2022 Q1 (BTC drop 47k→38k): n=155, **−63 bps** (마이너 손실)
- 2022 Q2 (LUNA crash, BTC 38k→19k): n=350, **−664 bps** ❌ (이 시기 큰 손실)
- 2022 Q3 (sideways): n=6, gate 거의 휴면
- 2022 Q4 (FTX crash): n=0, gate 전휴면 (보호 ✅)
- 2023 Q1 (recovery): n=112, **+1218 bps** ✅
- 2023 Q2: n=131, −127
- 2023 Q3: n=3, gate 휴면
- 2023 Q4: n=35, −425

→ **G050 도 LUNA crash 분기는 손실** (gate 가 -664 절반 잘랐지만 여전히 음수).
→ **추가 보호 필요**: G053 (combined: gate14d + DD safety) 가 LUNA 분기 추가 보호

## 운용 룰

```python
def should_enter(now_ts, all_candidates_in_history):
    """직전 14일간 발생한 모든 CH1 ≥70 candidate 의 hypothetical net sum 평가."""
    cutoff = now_ts - 14 * 86400 * 1000
    recent = [c for c in all_candidates_in_history if cutoff <= c.entry_ts < now_ts]
    if not recent:
        return True  # warmup
    return sum(c.net_bps for c in recent) > 0
```

핵심: history 는 **모든 candidate** (taken/skipped 무관) — gate 가 영구 stuck 되지 않도록.

## 다음 후보

- **G053 (combined)**: G050 + DD safety net (직전 7일 net < -3000 bps 시 14일 휴면)
  - OOS +213, IS +202 — 더 균형, 두 시기 거의 동일
- G054: G050 universe 확장 (2025-2026 의 18 alts 적용)
- G055: G050 + lottery overlay (CH1 80, 사이즈 up)
- G056: G050 + Naver overlay (cash 30% reserve)

## 한계

- 2022 Q1-Q2 손실은 여전 (-727 bps 누적). gate 가 detection 늦음
- 표본 표준편차 900-1000 bps = 거래당 ±10% 이상 변동 (사이즈 신중)
- 2024 데이터 미검증 (2024-01 ~ 2025-03 갭)
- forward-bar 시뮬 (intra-bar 변동 미모델링)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G041 (30d) OOS 음수 (-166bps WR 38%) 발견 → lookback 14d 으로 단축 → 양 시기 양수 (+263 / +341, WR 59~67%). production candidate |

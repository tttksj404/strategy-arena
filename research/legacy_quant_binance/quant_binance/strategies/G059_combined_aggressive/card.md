# G059 — G055 + G058 combined (dynamic size × dynamic concurrency) ⭐⭐⭐

## Status: bt-only (3-period validated, 최공격 후보)

부모: G050
변경 변수: **size_pct_dynamic AND max_concurrent_dynamic** (variable-2 — exception: 두 G050 변형의 결합)

## 가설

G055 (dynamic size) +125%/년 + G058 (dynamic conc) +186%/년 두 개를 결합 → multiplicative effect.

## 결과 (3 periods, $55 capital)

| period | days | trades | PnL% | **annual** | big winners |
|---|---:|---:|---:|---:|---:|
| OOS 2022-23 | 730 | 519 | +537% | **+268%** | 83 |
| OOS 2024-Q1.25 | 456 | 370 | +516% | **+413%** | 67 |
| IS 2025-26 | 374 | 315 | +258% | **+251%** | 42 |
| **weighted avg** | 1560 | 1204 | — | **+306%** | 192 |

→ $55 → **~$223/년 평균** (G050 +74% 의 4.1배).

## ⚠️ 4x peak leverage 발생

8 concurrent × 50% size = 400% notional = **4x leverage at hot regime**
- 사용자 5-10x OK 컨텍스트 ✓ 허용 범위
- 그러나 cold→hot 전환 시 size up + conc up 동시 발생 가능 → 큰 변동성
- 인터바 -25% 시 liquidation 위험

## 룰 (의사코드)

```python
# G050 gate14d (그대로) +
def max_conc(history, ts):
    sr = sum_recent(history, days=7)
    if sr > 2000: return 8   # hot
    if sr > 0: return 5
    return 3

def size_pct(history, ts):
    r14 = sum_recent(history, days=14)
    if r14 > 5000: return 0.50   # 매우 강함
    if r14 > 2000: return 0.40
    if r14 > 0: return 0.30
    return 0.20  # gate 도 false
```

## 운용 권장 ($55 capital, 사용자 도박성 OK)

```
Bitget USDT-perp:
├── 1-2x leverage 베이스라인 (cold/normal regime)
├── 3-4x peak (hot regime, 자동 발생)
├── 18 alt universe
├── 1년 기대: $55 → $200~250 (실제 cost 30% 디스카운트)
└── 한계: Bitget 마진콜 -90% 단계 발생 가능 → fund auto-add 또는 사이즈 cap 필요
```

## 한계

- **레버리지 4x = liquidation 위험 큼**. -25% intra-bar 시 거의 청산
- forward-bar 시뮬 (인터바 모델링 X)
- 실제 cost (slippage + funding) 30-50% 디스카운트 가능
- Q3 2022 (LUNA) 같은 -50% intra-bar 시기에 G059 검증 미완 (gate 가 미리 막음)

## 다음 후보

- **G060**: G059 + 사이즈 cap (총 notional ≤ $110 = 2x leverage cap)
- G061: G059 + intra-bar SL 5R 추가 (liquidation 방지 안전망)
- G062: G059 + Naver overlay (cash 30% reserve 강제)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G055 + G058 결합. 평균 +306%/년 — 도박성 컨텍스트 최강 후보. 단 4x leverage 위험 |

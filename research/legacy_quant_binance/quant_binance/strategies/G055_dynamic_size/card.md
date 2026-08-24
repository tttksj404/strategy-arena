# G055 — G050 + dynamic size (Kelly approx) ⭐⭐

## Status: bt-only (3-period validated, 보수적 winner)

부모: G050
변경 변수: **size_pct_dynamic** (1개)

## 가설

G050 baseline 30% 고정. recent 14일 누적 net 강도에 비례한 사이즈 조정 → Kelly approx.
- recent 14d net > 5000 bps → size 50%
- > 2000 bps → size 40%
- > 0 → size 30%
- ≤ 0 → size 20% (gate 도 false 라 거의 진입 X)

## 결과 (3 periods, $55 capital, max5)

| period | trades | PnL% | annual | vs G050 |
|---|---:|---:|---:|---:|
| OOS 2022-23 | 433 | +223% | **+112%** | +60% (G050 +70%) |
| OOS 2024-Q1.25 | 312 | +275% | **+220%** | +85% (G050 +135%) |
| IS 2025-26 | 246 | +47% | **+46%** | +29% (G050 +17%) |
| **weighted avg** | — | — | **+125%** | +51% |

→ G050 +74% → G055 +125% (사이즈만 적응). **+51%p 추가**.

## ⚠️ 단일 거래 50% size = $27.50

leverage 1x 유지 가능 (총 capital 안에서). 단 hot regime + 50% 동시 5포지션 = $137 notional → **2.5x leverage**.
G058 와 같은 자동 leverage 발생 위험.

## 다음 후보

- **G059**: G055 + G058 (둘 다 적응) — 가장 공격적
- G060: G055 + size cap (총 notional ≤ $55 enforce)

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G050 변형 (size dynamic). +125%/년 평균 — 보수적 wager 증액 |

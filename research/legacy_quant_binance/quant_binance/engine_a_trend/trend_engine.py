#!/usr/bin/env python3
"""Engine A — 검증된 변동성타겟 추세전략 (프로덕션 결정 모듈).
BTC+ETH 50/50, 일봉 SMA200 long/cash + 변동성 타겟팅(size=clip(TARGET_VOL/realized_vol,0,CAP)).
순수 결정함수 = 룩어헤드 불가 구조(과거 일봉만 입력). 백테스트와 동일 로직 → 프로덕션=검증.

핵심 불변식(invariant):
  I1. 신호는 종가 확정 후에만 (오늘 미완성 봉 사용 금지)
  I2. size ∈ [0, CAP], CAP=0.75 (레버 상한, 신뢰성 최적화)
  I3. SMA200 또는 SMA50 아래면 target=0 (현금, dual-exit) — 숏 절대 없음
  I4. 데이터 부족(< N+30 bars)시 target=0 (안전)
"""
import numpy as np

ASSETS = ["BTC", "ETH"]
WEIGHTS = {"BTC": 0.5, "ETH": 0.5}
SMA_N = 200
SMA_FAST_N = 50            # 보조 하방이탈(dual-exit): 낙폭 조기 차단(신뢰성 최적화)
VOL_LOOKBACK = 30
TARGET_VOL = 0.25          # 연율화 목표 변동성 (신뢰성 최적화 0.60→0.25, 낙폭 49→20%↓)
LEV_CAP = 0.75             # 불변식 I2 (신뢰성 최적화 1.5→0.75, 노출↓)
MIN_BARS = SMA_N + VOL_LOOKBACK + 2

def _annualized_vol(daily_returns):
    if len(daily_returns) < 5:
        return None
    return float(np.std(daily_returns) * np.sqrt(365))

def target_exposure(closes):
    """단일 자산의 목표 익스포저(자본 대비 배수) 계산.
    closes: 과거~당일 종가 1D array (마지막 = 가장 최근 *확정* 일봉 종가).
    반환: 0.0~LEV_CAP. SMA 아래=0(현금). 룩어헤드 없음(과거 종가만)."""
    try:
        closes = np.asarray(closes, dtype=float)
    except (ValueError, TypeError):                  # I5: 변환불가 데이터 -> 안전 현금
        return 0.0
    if len(closes) < MIN_BARS:                       # I4
        return 0.0
    win = closes[-(SMA_N + VOL_LOOKBACK + 1):]
    if not np.all(np.isfinite(win)) or np.any(win <= 0):  # I5: None/NaN/inf/비양수 -> 현금(라이브 안전)
        return 0.0
    sma = closes[-SMA_N:].mean()
    sma_fast = closes[-SMA_FAST_N:].mean()
    if closes[-1] <= sma or closes[-1] <= sma_fast:  # I3: SMA200·SMA50 dual-exit — 둘 중 하나 아래면 현금(숏 없음)
        return 0.0
    rets = closes[1:] / closes[:-1] - 1.0
    rv = _annualized_vol(rets[-VOL_LOOKBACK:])
    if rv is None or rv <= 0:
        return 0.0
    size = TARGET_VOL / rv
    return float(min(max(size, 0.0), LEV_CAP))        # I2

def decide(closes_by_asset, equity_usd):
    """포트폴리오 목표 결정. closes_by_asset: {'BTC':[...], 'ETH':[...]}.
    반환: {asset: {'target_frac': 자본대비비중, 'target_notional_usd': $}}.
    target_frac 합계 ≤ LEV_CAP (각 0.5*≤1.5). 숏 없음."""
    out = {}
    for a in ASSETS:
        exp = target_exposure(closes_by_asset.get(a, []))
        frac = WEIGHTS[a] * exp
        out[a] = {"target_frac": round(frac, 4),
                  "target_notional_usd": round(frac * equity_usd, 2),
                  "asset_exposure": round(exp, 3)}
    gross = sum(v["target_frac"] for v in out.values())
    assert gross <= LEV_CAP + 1e-9, f"invariant I2 violated: gross {gross}>CAP"  # 런타임 강제
    out["_gross_frac"] = round(gross, 4)
    return out

if __name__ == "__main__":
    # self-test: 합성 상승추세 -> 양의 익스포저, 하락추세 -> 0
    up = list(np.linspace(100, 300, 260))
    dn = list(np.linspace(300, 100, 260))
    print("UPTREND  BTC exposure:", target_exposure(up))
    print("DOWNTREND BTC exposure:", target_exposure(dn), "(must be 0.0)")
    print("decide($100):", decide({"BTC": up, "ETH": up}, 100))

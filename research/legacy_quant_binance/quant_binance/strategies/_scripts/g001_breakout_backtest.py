"""
G001 BTC Breakout Lottery — 30d 백테스트 검증 스크립트

목적: 사용자에게 G001 룰을 제안하기 전에 실제 BTC 30일 데이터로 검증.
"30~40% 승률, 비대칭 R" 같은 통설이 이 룰셋에서 실제로 나오는지 측정.

Forward-bar 시뮬레이션 (lookahead bias 있음 — TP/SL 어느게 먼저 맞는지는
intra-bar 정보 필요한데 OHLC 만 있으니 동시 도달 시 SL 우선 보수적 처리).
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[3]
HIST = REPO / "quant_runtime" / "historical" / "BTCUSDT"

# ====== 룰 파라미터 ======
TF_SIGNAL = "5m"
LOOKBACK_1H_HIGH = 12          # 직전 1h 12봉의 고점
VOL_MULTIPLIER = 1.5           # 5m 거래량 > 20봉 평균 × 1.5
VOL_LOOKBACK = 20
ATR_LEN_1H = 14
ATR_LOOKBACK_30 = 30           # 1h ATR > 30봉 평균 × 0.7
ATR_FILTER_MULT = 0.7
ATR_LEN_5M = 14                # 5m ATR (SL 계산용)
SL_ATR_MULT = 1.5
TP1_R = 1.5
TP2_R = 3.0
HOLD_BARS_5M = 48              # 4 시간
MAX_POS = 1
MAX_ENTRIES_PER_DAY = 3
COST_BPS_RT = 16.0             # 진입+청산 합산 (시장가 가정 — 보수적)

# ====== 데이터 로드 ======
def load(tf):
    return json.loads((HIST / f"{tf}.json").read_text())

bars_5m = load("5m")
bars_1h = load("1h")

# open_time 정렬
bars_5m.sort(key=lambda b: b["open_time"])
bars_1h.sort(key=lambda b: b["open_time"])

# 가격/거래량 float 변환
def f(b, k):
    return float(b[k])

# ====== 인디케이터 ======
def atr(bars, idx, length):
    """idx 번 봉까지 포함한 ATR(length)"""
    if idx < length:
        return None
    trs = []
    for i in range(idx - length + 1, idx + 1):
        if i == 0:
            tr = f(bars[i], "high_price") - f(bars[i], "low_price")
        else:
            high = f(bars[i], "high_price")
            low = f(bars[i], "low_price")
            prev_close = f(bars[i - 1], "close_price")
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return mean(trs)

def vol_mean_5m(bars, idx, n):
    if idx < n:
        return None
    return mean(f(bars[i], "base_volume") for i in range(idx - n + 1, idx + 1))

# 1h 봉의 고점 lookback
def max_high_1h_before(t_ms, k):
    """t_ms 이전 1h 봉 k 개의 high 최댓값"""
    # t_ms 보다 작은 마지막 1h open_time 의 인덱스
    idx = -1
    for i, b in enumerate(bars_1h):
        if b["open_time"] >= t_ms:
            break
        idx = i
    if idx < k:
        return None
    return max(f(bars_1h[i], "high_price") for i in range(idx - k + 1, idx + 1))

# 1h 봉의 ATR + ATR 평균
def atr_1h_filter(t_ms):
    idx = -1
    for i, b in enumerate(bars_1h):
        if b["open_time"] >= t_ms:
            break
        idx = i
    if idx < ATR_LEN_1H + ATR_LOOKBACK_30:
        return None, None
    cur_atr = atr(bars_1h, idx, ATR_LEN_1H)
    atrs = [atr(bars_1h, i, ATR_LEN_1H) for i in range(idx - ATR_LOOKBACK_30 + 1, idx + 1)]
    atrs = [a for a in atrs if a is not None]
    if not atrs:
        return cur_atr, None
    return cur_atr, mean(atrs)

# ====== 백테스트 ======
trades = []
in_position = None
entries_per_day = {}

for i in range(max(VOL_LOOKBACK, ATR_LEN_5M) + 1, len(bars_5m)):
    bar = bars_5m[i]
    t_ms = bar["open_time"]
    day = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).date().isoformat()

    # 1) 포지션 보유 중이면 청산 시뮬
    if in_position is not None:
        pos = in_position
        bars_held = i - pos["entry_idx"]
        high = f(bar, "high_price")
        low = f(bar, "low_price")

        # 보수적: low 가 SL 에 먼저 닿으면 SL (intra-bar 동시 가정 시 패자 우선)
        sl_hit = low <= pos["sl"]
        tp1_hit = high >= pos["tp1"] and not pos["tp1_done"]
        tp2_hit = high >= pos["tp2"]

        # 우선순위: SL 우선 (보수적). 단 TP1 이미 done 이면 TP1->진입가 SL 이동
        active_sl = pos["sl_after_tp1"] if pos["tp1_done"] else pos["sl"]
        sl_active_hit = low <= active_sl

        closed = False
        outcome_r = 0.0

        if sl_active_hit and tp2_hit:
            # 같은 봉에 둘 다 닿으면 보수적: SL 먼저
            # 현 사이즈 (TP1 후라면 50%) × −1R 또는 0R(진입가 SL)
            if pos["tp1_done"]:
                outcome_r += 0.5 * 1.5  # 이미 TP1 50% 잡았던 부분 그대로
                outcome_r += 0.5 * 0.0  # 나머지 50% 진입가 SL = 0R
            else:
                outcome_r += -1.0       # 전체 -1R
            closed = True

        elif sl_active_hit:
            if pos["tp1_done"]:
                outcome_r += 0.5 * 1.5
                outcome_r += 0.5 * 0.0
            else:
                outcome_r += -1.0
            closed = True

        elif tp2_hit:
            if pos["tp1_done"]:
                outcome_r += 0.5 * 1.5
                outcome_r += 0.5 * 3.0
            else:
                outcome_r += 0.5 * 1.5
                outcome_r += 0.5 * 3.0
            closed = True

        elif tp1_hit:
            pos["tp1_done"] = True
            pos["sl_after_tp1"] = pos["entry"]   # 진입가 SL

        elif bars_held >= HOLD_BARS_5M:
            # 시간 정지 — 종가로 청산
            close = f(bar, "close_price")
            r_at_close = (close - pos["entry"]) / (pos["entry"] - pos["sl"])
            if pos["tp1_done"]:
                outcome_r += 0.5 * 1.5
                outcome_r += 0.5 * r_at_close
            else:
                outcome_r += r_at_close
            closed = True

        if closed:
            cost_r_units = (COST_BPS_RT / 10000.0) * pos["entry"] / (pos["entry"] - pos["sl"])
            net_r = outcome_r - cost_r_units
            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": t_ms,
                "entry_price": pos["entry"],
                "sl": pos["sl"],
                "tp1": pos["tp1"],
                "tp2": pos["tp2"],
                "outcome_r_gross": outcome_r,
                "cost_r": cost_r_units,
                "net_r": net_r,
                "tp1_done": pos["tp1_done"],
                "bars_held": bars_held,
            })
            in_position = None
        # 포지션 보유 중이면 새 진입 금지 — continue
        continue

    # 2) 일일 진입 한도 체크
    if entries_per_day.get(day, 0) >= MAX_ENTRIES_PER_DAY:
        continue

    # 3) 진입 조건
    close = f(bar, "close_price")
    high_1h_lb = max_high_1h_before(t_ms, LOOKBACK_1H_HIGH)
    if high_1h_lb is None:
        continue
    if close <= high_1h_lb:
        continue

    vol_5m_mean = vol_mean_5m(bars_5m, i - 1, VOL_LOOKBACK)  # 직전 봉까지 평균
    if vol_5m_mean is None:
        continue
    if f(bar, "base_volume") < VOL_MULTIPLIER * vol_5m_mean:
        continue

    cur_atr_1h, mean_atr_1h = atr_1h_filter(t_ms)
    if cur_atr_1h is None or mean_atr_1h is None:
        continue
    if cur_atr_1h < ATR_FILTER_MULT * mean_atr_1h:
        continue

    # 4) ATR(14) on 5m → SL distance
    atr_5m = atr(bars_5m, i, ATR_LEN_5M)
    if atr_5m is None or atr_5m <= 0:
        continue

    sl_dist = SL_ATR_MULT * atr_5m
    sl = close - sl_dist
    tp1 = close + TP1_R * sl_dist
    tp2 = close + TP2_R * sl_dist

    in_position = {
        "entry_idx": i,
        "entry_time": t_ms,
        "entry": close,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp1_done": False,
        "sl_after_tp1": sl,
    }
    entries_per_day[day] = entries_per_day.get(day, 0) + 1

# 마지막 포지션 미청산 시 강제 종료
if in_position is not None:
    pos = in_position
    last = bars_5m[-1]
    close = f(last, "close_price")
    r_at_close = (close - pos["entry"]) / (pos["entry"] - pos["sl"])
    if pos["tp1_done"]:
        outcome_r = 0.5 * 1.5 + 0.5 * r_at_close
    else:
        outcome_r = r_at_close
    cost_r_units = (COST_BPS_RT / 10000.0) * pos["entry"] / (pos["entry"] - pos["sl"])
    trades.append({
        "entry_time": pos["entry_time"],
        "exit_time": last["open_time"],
        "entry_price": pos["entry"],
        "sl": pos["sl"],
        "tp1": pos["tp1"],
        "tp2": pos["tp2"],
        "outcome_r_gross": outcome_r,
        "cost_r": cost_r_units,
        "net_r": outcome_r - cost_r_units,
        "tp1_done": pos["tp1_done"],
        "bars_held": len(bars_5m) - 1 - pos["entry_idx"],
        "forced_close": True,
    })

# ====== 결과 집계 ======
n = len(trades)
if n == 0:
    print("진입 0건. 룰이 너무 엄격하거나 데이터에서 매칭 안 됨.")
else:
    wins = [t for t in trades if t["net_r"] > 0]
    losses = [t for t in trades if t["net_r"] <= 0]
    tp1_only = [t for t in trades if t["tp1_done"] and t["net_r"] > 0 and t["net_r"] < 1.5]
    tp2_full = [t for t in trades if t["net_r"] > 1.5]
    full_sl = [t for t in trades if t["net_r"] <= -0.9 and not t["tp1_done"]]

    total_r_net = sum(t["net_r"] for t in trades)
    total_r_gross = sum(t["outcome_r_gross"] for t in trades)
    avg_r_net = total_r_net / n
    avg_r_gross = total_r_gross / n

    print("=" * 60)
    print("G001 BTC Breakout Lottery — 30일 백테스트")
    print("=" * 60)
    print(f"기간: 2026-03-28 ~ 2026-04-27 (30 days, BTCUSDT 5m + 1h)")
    print(f"룰: 5m close > 직전 1h 12봉 고점 + 5m vol > 20봉 평균×1.5 + 1h ATR > 30봉×0.7")
    print(f"     SL = 1.5×ATR(14,5m), TP1 +1.5R(50%), TP2 +3R, 시간정지 4h")
    print(f"     Cost = 16 bps round-trip (시장가 가정)")
    print()
    print(f"진입 수:           {n}")
    print(f"  승 (net R > 0):  {len(wins)}  ({len(wins)/n:.1%})")
    print(f"  패 (net R ≤ 0):  {len(losses)}  ({len(losses)/n:.1%})")
    print(f"  TP2 도달 (≥1.5R 이상 누적): {len(tp2_full)} ({len(tp2_full)/n:.1%})")
    print(f"  TP1 만 잡고 SL: {len(tp1_only)} ({len(tp1_only)/n:.1%})")
    print(f"  풀 SL (TP1 미도달): {len(full_sl)} ({len(full_sl)/n:.1%})")
    print()
    print(f"누적 R (gross):    {total_r_gross:+.2f}")
    print(f"누적 R (net, 비용 후): {total_r_net:+.2f}")
    print(f"평균 R / 거래 (gross): {avg_r_gross:+.3f}")
    print(f"평균 R / 거래 (net):   {avg_r_net:+.3f}")
    print()
    print(f"평균 보유 봉:      {sum(t['bars_held'] for t in trades) / n:.1f} (×5m = {sum(t['bars_held'] for t in trades) / n * 5:.0f}분)")
    print(f"진입 빈도:         {n/30:.2f} / 일")
    print()
    # 자본 $50, 5x 레버리지, 노티오널 $250 가정 → −1R = −$2.5 (자본 5%)
    EQUITY = 50.0
    R_USD = EQUITY * 0.05  # 1R = 5% of equity (with 5x lev + 1.5×ATR stop)
    print(f"= $50 자본, 1R ≈ ${R_USD:.2f} (= 자본 5%) 가정 시:")
    print(f"  누적 PnL (gross): ${total_r_gross * R_USD:+.2f}")
    print(f"  누적 PnL (net):   ${total_r_net * R_USD:+.2f}")
    print(f"  최종 자본:        ${EQUITY + total_r_net * R_USD:.2f}")

# JSON 저장
out = REPO / "quant_binance" / "strategies" / "_scripts" / "g001_backtest_result.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "rules": {
        "lookback_1h_high": LOOKBACK_1H_HIGH,
        "vol_multiplier": VOL_MULTIPLIER,
        "vol_lookback": VOL_LOOKBACK,
        "atr_filter_mult": ATR_FILTER_MULT,
        "sl_atr_mult": SL_ATR_MULT,
        "tp1_r": TP1_R,
        "tp2_r": TP2_R,
        "hold_bars_5m": HOLD_BARS_5M,
        "cost_bps_rt": COST_BPS_RT,
    },
    "summary": {
        "trade_count": n,
        "wins": len([t for t in trades if t["net_r"] > 0]) if n else 0,
        "losses": len([t for t in trades if t["net_r"] <= 0]) if n else 0,
        "total_r_gross": sum(t["outcome_r_gross"] for t in trades) if n else 0,
        "total_r_net": sum(t["net_r"] for t in trades) if n else 0,
        "avg_r_net": (sum(t["net_r"] for t in trades) / n) if n else 0,
    },
    "trades": trades,
}, indent=2, default=str))
print(f"\n결과 저장: {out.relative_to(REPO)}")

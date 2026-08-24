# PB100 — NFI 룰 (NostalgiaForInfinityX, internal version v11.3.133, repo head 2026-04-28)

> Source: `https://raw.githubusercontent.com/iterativv/NostalgiaForInfinity/main/NostalgiaForInfinityX.py`
> File size: 1.6MB / 38,976 LOC. 본 문서는 head + grep 기반 추출 요약 (전체 dump X).
> 파생 변종 `NostalgiaForInfinityX2.py` (610KB) 도 존재하나 본 문서는 X 기준.

---

## 0. 기본 클래스 변수 (line 120-245)

```python
INTERFACE_VERSION = 3            # freqtrade strategy v3 (entry/exit semantics)
timeframe         = "5m"         # 메인 TF
res_timeframe     = "none"       # 추가 resample 안 씀
info_timeframe_15m = "15m"
info_timeframe_1h  = "1h"
info_timeframe_1d  = "1d"
has_BTC_base_tf   = True         # BTC 5m informative 사용
has_BTC_info_tf   = True         # BTC 1h informative 사용

minimal_roi   = {"0": 100.0}     # 사실상 ROI exit 비활성 → custom_exit 가 전부
stoploss      = -0.99            # 99% 까지 hold (custom_exit 가 책임)
trailing_stop = False            # built-in trailing 미사용
use_custom_stoploss = False      # custom_stoploss 함수 미정의 (없음)
process_only_new_candles = True
startup_candle_count = 480       # 480 캔들(5m → 40h) 워밍업
position_adjustment_enable = True  # DCA(rebuy) 사용
ignore_roi_if_entry_signal = True
exit_profit_only = False
exit_profit_offset = 0.01
profit_max_enabled = True
profit_max_threshold = 0.03
```

**주요 함의**:
- ROI/stoploss/trailing 모두 비활성 → 모든 exit 결정은 `custom_exit()` 한 함수가 처리. **포팅 시 가장 옮기기 어려운 부분** (수십 종 sell signal + 모드별 분기).
- `position_adjustment_enable=True` + `rebuy_pcts_n_*` → 손실 -3~-12% 단계마다 추가 매수 (피라미딩). $50 자본에는 위험, 비활성 권장.
- `can_short` 변수 없음 → **long-only 전략**. 양방향 변환은 사용자 책임.

---

## 1. 사용 인디케이터 (populate_indicators 5m TF, line 27519~27710)

### 라이브러리
```python
import talib.abstract as ta            # 주력 (RSI/EMA/SMA/CCI/MFI/ADX/ATR/MAMA/KAMA/STOCHF/STOCHRSI/LINEARREG/TRANGE)
import pandas_ta as pta                 # cti
import freqtrade.vendor.qtpylib.indicators as qtpylib  # bollinger_bands, heikinashi, crossed_below
from technical.indicators import RMI    # Relative Momentum Index
# 자체 헬퍼: chaikin_money_flow, williams_r, ewo, ema_vwma_osc, t3_average, vwap_bands, pivot_points, heikin_ashi
```

### 5m 시그널 후보 (모두 `populate_indicators` 결과)

| 카테고리 | 컬럼 | 파라미터 | 비고 |
|---|---|---|---|
| Momentum | `rsi_4`, `rsi_14`, `rsi_84`, `rsi_112` | RSI 다중 lookback | 14가 메인 트리거 |
| Trend EMA | `ema_8/12/13/16/20/25/26/50/100/200` | 10종 | 200이 regime 필터 |
| Trend SMA | `sma_15/28/30/75/200` | + `sma_200_dec_20/24` (하락 플래그) |
| Volatility BB | `bb40_2_low/mid/delta`, `bb20_2_low/mid/upp`, `bb20_3_low/mid/upp` | 40-2σ + 20-2σ + 20-3σ |
| Volatility KC | `kc_upperband_28_1`, `kc_lowerband_28_1` | sma_28 ± SMA(TRANGE,28) |
| BB derived | `bb20_width`, `bb20_delta`, `closedelta`, `tail` |  | squeeze/wick 측정 |
| Volume Flow | `cmf` (20), `mfi` (14) |  | `chaikin_money_flow` 자체 구현 |
| Williams %R | `r_14/24/32/64/96/480` | 6종 lookback |  |
| Oscillator | `cti` (length=20), `crsi` (3,2,100) | pta + 자체 합성 | crsi = (RSI3 + RSI(updown,2) + ROC100) / 3 |
| EMA-VWMA Osc | `ema_vwma_osc_32/64/96` | 자체 함수 |  |
| EWO | `ewo` (50, 200) | Elliott Wave Oscillator |
| Other osc | `cci` (20), `cci_25`, `rmi_17` (mom=4) |  |  |
| Stoch | `fastd`, `fastk` (STOCHF 5,3,0,3,0), `srsi_fk`, `srsi_fd` (STOCHRSI 15,20,2,2) |  |  |
| ADX | `adx` |  | 트렌드 강도 |
| Adaptive MA | `mama`, `fama` (0.25, 0.025), `kama` (84) | + `mama_diff` |  |
| Linreg | `hh_20`, `ll_20`, `avg_close_20`, `linreg_val_20` |  | 채널 ± linreg 잔차 |
| Heikin Ashi | `ha_open/close/high/low`, `ha_closedelta`, `ha_tail` |  | ClucHA 류 |
| ATR | `atr` (14), `trange`, `range_ma_28` |  | 변동성 |
| VWAP | `vwap_upperband/middleband/lowerband`, `vwap_width` (20, 1σ) |  | 자체 함수 |
| Volume MA | `vma_10/20`, `vol_osc`, `volume_mean_4/12/24` |  | volume 필터 |
| Range/Close max | `close_max_48`, `close_max_288`, `close_delta` |  | 로컬 천장 추적 |
| Pump protection | `tpct_change_0/2/12/144`, `hl_pct_change_36` |  | 윗꼬리·급등 차단 |

### 멀티-TF 인디케이터 (informative)
- **15m**: rsi_14, ema_12/26, bb20_2_low, close/open (중간 기간 dip 확인)
- **1h**: ema_200, sma_200, sma_200_1h, rsi_14, cti, crsi, r_14, r_480, ewo, mfi, ema_fast/slow_len, hl_pct_change_6/12/24/36/48, sup_level, btc_not_downtrend, not_downtrend
- **1d**: pivot/res1~3/sup1~3 (Fibonacci pivots), heikin_ashi smoothed, top_traded/top_grossing 메트릭
- **BTC**: 5m + 1h informative (`btc_pct_close_max_72_5m`, `btc_not_downtrend_1h`, `btc_tpct_change_144_5m`)

> **인디케이터 총 개수**: 5m 본 TF 만 약 80~100개 컬럼. 멀티-TF merge 후 (15m·1h·1d·BTC) 총 200~300개 컬럼.

---

## 2. 진입 시그널 — 총 74개 (`buy_condition_1` ~ `buy_condition_74`)

각 condition 은 `buy_protection_params[index]` (공통 ema/sma200 trend, safe_dips, safe_pump 필터) + condition 별 logic block (수십 줄, OR 조합 다수). 모든 condition 은 enable flag 로 on/off.

### 모드 분류 (코드 주석 + tag 기준)

| 모드 | 인덱스 범위 | 청산 핸들러 | 설명 |
|---|---|---|---|
| Semi swing | 1-30, 37-65 | `sell_signals` | 5m 메인. 로컬 dip 진입, 다양한 트렌드 컨텍스트 |
| Long mode | 31-36 | `sell_long_mode` | 며칠~주 단위 보유 |
| Quick mode | 58-65 (subset) | `sell_quick_mode` | 빠른 청산 |
| 15m semi swing | 41-57 | sell_signals | 15m TF 우위 트리거 |
| Rapid mode | 66-72 | `sell_rapid_mode` | 빠른 회전. mama/fama/kama + r_14 deep oversold |
| Half mode | 73-74 | `sell_half_mode` | 절반 사이즈, 슬롯 ≥2 여유 시만 |

### Top 10 — 주제별 대표 진입 시그널 (코드 주석 빈도 + 문법 패턴 기준)

> "동일 주제 condition 이 여러 개 있고, 그 중 첫 등장 인덱스 = 가장 자주 쓰이는 prototype" 가정.

| # | 시그널 | 인덱스 (대표) | 핵심 조건 (의사코드) | 우리 시스템 포팅 |
|---|---|---|---|---|
| 1 | **Local dip + RSI/MFI oversold** | #1 (Semi swing) | `(close - min(open[12]))/min > 0.027` AND `rsi_14<36` AND `mfi<36` AND `r_480_1h>-99` AND OR-블록 (cmf, ewo, sma_30, etc.) | 쉬움 (모든 인디 우리 보유) |
| 2 | **Local dip — short window momentum** | #2 (Semi swing) | rsi_14 + cti + bb20 lower 결합 OR-체인 | 쉬움 |
| 3 | **Local dip with downtrend filter** | #9, #13 (Downtrend) | sma_200_dec True AND r_96<-90 AND ema_26-ema_12 spread AND bb20_2_low cross | 쉬움 |
| 4 | **Strong uptrend + local dip** | #14, #19, #24 (Uptrend) | sma_200 rising AND ema_12>ema_26 AND close<sma_30*0.95 + RSI/cti oversold | 쉬움 |
| 5 | **Daily pivot bounce (swing)** | #22 (Swing) | close > sup1_1d (Fibonacci pivot) AND BTC not downtrend AND local dip | 중간 (pivot 자체 함수 포팅 필요) |
| 6 | **Cross-above (EMA/SMA breakout)** | #16 (Cross above) | ema_12 cross above ema_26 + RSI bounce | 쉬움 |
| 7 | **Deep buy (extreme oversold)** | #17 (Deep buy) | rsi_14<15-20 AND r_96<-95 AND cti<-0.95 AND bb20_3_low cross | 쉬움 |
| 8 | **15m TF dip with 1h uptrend** | #43, #47 (15m) | 15m bb20_2_low cross AND 1h ema slow rising AND 1h not downtrend | 중간 (15m informative merge) |
| 9 | **Squeeze momentum (KC inside BB)** | #64 (Squeeze) | bb20 inside kc_28 AND mama_diff<0 AND volume_mean spike | 중간 (squeeze metric) |
| 10 | **Rapid mode — mama/fama/kama trio** | #66 (Rapid) | `kama>fama` AND `fama>mama*0.981` AND `r_14<-61` AND `mama_diff<-0.025` AND `cti<-0.715` AND `rsi_84/112<60` AND OR-블록 | 중간 (MAMA/FAMA pandas-ta 매핑 확인 필요) |

### 공통 protection layer (모든 condition 자동 적용)
```
ema_fast > ema_200            (ema_fast_len 가변)
ema_slow_1h > ema_200_1h
close > ema_fast 또는 ema_slow_1h
sma_200 rising / sma_200_1h rising  (shift N 비교)
tpct_change_{0,2,12,144} < safe_dips_threshold
hl_pct_change_{6,12,24,36,48}_1h < safe_pump_threshold
btc_not_downtrend_1h
close vs daily pivot (close_over_pivot_type, close_under_pivot_type)
```
→ 사실상 "BTC가 죽지 않았고, 알트 자체도 일정 트렌드 위에 있고, 최근 윗꼬리가 작을 때만 진입". **공통 protection 만 옮겨도 NFI의 1차 효과의 절반은 재현됨.**

---

## 3. 청산 룰 (`custom_exit`, line 26991-)

### 분기 구조 (enter_tag 기반 라우팅)
```
if is_long_mode (tag 31-36):           sell_long_mode()
elif tag in {empty,58-65}:              sell_quick_mode()
elif tag in rapid_mode_tags (66-72):    sell_rapid_mode()
elif tag in half_mode_tags (73-74):     sell_half_mode()
else:                                    sell_signals()  # 기본
fallback:                                sell_stoploss()  # 손실 -X% 컷
fallback:                                sell over EMA200 / main profit targets
```

### 핵심 메커니즘
- **ROI 계단형**: NFI 는 freqtrade 의 `minimal_roi` 를 사용 안 함 (`{"0":100.0}` = 무한). 대신 `custom_exit` 안에서 `current_profit` + `max_profit` (트레일 고점) + `max_loss` (트레일 저점) + 5캔들 lookback 으로 동적 판정.
- **Custom stoploss**: `use_custom_stoploss=False`. 즉 `custom_stoploss()` 함수도 정의 안 함 (`def custom_stoploss` grep 0건). 손절은 `sell_stoploss()` 가 `custom_exit` 내부에서 처리.
- **Trailing stop**: built-in 미사용. 대신 mode 별 sell 함수가 `max_profit - current_profit > X` 류 trail-from-peak 조건을 자체 구현.
- **Profit max**: `profit_max_enabled=True`, threshold 0.03 → +3% 도달 후 회귀 시 청산.
- **Position adjustment (DCA)**: `position_adjustment_enable=True`. 손실 -3~-12% 단계마다 추가 매수 (mode 별 `rebuy_pcts_n_0~5`).

---

## 4. 멀티-TF 페어 (`informative_pairs`, line 27245-)

```python
informative_pairs = [(pair, "1h"), (pair, "1d"), (pair, "15m") for each pair]
                    + [("BTC/<stake>", "1d"), ("BTC/<stake>", "1h"), ("BTC/<stake>", "15m")]
```

- 알트 페어 각각: **5m (메인) + 15m + 1h + 1d** = 4 TF
- BTC 글로벌: **5m + 1h + 1d** (regime 필터)

---

## 5. 우리 환경 포팅 매핑

### pandas-ta 매핑

| NFI (TA-Lib) | pandas-ta | 비고 |
|---|---|---|
| `ta.RSI` | `pta.rsi` | 동일 |
| `ta.EMA/SMA` | `pta.ema/sma` | 동일 |
| `ta.CCI` | `pta.cci` | hlc3 source 옵션은 직접 계산 |
| `ta.MFI` | `pta.mfi` | 동일 |
| `ta.ADX` | `pta.adx` | DataFrame 반환 (`["ADX_14"]` 컬럼) |
| `ta.ATR` | `pta.atr` | 동일 |
| `ta.TRANGE` | `pta.true_range` | 동일 |
| `ta.STOCHF` | `pta.stoch` | 컬럼명 K/D 다름, 매핑 주의 |
| `ta.STOCHRSI` | `pta.stochrsi` | 동일 |
| `ta.LINEARREG` | `pta.linreg` | 동일 |
| `ta.MAX/MIN` | `df.rolling(N).max/min` | numpy 직접 |
| `ta.MAMA` | **미확인** (pta 에 없음) | 자체 구현 또는 talib 의존 — **포팅 시 #66 Rapid 영향** |
| `ta.KAMA` | `pta.kama` | 동일 |
| `qtpylib.bollinger_bands` | `pta.bbands` | 컬럼명 다름 (`BBL/BBM/BBU`) |
| `qtpylib.heikinashi` | `pta.ha` | 동일 |
| `qtpylib.crossed_below/above` | 직접: `(a.shift() > b.shift()) & (a < b)` | qtpylib 미사용 시 |
| `qtpylib.typical_price` | `(H+L+C)/3` 직접 | 1줄 |
| `pta.cti` | `pta.cti` | 그대로 |
| `RMI` (technical lib) | **미확인** | 자체 구현 필요 (Relative Momentum Index) |
| `chaikin_money_flow` | `pta.cmf` | 매핑 OK |
| `williams_r` | `pta.willr` | 동일 |
| `t3_average`, `ewo`, `ema_vwma_osc`, `vwap_bands`, `pivot_points` | **자체 구현 (NFI 내부 함수)** | 포팅 시 함수 단위 복사 |

### freqtrade-only 기능 → 우리 시스템 대체

| freqtrade 의존 | 대체 방안 |
|---|---|
| `IStrategy`, `populate_*` 시그니처 | 우리 `regime.py` 의 `decide(df_5m, df_15m, df_1h, df_1d, df_btc) → signal` 단일 함수 |
| `self.dp.get_pair_dataframe(pair, tf)` | 백테스트 엔진이 사전에 모든 TF 를 dict 으로 주입 |
| `merge_informative_pair` | `pd.merge_asof(df_5m, df_1h.add_suffix("_1h"), on="date")` |
| `Trade`, `LocalTrade`, `select_filled_orders` | 우리 backtester 의 `Position` 객체 |
| `custom_exit(trade, current_time, current_rate, current_profit, **kwargs)` | `should_exit(position, last_candle, prev5_candles) → bool` |
| `position_adjustment_enable` + DCA | **비권장** ($50 자본에서 마진콜 위험) — 단일 진입만 |
| `informative_pairs()` 등록 | `regime.py` 가 필요한 TF 명시 |
| `max_open_trades` slot 관리 | 우리 portfolio manager |

### Bitget USDT-perp 호환성
- NFI 는 `min_age_days=3` (신규 코인 필터) 외엔 거래소 비종속.
- BTC informative 는 Bitget 의 `BTC/USDT:USDT` perp 으로 매핑.
- `position_adjustment` (DCA) → Bitget perp 에서 가능하나 $50 + 5-10x 에선 권장 X.

---

## 6. 양방향 (long+short) 변환 가능성

### 평가
NFI 는 `can_short` 변수 부재 + 모든 condition 이 "local dip + oversold + uptrend filter" 패턴 → **본질적으로 long-only mean-reversion 전략**. 단순 inverse 변환은 위험.

### 변환 가능 시그널 (10개 중)

| # | 시그널 | inverse short 가능성 | 비고 |
|---|---|---|---|
| 1 | RSI/MFI oversold dip | 가능 (RSI>64 + MFI>64) | 함정: 알트 RSI 과매수는 종종 추가 상승 |
| 2 | Short window momentum dip | 가능 (mirror) | 동일 |
| 3 | Downtrend dip | **부적합** (이미 downtrend 가정 → short 시 pyramid) | 변환 X |
| 4 | Strong uptrend dip | **부적합** (uptrend 에서 long 진입) | 변환 X |
| 5 | Pivot bounce | 가능 (저항선 거부 short) | pivot res1/res2/res3 사용 |
| 6 | Cross above | 가능 (cross below short) | 데드크로스 단순 변환 |
| 7 | Deep oversold | 가능 (overbought: RSI>85, r_96>-5) | 매우 신중 |
| 8 | 15m dip + 1h uptrend | **부적합** | 변환 X |
| 9 | Squeeze | 가능 (방향 중립 → breakout 양방향) | breakout direction 추가 판정 필요 |
| 10 | Rapid mama/fama/kama | 가능 (mama>fama, mama_diff>+0.025 short) | mirror 가능 |

**변환 가능 비율: 6/10 = 60%** (단, 실측 검증 없이는 fwd-test 후 평가 필수)

### 추천 양방향 활용 패턴
1. NFI long signal **그대로 유지** + 신규 short signal 추가 (병렬 운영)
2. Short 트리거: **#1 mirror (RSI>70 + MFI>70 + r_96>-10)**, **#7 mirror (RSI>85)**, **#10 mirror (mama_diff>+0.025)** — 3종만 시작
3. BTC regime 필터 반전: `btc_not_uptrend_1h` 신규 정의 → short 시 BTC 가 죽고 있을 때만
4. $50 + 5-10x: short 1개당 진입 사이즈 long 의 50% (rapid liquidation 위험)

---

## 7. 구현 우선순위 (Phased Port)

| Phase | 범위 | 예상 LOC | 가치 |
|---|---|---|---|
| P1 (MVP) | 인디케이터 5m 본 TF 50종 + condition #1, #2, #17 (3개) + sell_quick_mode | ~600 | 1차 검증 |
| P2 | 멀티-TF (15m, 1h informative merge) + condition #14, #22, #43, #66 추가 | +400 | 멀티 TF 효과 측정 |
| P3 | 1d pivots + BTC global filter + 공통 protection layer 전체 | +300 | 진입 정확도 ↑ |
| P4 | sell_signals/sell_long_mode/sell_rapid_mode 전체 + custom_exit 라우팅 | +800 | 청산 quality |
| P5 | Short 변환 6종 추가 + 양방향 운영 | +400 | $50 도박 모드 |

> **Phase 별 backtest 게이트**: 각 phase 완료 시 strategy ID (예: PB100-S001 ~ PB100-S005) 로 등록 후 다음 phase 진행. 변수 1개 룰 준수.

---

## 8. 알려진 위험 / 함정

1. **DCA (position_adjustment_enable)**: 손실 단계마다 추가 매수 → $50 + 5-10x 에선 1번 깊은 dump 시 마진콜. **반드시 OFF.**
2. **`stoploss = -0.99`**: 99% drawdown 까지 hold 한다는 의미. custom_exit 가 작동 안 하면 사실상 무손절. 우리 시스템 포팅 시 **safety net 으로 hard stoploss -10~-15%** 별도 설정 필수.
3. **freqtrade 의 enter_tag 텍스트 분기**: 우리 시스템엔 enter_tag 개념이 없을 수 있음 → `signal_id` 컬럼으로 정수 ID 만 저장, 청산 함수가 ID 기반 라우팅.
4. **BTC informative 의존도**: NFI 의 ~30% protection 이 BTC 1h regime 의존. BTC pair 데이터 누락 시 시그널 90% 발생 안 함.
5. **MAMA/FAMA pandas-ta 미존재**: #66 Rapid mode 의 핵심. 자체 구현 또는 talib 의존 결정 필요.
6. **Long-only inverse 위험**: 알트 단기는 mean-reversion long 이 압도적으로 많은 이유 = 알트는 폭발적으로 오르고 천천히 빠짐. Short 변환 시 윗꼬리 청산 잦아 PnL 마이너스 가능성.

---

## 보고

```
[PB100-rules] NFI v11.3.133 (X.py 38976 LOC): indicators=80~100개(5m) + 멀티TF(15m/1h/1d/BTC) / 진입시그널=74종(top10 식별: dip×7, breakout×1, squeeze×1, rapid×1) / 청산=custom_exit 라우팅(long/quick/rapid/half/signals/stoploss 6모드, ROI/trailing/custom_stoploss 모두 비활성) / 멀티TF=[5m,15m,1h,1d]+BTC / short변환=60% (6/10개, 단 실증 검증 필수)
```

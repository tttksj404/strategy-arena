# Strategy Registry — quant_binance

> 단일 마스터 인덱스. 모든 전략(=파라미터 세트)을 ID·상태·핵심 가설·최신 성과 한 줄씩.
> **전략은 `overrides.json` 한 파일로 정의됨**. 코드 변경 없이 `STRATEGY_OVERRIDE_PATH=strategies/<ID>/overrides.json` 으로 갈아끼움.

## ⚠️ 검증 룰 (2026-04-28 추가)

새 전략을 "production candidate" / "winner" 라 부르기 전 **9-point checklist 100% PASS** 필수.
Memory: `feedback_quant_strategy_validation_process.md` 참조.

**현재 전 전략 retroactive 6축 정합성 평가**:

| 전략 | lottery | lev | 양방향 | ≥3/일 | 단기 | 빠검 | **합계** |
|---|---|---|---|---|---|---|---:|
| G050 | ❌ | ❌ | ❌ | ❌ | ❌ | ✓ | **1/6** |
| G058 | ❌ | △ | ❌ | ❌ | ❌ | ✓ | **1/6** |
| G059 | ❌ | ✓ | ❌ | ❌ | ❌ | ✓ | **2/6** |
| G004 | ✓ | ❌ | ❌ | ❌ | ❌ | ✓ | **2/6** |
| **G070** | ✓ | ✓ | ❌ | ❌ | ✓ | ✓ | **4/6 (top)** |

→ **모든 전략이 미달** (양방향 + ≥3건/일 둘 다 fail). G070 만 4/6 으로 가장 핏.
→ **9/9 PASS production candidate 없음** — 모두 DRAFT 상태
→ ≥3건/일 = alpha 와 양립 불가 (구조적 한계, G072 -363% 입증)
→ 양방향 = G071 별도 트랙 필요 (PB101 short 신호 마이닝)

## 사용 규칙 (반드시 지킬 것)

1. **새 전략 만들기**: `/strategy-new <ID> --base <prev_ID>` (또는 직접 `_template/` 복사)
2. **전략 평가**: `/strategy-eval <ID>` (백테스트 + 페이퍼라이브 자동 + card.md 결과 갱신 + 이 표 갱신)
3. **변수 1개 룰**: 새 전략은 직전 전략 대비 **단일 변수만** 변경. 다중 변수 변경 = 함정. 다중 변경이 필요하면 별도 ID 2개로 분리.
4. **Status 라벨**:
   - `live` — 실제 자본 배정 중
   - `paper` — 페이퍼 라이브 검증 중
   - `bt-only` — 백테스트만 통과, 라이브 미투입
   - `shelved` — 일시 보관 (다시 쓸 수도)
   - `dead` — 명백히 폐기. 가설·결론은 후속 전략에 상속됨

## 전략 비교 표

> 비고: `replay` 모드는 진입 의사결정만 평가 (PnL/승률/MDD/Sharpe N/A). closed-trade 메트릭 = `batch_backtest.py` (klines) 또는 페이퍼라이브 누적.

| ID | 가설 한 줄 | Status | 부모 | 변경 변수 | 거래 | 진입률 | Gross bps | Net bps | 승률 | Live PnL | 결론·다음 후보 |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| [S001](S001_baseline/card.md) | 현 기본 config 그대로 — 베이스라인 측정 | bt-only | — | (베이스) | 1,258 | 70% | **−1.16** | −17.16 | 35.9% | — | **신호 random 수준 + cost 못 이김. 진입 늘리기·레버리지 ↑ 모두 금지. 다음 후보: Universe=BTC 단독(S002), Cost 절감(S003), Holding 늘리기(S004)** |
| [G002](G002_mingogogo_ch1_72h/card.md) | PB001 CH1 (10-indicator 가중) × 72h horizon (3-day) | bt-only | S001 (PB001) | score_engine, holding, threshold, universe, decision_loop (5개, PB 신규) | 2,667 | — | **+237** | **+221** | **58.3%** | — | **🎯 Mingogogo 신호 = 진짜 alpha (3-day horizon 에서만). 374-day window. 다음: G003 universe 확장 검증, G004 lottery 변형** |
| [G003](G003_mingogogo_universe18/card.md) | G002 + universe 8→18 | **bull-regime only** ⚠️ | G002 | universe (1개) | **5,089** | — | +224 | **+208** | **59.1%** | — | ⚠️ **regime 의존성 발견 (G011): Q1+361 / Q2-26 / Q3+487 / Q4-21. Portfolio 시뮬 (G012): -$14.40/-28.8%. BTC 모멘텀 filter 무용 (G013). production X — paper-live $5 micro 필수** |
| [G004](G004_mingogogo_lottery_thr80/card.md) | G002 + threshold 70→80 | **bull-regime only** ⚠️ | G002 | entry_threshold (1개) | **40** | — | +994 | **+978** | **85.0%** | — | ⚠️ **분기별: Q1/Q2 무진입, Q3 35건 100%WR +1227, Q4 7건 0%WR -619. Q3 phenomenon. 라이브 X — Q3 회복기 detection 필요** |
| G011 robustness | quarterly split G003 검증 | — | G003 | — | — | — | — | — | — | — | ⚠️ **regime 의존 발견 (위 G003 비고 반영)** |
| G012 portfolio | $50 capital 5-pos 시뮬 | — | G003 | — | 312 | — | — | -$14.40 | — | — | ⚠️ **capacity 제약 시 -28.8%. 5089 후보 중 312 만 capture (6%). 사이즈/regime/우선순위 재설계 필요** |
| G013 regime gate | + BTC 7d momentum filter | dead | G003 | regime_filter | 3,889 | — | — | +213 | 59.5% | — | ⚠️ **BTC momentum 과 alt mean-reversion 비상관. filter 가 winner 만 잘라냄. 다른 regime metric 필요** |
| [G040](G040_vol_extreme_lottery/card.md) | Volatility ATR ratio ≥ 2.5/3.0 + hold 72h | bt-only | (신규) | 신호 클래스 신규 | 31~23 | — | +1475 | **+1459~+1524** | **100%** | — | 💎 **Q3 specialized lottery — 회복기 polynomial 진입 자동 감지. 100% WR but Q3 only. G041 보완 overlay** |
| [G041](G041_walk_forward_adaptive/card.md) | G003 + walk-forward adaptive gate (30일 net>0) | **OOS FAIL** ⚠️ | G003 | deployment_gate (1개) | 3,398 | — | — | +298 | 65.0% | — | ⚠️ **2022-2023 OOS 검증 실패 (n=792, mean=-166bps, WR=38%). 30일 lookback 이 LUNA/FTX 같은 빠른 regime 변화에 너무 느림. lookback fix → G050 참조** |
| [G050](G050_oos_validated_14d/card.md) ⭐⭐⭐ | G041 lookback 30d → **14d** (3-period validated) | **bt-only (paper-live ready)** | G041 | adaptive_lookback_days (1개) | **2,814 / 5,432 / 3,343** | — | — | OOS22+263 / OOS24+? / IS25+341 | 59~67% | — | **🎯🎯🎯 3-period robust ($55 max5/30% portfolio): OOS22-23 +70%/년 / OOS24-Q1.25 +135%/년 / IS25-26 +17%/년 = 평균 +74%/년 1560일 검증** |
| [G053](G053_combined_gate_dd/card.md) | G050 + DD safety net (gate14d + DD trigger) | bt-only (단독 trade-level 균등 but portfolio sim 음수) | G050 | dd_safety_net (1개) | 1,616 / 1,652 | — | — | OOS+213 / IS+202 | 60~62% | — | ⚠️ **trade-level OOS-IS gap 11bps (균등) but portfolio sim G050 보다 열등 — DD safety 가 winner 도 자름. G050 단독이 더 robust 발견** |
| [G055](G055_dynamic_size/card.md) ⭐⭐ | G050 + dynamic size (Kelly approx) | bt-only | G050 | size_pct_dynamic (1개) | 433 / 312 / 246 | — | — | — | — | — | **3-period: +112% / +220% / +46% (avg +125%/년). max5/dynamic 30~50%. G050 baseline 대비 +51%p** |
| [G058](G058_dynamic_conc/card.md) ⭐⭐⭐ | G050 + dynamic concurrency (3~8 by regime) | bt-only | G050 | max_concurrent_dynamic (1개) | 519 / 370 / 315 | — | — | — | — | — | **3-period: +163% / +252% / +146% (avg +186%/년). 2.4x peak leverage. hot regime 동시 winner skip 막음** |
| [G059](G059_combined_aggressive/card.md) ⭐⭐⭐⭐ | G055 + G058 combined (dynamic size × conc) | bt-only — TOP performer | G050 | 2 (variable-2 exception, parent 변형 결합) | 519 / 370 / 315 | — | — | — | — | — | **🚀 3-period: +268% / +413% / +251% (avg +306%/년). 4x peak leverage. $55→~$223/년. liquidation 위험 큼** |
| [G130](G130_winner_pattern_override/card.md) | live-ultra-aggressive override + winner 패턴 강제 | superseded by G131 | live-ultra-aggressive | universe 4종 + score≥70 + size 0.25 + lev 20x | 133 | — | — | +10.9bps | 44% | — | score=70 인 baseline. backtest 결과 G131 (score=75) 이 더 우수. G130 deprecated |
| [G131](G131_score75_locked/card.md) 🎯 | G130 + score_min 70 → 75 (backtest sweet spot) | superseded by G135 | G130 | mode_thresholds.futures_score_min (1개) | 76 (30d) / 57 (60d) | — | — | +33.8 / +22.5 bps | 51% / 51% | — | 4-coin + score≥75 = +2567/+1281 bps. G135 (score=76) 으로 marginal 개선 |
| [G135](G135_score76_refined/card.md) | G131 + score 75→76 (refined) | superseded by G150 | G131 | score_min | 55 (60d) | — | — | +24.9 bps | — | — | superseded — 사용자 인터뷰 답변 반영한 G150 이 trade 2.5배 + total +60% |
| [G150](G150_user_intent/card.md) 🎯⭐⭐⭐ | G135 + 사용자 인터뷰 4개 답변 (lev 20+size 0.40, max hold 72h, stop -35%, short X) | **paper-live ready** ⭐⭐⭐ | G135 | leverage + size + hold + stop (4개, 사용자 의도 lock) | **140 (60d)** | — | — | **+14.6 bps avg** | **50%** | — | **🎯⭐⭐⭐ 사용자 답변 직접 반영. 60d total +2050 bps (G135 +1369 대비 +50%). trade 수 2.5배 + WR +6pp. ~+500%/년 추정 (cost 디스카운트 후)** |
| [G160](G160_smart_hours/card.md) ⭐⭐ | G150 + UTC 12-13/15-17 entry gate (worst UTC 14 제외) | bt-only candidate | G150 | entry_hour_filter (1개) | 40 (60d est) | — | — | +54.7 bps | 55% | — | **G135 hour 분포 기반 추정. quality skew (G150 +14.6 → G160 +54.7 avg). UTC 14 -1058 worst 제외 효과. 60d 단일 분기 추정 → 다른 시기 OOS 필요** |
| [G161](G161_lottery_hours/card.md) ⭐⭐⭐ | G150 + UTC 13/16 only (extreme selective) | bt-only candidate | G150 | entry_hour_filter (1개) | 14 (60d est) | — | — | +180 bps | **79%** | — | **lottery extreme. UTC 13 (KST22) +1297 + UTC 16 (KST01) +1231 만 진입. 60d 14t / WR 79% / avg +180. ≥3건/일 미달 (0.23/day). G150 main + G161 lottery overlay 병행 가능** |
| [G132](G132_score80_1h/card.md) | G131 + score 75→80 (selective) | bt-only | G131 | score_min | 22 (60d) | — | — | +38.6 bps | 46% | — | n 작음, lottery 보조 |
| [G133](G133_score85_lottery/card.md) | G131 + score 75→85 (lottery extreme) | bt-only | G131 | score_min | 5 (60d) | — | — | +65.5 bps | 60% | — | jackpot 5/60d, 100% WR (n=5) |
| G134/G136/G137 lev/size 변형 | (deprecated) | — | G131 | leverage / size | — | — | — | — | — | — | backtest 영향 X (% return 기준). live 환경 자본 보호용 |
| [G139](G139_universe_expanded/card.md) | G131 + WIF/BONK universe 확장 | data 없음 | G131 | universe | — | — | — | — | — | — | WIF/BONK 데이터 fetch 필요 (현재 4-coin 만 가용) |
| [G140](G140_utc_filter/card.md) | G135 + UTC 06-09 entry gate | bt-only — REJECT (n부족) | G135 | entry hour filter (post-filter) | 12 (60d) | — | — | +31.8 bps | 66.7% | — | per-trade quality 우월 (WR +20pp, avg +12.9), trade 빈도 10배↓. 4h hold 환경에서 best hours = UTC 13/16 으로 시프트 |
| [G141](G141_hold1h/card.md) | G135 + holding 4h → 1h | bt-only — REJECT | G135 | holding_period (1개) | 126 (60d) | — | — | +1.3 bps | 40.5% | — | 1h hold = avg ≈ cost. 4h hold 가 단연 우월 |
| [G143](G143_short_enabled/card.md) | G135 + short_disabled false | bt-only — NEUTRAL | G135 | short flags (3개) | 127 (60d) | — | — | +17.9 bps | 45.7% | — | short slot 1건만 추가. extra_score_floor 5.0 도 빡빡 — G146 별도 short curve 검토 |
| [G144](G144_combo_utc_1h/card.md) | G140 + G141 combo | bt-only — REJECT | G135 | UTC + hold (post-filter) | 12 (60d) | — | — | +6.2 bps | 33.3% | — | UTC filter 강점이 1h hold 와 anti-synergy |
| [G070](G070_lottery_thr80_5x/card.md) 🎰 | **LOTTERY core** — G004 + 5x lev + 24h hold + ATR guard | bt-only (paper-live G070 active) | G004 | leverage + holding (2개) | 21 / 34 / 12 | — | — | OOS22+162 / OOS24+563 / IS25+94 | **75-88%** | — | **🎰 LOTTERY 정답: 거래당 +8~21%, WR 75-88%, 3-period avg +167%/년. 1주 1-2회 발화 (≥3/일 미달, alpha-frequency trade-off 입증). 사용자 5/6 컨텍스트 충족** |
| [G185](G185_size40_100usd/card.md) 💰🎰⭐⭐⭐ | G070 + size 0.30→0.40 ($100 capital lottery scaler) | **paper-live Oracle Cloud (active)** | G070 | size_pct_per_trade (1개) | **20 / 34 / 12** | — | — | OOS22+199 / OOS24+751 / IS25+126 | **85.0% / 88.2% / 75.0%** | live | **WR 84.9% / 연 $343 / 월 $28.55. 2026-04-29 Oracle 배포 (systemd g185-emulator)** |
| [G186](G186_size45_100usd/card.md) 💰⭐⭐⭐ | G185 + size 0.40→0.45 | **paper-live Oracle Cloud (active)** | G185 | size_pct_per_trade (1개) | 66 (3-period) | — | — | weighted +$1648 | **84.9%** | live | **PASS 9-point: WR 84.9% / 연 $385 / 월 $32.12. 2026-04-29 배포 (systemd g186-emulator). 사용자 월 $30 목표 cleanly pass** |
| [G187](G187_lev6_100usd/card.md) 💰⭐⭐⭐⭐ | G185 + leverage 5x→6x | **paper-live Oracle Cloud (active, TOP)** | G185 | leverage (1개) | 66 (3-period) | — | — | weighted +$1757 | **84.9%** | live | **🏆 G18x batch TOP — WR 84.9% / 연 $411 / 월 $34.27. 2026-04-29 배포. liquidation buffer 6x lev = -16.7% intra-bar** |
| [G188](G188_hold48_100usd/card.md) | G185 + holding 24h→48h | bt-only **REJECT** | G185 | holding_period_bars (1개) | 65 (3-period) | — | — | weighted +$1240 | **75.4%** | — | ❌ 9-point FAIL: 연 $290 < $300 / WR 75.4%. hold 늘리면 alpha decay 입증 |
| [G189](G189_thr85_100usd/card.md) | G185 + threshold 80→85 | bt-only **REJECT** | G185 | entry_threshold (1개) | 1 (3-period) | — | — | +$16 | 100% | — | ❌ 9-point FAIL: n=1 통계적 무의미. threshold 85 너무 빡빡 |
| [G190](G190_size45_lev6_100usd/card.md) 💰⭐⭐⭐⭐ | G186+G187 결합 (size 0.45 + lev 6x) | **paper-live Oracle (active)** | G185 | size + lev (2개, 결합 exception) | 66 (3-period) | — | — | +$1977 std / +$1963 stress | **84.9%** | live | **🏆 robust PASS std16+stress24. 연 $463 / 월 $38.55. peak notional 13.5x. 2026-04-29 배포** |
| [G191](G191_lev6_conc8_100usd/card.md) 💰⭐⭐⭐⭐⭐ | G187 + max_concurrent 5→8 | **paper-live Oracle (active, TOP)** | G187 | max_concurrent (1개) | 83 (3-period) | — | — | +$2272 std / +$2256 stress | **87.9%** | live | **🏆🏆 ALL-TIME TOP — robust PASS. 연 $532 / 월 $44.29 / WR 87.9% / n=83. peak notional 19x. 2026-04-29 배포** |
| [G192](G192_lev6_atr6_100usd/card.md) 💰⭐⭐⭐ | G187 + atr_guard 8%→6% | **paper-live Oracle (active)** | G187 | atr_volatility_guard (1개) | 66 (3-period) | — | — | +$1686 std / +$1674 stress | **84.9%** | live | **robust PASS. 연 $395 / 월 $32.88. tighter risk filter (high-vol 알트 더 차단). 2026-04-29 배포** |
| [G193](G193_lev6_thr78_100usd/card.md) | G187 + threshold 80→78 | bt-only **REJECT** | G187 | entry_threshold (1개) | 174 (3-period) | — | — | +$1115 | **64.4%** | — | ❌ 9-point FAIL: WR 64.4% < 70%. n 2.6x 늘었지만 quality 급락 (G072 패턴 재현, alpha-frequency tradeoff) |
| [G194](G194_lev6_hold16_100usd/card.md) | G187 + holding 24h→16h | bt-only **REJECT** | G187 | holding_period_bars (1개) | 66 (3-period) | — | — | +$1176 | **80.3%** | — | ❌ 9-point FAIL: 연 $275 < $300, IS25 음수. hold 짧으면 alpha decay (G141 1h fail 패턴 약화 버전) |
| [G210](G210_g191_drop_dead/card.md) 💰⭐⭐⭐⭐⭐ | G191 + universe 18→15 (WIF/LTC/BTC 제외) | **paper-live Oracle (active, NEW TOP)** | G191 | universe (1개, dead weight 제외) | 81 (3-period) | — | — | +$2291 std / +$2276 stress | **90.1%** | live | **🏆🏆🏆 NEW ALL-TIME TOP — robust PASS. 연 $536 / 월 $44.67 / WR 90.1% (G191 대비 +2.2pp). 같은 alpha 에서 dead weight 제거 = pure improvement. 2026-04-29 배포** |
| [G220](G220_g191_top10/card.md) 💰⭐⭐⭐⭐ | G191 + universe TOP 10 only | **paper-live Oracle (active)** | G191 | universe (1개, top10 concentration) | 66 (3-period) | — | — | +$1920 std / +$1907 stress | **90.9%** | live | **robust PASS. WR 90.9% (G191 대비 +3.0pp) 가장 높음. 단 n=66 (G191 83 → 66, 20% drop). monthly $37.44. quality > quantity. 2026-04-29 배포** |
| G230 (top5) | G191 + universe TOP 5 only | bt-only **REJECT** | G191 | universe (1개) | 34 | — | — | +$1038 | 88.2% | — | ❌ FAIL: 연 $243 < $300. 너무 좁아 alpha 시점 놓침 (n 60% drop). top10 (G220) 이 sweet spot |
| G072 freq (thr60/12h/3x) | thr60 으로 ≥3/일 시도 | dead | G050 | threshold + hold (2개) | 4375/2981/2282 | — | — | -363 / -255 / -562 | 47% | — | ⚠️ **빈도 ↑ → alpha 사라짐 (모든 period 음수). 사용자 ≥3/일 목표는 lottery 와 양립 불가 입증** |
| [G007](G007_intra_bar_NEGATIVE/card.md) | G003 + intra-bar TP/SL (안전장치 시도) | **dead** | G003 | exit_logic (1개) | 1,609 | — | −9 | **−25** | 36.7% | — | ⚠️ **NEGATIVE — TP/SL 자체가 alpha 파괴. SL 63% hit → winner 미리 stop-out. 결론: G003 = 레버리지 X / spot or 1x / size ≤10%** |
| [G010](G010_short_inversion_NEGATIVE/card.md) | G003 + 단순 short inversion | **dead** | G003 | direction (1개) | 1,565 | — | −19 | **−35** | 36.2% | — | ⚠️ **NEGATIVE — Mingogogo CH1 = mean-reversion 본질, 단순 invert 부적합. Short 전략은 PB101/PB103 별도 트랙 필요** |

## 사고 사이클 (반복할 것)

```
1. 가설 1줄 (정량 예측 포함)
   예: "거래량 필터 1.5x → 1.2x 완화 시 진입 빈도 3배, 승률 5%p↓ 예상"
2. 변수 1개만 변경 → 새 ID 발급 (S002 등)
3. /strategy-eval <ID>  →  백테스트 + 페이퍼 동시 평가, 결과 자동 기록
4. card.md 의 "결론" 섹션에 한 줄 누적
5. 결론 위에서 다음 가설 → 다음 ID
```

**규칙 어김 = 학습 누적 안 됨**. 같은 전략을 다른 이름으로 짓거나 (변수 여러 개 동시 변경) 같은 함정 반복.

## 참고

- **자본 컨텍스트**: $50 자본 + 도박성 OK + 거래당 큰 수익 추구.
  → 진입 빈도 ≥ 3건/일, 거래당 expectancy +0.5R↑, 5~10x 레버리지, 롱·숏 양방향 권장 (CLAUDE.md "처방 A" 참조).
- **베이스라인 (S001)** 은 변경 금지. 측정 기준선 역할.
- **runs/ 폴더**: 각 전략의 평가 결과 요약 JSON 누적. 원본 raw 트레이스(`decisions.jsonl` 등)는 `quant_runtime/` 또는 `iCloudDrive/quant_archive/quant_runtime/` 에 별도 보관.
- **`_playbook/`**: G-시리즈 전략 발급의 인용 출처. 신뢰도 ≥3.0 인 PB만 G-전략 발급 가능. master 인덱스: [_playbook/PLAYBOOK.md](_playbook/PLAYBOOK.md). 현재 PB001 (Mingogogo 8채널, 신뢰도 3.2), PB100 (freqtrade NFI, 신뢰도 4.6) 등록.

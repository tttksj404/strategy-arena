# G010 — G003 + 단순 short inversion (NEGATIVE result)

## Status: dead — 학습용 보존

부모: G003
변경 변수: **direction** (long → short, CH1 score logic invert)

## 가설

PB001 CH1 의 10지표 (RSI/MFI/Stoch/CCI/W%R/BB%B oversold detection) 를 invert 하면 (overbought detection), short 신호로도 작동할 것이다.

## 결과 (NEGATIVE)

| | side | n | net bps | WR | SL hit% |
|---|---|---:|---:|---:|---:|
| G003 (long) | long | 5089 | **+208** | **59.1%** | — |
| G010 (short, intra-bar TP/SL) | short | 1565 | **−35** | 36.2% | 63.6% |

(intra-bar 로 측정 — close-to-close short 은 미실험)

## 결론

**Mingogogo CH1 은 본질이 mean-reversion (oversold bounce) → long-only**:
- 10지표 oversold detection 의 핵심 thesis = "바닥권 탈출 후 회귀"
- 크립토는 long-bias drift 가 있어 mean-reversion 이 long 측에서만 EV 양수
- "세력 매집" (CH6) 도 본질 long 흐름
- Invert 한 short score 는 단순 "overbought" 검출에 그치고 squeeze hunting / distribution pattern 미포함

→ **PB001 으로 short 시도는 부적합**. 별도 short 알고리즘 필요:
- PB101 (Passivbot) 의 short side 룰 차용
- 또는 별도 마이닝 (squeeze 기반, funding extreme 기반 — PB103 차용)

## 의의

사용자 "양방향" 컨텍스트는 single-PB 단순 invert 로 충족 X. 다음:
- G013 (가칭) = PB101 Passivbot short 룰 별도 포팅
- G014 (가칭) = PB103 funding 극단 short 시그널 차용
- G003 (long) + 별도 short 전략 = portfolio 양방향

## 변경 이력

| 날짜 | 사유 |
|---|---|
| 2026-04-28 | G003 변형 (direction). 단순 invert 실패. dead. 학습용 보존. short 전략은 PB101/PB103 별도 트랙 |

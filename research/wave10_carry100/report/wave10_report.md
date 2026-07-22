# Wave-10 리포트 -- $100 자본 반노셔널 캐리 ($90 활성자본)

사전등록: research/wave10_carry100/SPEC.md. 배경: wave-8은 W2c(델타중립 숏퍼프+롱현물, 7d 펀딩 APR>15% 진입/7.5% 청산)를 4쌍 풀사이즈(top4 균등분할, 레그당 최대 100%)로 판정했고 gross notional이 활성자본의 2배($180 > $90)라 $100에서 실행 불가라고 결론지었다. wave-10은 신호·비용·체결 규약은 그대로 두고 **사이징(레그당 고정 비중)과 동시 쌍 수만** 바꿔, gross를 활성자본($90) 이하로 강제한 4개 구성이 나머지 강건성 게이트(MC/블록셔플/전기간수익/OOS)까지 통과하는지 검증한다.

자본 규약: 총자본 $100 / 현금버퍼 10% / 활성자본 $90 / 최소주문 $5.00. 비용: 메이커 0.02%/레그 + 슬리피지(메이저 1bp/알트 3bp), 펀딩 8h 실적립, 신호는 t종가 확정 -> t+1시가 체결 (룩어헤드 없음). 신호·유니버스·체결 로직은 research.wave1.fam_funding (funding_score/carry_position/load_markets) 임포트 재사용, 비용 상수는 research.wave2.funding.W2_MAKER_FEE_RATE + research.wave1.costs.slippage_rate 임포트 재사용.

## 구성 (사전등록 4개, 사후 추가 없음)

| Config | 쌍 수 | 레그당 비중 | 레그 $ (@ $90 활성자본) | gross $ | gross 배수 | 진입 임계 | 정의 |
|---|---:|---:|---:|---:|---:|---:|---|
| C1 | 1 | 50% | $45.00 | $90.00 | 1.00x | 15% APR | 1 pair, 50% active capital per leg ($45/$45 @ $90 active), gross 1.0x |
| C2 | 1 | 40% | $36.00 | $72.00 | 0.80x | 15% APR | 1 pair, 40% active capital per leg ($36/$36 @ $90 active), gross 0.8x buffer |
| C3 | 2 | 25% | $22.50 | $90.00 | 1.00x | 15% APR | 2 pairs, 25% active capital per leg each ($22.5/$22.5 x2 @ $90 active), gross 1.0x |
| C4 | 1 | 45% | $40.50 | $81.00 | 0.90x | 25% APR | 1 pair, 45% active capital per leg ($40.5/$40.5 @ $90 active) + entry threshold literally 25% APR per spec text (higher than W2c's 15% baseline -- see module docstring; mechanically LOWERS utilization, contradicting the spec's own 'raise utilization' intent, implemented as-written and flagged, not corrected) |

## C4 스펙 불일치 참고사항 (정직성 고지, 수치 보정 없음)

원 지시문: "C4: 1쌍, 레그당 45% + 진입임계 완화 25%APR(가동률↑ 시도)". W2c/기본 임계값은 15% APR이며, 이 코드베이스의 기존 용례(W2b: W2a의 8%APR -> 5%APR을 "가동률↑"로 명시, research/wave2/SPEC.md)에서 "완화"는 **더 낮은** 임계값을 의미한다. 25%APR은 15%APR보다 **높으므로** 기계적으로 진입 조건이 더 까다로워져 가동률이 낮아진다 -- 지시문의 수치(25%)와 의도(가동률↑) 문구가 서로 모순된다. 사전등록 수치를 사후에 임의로 고치는 것 자체가 이 wave의 금지 사항이므로, **지시된 수치(25% APR)를 문자 그대로 구현**하고 그 결과(대개 가동률이 W2c 기준선보다 낮아짐)를 정직하게 아래 표에 반영했다. 의도대로 "가동률을 높이는" 실험이 필요하다면 임계값을 15%APR보다 낮춰(예: 10%APR 이하) 재등록해야 한다.

## 게이트 결과 (A 실행가능성 / B MC 1e4부트스트랩 / C 블록셔플90일MDD / D 전기간비용후수익 / E 휴면기OOS)

| Config | A 실행가능성 | B MC(p05/ruin) | C 블록MDD p95 | D 전기간수익 | E OOS(휴면기) | Overall | 실패/라벨 사유 |
|---|---|---|---|---|---|---|---|
| C1 | PASS (레그$45.00, gross1.00x) | PASS (p05=$130.31, ruin=0.00%) | PASS (2.29%) | PASS (59.38%) | PASS (return=0.21%, trades=2) | **PASS** | - |
| C2 | PASS (레그$36.00, gross0.80x) | PASS (p05=$123.64, ruin=0.00%) | PASS (1.82%) | PASS (45.40%) | PASS (return=0.17%, trades=2) | **PASS** | - |
| C3 | PASS (레그$22.50, gross1.00x) | PASS (p05=$135.99, ruin=0.00%) | PASS (1.09%) | PASS (54.92%) | PASS (return=0.11%, trades=2) | **PASS** | - |
| C4 | PASS (레그$40.50, gross0.90x) | PASS (p05=$127.85, ruin=0.00%) | PASS (2.05%) | PASS (52.78%) | UNTESTED_IN_OOS (무포지션) | **UNTESTED_IN_OOS** | 휴면 |

## 펀딩 레짐별 $100 기준 연환산 기대수익률

고펀딩기 = 2020/2021/2024 (실측 이력 슬라이스, W2c 자체 연도별 수익도 이 3개 연도가 최고치). 저펀딩기(현재) = OOS 2025-10-01~데이터 종료(2026-07-14) (실측). "$100기준 연이익" = 활성자본($90) x 해당 구간 연환산수익률 (현금버퍼 $10은 무이자 대기 가정, 보수적).

| Config | 2020 연환산 | 2021 연환산 | 2024 연환산 | 고펀딩기 평균(연환산) | 고펀딩기 평균 $100기준 연이익 | 현재(저펀딩 OOS) 연환산 | 현재 $100기준 연이익 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | 13.41% | 21.36% | 15.73% | 16.84% | $15.15 | 0.27% | $0.24 |
| C2 | 10.61% | 16.76% | 12.53% | 13.30% | $11.97 | 0.22% | $0.20 |
| C3 | 12.19% | 23.55% | 9.95% | 15.23% | $13.71 | 0.14% | $0.13 |
| C4 | 11.89% | 18.94% | 15.42% | 15.42% | $13.88 | 0.00% | $0.00 |

## 판정

- 전체 게이트(A-D) 통과: **4/4** (그중 OOS까지 실측 확인된 완전 PASS: **3/4**)
- PASS: C1, C2, C3
- UNTESTED_IN_OOS (A-D 통과, 휴면기 무포지션이라 OOS 실측 불가): C4
- FAIL: 없음

- wave-8 판정(4쌍 풀사이즈, gross $180 > 활성자본 $90 -> 전량 실행 불가)과 달리, wave-10의 4개 구성은 모두 사전 설계 단계에서 gross <= 활성자본이 되도록 사이징했다 (게이트 A는 사실상 설계로 보장됨 -- 진짜 판정은 B/C/D/E에서 갈린다).

## 모델링 노트

- 사이징 규칙 변경만 허용: 신호(funding_score/carry_position), 유니버스(wave-1 40심볼 eligible universe), 체결 타이밍(t종가 신호 -> t+1시가 체결)은 W2c와 동일 임포트. 유일한 차이는 랭킹 상위 K(top_k=쌍 수)까지의 각 심볼에 1/len(ranked) 대신 **고정 레그 비중**을 배정하는 것 (research/wave10_carry100/engine.py의 run_fixed_fraction_portfolio).
- 델타중립: 스팟 롱과 퍼프 숏이 동일한 weight 값 하나로 함께 구동되므로(intraday = spot_ret - perp_ret + funding, weights 곱) 구조적으로 델타중립이 보장된다. tests/test_wave10_engine.py가 가격이 움직이되 베이시스가 0인 합성 시장으로 이를 회귀 검증한다.
- MC/블록셔플 방법론은 research/wave8_capital/run_capital100.py의 _simulate_mc/_block_shuffle과 동일 (경로 수, $100/$90/$10 기준, 90일 블록) -- wave-8 판정과 비교 가능하도록 유지.
- 이 리포트는 paper 백테스트 결과이며 실계좌 주문을 실행하지 않는다. 메이커 체결률은 실측이 아닌 가정이다 (W2c와 동일한 한계, research/wave2/SPEC.md).


# Paper forward-validation status

생성 시각: `2026-07-16T06:24:07.606320+00:00`

실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.
메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.
펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.

| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |
|---|---:|---|---:|---|
| W2c | 299.9600 | DEXEUSDT spot long 100.00 USDT @ 36.95<br>DEXEUSDT perp short 100.00 USDT @ 37.07 | 0.000000 | 2026-07-16 |
| F1e | 299.8800 | BTCUSDT spot long 300.00 USDT @ 64848<br>BTCUSDT perp short 300.00 USDT @ 64850.9 | 0.000000 | 2026-07-16 |
| W3c | 299.9914 | SYNUSDT perp long 5.02 USDT @ 0.2363<br>TLMUSDT perp long 3.15 USDT @ 0.001603<br>MMTUSDT perp long 7.48 USDT @ 0.1695<br>STGUSDT perp short 14.96 USDT @ 0.1413<br>HOMEUSDT perp short 7.75 USDT @ 0.01475<br>EPICUSDT perp short 4.66 USDT @ 0.4337 | 0.000000 | 2026-07-16 |
| W3d | 300.0000 | 현금 | 0.000000 | 2026-07-16 |

## 후보별 신호

- `W2c`: W2c carry selected: DEXEUSDT; 최근 손익 -0.040000 USDT; maker fee 0.040000 USDT
- `F1e`: F1e carry selected: BTCUSDT; 최근 손익 -0.120000 USDT; maker fee 0.120000 USDT
- `W3c`: W3c weekly momentum targets: SYNUSDT:long, TLMUSDT:long, MMTUSDT:long, STGUSDT:short, HOMEUSDT:short, EPICUSDT:short; 최근 손익 -0.008604 USDT; maker fee 0.008604 USDT
- `W3d`: W3d weekly momentum targets: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT

원장 경로: `research/paper/ledger/paper_ledger.jsonl`

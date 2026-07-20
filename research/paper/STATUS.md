# Paper forward-validation status

생성 시각: `2026-07-20T00:11:38.530609+00:00`

실주문: **금지**. 주문 엔드포인트·API 키·서명 기능을 사용하지 않는다.
메이커 체결가정: 최신 공개 1D 바의 종가를 mid 추정 체결가로 사용하고, 진입·청산 각 leg에 0.02% maker fee를 적용하며 슬리피지는 0으로 둔다.
펀딩: 보유 perp notional × 공개 funding rate × 경과시간/8h로 가상 적립한다. 양수 funding에서 perp short는 수취한다.

| 후보 | 가상 에쿼티(USDT) | 오픈 포지션 | 누적 펀딩(USDT) | 최근 실행일 |
|---|---:|---|---:|---|
| W2c | 299.9200 | 현금 | 0.000000 | 2026-07-20 |
| F1e | 300.4680 | BTCUSDT spot long 299.88 USDT @ 64894<br>BTCUSDT perp short 299.88 USDT @ 64805.4 | 0.164667 | 2026-07-20 |
| W3c | 303.2357 | DEXEUSDT perp long 7.72 USDT @ 34.98<br>SYNUSDT perp long 5.89 USDT @ 0.2148<br>KAITOUSDT perp long 9.72 USDT @ 0.9395<br>HOMEUSDT perp short 5.35 USDT @ 0.00743<br>PARTIUSDT perp short 12.49 USDT @ 0.0307<br>ROBOUSDT perp short 12.87 USDT @ 0.01205 | -0.131761 | 2026-07-20 |
| W3d | 300.0000 | 현금 | 0.000000 | 2026-07-20 |

## 후보별 신호

- `W2c`: W2c carry selected: cash; 최근 손익 -0.040000 USDT; maker fee 0.040000 USDT
- `F1e`: F1e carry selected: BTCUSDT; 최근 손익 0.587954 USDT; maker fee 0.000048 USDT
- `W3c`: W3c weekly momentum targets: DEXEUSDT:long, SYNUSDT:long, KAITOUSDT:long, HOMEUSDT:short, PARTIUSDT:short, ROBOUSDT:short; 최근 손익 3.244262 USDT; maker fee 0.015263 USDT
- `W3d`: W3d weekly momentum targets: cash; 최근 손익 0.000000 USDT; maker fee 0.000000 USDT

원장 경로: `research/paper/ledger/paper_ledger.jsonl`

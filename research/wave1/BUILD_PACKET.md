# BUILDER PACKET — wave1 리서치 파이프라인 구현

너는 빌더다. `research/wave1/SPEC.md`를 먼저 읽고, 아래 계약대로 파이프라인을 **코드로만** 구현하라. 네 샌드박스에서 네트워크 호출·데이터 다운로드는 하지 마라(오케스트레이터가 실행한다). 단, 합성 데이터 단위테스트는 작성하고 pytest로 돌려 통과시켜라.

## 환경
- Python 3.12, 설치됨: pandas 3.0.3, numpy, requests. 그 외 서드파티 금지(stdlib OK).
- Windows. 경로는 전부 `pathlib`, 인코딩 명시 `utf-8`.
- 모듈당 ≤ 400줄, 함수 단위 테스트 가능하게.

## 파일 배치 (전부 `research/wave1/` 아래)
```
common.py        # 캐시 IO(parquet 대신 csv.gz), ts 유틸, 무결성 체크
fetch_binance.py # klines(spot/fapi)+fundingRate 페이지네이션 페처 → cache/
fetch_bitget.py  # v2 candles/history-candles/history-fund-rate 페처 → cache/
costs.py         # 수수료·슬리피지·펀딩 부과 (SPEC 비용 모델)
backtest.py      # 이벤트 루프: 시그널 t종가 → t+1시가 체결, 스탑 처리, 펀딩 8h 부과, 에쿼티 곡선
fam_funding.py   # F1a~F1f (SPEC 그리드)
fam_tsmom.py     # F2a~F2f
fam_session.py   # F3-H1/H2/H3 (통계 분석 위주 + 비용후 시뮬)
gates.py         # 게이트 평가기: 지표 계산 + PASS/FAIL 표 (아래 정의)
run_wave1.py     # CLI: --stage fetch|run|gates|report (스테이지별 실행, 캐시 재사용)
tests/           # pytest: test_no_lookahead.py, test_costs.py, test_funding_sign.py, test_donchian.py, test_voltarget.py
```

## API 실측 포맷 (프로브 완료 — 이대로 파싱)
- Binance fapi klines: `GET https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=1500&startTime=<ms>` → `[[openTime,o,h,l,c,vol,closeTime,quoteVol,...],...]` 오름차순, startTime 페이지네이션.
- Binance spot: `GET https://api.binance.com/api/v3/klines` 동일 포맷.
- Binance funding: `GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=X&startTime=<ms>&limit=1000` → `[{symbol,fundingTime,fundingRate,markPrice},...]` 오름차순. BTC 최초 1568102400000.
- Binance 거래소정보: `GET https://fapi.binance.com/fapi/v1/exchangeInfo`, 24h 볼륨: `GET https://fapi.binance.com/fapi/v1/ticker/24hr` (quoteVolume).
- Bitget candles: `GET https://api.bitget.com/api/v2/mix/market/candles?symbol=SPYUSDT&productType=usdt-futures&granularity=1H&limit=500` → `{code:'00000',data:[[tsMs,o,h,l,c,baseVol,quoteVol],...]}` 오름차순. 과거는 `/history-candles` + `endTime=<earliest-1>` 워크백 (1회 ≤200개).
- Bitget funding: `GET https://api.bitget.com/api/v2/mix/market/history-fund-rate?symbol=X&productType=usdt-futures&pageSize=100&pageNo=N` → `{data:[{symbol,fundingRate,fundingTime},...]}` **내림차순**, 깊이 ~133일(pageNo 5부터 빈 배열).
- Bitget 계약목록: `GET https://api.bitget.com/api/v2/mix/market/contracts?productType=usdt-futures` (symbol, makerFeeRate 0.0002, takerFeeRate 0.0006, minTradeUSDT 5).
- Stooq: `GET https://stooq.com/q/d/l/?s=spy.us&i=d` → CSV Date,Open,High,Low,Close,Volume.
- 페처 공통: 재시도 3회 백오프, 요청간 sleep 0.15s, 캐시 히트 시 재다운로드 스킵(`--force` 플래그), User-Agent 헤더.

## 백테스트 계약 (테스트로 강제)
1. **노 룩어헤드**: 시그널 시리즈를 +1바 시프트 후 체결. 테스트: 합성 계단 가격에서 진입 체결가 == 시그널 다음 바 시가.
2. **펀딩 부호**: funding>0 & 롱 → 자본 감소, 숏 → 증가. 테스트로 양방향 검증.
3. **비용**: 진입·청산 각각 노셔널×(수수료+슬리피지). 델타중립(F1)은 4레그(현물 진입/청산 0.10%+슬립, 퍼프 진입/청산 0.06%+슬립).
4. **F1 PnL**: 두 레그 실가격 시리즈 사용 — hedged_pnl = (spot_ret − perp_ret)×노셔널 + 펀딩수취 − 비용. 현물이 없는 심볼은 유니버스에서 제외하고 로그.
5. **볼 타게팅**: pos_frac = min(lev_cap, target_vol/realized_vol20). realized_vol==0 가드.
6. **스탑**: 바 low/high가 스탑 터치 시 스탑가 체결, 갭 넘으면 시가 체결.
7. 에쿼티는 복리(자본 재투자), 일별 mark-to-market.

## gates.py 계산 정의
- 지표: total_ret, CAGR, sharpe(일수익 연율화 √365), sortino, calmar(CAGR/MDD), MDD, profit_factor, win_rate, turnover, exposure, n_trades.
- 게이트2 과최적화: 이웃 파라미터(W±, θ±, N±20%) 재실행 → IS 샤프 표준편차/평균 < 0.5 & IS/OOS 샤프 발산 < 2x.
- 게이트3 WF: 연도별 수익 표 → 양(+)해 비율 ≥ 60%.
- 게이트4 OOS: 홀드아웃 비용후 총수익 > 0.
- 게이트5 MC: 트레이드 부트스트랩 1e4, 5%분위 최종자본 > 초기. (트레이드 < 20개면 UNDETERMINED)
- 게이트7: 슬리피지×2에서 OOS 수익 부호 유지.
- 게이트9 Kelly: OOS 트레이드 평균/분산 기반 f* > 0. 리포트에 0.25f 값.
- 게이트16 파산: $300 시작 MC에서 P(최종<$150) < 5%.
- 게이트10~15: 수치 리포트(판정 기준 SPEC).
- 게이트17 regime: 2022(약세, 데이터 있으면)/2024(강세)/2025(횡보) 구간별 부호.
- 게이트18: 에쿼티 일수익 vs BTC 일수익 상관.
- 출력: `results/<cand>.json` + `results/gates_summary.md` 표.

## run_wave1.py
- `python research/wave1/run_wave1.py --stage fetch` → 유니버스 확정(펀딩≥24mo∧Bitget상장∧현물존재, 볼륨 상위 40) 후 전 데이터 캐시. 유니버스 목록 `cache/universe.json` 저장.
- `--stage run` → 15후보 전부 실행, results/ 저장. `--only F1a` 지원.
- `--stage gates` → 게이트 평가. `--stage report` → REGISTRY.md 갱신 + report/wave1_report.md 생성(표+정직 verdict 문구 자동).
- 진행 출력은 10심볼/10후보당 1줄.

## 완료 기준 (전부 만족해야 done)
1. pytest 전부 통과 (네트워크 없이).
2. `python -c "import research.wave1.run_wave1"` 급 임포트 에러 0 — 단 패키지 임포트 대신 스크립트 상대실행 기준으로 하라(`sys.path` 셋업 포함).
3. 각 파일 헤더에 한 줄 목적 주석.
4. 보고는 50자 이내 + 변경 파일 목록만. 장황 설명 금지.

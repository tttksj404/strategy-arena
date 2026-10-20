# 데이터 정합성 수정 리포트 -- Binance spot klines 페이지네이션 절단 버그

수정일 2026-07-22. 대상: `research/wave1/fetch_binance.py`의 `fetch_klines()`. 이 문서는 버그 -> 수정 ->
재수집 -> 재검증까지의 전 과정과, 교정 전/후 수치를 가감 없이 대비한다. **유리한 방향으로만 보고하지
않는다**: 아래 표가 보여주듯 교정 후 전기간 수익률은 모든 후보에서 소폭 **하락**했고 MDD는 **상승**했다.

## 0. 요약

- **버그**: spot klines 요청에 fapi(퍼프) 전용 한도인 `limit=1500`을 그대로 사용. 스팟 API 실제 한도는
  1000이라 응답이 1000행에서 조용히 잘리고, `len(page) < 1500`이 "더 이상 데이터 없음"으로 오판되어
  페이지네이션이 1페이지 만에 멈췄다. 유니버스 40종목 중 32종목의 스팟 데이터가 상장일+1000일 지점에서
  끊겨 있었다 (예: BTCUSDT 2019-09-01~2022-05-27, 1000행 -- 같은 기간 퍼프는 2502행, 2026-07-14까지 정상).
- **수정**: `fetch_klines()`가 market별로 올바른 한도(fapi=1500, spot=1000)를 사용하도록 수정하고, 페이지
  종료 조건도 같은 한도를 참조하도록 정정. 회귀 테스트 4개 추가.
- **재수집**: 유니버스 40종목 전체 스팟 1d를 `--force`로 재수집. **40/40 성공, 40/40이 퍼프 종료일과
  정확히 일치**(diff=0일, 전부 ±3일 이내 판정 통과).
- **핵심 판정 (저펀딩기 가동률)**: 2025-10-01 이후 저펀딩 OOS 구간의 가동률은 **교정 전후로 정확히
  동일**하다 (W2c/C1/C2/C3: OOS 거래수 2건, 활성일 7/287일(2.44%) 불변; C4: 0건, 0% 불변). 즉 **"저펀딩
  레짐이라 대기 중"이라는 기존 설명은 이 구간에 한정하면 데이터 누락이 아니라 실제 현상이었다** --
  버그가 넓혀준 유니버스로도 최근 9.5개월간 15% APR 진입 임계를 넘긴 심볼이 추가로 나타나지 않았다.
- **반대급부**: 과거 구간(주로 2023~2025)의 기회 상실은 실재했다. 교정 후 총 거래수는 모든 후보에서
  증가했지만(W2c 489->601, C1 261->283, C2 261->283, C3 403->449, C4 172->186), 그 결과로 **전기간
  수익률은 오히려 소폭 하락**하고 **MDD는 상승**했다 (아래 6~7절). 새로 드러난 과거 거래들이 순수하게
  플러스였던 것은 아니라는 뜻이다.

## 1. 확인된 버그

`research/wave1/fetch_binance.py`의 `fetch_klines()` (수정 전):

```python
params: dict[str, str | int] = {
    "symbol": request.symbol,
    "interval": request.interval,
    "limit": 1500,          # <- fapi 전용 한도를 spot에도 그대로 사용
    "startTime": cursor,
}
...
if next_cursor <= cursor or len(page) < 1500:   # <- 같은 상수로 종료 판정
    break
```

Binance `/api/v3/klines`(spot)의 실제 서버측 한도는 1000이다. `/fapi/v1/klines`(퍼프)만 1500을 허용한다.
`integrity_report()`(research/wave1/common.py)는 모노토닉/중복/갭/거래량 등 **내부** 정합성만 검사해
"너무 일찍 끝났다"는 잡아내지 못했기 때문에, 이 버그는 파이프라인의 모든 무결성 게이트를 통과한 채
방치되어 있었다.

증거 (재수집 전 실측, BTCUSDT):

| 항목 | Spot (버그) | Perp (정상) |
|---|---|---|
| 행수 | 1,000 | 2,502 |
| 시작 | 2019-09-01 | 2019-09-08 |
| 종료 | **2022-05-27** | **2026-07-14** |

엔진(`research/wave1/fam_funding.py`, `research/wave10_carry100/engine.py`)은 spot·perp 양쪽이
모두 non-NaN인 날에만 심볼을 편입하므로, 가짜 손익이 생기지는 않았다(추정치가 부풀려지지 않음) -- 대신
**기회가 조용히 소실**되는 방향으로만 작동했다. 이 발견은 이번 작업에서 처음 한 게 아니라
`research/wave11_yield/fetch_y11.py`가 자체 스코프(wave11 전용 로컬 캐시) 안에서 독립적으로도 확인한
바 있다 (모듈 상단 docstring, 40종목 중 32종목 1000행 절단으로 동일 진단).

## 2. 수정 내용

`research/wave1/fetch_binance.py::fetch_klines()`:

```python
limit = 1500 if request.market == "fapi" else 1000
...
"limit": limit,
...
if next_cursor <= cursor or len(page) < limit:
    break
```

fapi 경로는 그대로 1500을 유지(변경 없음). 회귀 테스트 4건 추가
(`research/wave1/tests/test_fetch_binance.py`, 모의 세션으로 실제 네트워크 없이 검증):

1. `test_spot_market_requests_limit_1000_and_paginates_past_a_full_first_page` -- spot 요청이
   limit=1000을 쓰고, 꽉 찬 첫 페이지(1000행) 뒤 두 번째 페이지(500행)까지 정확히 이어붙이는지 (총
   1500행), 두 번째 요청의 `startTime`이 첫 페이지 마지막 행 openTime+1ms로 정확히 이어지는지 검증.
2. `test_fapi_market_still_requests_limit_1500` -- fapi 경로가 회귀 없이 1500을 그대로 요청하는지
   (스팟 수정이 퍼프 경로를 건드리지 않았는지의 가드).
3. `test_spot_single_short_page_does_not_paginate_further` -- 첫 페이지가 이미 짧으면(200행) 추가
   요청을 보내지 않는지.
4. `test_string_millisecond_timestamp_is_numeric_before_datetime_conversion` -- 기존
   `test_fetch_bitget.py`와 동일한 취지의 회귀 가드(단일 스팟 페이지도 pandas FutureWarning 없이
   변환되는지).

```
$ python -m pytest research/wave1/tests/test_fetch_binance.py -v
4 passed in 1.86s
```

## 3. 재수집 검증표 (유니버스 40종목, spot 1d, --force)

`research/wave1/cache/binance_spot_*_1d.csv.gz`를 fixed `fetch_klines()`로 전량 재수집. 덮어쓰기 전
원본은 같은 디렉터리에 `binance_spot_{symbol}_1d_prefix.csv.gz`로 보존했다(gitignore 대상 디렉터리라
git 이력이 없으므로 수동 백업이 유일한 복구 경로).

결과: **40/40 성공, 40/40이 퍼프 종료일과 diff=0.0일** (판정 기준 ±3일 이내 전부 통과). 사전 조사대로
32종목이 실제로 절단되어 있었고(재수집 후 행수 대폭 증가), 8종목(ONDOUSDT/ENAUSDT/VANRYUSDT/JTOUSDT/
ETHFIUSDT/TIAUSDT/ORDIUSDT/WIFUSDT -- 전부 최근 상장이라 1000일 한도에 애초에 도달하지 않음)은
버그 영향이 없었다(행수 불변).

| Symbol | 이전 행수 | 이전 종료일 | 이후 행수 | 이후 시작~종료 | Perp 종료일 | 오차(일) |
|---|---:|---|---:|---|---|---:|
| ETHUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| BTCUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| SOLUSDT | 1000 | 2023-05-07 | 2164 | 2020-08-11~2026-07-14 | 2026-07-14 | 0.0 |
| XRPUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| DOGEUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| ADAUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| WLDUSDT | 1000 | 2026-04-18 | 1087 | 2023-07-24~2026-07-14 | 2026-07-14 | 0.0 |
| NEARUSDT | 1000 | 2023-07-10 | 2100 | 2020-10-14~2026-07-14 | 2026-07-14 | 0.0 |
| ONDOUSDT | 460 | 2026-07-14 | 460 | 2025-04-11~2026-07-14 | 2026-07-14 | 0.0 |
| SUIUSDT | 1000 | 2026-01-26 | 1169 | 2023-05-03~2026-07-14 | 2026-07-14 | 0.0 |
| LINKUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| BCHUSDT | 1000 | 2022-08-23 | 2421 | 2019-11-28~2026-07-14 | 2026-07-14 | 0.0 |
| AAVEUSDT | 1000 | 2023-07-11 | 2099 | 2020-10-15~2026-07-14 | 2026-07-14 | 0.0 |
| LTCUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| SKLUSDT | 1000 | 2023-08-27 | 2052 | 2020-12-01~2026-07-14 | 2026-07-14 | 0.0 |
| XLMUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| ENAUSDT | 834 | 2026-07-14 | 834 | 2024-04-02~2026-07-14 | 2026-07-14 | 0.0 |
| AVAXUSDT | 1000 | 2023-06-18 | 2122 | 2020-09-22~2026-07-14 | 2026-07-14 | 0.0 |
| UNIUSDT | 1000 | 2023-06-13 | 2127 | 2020-09-17~2026-07-14 | 2026-07-14 | 0.0 |
| TRXUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| DOTUSDT | 1000 | 2023-05-14 | 2157 | 2020-08-18~2026-07-14 | 2026-07-14 | 0.0 |
| FILUSDT | 1000 | 2023-07-11 | 2099 | 2020-10-15~2026-07-14 | 2026-07-14 | 0.0 |
| VANRYUSDT | 957 | 2026-07-14 | 957 | 2023-12-01~2026-07-14 | 2026-07-14 | 0.0 |
| ARBUSDT | 1000 | 2025-12-16 | 1210 | 2023-03-23~2026-07-14 | 2026-07-14 | 0.0 |
| TUSDT | 1000 | 2024-11-20 | 1601 | 2022-02-25~2026-07-14 | 2026-07-14 | 0.0 |
| LDOUSDT | 1000 | 2025-02-01 | 1528 | 2022-05-09~2026-07-14 | 2026-07-14 | 0.0 |
| INJUSDT | 1000 | 2023-07-17 | 2093 | 2020-10-21~2026-07-14 | 2026-07-14 | 0.0 |
| JTOUSDT | 951 | 2026-07-14 | 951 | 2023-12-07~2026-07-14 | 2026-07-14 | 0.0 |
| ETHFIUSDT | 849 | 2026-07-14 | 849 | 2024-03-18~2026-07-14 | 2026-07-14 | 0.0 |
| TIAUSDT | 988 | 2026-07-14 | 988 | 2023-10-31~2026-07-14 | 2026-07-14 | 0.0 |
| CRVUSDT | 1000 | 2023-05-11 | 2160 | 2020-08-15~2026-07-14 | 2026-07-14 | 0.0 |
| OPUSDT | 1000 | 2025-02-24 | 1505 | 2022-06-01~2026-07-14 | 2026-07-14 | 0.0 |
| ORDIUSDT | 981 | 2026-07-14 | 981 | 2023-11-07~2026-07-14 | 2026-07-14 | 0.0 |
| FETUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| WIFUSDT | 862 | 2026-07-14 | 862 | 2024-03-05~2026-07-14 | 2026-07-14 | 0.0 |
| HBARUSDT | 1000 | 2022-06-24 | 2481 | 2019-09-29~2026-07-14 | 2026-07-14 | 0.0 |
| ETCUSDT | 1000 | 2022-05-27 | 2509 | 2019-09-01~2026-07-14 | 2026-07-14 | 0.0 |
| APTUSDT | 1000 | 2025-07-14 | 1365 | 2022-10-19~2026-07-14 | 2026-07-14 | 0.0 |
| APEUSDT | 1000 | 2024-12-10 | 1581 | 2022-03-17~2026-07-14 | 2026-07-14 | 0.0 |
| PENDLEUSDT | 1000 | 2026-03-28 | 1108 | 2023-07-03~2026-07-14 | 2026-07-14 | 0.0 |

미달 심볼: **없음** (40/40 전부 판정 통과).

## 4. 유니버스 유효 심볼수 시계열 (spot+perp 동시 커버 연도, /40)

연도별로 "그 해의 어느 하루라도 spot과 perp가 동시에 존재하는" 심볼 수. `research/wave1/cache`의 실제
spot/perp 캐시 파일을 직접 스캔해 계산(전략 임계값과 무관한 순수 데이터 가용성 지표):

| 연도 | 교정 전 | 교정 후 | 변화 |
|---|---:|---:|---:|
| 2019 | 3 | 3 | 0 |
| 2020 | 20 | 20 | 0 |
| 2021 | 21 | 21 | 0 |
| 2022 | 26 | 26 | 0 |
| 2023 | 22 | 35 | **+13** |
| 2024 | 16 | 39 | **+23** |
| 2025 | 15 | 40 | **+25** |
| 2026 | 11 | 40 | **+29** |

2020~2022는 원래도 커버리지 손실이 크지 않았다(그 시절 랭킹 상위권이 대부분 이미 완전 커버 심볼이었기
때문). 손실은 2023년부터 급격히 커지고, **가장 최근 구간(2025~2026, 바로 지금 문제되는 저펀딩
OOS 구간)에서 최대**였다 -- 2026년 기준 40종목 중 겨우 11종목만 스팟 데이터가 살아 있었다.

## 5. 재실행 (exit code)

| 단계 | 명령 | 결과 |
|---|---|---|
| 회귀 테스트 | `pytest research/wave1/tests/test_fetch_binance.py` | 4 passed |
| 전체 테스트 (수정 직후, 재수집 전 베이스라인) | `pytest --import-mode=importlib --ignore=research/wave1-rwa` | **134 passed** (359.72s) |
| 재수집 | `research/wave1/cache/binance_spot_*_1d.csv.gz` x40, `--force` | 40/40 OK, exit 0 |
| wave2 | `run_wave2.py --stage fetch\|run --only W2c\|gates --only W2c\|report` | exit 0 (4단계 전부) |
| wave10 | `run_wave10.py --stage all` (C1~C4) | exit 0, C1/C2/C3 PASS, C4 UNTESTED_IN_OOS(휴면, 변화없음) |
| deep_validate | `python -m research.validation.deep_validate` | exit 0, 5개 후보 JSON + DEEP_REPORT.md 재생성 |
| 전체 테스트 (모든 재실행 이후, 1차) | `pytest --import-mode=importlib --ignore=research/wave1-rwa` | **1 failed, 133 passed** (아래 참고) |
| 전체 테스트 (스냅샷 보정 후, 최종) | 〃 | **134 passed**, exit 0 |

`--import-mode=importlib --ignore=research/wave1-rwa`를 쓰는 이유: 이 저장소는 애초에 루트에서 맨
`pytest`로 전량 수집이 안 된다 (`research/wave1-rwa`는 `src.engine`을 임포트하는 별도 서브프로젝트라
루트 sys.path로는 애초에 못 도는 구조이고, `research/wave11_yield/tests/test_wave11.py`와
`research/wave11_multi_factor/tests/test_wave11.py`, `research/wave9_capital_native/tests/test_wave9.py`와
`research/wave9_100usd/tests/test_wave9.py`는 각각 `tests/__init__.py`가 없어 동일 basename 충돌이
난다). 전부 이번 수정과 무관한 기존 구조적 이슈이며(건드린 파일과 완전히 무관한 디렉터리), git
디폴트 임포트 모드(prepend)에서만 재현되고 `--import-mode=importlib`로는 정상 수집된다.

### 5.1 재실행 후 발견된 테스트 실패 1건 (수정함)

전체 재실행 이후 1차 전체 테스트에서 `research/wave4_leverage/tests/test_leverage.py::
test_l1_reconciles_with_wave_engine_cagr_and_mdd[W2c-SYM]`이 실패했다. 이 테스트는 고정 시드
(20\_260\_716)로 `research/wave1/cache`에서 W2c를 **직접 재계산**해 MC p05를 하드코딩된 스냅샷
`746.3225248264143`과 비교하는데(참고: 이 값은 6절의 교정 전 deep_validate MC p05와 정확히 같다 --
같은 절단 데이터로 계산됐기 때문), 캐시가 교정되며 실제 값이 `720.40882704391`로 legitimately
바뀌어 `rel=0.01` 허용오차를 벗어났다(차이 약 -3.5%). 즉 **이 실패는 회귀가 아니라, 옛(버그) 데이터의
스냅샷 값이 그대로 하드코딩돼 있던 테스트가 정정된 데이터를 정확히 잡아낸 것**이다. 허용오차를
느슨하게 풀거나 assert를 지우는 대신, 새로 관측된 올바른 값으로 스냅샷을 갱신하고 왜 바뀌었는지
주석으로 남겼다 (변경 전/후 값을 코드 주석에 그대로 남겨 이력 추적 가능). 다른 wave(wave5/wave7/wave9
등)의 캐시-의존 테스트들은 전량 재실행에서 전부 통과해 유사한 스냅샷 문제가 없음을 확인했다.

## 6. 교정 전/후 비교 -- W2c (wave2)

| 지표 | 교정 전 | 교정 후 | 변화 |
|---|---:|---:|---:|
| 전기간 수익률 (2019-09-01~2026-07-14, $300 기준) | +181.54% ($844.61) | **+176.68%** ($830.05) | **-4.86pp** |
| 전기간 실현 MDD | 1.775% | 1.885% | +0.11pp |
| 총 거래수 | 489 | **601** | +112 |
| IS 누적수익 (`metrics.is.total_ret`) | +179.75% | +174.92% | -4.82pp |
| OOS 누적수익 (`metrics.oos.total_ret`) | +0.63949% | +0.63949% | **0.000pp (불변)** |
| OOS 거래수 | 2 | 2 | 불변 |
| OOS 가동일 (2025-09-30 이후 287일 중) | 7일 (2.44%) | 7일 (2.44%) | **불변** |
| 19게이트 PASS/FAIL/UNDETERMINED | 12/5/2 | 11/6/2 | **gate17(regime) PASS->FAIL** |
| gate17 2025_sideways 구간 수익 | +0.0492% | **-0.5464%** | 부호 전환(+→-) |
| deep_validate MC(거래부트스트랩) p05 | $746.32 | $720.68 | -$25.64 |
| deep_validate DSR score | 1.8188 | 1.8197 | +0.0009 (사실상 불변) |
| deep_validate overall | FAIL (bitget_sign) | FAIL (bitget_sign, 사유 동일) | 불변 |

gate17("regime") 은 2022_bear/2024_bull/2025_sideways 세 구간 수익이 전부 양수여야 PASS다. 교정 전에는
2025_sideways가 간신히 +0.05%였는데, 교정 후 정확히 -0.55%로 뒤집혀 FAIL로 전환됐다 -- **교정이 유리한
쪽으로만 작동하지 않았다는 직접적 증거**다. W2c의 전체 판정(REGISTRY.md)은 교정 전에도 이미 FAIL이었고
(sharpe/sortino/calmar/recovery_factor/capacity 게이트가 원래도 FAIL) 교정 후에도 FAIL로 동일하다 --
최종 판정 자체는 바뀌지 않았지만, FAIL 사유의 구성은 더 나빠졌다.

deep_validate의 `bitget_sign` FAIL과 `cache_integrity` FAIL은 **이번 수정과 무관한 기존 이슈**다 (funding
파일 해시 문제이며, mismatch 목록이 교정 전/후 완전히 동일함을 확인함 -- `research/wave2/cache_manifest.json`이
`W2G_SYMBOLS`+유니버스 40종목의 funding 파일만 등록하는데 `deep_validate._cache_integrity()`는 캐시 내
모든 `*_funding_*.csv.gz`를 스캔해서 미등록 심볼들을 전부 "missing manifest entry"로 표시하는 구조적
gap; 스팟 버그와 무관, 이번 스코프 밖).

## 7. 교정 전/후 비교 -- wave10 C1~C4

두 가지 MDD를 구분해서 기재한다: **블록MDD p95(gate_c)** = 90일 블록 셔플 1000경로의 p95(wave10/wave11
리포트가 표준으로 쓰는 스트레스 지표), **전기간 실현MDD** = 실제 히스토리 equity 곡선의 단순 최대낙폭
(재계산·셔플 없음). 둘은 다른 개념이므로 행을 분리했다 (초안에서 이 둘을 한 행에 섞어 적은 오류를
발견해 정정함).

| Config | 지표 | 교정 전 | 교정 후 | 변화 |
|---|---|---:|---:|---:|
| C1 | 전기간수익 (gate_d) | 59.38% | **56.19%** | -3.19pp |
| C1 | 블록MDD p95 (gate_c) | 2.289% | 2.688% | +0.40pp |
| C1 | 전기간 실현MDD | 2.451% | **4.125%** | +1.67pp |
| C1 | MC p05 (gate_b) | $130.31 | $127.95 | -$2.36 |
| C1 | 총 거래수 | 261 | 283 | +22 |
| C1 | 고펀딩기 평균 연환산 | 16.84% | 15.94% | -0.90pp |
| C1 | 저펀딩기(OOS) 연환산 / 거래수 / 가동률 | 0.271% / 2건 / 2.44% | 0.271% / 2건 / 2.44% | **불변** |
| C1 | overall | PASS | PASS | 불변 |
| C2 | 전기간수익 | 45.40% | 43.07% | -2.33pp |
| C2 | 블록MDD p95 | 1.820% | 2.214% | +0.39pp |
| C2 | 전기간 실현MDD | 1.961% | 3.312% | +1.35pp |
| C2 | MC p05 | $123.64 | $121.78 | -$1.86 |
| C2 | 저펀딩기(OOS) 연환산/거래수/가동률 | 0.220%/2건/2.44% | 0.220%/2건/2.44% | **불변** |
| C2 | overall | PASS | PASS | 불변 |
| C3 | 전기간수익 | 54.92% | 52.79% | -2.13pp |
| C3 | 블록MDD p95 | 1.092% | 1.157% | +0.07pp |
| C3 | 전기간 실현MDD | 1.172% | 1.563% | +0.39pp |
| C3 | MC p05 | $135.99 | $134.13 | -$1.86 |
| C3 | 저펀딩기(OOS) 연환산/거래수/가동률 | 0.140%/2건/2.44% | 0.140%/2건/2.44% | **불변** |
| C3 | overall | PASS | PASS | 불변 |
| C4 | 전기간수익 | 52.78% | 51.20% | -1.58pp |
| C4 | 블록MDD p95 | 2.055% | 2.053% | -0.002pp (사실상 불변) |
| C4 | 전기간 실현MDD | 2.206% | 2.206% | 불변 |
| C4 | MC p05 | $127.85 | $126.51 | -$1.34 |
| C4 | 저펀딩기(OOS) 연환산/거래수/가동률 | 0.0%/0건/0.0% | 0.0%/0건/0.0% | **불변 (여전히 휴면)** |
| C4 | overall | UNTESTED_IN_OOS | UNTESTED_IN_OOS | 불변 |

패턴은 W2c와 완전히 일치한다: **과거 구간(고펀딩기 포함)의 수익은 소폭 낮아지고 두 MDD 지표 모두 높아지는
(C4 제외) 방향으로 전부 수정됐고, 저펀딩 OOS 구간의 실적/가동률은 4개 config 전부 소수점까지 완전히
동일**했다. C4만 실현MDD·블록MDD 둘 다 사실상 불변인데, C4는 진입임계가 25%APR로 등록돼 있어(W2c/C1~C3의
15%APR보다 높음 -- `research/wave10_carry100/configs.py`의 C4 주석 및 `wave10_report.md`의 "C4 스펙
불일치 참고사항" 참고) 원래도 거래가 가장 드물었고, 새로 편입된 심볼들이 C4의 최대낙폭 구간 자체를
건드리지 않았기 때문으로 보인다.

## 8. 저펀딩기(2025-10~) 가동률 판정 -- 핵심 질의에 대한 답

**질문**: "저펀딩 레짐이라 대기 중"이라는 기존 설명이 사실은 스팟 데이터 누락 때문이었을 가능성이 있는가?

**답: 아니다 (이 구간에 한정하면).** W2c, C1, C2, C3, C4 다섯 후보 전부 2025-09-30 이후 287일 구간의
거래수·활성일수·연환산수익이 교정 전후로 **완전히 동일**했다 (6~7절 표). 유니버스가 최대 40->11(2026년
기준, 4절)까지 줄어 있었던 과거 시점과 달리, 지금은 40종목 전부의 스팟 데이터가 이미 살아 있었으므로
버그가 "지금 이 순간"의 진입 여부에는 애초에 영향을 줄 수 없는 상태였다 -- 즉 **버그가 있어도 없어도
현재는 40종목 중 15% APR 임계를 넘긴 심볼이 나타나지 않는다는 사실 자체는 바뀌지 않는다.**

다만 이것이 버그가 무해했다는 뜻은 아니다: 3~4절이 보여주듯 **과거(주로 2023~2025) 구간에서는 유니버스가
실제로 조용히 줄어 있었고, 그 결과 그 시기의 거래 기회를 놓쳤다** (거래수가 후보별로 14~112건 증가:
C4 +14, C1/C2 +22, C3 +46, W2c +112).
다만 그 영향은 수익률을 끌어올리는 방향이 아니라 **오히려 전기간 수익을 낮추고 MDD를 높이는 방향**으로
나타났다 -- 놓친 기회들이 사후적으로는 손실 쪽에 가까웠다는 뜻이다. 결론적으로 "지금 대기 중인 이유"에
대한 기존 설명은 정정할 필요가 없지만, "과거 백테스트 성과가 데이터 결손으로 인해 실제보다 좋게 나왔을
가능성"은 사실이었고 이번 수정으로 해소됐다.

## 9. wave-11 Y4 vs C1 재비교

`research/wave11_yield/report/wave11_report.md`는 스스로 "C1은 구버전(절단된) 데이터 기준" 이라고
고지했었다 (Y1~Y6는 wave11 자체 교정 캐시로 공정 비교했지만, 비교 기준인 C1은 wave10의 기존 JSON을
그대로 인용). 이번 수정으로 C1이 wave1 캐시 기준으로 정식 재계산됐으므로, 처음으로 같은 세대의 데이터로
Y4와 C1을 비교할 수 있다.

두 리포트 모두 "블록MDD p95"는 같은 방법론(90일 블록 셔플, wave10 gates.py 기준 -- wave11의 SPEC.md도
"wave10 gates.py와 동일 경로수/자본기준/블록길이"라고 명시)이므로 그대로 비교 가능하다. 이 절의 표는
wave10 gate_c 값만 쓴다 (7절에서 도입한 "전기간 실현MDD"는 wave11 리포트에 대응 지표가 없어 여기서는
쓰지 않음).

| 지표 | Y4 (wave11, 불변) | C1 (구버전, 무효 비교였음) | C1 (교정 후, 이번 리포트) |
|---|---:|---:|---:|
| 고펀딩기 평균 연환산 | 17.88% | 16.84% | **15.94%** |
| 블록MDD p95 (gate_c) | 3.86% | 2.29% | **2.688%** |
| MC p05 | $132.90 | $130.31 | **$127.95** |

재비교 (Y4 vs 교정된 C1):

- 고펀딩기 연환산: Y4(17.88%) > 교정C1(15.94%), **격차 +1.94pp로 오히려 확대** (구버전 대비 격차는
  +1.04pp였음 -- C1 자체 수익이 낮아졌기 때문).
- 블록MDD p95: Y4(3.86%) vs 교정C1(2.688%) -- **Y4가 여전히 더 높다(나쁘다)**, 다만 격차는 좁혀졌다
  (구버전 대비 +1.57pp -> 교정 후 +1.17pp). 순위 역전은 없다 -- Y4는 원래도, 교정 후에도 C1보다 MDD가
  높다.
- MC p05: Y4($132.90) > 교정C1($127.95), 격차 $4.95로 확대(구버전 대비 $2.59였음).
- wave11의 승격 조건("고펀딩기 연환산 > C1 ∧ MDD <= C1의 2배")을 교정된 C1 기준(15.94%, 2.688% x2=5.377%)
  으로 재적용해도 **Y4(17.88%, 3.86%)는 두 조건을 모두 통과** -- Y4의 승격 판정은 교정 후에도 **그대로
  유지**된다 (MDD 조건의 통과 여유폭은 구버전 4.58% 문턱보다 5.377% 문턱이 더 커져 오히려 넉넉해졌지만,
  이는 Y4가 개선된 게 아니라 비교 기준인 C1의 실측 MDD 자체가 올라갔기 때문이다).

## 10. 영향받는 wave 목록 (이번 작업에서 재실행하지 않음 -- 후속 필요)

지시된 재실행 범위는 wave2 W2c / wave10 C1~C4 / deep_validate 뿐이었다. `research/wave1/cache`를 직접
또는 `research.wave1.fam_funding.load_markets`/`research.wave1.fam_funding.run_cached`를 통해
소비하는 아래 파이프라인들은 **동일한 교정된 캐시 위에 있지만 이번에 재실행하지 않았고, 그 결과 JSON은
여전히 교정 전 데이터를 반영한다**:

| Wave | 파일 | 비고 |
|---|---|---|
| wave1 F1 family (F1a~F1f) | `research/wave1/results/F1*.json` | `run_wave1.py --stage run` 미실행. F1e/F1f는 deep_validate가 그대로 재사용(5절 확인: byte-identical) |
| wave2 나머지 후보 (W2a/b/d/e/f/g) | `research/wave2/results/W2*.json` | `--only W2c`만 재실행, 나머지는 미실행 |
| wave4_leverage | `research/wave4_leverage/sweep.py` | `cache_dir = research/wave1/cache` 직접 사용 확인 |
| wave5 | `research/wave5/run_wave5.py` | `CACHE_DIR = wave1/cache` 확인 (W5c는 `binance_spot_` 직접 로드) |
| wave7 | `research/wave7/engine_w7.py` | `cache_dir = root/research/wave1/cache` 확인 |
| wave9_100usd | `research/wave9_100usd/engine_w9.py` | W2c/W3c 결과를 재료로 블렌딩(`run_all(w2c_path, w3c_path, ...)`) -- W2c가 갱신됐으므로 간접 영향, 재실행 시 수치 변경 가능 |
| paper (paper trading) | `research/paper/track.py` | `fam_funding`의 신호 함수(`carry_position`/`funding_score`)를 임포트 -- 라이브 신호 계산 경로이므로 영향 확인 권장 |

## 11. 별도 발견 -- research/wave3/cache도 동일 버그 (이번 스코프 밖, 미수정)

`research/wave3/fetch.py`는 `research.wave1.fetch_binance.fetch_klines`를 **그대로 임포트해서** 자체
캐시(`research/wave3/cache/`, wave1과 완전히 별개 디렉터리, 종목 332개)를 채운다. 즉 이번에 고친 함수
자체는 wave3 재수집에도 자동 적용되지만, **wave3의 기존 캐시 파일은 아직 재수집 전이라 여전히 절단돼
있다.** 실측 확인:

```
wave3 spot 파일 332개 중 정확히 1000행(절단 시그니처) = 170개
예: binance_spot_1INCHUSDT_1d.csv.gz -> 1000행, 2020-12-25~2023-09-20 (2026-07-14까지 가야 정상)
```

wave3 자체(`research/wave3/run_wave3.py`, W3c/W3d) 및 wave3 캐시에 의존하는
`wave9_capital_native`/`wave10_event_driven`/`wave11_multi_factor`가 전부 이 캐시를 쓴다. 이번 작업
지시 범위(`research/wave1/cache`)를 벗어나므로 재수집하지 않았고, 별도 후속 작업으로 분리해 flag
했다 (spawn_task).

## 12. 정직성 노트 / 보존 파일

- 아래 파일은 전부 덮어쓰기 **전** 원본이다 (`git status`로 추적되는 JSON들도 이번 지시에 따라 추가로
  수동 백업):
  - `research/wave1/cache/binance_spot_{symbol}_1d_prefix.csv.gz` x40
  - `research/wave2/results/W2c_prefix.json`
  - `research/wave10_carry100/results/C{1,2,3,4}_prefix.json`
  - `research/validation/results/deep_{W2c,F1e,F1f,W3c,W3d}_prefix.json`
  - `research/wave2/cache_manifest_prefix.json`
- 유리한 방향으로 조정한 수치는 없다. 6~9절의 모든 숫자는 재실행 결과 JSON에서 그대로 뽑았다(재계산·
  반올림 방향 조정 없음). 하락한 수치(전기간수익, MC p05, gate17)도 그대로 보고했다.
- `research/wave2/cache_manifest.json`의 `bitget_sign`/`cache_integrity` FAIL은 이번 스팟 버그와
  무관한 기존 이슈이며 mismatch 목록이 교정 전후 완전히 동일함을 직접 확인했다 (6절).

## 13. 재현 명령

```bash
# 회귀 테스트
python -m pytest research/wave1/tests/test_fetch_binance.py -v

# 전체 테스트 (이 저장소는 루트 rootdir에서 맨 pytest로 전량 수집이 안 됨 -- 5절 참고)
python -m pytest -q --import-mode=importlib --ignore="research/wave1-rwa"

# wave2 (W2c만)
python -m research.wave2.run_wave2 --stage fetch
python -m research.wave2.run_wave2 --stage run --only W2c
python -m research.wave2.run_wave2 --stage gates --only W2c
python -m research.wave2.run_wave2 --stage report

# wave10 (C1~C4 전체)
python -m research.wave10_carry100.run_wave10 --stage all

# 심층검증 (W2c 포함 5개 후보)
python -m research.validation.deep_validate
```

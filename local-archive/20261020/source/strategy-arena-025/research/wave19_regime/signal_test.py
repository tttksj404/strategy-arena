# Wave-19 STAGE 1 -- 펀딩 스파이크 예측 신호 검정 (SPEC.md "성패를 가르는 단 하나의 질문").
#
# 이 모듈이 FAIL을 내면 stage 2(engine19.py/gates19.py/R0-R4)는 만들지 않는다 -- SPEC.md의
# 절대 규칙. PASS/FAIL을 가르는 임계값·계수는 전부 이 파일에 사전 고정돼 있고, 실제 결과를
# 계산한 뒤 거꾸로 조정하지 않는다(과최적화 금지, "계수를 데이터 보고 조정 금지").
#
# 특징 3개(OI 5d증가율/5d가격수익/3d펀딩추세) 동일가중 합이 SPEC.md 원안이지만, OI는 실제로
# 뺐다 -- Binance `/futures/data/openInterestHist`를 라이브로 재확인한 결과
# (check_oi_history_limit, 아래) period=1d/limit=500을 줘도 30일치만 반환된다. 이미
# research/wave15_diverse/signals15.py가 동일 엔드포인트에서 같은 결론을 문서화했었고(그 모듈
# L48-51 주석), 이번에 다시 라이브로 재확인해도 같다. 2019-2026 7년 백테스트에 30일 데이터는
# 특징으로 쓸 수 없으므로 SPEC.md가 사전 승인한 폴백("부족하면 특징에서 OI 제외하고 2특징으로
# 축소")을 그대로 쓴다. 30일 구간 한정 참고용 기술통계만 별도로 남긴다
# (supplementary_oi_descriptive) -- PASS/FAIL에는 관여하지 않는다.
#
# 남은 2특징의 계수는 1.0/1.0(동일가중 합, SPEC.md 문구 그대로) -- 학습 없음. 진입임계값
# theta는 SPEC.md에 숫자가 없어 이 파일이 정해야 하는데, 결과를 보고 정한 게 아니라는 점이
# 핵심이다: theta=1.0(원점수 합 기준의 소박한 반올림 수)을 이 파일 작성 시점에 고정하고, 이
# 값 하나로만 PASS/FAIL을 낸다. {0.5,1.5,2.0}에서의 민감도는 참고용으로만 별도 표기한다(그
# 표가 PASS/FAIL을 바꾸지 않는다 -- 사후에 더 유리한 theta로 갈아타지 않는다).

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Final

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import requests

from research.wave1.common import json_safe, save_json
from research.wave1.fam_funding import FundingMarket, funding_score
from research.wave13_liquidity import costs_measured
from research.wave13_liquidity import universe_liquidity as ul
from research.wave13_liquidity.configs13 import get_config as get_wave13_config

L4_CONFIG: Final = get_wave13_config("L4")  # top200 거래대금, 12mo 히스토리 하한 -- stage-2 안정 레그(I5)와 동일 유니버스, 재사용

BASE_DIR: Final = Path(__file__).resolve().parent
RESULTS_DIR: Final = BASE_DIR / "results"

# ---------------------------------------------------------------------------
# 사전 고정 상수 -- 어떤 라벨/결과도 계산하기 전에 고정("계수 사전고정").
# ---------------------------------------------------------------------------

SPIKE_APR_THRESHOLD: Final = 0.15  # "펀딩 스파이크" = L4/carry_position이 이미 쓰는 진입 임계값(research.wave1.fam_funding), 새로 만들지 않음
LABEL_HORIZON_MIN_DAYS: Final = 3
LABEL_HORIZON_MAX_DAYS: Final = 7  # SPEC.md "3~7일 내"
PRICE_RET_WINDOW_DAYS: Final = 5  # SPEC.md "5d 가격수익"
FUNDING_TREND_WINDOW_DAYS: Final = 3  # SPEC.md "3d 펀딩추세"
ZSCORE_WINDOW_DAYS: Final = 90  # research/wave15_diverse/signals15.py ZSCORE_WINDOW_DAYS 승계(신규 도출 아님)
ENTRY_Z_THRESHOLD: Final = 1.0  # 원점수 합(계수 1.0/1.0) 기준 소박한 반올림값 -- 결과를 본 뒤 정한 값이 아님
THRESHOLD_SENSITIVITY_GRID: Final[tuple[float, ...]] = (0.5, 1.0, 1.5, 2.0)  # 참고용 -- PASS/FAIL에 관여하지 않음
HOLD_DAYS: Final = LABEL_HORIZON_MAX_DAYS  # 진입~청산 보유일 = 라벨 창 상한과 동일(예측 창 전체를 보유)

OI_ENDPOINT: Final = "https://fapi.binance.com/futures/data/openInterestHist"
FUNDING_HISTORY_ENDPOINT: Final = "https://fapi.binance.com/fapi/v1/fundingRate"
OI_CHECK_SYMBOLS: Final[tuple[str, ...]] = ("BTCUSDT", "ETHUSDT")
OI_SUPPLEMENTARY_SYMBOLS: Final[tuple[str, ...]] = (
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT",
    "AVAXUSDT", "LINKUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT", "SUIUSDT", "NEARUSDT", "APTUSDT",
)  # L4 유니버스 상위 거래대금권과 대략 겹치는 대표 표본 -- 30일 창 보조 기술통계 전용, 승격/판정 미반영

SIGNIFICANCE_ALPHA: Final = 0.05
BOOT_PATHS: Final = 10_000  # repo 전역 MC_PATHS 관례(gates13/18, deep_stats)와 동일 자릿수
SEED: Final = 20_260_724  # 오늘(동결) 날짜 기반 리터럴 시드 -- wave10-18의 "per-wave 리터럴 시드" 관례


# ---------------------------------------------------------------------------
# 0) OI 히스토리 제한 라이브 재확인 (SPEC.md "제한 확인").
# ---------------------------------------------------------------------------


def check_oi_history_limit(symbols: tuple[str, ...] = OI_CHECK_SYMBOLS, period: str = "1d", limit: int = 500) -> dict[str, Any]:
    """Binance `/futures/data/openInterestHist`가 실제로 며칠치를 돌려주는지 라이브로 확인한다.
    research/wave15_diverse/signals15.py가 이미 동일 결론(약 30일)을 문서화했지만(그 모듈
    L48-51), SPEC.md는 이 wave 자체가 "제한 확인"하라고 명시적으로 요구하므로 재확인한다."""
    per_symbol: dict[str, Any] = {}
    for symbol in symbols:
        try:
            response = requests.get(OI_ENDPOINT, params={"symbol": symbol, "period": period, "limit": limit}, timeout=(5.0, 20.0))
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                per_symbol[symbol] = {"n_points": 0, "error": f"empty/non-list response: {str(data)[:200]}"}
                continue
            first_ts = pd.to_datetime(int(data[0]["timestamp"]), unit="ms", utc=True)
            last_ts = pd.to_datetime(int(data[-1]["timestamp"]), unit="ms", utc=True)
            per_symbol[symbol] = {
                "n_points": len(data),
                "requested_limit": limit,
                "period": period,
                "earliest": first_ts.isoformat(),
                "latest": last_ts.isoformat(),
                "span_days": int((last_ts - first_ts).days),
            }
        except (requests.RequestException, ValueError, KeyError) as error:
            per_symbol[symbol] = {"n_points": None, "error": str(error)}
    spans = [row.get("span_days") for row in per_symbol.values() if isinstance(row.get("span_days"), int)]
    max_span = max(spans) if spans else None
    sufficient = max_span is not None and max_span >= 365
    return {
        "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "endpoint": OI_ENDPOINT,
        "per_symbol": per_symbol,
        "max_observed_span_days": max_span,
        "sufficient_for_multiyear_backtest": sufficient,
        "conclusion": (
            "히스토리가 충분히 길면(>=365일) OI 특징을 포함한다 -- 이번 실측에서는 도달하지 않을 "
            "것으로 예상된다(wave15 선행 확인과 일치할 경우)."
            if sufficient
            else "약 30일로 제한 확인됨 -- 2019-2026 7년 백테스트에는 부족. OI 특징을 제외하고 "
            "5d가격수익+3d펀딩추세 2특징으로 축소한다(SPEC.md 사전승인 폴백)."
        ),
        "prior_finding_reference": "research/wave15_diverse/signals15.py L48-51 (동일 엔드포인트, 동일 결론의 선행 라이브 확인)",
    }


def supplementary_oi_descriptive(symbols: tuple[str, ...] = OI_SUPPLEMENTARY_SYMBOLS, period: str = "1d", limit: int = 500) -> dict[str, Any]:
    """SPEC.md "가능한 범위로" -- 30일 창에서라도 OI 5d증가율과 이후 펀딩 추세의 관계를 최소한
    들여다본다. 심볼당 표본이 절대적으로 작아(~30일) 통계적 결론을 낼 수 없다 -- PASS/FAIL 판정에는
    쓰이지 않고, report에는 참고용 서술로만 남는다."""
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        try:
            oi_response = requests.get(OI_ENDPOINT, params={"symbol": symbol, "period": period, "limit": limit}, timeout=(5.0, 20.0))
            oi_response.raise_for_status()
            oi_data = oi_response.json()
            if not isinstance(oi_data, list) or len(oi_data) < 10:
                continue
            oi_series = pd.Series(
                {pd.to_datetime(int(item["timestamp"]), unit="ms", utc=True): float(item["sumOpenInterest"]) for item in oi_data}
            ).sort_index()

            funding_response = requests.get(FUNDING_HISTORY_ENDPOINT, params={"symbol": symbol, "limit": 1000}, timeout=(5.0, 20.0))
            funding_response.raise_for_status()
            funding_data = funding_response.json()
            if not isinstance(funding_data, list) or not funding_data:
                continue
            funding_native = pd.Series(
                {pd.to_datetime(int(item["fundingTime"]), unit="ms", utc=True): float(item["fundingRate"]) for item in funding_data}
            ).sort_index()
            funding_native = funding_native[funding_native.index >= oi_series.index[0] - pd.Timedelta(days=1)]
            if len(funding_native) < 6:
                continue

            realized_apr = funding_score(funding_native, LABEL_HORIZON_MAX_DAYS).resample("1D").last().reindex(oi_series.index)
            oi_growth_5d = (oi_series / oi_series.shift(PRICE_RET_WINDOW_DAYS) - 1.0).reindex(oi_series.index)
            forward_apr = realized_apr.shift(-LABEL_HORIZON_MIN_DAYS)
            paired = pd.concat([oi_growth_5d.rename("oi_growth_5d"), forward_apr.rename("forward_apr")], axis=1).dropna()
            if len(paired) < 5:
                continue
            has_variance = paired["oi_growth_5d"].std() > 0 and paired["forward_apr"].std() > 0
            correlation = float(paired["oi_growth_5d"].corr(paired["forward_apr"])) if has_variance else None
            rows.append({"symbol": symbol, "n_paired_days": int(len(paired)), "oi_growth_vs_forward_apr_correlation": correlation})
        except (requests.RequestException, ValueError, KeyError):
            continue
    correlations = [row["oi_growth_vs_forward_apr_correlation"] for row in rows if row["oi_growth_vs_forward_apr_correlation"] is not None]
    return {
        "note": "30일/심볼 표본 -- 통계적으로 결론 낼 수 없음. 참고용 서술 전용, PASS/FAIL 미반영.",
        "per_symbol": rows,
        "n_symbols_with_data": len(rows),
        "mean_correlation_across_symbols": float(np.mean(correlations)) if correlations else None,
    }


# ---------------------------------------------------------------------------
# 1) 유니버스 로드 (network 불필요 -- wave13/18과 동일 캐시 재사용, 신규 fetch 없음).
# ---------------------------------------------------------------------------


def load_universe() -> tuple[tuple[str, ...], dict[str, FundingMarket]]:
    symbols = ul.verify_cache_and_load_symbols(L4_CONFIG)
    markets = ul.load_markets_for_symbols(symbols)
    return symbols, markets


# ---------------------------------------------------------------------------
# 2) 프레임 구성. 펀딩 계열만 심볼별 루프(funding_score가 8h 해상도 원본 시리즈에서 동작하기
#    때문 -- engine13/18과 동일한 이유로 동일한 루프 모양을 쓴다). 가격/z-score/합성/신호/라벨은
#    프레임 전체에 대해 벡터화(pandas가 컬럼=심볼별로 자동 broadcast).
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UniverseFrames:
    spot_open: pd.DataFrame
    spot_close: pd.DataFrame
    perp_open: pd.DataFrame
    perp_close: pd.DataFrame
    funding_daily: pd.DataFrame
    realized_apr_7d: pd.DataFrame  # 라벨의 원천 -- L4와 동일한 7d 펀딩 APR
    composite: pd.DataFrame  # z(5d가격수익)+z(3d펀딩추세), 아직 shift 전
    label: pd.DataFrame  # 1.0/0.0/NaN(미래데이터 부족 = 알 수 없음)


def _rolling_zscore_frame(raw: pd.DataFrame, window: int = ZSCORE_WINDOW_DAYS) -> pd.DataFrame:
    mean = raw.rolling(window, min_periods=window).mean()
    std = raw.rolling(window, min_periods=window).std()
    z = (raw - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def _forward_max_with_validity(frame: pd.DataFrame, min_offset: int, max_offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frame.shift(-k), k in [min_offset,max_offset]의 행별 최댓값(NaN 제외) -- (forward_max,
    all_nan_mask)를 돌려준다. all_nan_mask는 창 안의 모든 offset이 미지(未知)인 셀(해당 심볼
    히스토리 끝에 너무 가까워 그만큼 미래를 볼 수 없는 경우)을 표시한다 -- 호출자는 이걸
    "스파이크 없음"이 아니라 "라벨 모름"으로 취급해야 한다."""
    shifted = [frame.shift(-k).to_numpy() for k in range(min_offset, max_offset + 1)]
    stack = np.stack(shifted, axis=0)
    all_nan = np.all(np.isnan(stack), axis=0)
    filled = np.where(np.isnan(stack), -np.inf, stack)
    with np.errstate(invalid="ignore"):
        forward_max = np.nanmax(filled, axis=0)
    forward_max = np.where(all_nan, np.nan, forward_max)
    return (
        pd.DataFrame(forward_max, index=frame.index, columns=frame.columns),
        pd.DataFrame(all_nan, index=frame.index, columns=frame.columns),
    )


def build_universe_frames(markets: dict[str, FundingMarket]) -> UniverseFrames:
    spot_open: dict[str, pd.Series] = {}
    spot_close: dict[str, pd.Series] = {}
    perp_open: dict[str, pd.Series] = {}
    perp_close: dict[str, pd.Series] = {}
    funding_daily: dict[str, pd.Series] = {}
    funding_apr_7d: dict[str, pd.Series] = {}
    funding_apr_3d: dict[str, pd.Series] = {}
    for symbol, market in markets.items():
        spot_daily = market.spot.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        perp_daily = market.perp.resample("1D").agg({"open": "first", "close": "last"}).dropna()
        spot_open[symbol] = spot_daily["open"]
        spot_close[symbol] = spot_daily["close"]
        perp_open[symbol] = perp_daily["open"]
        perp_close[symbol] = perp_daily["close"]
        funding_daily[symbol] = market.funding.resample("1D").sum()
        funding_apr_7d[symbol] = funding_score(market.funding, LABEL_HORIZON_MAX_DAYS).resample("1D").last()
        funding_apr_3d[symbol] = funding_score(market.funding, FUNDING_TREND_WINDOW_DAYS).resample("1D").last()

    # .astype(float) below is load-bearing, not cosmetic: a handful of L4-universe symbols
    # (e.g. AEROUSDT) have a PERP cache but zero rows of SPOT data -- an empty per-symbol
    # Series defaults to object dtype, which then infects that one column of the combined
    # wide frame (pandas keeps a column object-dtype if any contributing Series was object,
    # even though every actual value ends up NaN). Downstream code calls .to_numpy() +
    # np.isnan() on these frames (compute_return_comparison), which raises on true object
    # dtype -- forcing float64 here converts the all-NaN Python-float object column into a
    # normal NaN float64 column (verified empirically), and the existing .notna()-based
    # validity masks already exclude these symbol/days regardless (no economic content lost,
    # they were never tradeable -- no spot leg).
    spot_open_frame = pd.DataFrame(spot_open).sort_index().astype(float)
    idx = spot_open_frame.index
    columns = spot_open_frame.columns
    spot_close_frame = pd.DataFrame(spot_close).reindex(index=idx, columns=columns).astype(float)
    perp_open_frame = pd.DataFrame(perp_open).reindex(index=idx, columns=columns).astype(float)
    perp_close_frame = pd.DataFrame(perp_close).reindex(index=idx, columns=columns).astype(float)
    funding_daily_frame = pd.DataFrame(funding_daily).reindex(index=idx, columns=columns).fillna(0.0)
    funding_apr_7d_frame = pd.DataFrame(funding_apr_7d).reindex(index=idx, columns=columns)
    funding_apr_3d_frame = pd.DataFrame(funding_apr_3d).reindex(index=idx, columns=columns)

    price_ret_5d_frame = perp_close_frame / perp_close_frame.shift(PRICE_RET_WINDOW_DAYS) - 1.0
    funding_trend_3d_frame = funding_apr_3d_frame - funding_apr_3d_frame.shift(FUNDING_TREND_WINDOW_DAYS)
    z_price_frame = _rolling_zscore_frame(price_ret_5d_frame)
    z_trend_frame = _rolling_zscore_frame(funding_trend_3d_frame)
    composite_frame = z_price_frame + z_trend_frame  # 동일가중 합(계수 1.0/1.0) -- OI 항 제외(모듈 docstring 참조)

    forward_max_apr, forward_all_nan = _forward_max_with_validity(funding_apr_7d_frame, LABEL_HORIZON_MIN_DAYS, LABEL_HORIZON_MAX_DAYS)
    label_values = np.where(forward_all_nan.to_numpy(), np.nan, (forward_max_apr.to_numpy() > SPIKE_APR_THRESHOLD).astype(float))
    label_frame = pd.DataFrame(label_values, index=idx, columns=columns)

    return UniverseFrames(
        spot_open=spot_open_frame,
        spot_close=spot_close_frame,
        perp_open=perp_open_frame,
        perp_close=perp_close_frame,
        funding_daily=funding_daily_frame,
        realized_apr_7d=funding_apr_7d_frame,
        composite=composite_frame,
        label=label_frame,
    )


# ---------------------------------------------------------------------------
# 3) 신호/스파이크 이벤트 + 원가 포함 순방향 수익 프레임.
# ---------------------------------------------------------------------------


def _extract_true_cells(frame: pd.DataFrame) -> list[tuple[pd.Timestamp, str]]:
    rows, cols = np.where(frame.to_numpy())
    index = frame.index
    columns = frame.columns
    return [(pd.Timestamp(index[r]), str(columns[c])) for r, c in zip(rows.tolist(), cols.tolist())]


def signal_frames(frames: UniverseFrames, entry_z: float = ENTRY_Z_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(signal_bool, event_bool). signal_bool(t)는 composite.shift(1)을 쓴다 -- day t의 결정은
    day t-1 종가까지의 정보만 쓸 수 있다(research.wave1.fam_funding.carry_position과 동일한
    point-in-time 관례). event_bool은 연속 'on' 구간의 첫날만 표시한다(rising edge) -- 아래
    precision/return 분석은 매일이 아니라 이벤트 단위로 평가해서, 하나의 지속 국면을 여러 개의
    거의-동일한 '예측'으로 의사복제(pseudo-replicate)하지 않는다."""
    signal_score = frames.composite.shift(1)
    signal_bool = (signal_score > entry_z).fillna(False)
    # NOTE: must be shift(1, fill_value=False), NOT shift(1).fillna(False). A bool-dtype
    # frame's .shift(1) alone upcasts to object dtype to hold the introduced NaN at the
    # boundary; `~` on that object column then applies Python's *bitwise* ~ to each element
    # (~True==-2, ~False==-1 -- both nonzero/truthy), so a later `&` against a proper bool
    # frame silently treats every cell as True and event_bool degenerates to == signal_bool
    # (verified empirically while building this module -- tests/test_wave19.py pins this).
    # shift(..., fill_value=False) fills during the shift itself and keeps bool dtype throughout.
    event_bool = signal_bool & ~signal_bool.shift(1, fill_value=False)
    return signal_bool, event_bool


def spike_onset_frame(frames: UniverseFrames) -> pd.DataFrame:
    spike_bool = (frames.realized_apr_7d > SPIKE_APR_THRESHOLD).fillna(False)
    return spike_bool & ~spike_bool.shift(1, fill_value=False)


def forward_return_from_frames(frames: UniverseFrames, cost_rate_frame: pd.DataFrame, hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """단일 진입(t의 open)/단일 청산(t+H의 close) 실현수익 -- 순수 계산부(비용 프레임은 이미
    만들어져 주입된다). 스파이크 예측 신호를 검정하는 목적에는 engine13/18의 일별 재컴파운딩
    루프보다 이 방식(고정 노셔널의 단순 보유수익)이 더 정확하다 -- 포지션을 매일 재투자하지 않는
    개별 트레이드 평가이기 때문. `cost_rate_frame`을 인자로 분리한 이유는 tests/test_wave19.py가
    costs_measured/universe_liquidity 캐시를 건드리지 않고도(engine13/18 테스트 관례와 동일하게
    평평한 합성 비용 프레임을 주입해) 이 산식 자체를 고정된 값으로 검증할 수 있게 하기 위함이다."""
    exit_spot_close = frames.spot_close.shift(-hold_days)
    exit_perp_close = frames.perp_close.shift(-hold_days)
    price_leg_return = exit_spot_close / frames.spot_open - exit_perp_close / frames.perp_open
    funding_window_sum = sum(frames.funding_daily.shift(-k) for k in range(hold_days + 1))

    entry_cost = cost_rate_frame
    exit_cost = cost_rate_frame.shift(-hold_days)
    forward_return = price_leg_return + funding_window_sum - entry_cost - exit_cost

    valid = (
        frames.spot_open.notna()
        & frames.perp_open.notna()
        & exit_spot_close.notna()
        & exit_perp_close.notna()
        & entry_cost.notna()
        & exit_cost.notna()
    )
    return forward_return.where(valid)


def build_forward_return_frame(frames: UniverseFrames, symbols: tuple[str, ...], hold_days: int = HOLD_DAYS) -> pd.DataFrame:
    """비용은 wave13 실측매핑을 그대로 재사용(entry/exit 각 1회, engine13과 동일한 '2레그 왕복
    1회' 단가) -- research/wave13_liquidity/cache의 기존 캐시만 읽는다(신규 fetch 없음)."""
    mapping = costs_measured.fit_mapping()
    quote_volume_frame = ul.load_quote_volume_frame(symbols)
    cost_rate_frame = costs_measured.build_cost_rate_frame(quote_volume_frame, symbols, mapping, stress_multiplier=1.0)
    cost_rate_frame = cost_rate_frame.reindex(index=frames.spot_open.index, columns=frames.spot_open.columns)
    cost_rate_frame = cost_rate_frame.fillna(costs_measured.cost_rate_from_bp(mapping.worst_bp, 1.0))
    return forward_return_from_frames(frames, cost_rate_frame, hold_days)


# ---------------------------------------------------------------------------
# 4) precision / recall / lead-time.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PrecisionRecallResult:
    n_events_total: int
    n_events_known_label: int
    n_events_unknown_label: int
    precision: float | None
    base_rate: float | None
    n_valid_label_population: int
    n_spike_onsets: int
    recall: float | None
    recall_hits: int
    lead_time_days: dict[str, float | int | None]
    signal_records: tuple[tuple[pd.Timestamp, str, float], ...]  # (date, symbol, label) -- known-label events only, feeds the bootstrap


def compute_precision_recall(frames: UniverseFrames, signal_bool: pd.DataFrame, event_bool: pd.DataFrame) -> PrecisionRecallResult:
    events = _extract_true_cells(event_bool)
    signal_records: list[tuple[pd.Timestamp, str, float]] = []
    for date, symbol in events:
        label_value = frames.label.at[date, symbol]
        if pd.isna(label_value):
            continue
        signal_records.append((date, symbol, float(label_value)))

    label_values_population = frames.label.to_numpy()
    valid_population_mask = ~np.isnan(label_values_population)
    n_valid_population = int(valid_population_mask.sum())
    base_rate = float(np.nanmean(label_values_population)) if n_valid_population > 0 else None
    precision = float(np.mean([lbl for _, _, lbl in signal_records])) if signal_records else None

    onsets = _extract_true_cells(spike_onset_frame(frames))
    recall_hits = 0
    lead_times: list[int] = []
    signal_index = signal_bool.index
    for date, symbol in onsets:
        window_dates = [date - pd.Timedelta(days=k) for k in range(LABEL_HORIZON_MIN_DAYS, LABEL_HORIZON_MAX_DAYS + 1)]
        active_days = [d for d in window_dates if d in signal_index and bool(signal_bool.at[d, symbol])]
        if active_days:
            recall_hits += 1
            lead_times.append(int((date - min(active_days)).days))
    recall = (recall_hits / len(onsets)) if onsets else None

    lead_array = np.asarray(lead_times, dtype=float)
    lead_summary: dict[str, float | int | None] = {
        "n": len(lead_times),
        "mean": float(lead_array.mean()) if lead_array.size else None,
        "median": float(np.median(lead_array)) if lead_array.size else None,
        "p25": float(np.quantile(lead_array, 0.25)) if lead_array.size else None,
        "p75": float(np.quantile(lead_array, 0.75)) if lead_array.size else None,
        "min": float(lead_array.min()) if lead_array.size else None,
        "max": float(lead_array.max()) if lead_array.size else None,
    }

    return PrecisionRecallResult(
        n_events_total=len(events),
        n_events_known_label=len(signal_records),
        n_events_unknown_label=len(events) - len(signal_records),
        precision=precision,
        base_rate=base_rate,
        n_valid_label_population=n_valid_population,
        n_spike_onsets=len(onsets),
        recall=recall,
        recall_hits=recall_hits,
        lead_time_days=lead_summary,
        signal_records=tuple(signal_records),
    )


# ---------------------------------------------------------------------------
# 5) 조건부 수익 vs 무작위 진입 (부트스트랩).
# ---------------------------------------------------------------------------


def _bootstrap_one_sample(values: np.ndarray, seed: int, paths: int = BOOT_PATHS) -> dict[str, Any]:
    """H0: mean(values) <= 0. p_le_zero = P(부트스트랩 평균 <= 0)."""
    if values.size == 0:
        return {"mean": None, "p05": None, "p95": None, "p_le_zero": None, "win_rate": None, "n": 0, "paths": paths}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(paths, values.size))
    boot_means = values[idx].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "p05": float(np.quantile(boot_means, 0.05)),
        "p95": float(np.quantile(boot_means, 0.95)),
        "p_le_zero": float(np.mean(boot_means <= 0.0)),
        "win_rate": float(np.mean(values > 0.0)),
        "n": int(values.size),
        "paths": paths,
    }


def _bootstrap_two_sample(a: np.ndarray, b: np.ndarray, seed: int, paths: int = BOOT_PATHS, sample_size: int | None = None) -> dict[str, Any]:
    """두 그룹의 평균 차(a-b) 부트스트랩. `sample_size`(기본 a.size)로 양쪽을 동일 크기로
    리샘플 -- b(무작위 모집단 전체)가 아주 클 수 있어(수십만) 매 반복 b.size개를 통째로
    리샘플하면 메모리가 터진다. a와 '같은 개수(a.size)만큼 무작위로 뽑았다면'이 SPEC.md의
    '무작위 진입'과 정확히 대응하는 비교이기도 하다."""
    if a.size == 0 or b.size == 0:
        return {"mean_a": None, "mean_b": None, "mean_diff": None, "p05": None, "p95": None, "p_diff_le_zero": None, "n_a": int(a.size), "n_b": int(b.size), "paths": paths}
    n = sample_size if sample_size is not None else a.size
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, a.size, size=(paths, n))
    idx_b = rng.integers(0, b.size, size=(paths, n))
    diffs = a[idx_a].mean(axis=1) - b[idx_b].mean(axis=1)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(a.mean() - b.mean()),
        "p05": float(np.quantile(diffs, 0.05)),
        "p95": float(np.quantile(diffs, 0.95)),
        "p_diff_le_zero": float(np.mean(diffs <= 0.0)),
        "n_a": int(a.size),
        "n_b": int(b.size),
        "resample_n": int(n),
        "paths": paths,
    }


def _bootstrap_precision(labels: np.ndarray, base_rate: float | None, seed: int, paths: int = BOOT_PATHS) -> dict[str, Any]:
    """H0: precision <= base_rate. base_rate는 모집단(수십만 셀) 평균이라 표본오차가 무시할
    만큼 작다고 보고 고정 상수로 취급한다(신호 이벤트 쪽만 리샘플)."""
    if labels.size == 0 or base_rate is None:
        return {"precision": None, "base_rate": base_rate, "lift": None, "p05": None, "p95": None, "p_le_base_rate": None, "n": int(labels.size), "paths": paths}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, labels.size, size=(paths, labels.size))
    boot_precision = labels[idx].mean(axis=1)
    precision = float(labels.mean())
    return {
        "precision": precision,
        "base_rate": float(base_rate),
        "lift": (precision / base_rate) if base_rate > 0.0 else None,
        "p05": float(np.quantile(boot_precision, 0.05)),
        "p95": float(np.quantile(boot_precision, 0.95)),
        "p_le_base_rate": float(np.mean(boot_precision <= base_rate)),
        "n": int(labels.size),
        "paths": paths,
    }


@dataclass(frozen=True, slots=True)
class ReturnComparisonResult:
    signal_returns_summary: dict[str, Any]
    unconditional_random_bootstrap: dict[str, Any]
    time_matched_paired_bootstrap: dict[str, Any]
    n_events_with_return: int
    n_events_dropped_no_return_data: int
    n_events_dropped_no_paired_control: int


def compute_return_comparison(frames: UniverseFrames, event_bool: pd.DataFrame, forward_return: pd.DataFrame, seed: int = SEED) -> ReturnComparisonResult:
    """1차(결정) 비교 = time_matched_paired: 같은 날 다른(무작위) 심볼과 짝지어 대조 -- 특정
    시기(강세장 등)에 신호가 몰려서 생기는 착시를 제거한다. unconditional(전체 모집단 대조)은
    맥락 참고용으로만 병기."""
    events = _extract_true_cells(event_bool)
    return_valid = forward_return.notna()

    signal_return_records: list[tuple[pd.Timestamp, str, float]] = []
    for date, symbol in events:
        value = forward_return.at[date, symbol]
        if pd.isna(value):
            continue
        signal_return_records.append((date, symbol, float(value)))
    signal_returns = np.asarray([value for *_rest, value in signal_return_records], dtype=float)

    all_valid_returns = forward_return.to_numpy()[return_valid.to_numpy()]
    all_valid_returns = all_valid_returns[~np.isnan(all_valid_returns)]

    rng = np.random.default_rng(seed)
    paired_diffs: list[float] = []
    n_dropped_no_pair = 0
    for date, symbol, signal_return in signal_return_records:
        row = return_valid.loc[date]
        other_symbols = row.index[row.to_numpy() & (row.index != symbol)]
        if len(other_symbols) == 0:
            n_dropped_no_pair += 1
            continue
        pick = other_symbols[int(rng.integers(0, len(other_symbols)))]
        control_return = float(forward_return.at[date, pick])
        paired_diffs.append(signal_return - control_return)
    paired_array = np.asarray(paired_diffs, dtype=float)

    return ReturnComparisonResult(
        signal_returns_summary=_bootstrap_one_sample(signal_returns, seed + 107),
        unconditional_random_bootstrap=_bootstrap_two_sample(signal_returns, all_valid_returns, seed + 101),
        time_matched_paired_bootstrap=_bootstrap_one_sample(paired_array, seed + 103),
        n_events_with_return=int(signal_returns.size),
        n_events_dropped_no_return_data=len(events) - len(signal_return_records),
        n_events_dropped_no_paired_control=n_dropped_no_pair,
    )


def threshold_sensitivity_table(frames: UniverseFrames, forward_return: pd.DataFrame, grid: tuple[float, ...] = THRESHOLD_SENSITIVITY_GRID) -> list[dict[str, Any]]:
    """참고용: theta를 바꾸면 precision/lift/평균수익이 어떻게 움직이는지. PASS/FAIL 판정은
    오직 ENTRY_Z_THRESHOLD(1.0) 한 번의 결과만 쓴다 -- 이 표는 그 결정 이후 부가 정보로만
    싣는다(사후에 더 유리한 theta로 갈아타지 않는다)."""
    rows: list[dict[str, Any]] = []
    for theta in grid:
        signal_bool, event_bool = signal_frames(frames, entry_z=theta)
        pr = compute_precision_recall(frames, signal_bool, event_bool)
        labels = np.asarray([lbl for _, _, lbl in pr.signal_records], dtype=float)
        precision_boot = _bootstrap_precision(labels, pr.base_rate, SEED + int(theta * 1000) + 1)
        returns = compute_return_comparison(frames, event_bool, forward_return, seed=SEED + int(theta * 1000) + 2)
        rows.append(
            {
                "theta": theta,
                "n_events": pr.n_events_total,
                "n_events_known_label": pr.n_events_known_label,
                "precision": pr.precision,
                "base_rate": pr.base_rate,
                "lift": precision_boot["lift"],
                "recall": pr.recall,
                "signal_mean_return": returns.signal_returns_summary["mean"],
                "paired_mean_diff": returns.time_matched_paired_bootstrap["mean"],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 6) 최상위 오케스트레이션.
# ---------------------------------------------------------------------------


def run_signal_test() -> dict[str, Any]:
    oi_check = check_oi_history_limit()
    oi_supplementary = supplementary_oi_descriptive()

    symbols, markets = load_universe()
    frames = build_universe_frames(markets)
    signal_bool, event_bool = signal_frames(frames, entry_z=ENTRY_Z_THRESHOLD)
    forward_return = build_forward_return_frame(frames, symbols, HOLD_DAYS)

    precision_recall = compute_precision_recall(frames, signal_bool, event_bool)
    labels_array = np.asarray([lbl for _, _, lbl in precision_recall.signal_records], dtype=float)
    precision_boot = _bootstrap_precision(labels_array, precision_recall.base_rate, SEED + 1)
    returns = compute_return_comparison(frames, event_bool, forward_return, seed=SEED + 2)
    sensitivity = threshold_sensitivity_table(frames, forward_return)

    precision_significant = (
        precision_boot["precision"] is not None
        and precision_boot["base_rate"] is not None
        and precision_boot["precision"] > precision_boot["base_rate"]
        and precision_boot["p_le_base_rate"] is not None
        and precision_boot["p_le_base_rate"] < SIGNIFICANCE_ALPHA
    )
    paired = returns.time_matched_paired_bootstrap
    return_significant = paired["mean"] is not None and paired["mean"] > 0.0 and paired["p_le_zero"] is not None and paired["p_le_zero"] < SIGNIFICANCE_ALPHA
    stage1_pass = bool(precision_significant and return_significant)

    reasons: list[str] = []
    if precision_recall.n_events_known_label == 0:
        reasons.append("신호가 한 번도 발화하지 않았거나(threshold 미도달) 전부 라벨 미지 구간 -- 검정 불가")
    if not precision_significant:
        reasons.append("정밀도가 base rate 대비 유의하게 높지 않음(precision<=base_rate 이거나 p_le_base_rate>=0.05)")
    if not return_significant:
        reasons.append("신호 조건부 수익이 무작위(동일자 타심볼 대조) 대비 유의하게 높지 않음(평균<=0 이거나 p_le_zero>=0.05)")

    n_signal_events = precision_recall.n_events_known_label
    low_power_warning = n_signal_events < 30

    honesty_notes = [
        "OI 5d증가율 특징은 뺐다 -- Binance openInterestHist가 라이브로도 ~30일만 반환(2019-2026 "
        "7년 백테스트에 미달, oi_history_check 참조). SPEC.md가 사전승인한 2특징(5d가격수익+3d "
        "펀딩추세) 폴백을 썼다.",
        "funding_trend_3d 특징은 라벨(미래 7d 펀딩 APR)과 같은 펀딩 시계열에서 파생된다 -- 둘 다 "
        "'펀딩이 최근 오르는 중이면 계속 오른다'는 자기상관을 일부 공유할 수 있다. 이건 진짜 "
        "예측력과 완전히 구별하기 어렵다 -- precision_vs_base_rate와 time_matched_paired_return "
        "검정이 유의하게 나와도, '독립적인 조기경보'라기보다 '이미 시작된 추세의 연장 포착'일 "
        "가능성이 있다는 점을 report에 명시한다.",
        f"신호 이벤트 표본 n={n_signal_events}"
        + (" -- 30 미만, 검정력 낮음. 통계적으로 유의해도 소표본 결과라는 한계를 함께 고려해야 한다." if low_power_warning else "."),
        "theta=1.0은 이 파일 작성 시점에 고정한 값이고 결과를 본 뒤 바꾸지 않았다. 0.5/1.5/2.0 "
        "민감도 표는 참고용일 뿐 PASS/FAIL 판정에 관여하지 않는다.",
        "signal_events는 연속 'on' 구간의 첫날만 센다(중복 계수 방지) -- recall의 '탐지' 판정은 "
        "이벤트가 아니라 signal_bool 자체를 쓴다(더 관대한 재현율 정의, precision과 의도적으로 비대칭).",
        "다중검정: 이 wave 자체가 SPEC.md 사전등록상 '누적 116회'에 포함되는 신규 시행이다. 여기서 "
        "쓰는 유의성 기준(alpha=0.05)은 이 116회 전체에 대해 보정되지 않은 단일 검정 기준이다 -- "
        "참고용 DSR 보정과 동일하게, wave10-18의 공통 원칙상 이 wave 자신의 승격 판정에는 그 "
        "보정을 적용하지 않지만 한계로 명시한다.",
    ]

    payload: dict[str, Any] = {
        "stage": "signal_test",
        "spec_ref": "research/wave19_regime/SPEC.md 1단계",
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "methodology": {
            "feature_formula": "z(5d 가격수익) + z(3d 펀딩추세) -- 동일가중 합(계수 1.0/1.0), OI 항 제외(사유는 oi_history_check)",
            "zscore_window_days": ZSCORE_WINDOW_DAYS,
            "zscore_window_source": "research/wave15_diverse/signals15.py ZSCORE_WINDOW_DAYS 승계",
            "entry_threshold_z": ENTRY_Z_THRESHOLD,
            "entry_threshold_note": "이 파일에 사전 고정, 결과를 본 뒤 조정하지 않음. 민감도는 threshold_sensitivity_reference_only(참고용)",
            "spike_label": f"신호 시점 t 이후 [t+{LABEL_HORIZON_MIN_DAYS}, t+{LABEL_HORIZON_MAX_DAYS}]일 내 7d 펀딩 APR > {SPIKE_APR_THRESHOLD:.0%}",
            "universe": "research.wave13_liquidity.configs13 L4 (top200 거래대금, 12mo 히스토리 하한) -- wave13/18과 동일 캐시 재사용, 신규 fetch 없음",
            "hold_days_for_return_test": HOLD_DAYS,
            "cost_model": "research.wave13_liquidity.costs_measured (Bitget 실측, wave13 승계) -- 진입/청산 각 1회(왕복 2회) 부과",
            "return_comparison_design": "1차 판정 = time-matched paired(같은 날 다른 심볼) 대조. unconditional(전체 모집단) 대조는 참고용 병기.",
        },
        "oi_history_check": oi_check,
        "oi_supplementary_descriptive_30d": oi_supplementary,
        "population": {
            "n_symbols": len(symbols),
            "date_range": [str(pd.Timestamp(frames.spot_open.index[0]).date()), str(pd.Timestamp(frames.spot_open.index[-1]).date())],
            "n_valid_label_population": precision_recall.n_valid_label_population,
            "base_rate": precision_recall.base_rate,
        },
        "signal_events": {
            "n_events_total": precision_recall.n_events_total,
            "n_events_known_label": precision_recall.n_events_known_label,
            "n_events_unknown_label_excluded": precision_recall.n_events_unknown_label,
        },
        "precision_recall_leadtime": {
            "precision": precision_recall.precision,
            "recall": precision_recall.recall,
            "recall_hits": precision_recall.recall_hits,
            "n_spike_onsets": precision_recall.n_spike_onsets,
            "lead_time_days": precision_recall.lead_time_days,
        },
        "precision_vs_base_rate_bootstrap": precision_boot,
        "return_comparison": {
            "signal_group": returns.signal_returns_summary,
            "unconditional_random_bootstrap": returns.unconditional_random_bootstrap,
            "time_matched_paired_bootstrap": returns.time_matched_paired_bootstrap,
            "n_events_with_return": returns.n_events_with_return,
            "n_events_dropped_no_return_data": returns.n_events_dropped_no_return_data,
            "n_events_dropped_no_paired_control": returns.n_events_dropped_no_paired_control,
        },
        "threshold_sensitivity_reference_only": sensitivity,
        "pass_fail": {
            "precision_significant": precision_significant,
            "return_significant_time_matched": return_significant,
            "stage1_pass": stage1_pass,
            "low_power_warning_n_lt_30": low_power_warning,
            "reasons_if_fail": reasons,
        },
        "honesty_notes": honesty_notes,
    }
    return payload


def main() -> int:
    payload = run_signal_test()
    save_json(RESULTS_DIR / "signal_report.json", json_safe(payload))
    pf = payload["pass_fail"]
    pr = payload["precision_recall_leadtime"]
    rc = payload["return_comparison"]
    print(
        f"signal_test: n_events={payload['signal_events']['n_events_known_label']} "
        f"precision={pr['precision']} base_rate={payload['population']['base_rate']} "
        f"lift={payload['precision_vs_base_rate_bootstrap']['lift']} "
        f"paired_mean_diff={rc['time_matched_paired_bootstrap']['mean']} "
        f"p_le_zero={rc['time_matched_paired_bootstrap']['p_le_zero']} "
        f"-> STAGE1={'PASS' if pf['stage1_pass'] else 'FAIL'}"
    )
    return 0 if pf["stage1_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOT_PATHS",
    "ENTRY_Z_THRESHOLD",
    "FUNDING_TREND_WINDOW_DAYS",
    "HOLD_DAYS",
    "LABEL_HORIZON_MAX_DAYS",
    "LABEL_HORIZON_MIN_DAYS",
    "PRICE_RET_WINDOW_DAYS",
    "SIGNIFICANCE_ALPHA",
    "SPIKE_APR_THRESHOLD",
    "THRESHOLD_SENSITIVITY_GRID",
    "ZSCORE_WINDOW_DAYS",
    "PrecisionRecallResult",
    "ReturnComparisonResult",
    "UniverseFrames",
    "build_forward_return_frame",
    "build_universe_frames",
    "check_oi_history_limit",
    "compute_precision_recall",
    "compute_return_comparison",
    "forward_return_from_frames",
    "load_universe",
    "run_signal_test",
    "signal_frames",
    "spike_onset_frame",
    "supplementary_oi_descriptive",
    "threshold_sensitivity_table",
]

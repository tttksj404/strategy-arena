# Wave-19 STAGE 1 (signal_test.py) 검증. 2단계(engine19/gates19/R0-R4)는 신호 검정이 FAIL한
# 관계로 만들어지지 않았다(research/wave19_regime/report/wave19_report.md 참조) -- 이 파일은
# 실제로 존재하는 stage-1 로직만 검증한다.
#
# 아래 중 세 개는 개발 중 실제로 발견·수정한 버그를 고정하는 회귀 테스트다:
#   - test_signal_event_frame_marks_only_the_first_day_of_each_active_streak
#   - test_spike_onset_frame_marks_only_the_first_day_of_each_spike_streak
#     (bool 프레임을 .shift(1).fillna(False) 한 뒤 `~`를 적용하면 object dtype으로 승격되고
#     Python의 비트 반전(~True==-2, ~False==-1, 둘 다 truthy)이 적용돼 event_bool이
#     signal_bool과 항상 같아지는 버그 -- signal_test.py는 shift(1, fill_value=False)로 고쳤다)
#   - test_build_universe_frames_handles_a_symbol_with_zero_spot_rows_without_crashing
#     (스팟 데이터가 0행인 심볼(L4 유니버스의 AEROUSDT 실제 사례)이 object dtype 컬럼을 만들어
#     np.isnan()에서 TypeError를 내던 버그 -- .astype(float)로 고쳤다)
#   - test_bootstrap_two_sample_handles_a_much_larger_second_population_without_blowing_up
#     (무작위 대조군 전체(수십만 개)를 매 부트스트랩 반복마다 통째로 리샘플하면 메모리가
#     터지는 버그 -- sample_size를 신호 쪽 표본 크기로 고정해 고쳤다)

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    repository_root = Path(__file__).resolve().parents[3]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import pytest

from research.wave1.fam_funding import FundingMarket, funding_score
from research.wave19_regime import signal_test as st


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _synthetic_market(periods: int = 200, seed: int = 0, symbols: tuple[str, ...] = ("AAAUSDT", "BBBUSDT")) -> dict[str, FundingMarket]:
    """Seeded random-walk prices + noisy funding -- enough variance for non-degenerate rolling
    z-scores (unlike test_wave18.py's own deterministic geometric fixtures, which are too smooth
    for a 90d rolling std to ever be nonzero)."""
    rng = np.random.default_rng(seed)
    daily_index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    funding_index = pd.date_range("2023-01-01", periods=periods * 3, freq="8h", tz="UTC")
    markets: dict[str, FundingMarket] = {}
    for i, symbol in enumerate(symbols):
        base = 100.0 * (1 + i)
        returns = rng.normal(0.0003, 0.02, periods)
        close = base * np.cumprod(1.0 + returns)
        open_ = np.concatenate([[base], close[:-1]])
        spot = pd.DataFrame({"open": open_, "close": close}, index=daily_index)
        perp = pd.DataFrame({"open": open_ * 0.999, "close": close * 0.999}, index=daily_index)
        funding = pd.Series(rng.normal(0.0001, 0.0004, periods * 3), index=funding_index, name="funding_rate")
        markets[symbol] = FundingMarket(spot=spot, perp=perp, funding=funding)
    return markets


def _minimal_frames(index: pd.DatetimeIndex, columns: list[str], **overrides: pd.DataFrame) -> st.UniverseFrames:
    """UniverseFrames with every field defaulted to a harmless constant frame -- tests override
    only the field(s) the function under test actually reads."""
    ones = pd.DataFrame(1.0, index=index, columns=columns)
    zeros = pd.DataFrame(0.0, index=index, columns=columns)
    defaults: dict[str, pd.DataFrame] = dict(
        spot_open=ones,
        spot_close=ones,
        perp_open=ones,
        perp_close=ones,
        funding_daily=zeros,
        realized_apr_7d=zeros,
        composite=zeros,
        label=zeros,
    )
    defaults.update(overrides)
    return st.UniverseFrames(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1) 사전 고정 상수.
# ---------------------------------------------------------------------------


def test_frozen_constants_match_spec() -> None:
    assert st.SPIKE_APR_THRESHOLD == pytest.approx(0.15)
    assert st.LABEL_HORIZON_MIN_DAYS == 3
    assert st.LABEL_HORIZON_MAX_DAYS == 7
    assert st.PRICE_RET_WINDOW_DAYS == 5
    assert st.FUNDING_TREND_WINDOW_DAYS == 3
    assert st.ENTRY_Z_THRESHOLD == pytest.approx(1.0)
    assert st.HOLD_DAYS == st.LABEL_HORIZON_MAX_DAYS
    assert st.SIGNIFICANCE_ALPHA == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 2) 계수 사전고정 -- composite는 두 z-score의 "동일가중 합"(계수 1.0/1.0)이지, wave15 C1처럼
#    0.5/0.5 평균이거나 다른 어떤 가중도 아니다.
# ---------------------------------------------------------------------------


def test_composite_is_unweighted_sum_of_price_and_funding_trend_zscores() -> None:
    markets = _synthetic_market(periods=200, seed=1)
    frames = st.build_universe_frames(markets)

    perp_close_frame = pd.DataFrame(
        {symbol: market.perp.resample("1D").agg({"close": "last"})["close"] for symbol, market in markets.items()}
    ).reindex(frames.composite.index)
    funding_apr_3d_frame = pd.DataFrame(
        {symbol: funding_score(market.funding, st.FUNDING_TREND_WINDOW_DAYS).resample("1D").last() for symbol, market in markets.items()}
    ).reindex(frames.composite.index)
    price_ret = perp_close_frame / perp_close_frame.shift(st.PRICE_RET_WINDOW_DAYS) - 1.0
    funding_trend = funding_apr_3d_frame - funding_apr_3d_frame.shift(st.FUNDING_TREND_WINDOW_DAYS)
    expected = st._rolling_zscore_frame(price_ret) + st._rolling_zscore_frame(funding_trend)

    pd.testing.assert_frame_equal(frames.composite, expected, check_names=False)


# ---------------------------------------------------------------------------
# 3) 룩어헤드 방지 -- 신호(feature) 쪽.
# ---------------------------------------------------------------------------


def test_composite_signal_has_no_lookahead_from_future_price_or_funding_changes() -> None:
    markets_a = _synthetic_market(periods=200, seed=7)
    frames_a = st.build_universe_frames(markets_a)

    markets_b = _synthetic_market(periods=200, seed=7)  # byte-identical baseline
    late_perp_ts = markets_b["AAAUSDT"].perp.index[190]
    markets_b["AAAUSDT"].perp.loc[late_perp_ts, "close"] *= 5.0
    late_funding_ts = markets_b["AAAUSDT"].funding.index[190 * 3]
    markets_b["AAAUSDT"].funding.loc[late_funding_ts] += 0.05
    frames_b = st.build_universe_frames(markets_b)

    early_cutoff = frames_a.composite.index[100]  # >90d z-score window clear of day-0, well before day-190 perturbation
    pd.testing.assert_series_equal(
        frames_a.composite.loc[:early_cutoff, "AAAUSDT"], frames_b.composite.loc[:early_cutoff, "AAAUSDT"], check_names=False
    )
    # sanity: the fixture actually exercises a change somewhere at/after the perturbation
    assert not frames_a.composite.loc[early_cutoff:, "AAAUSDT"].equals(frames_b.composite.loc[early_cutoff:, "AAAUSDT"])


# ---------------------------------------------------------------------------
# 4) 룩어헤드 방지 -- 라벨 쪽 + 정확한 윈도우 경계([t+3, t+7] 포함).
# ---------------------------------------------------------------------------


def _label_from_apr(apr: pd.Series) -> pd.Series:
    frame = apr.to_frame("SYM")
    forward_max, all_nan = st._forward_max_with_validity(frame, st.LABEL_HORIZON_MIN_DAYS, st.LABEL_HORIZON_MAX_DAYS)
    values = np.where(all_nan.to_numpy(), np.nan, (forward_max.to_numpy() > st.SPIKE_APR_THRESHOLD).astype(float))
    return pd.Series(values.ravel(), index=apr.index, name="SYM")


def test_label_has_no_lookahead_and_respects_exact_window_bounds() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="D", tz="UTC")
    apr = pd.Series(0.05, index=idx)
    spike_day = idx[20]
    apr.loc[spike_day] = 0.20  # single-day spike, above threshold

    label = _label_from_apr(apr)
    expected_positive_days = {spike_day - pd.Timedelta(days=k) for k in range(st.LABEL_HORIZON_MIN_DAYS, st.LABEL_HORIZON_MAX_DAYS + 1)}
    known_cutoff = idx[-1] - pd.Timedelta(days=st.LABEL_HORIZON_MAX_DAYS)
    for day in idx:
        if day in expected_positive_days:
            assert label.loc[day] == 1.0, day
        elif day <= known_cutoff:
            assert label.loc[day] == 0.0, day
    # "unknown" (all_nan) triggers only once even the NEAREST offset (t+MIN_DAYS) falls off the
    # end of the frame -- i.e. exactly the last LABEL_HORIZON_MIN_DAYS rows, not MAX_DAYS: a day
    # that can still see SOME (if not all) of its forward window gets a real 0/1, not NaN (a
    # deliberate, documented leniency in _forward_max_with_validity, not an off-by-one bug).
    assert label.loc[idx[-st.LABEL_HORIZON_MIN_DAYS] :].isna().all()
    assert not label.loc[idx[-st.LABEL_HORIZON_MAX_DAYS] : idx[-st.LABEL_HORIZON_MIN_DAYS - 1]].isna().any()

    # a change WELL OUTSIDE an early day's own forward window must not move that day's value
    apr_far = apr.copy()
    apr_far.loc[idx[35]] = 0.99
    label_far = _label_from_apr(apr_far)
    assert label_far.loc[idx[5]] == label.loc[idx[5]]


# ---------------------------------------------------------------------------
# 5)+6) 회귀: rising-edge(첫날만) 추출 -- shift(1).fillna(False) + `~` object-dtype 버그.
# ---------------------------------------------------------------------------


def test_signal_event_frame_marks_only_the_first_day_of_each_active_streak() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    composite = pd.DataFrame({"AAAUSDT": [0.0, 2.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0]}, index=idx)
    frames = _minimal_frames(idx, ["AAAUSDT"], composite=composite)

    signal_bool, event_bool = st.signal_frames(frames, entry_z=1.0)

    assert signal_bool["AAAUSDT"].tolist() == [False, False, True, True, True, False, True, False, True, True]
    assert event_bool["AAAUSDT"].tolist() == [False, False, True, False, False, False, True, False, True, False]
    # the bug this pins made event_bool identically equal signal_bool -- guard that directly too
    assert int(event_bool["AAAUSDT"].sum()) < int(signal_bool["AAAUSDT"].sum())


def test_spike_onset_frame_marks_only_the_first_day_of_each_spike_streak() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC")
    apr = pd.DataFrame({"AAAUSDT": [0.05, 0.20, 0.20, 0.05, 0.20, 0.05]}, index=idx)
    frames = _minimal_frames(idx, ["AAAUSDT"], realized_apr_7d=apr)

    onset = st.spike_onset_frame(frames)

    assert onset["AAAUSDT"].tolist() == [False, True, False, False, True, False]


# ---------------------------------------------------------------------------
# 7) precision / recall / lead-time -- hand-worked example.
# ---------------------------------------------------------------------------


def test_compute_precision_recall_matches_hand_worked_example() -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="D", tz="UTC")
    columns = ["AAAUSDT", "BBBUSDT"]

    composite = pd.DataFrame(0.0, index=idx, columns=columns)
    composite.loc[idx[4], "AAAUSDT"] = 2.0  # -> AAA signal/event fires on day idx[5] only
    composite.loc[idx[9], "BBBUSDT"] = 2.0  # -> BBB signal/event fires on day idx[10] only

    realized_apr = pd.DataFrame(0.05, index=idx, columns=columns)
    realized_apr.loc[idx[8], "AAAUSDT"] = 0.20  # AAA spike onset on day idx[8] (window back to idx[8]-7..idx[8]-3 catches idx[5])
    realized_apr.loc[idx[19], "BBBUSDT"] = 0.20  # BBB spike onset on day idx[19] (window idx[12]..idx[16] MISSES idx[10])

    label = pd.DataFrame(0.0, index=idx, columns=columns)
    label.loc[idx[5], "AAAUSDT"] = 1.0  # the AAA event is a true positive; the BBB event (day idx[10]) stays a false positive (0.0)

    frames = _minimal_frames(idx, columns, composite=composite, realized_apr_7d=realized_apr, label=label)
    signal_bool, event_bool = st.signal_frames(frames, entry_z=1.0)
    result = st.compute_precision_recall(frames, signal_bool, event_bool)

    assert result.n_events_total == 2
    assert result.n_events_known_label == 2
    assert result.precision == pytest.approx(0.5)  # 1 true positive / 2 events
    assert result.n_valid_label_population == 40  # 20 days x 2 symbols, all known
    assert result.base_rate == pytest.approx(1.0 / 40.0)
    assert result.n_spike_onsets == 2
    assert result.recall_hits == 1
    assert result.recall == pytest.approx(0.5)
    assert result.lead_time_days["n"] == 1
    assert result.lead_time_days["mean"] == pytest.approx(3.0)  # idx[8] - idx[5]


# ---------------------------------------------------------------------------
# 8) 순방향 실현수익 산식 -- 손으로 계산한 값과 대조.
# ---------------------------------------------------------------------------


def test_forward_return_from_frames_matches_hand_computed_pnl() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    columns = ["AAAUSDT"]
    spot_open = pd.DataFrame(100.0, index=idx, columns=columns)
    spot_close = pd.DataFrame(100.0, index=idx, columns=columns)
    spot_close.loc[idx[3], "AAAUSDT"] = 110.0
    perp_open = pd.DataFrame(100.0, index=idx, columns=columns)
    perp_close = pd.DataFrame(100.0, index=idx, columns=columns)
    perp_close.loc[idx[3], "AAAUSDT"] = 105.0
    funding_daily = pd.DataFrame(0.001, index=idx, columns=columns)
    frames = _minimal_frames(
        idx, columns, spot_open=spot_open, spot_close=spot_close, perp_open=perp_open, perp_close=perp_close, funding_daily=funding_daily
    )
    cost_rate_frame = pd.DataFrame(0.0006, index=idx, columns=columns)

    forward_return = st.forward_return_from_frames(frames, cost_rate_frame, hold_days=3)

    expected = (110.0 / 100.0 - 105.0 / 100.0) + 4 * 0.001 - 2 * 0.0006  # price leg + 4 days funding(0..3 incl.) - entry/exit cost
    assert forward_return.loc[idx[0], "AAAUSDT"] == pytest.approx(expected)
    assert forward_return.loc[idx[7] :, "AAAUSDT"].isna().all()  # last 3 days can't see hold_days=3 ahead -- must be NaN, not 0


# ---------------------------------------------------------------------------
# 9) 회귀: 스팟 데이터 0행 심볼(AEROUSDT 실제 사례) -- object dtype -> np.isnan() 크래시.
# ---------------------------------------------------------------------------


def test_build_universe_frames_handles_a_symbol_with_zero_spot_rows_without_crashing() -> None:
    markets = _synthetic_market(periods=120, seed=3, symbols=("AAAUSDT",))
    empty_ohlc = pd.DataFrame({"open": pd.Series(dtype=float), "close": pd.Series(dtype=float)}, index=pd.DatetimeIndex([], tz="UTC"))
    ghost_funding_index = pd.date_range("2023-01-01", periods=120 * 3, freq="8h", tz="UTC")
    markets["GHOSTUSDT"] = FundingMarket(
        spot=empty_ohlc,
        perp=markets["AAAUSDT"].perp.copy(),
        funding=pd.Series(0.0001, index=ghost_funding_index, name="funding_rate"),
    )

    frames = st.build_universe_frames(markets)

    assert frames.spot_open["GHOSTUSDT"].dtype == np.float64
    assert frames.spot_open["GHOSTUSDT"].isna().all()
    np.isnan(frames.spot_open.to_numpy())  # must not raise TypeError (pre-fix: object dtype)

    cost_rate_frame = pd.DataFrame(0.0005, index=frames.spot_open.index, columns=frames.spot_open.columns)
    forward_return = st.forward_return_from_frames(frames, cost_rate_frame, hold_days=st.HOLD_DAYS)
    assert forward_return["GHOSTUSDT"].isna().all()  # never tradeable -- no spot leg, ever
    assert np.isfinite(forward_return["AAAUSDT"].dropna().to_numpy()).all()


# ---------------------------------------------------------------------------
# 10)+11) 부트스트랩 원시 함수 -- 극단값에서 방향이 맞는지.
# ---------------------------------------------------------------------------


def test_bootstrap_precision_detects_clear_lift_and_rejects_no_lift() -> None:
    strong = np.ones(200)
    strong_result = st._bootstrap_precision(strong, base_rate=0.1, seed=1)
    assert strong_result["p_le_base_rate"] == pytest.approx(0.0)
    assert strong_result["lift"] == pytest.approx(10.0)

    no_lift = np.array([1.0] * 50 + [0.0] * 50)  # sample precision == base_rate exactly
    no_lift_result = st._bootstrap_precision(no_lift, base_rate=0.5, seed=2)
    assert no_lift_result["p_le_base_rate"] > 0.3  # bootstrap centered ON base_rate -- nowhere near <0.05 significant

    empty_result = st._bootstrap_precision(np.array([]), base_rate=0.1, seed=3)
    assert empty_result["precision"] is None and empty_result["n"] == 0


def test_bootstrap_one_sample_detects_clear_positive_and_rejects_negative() -> None:
    positive = np.full(200, 0.01)
    positive_result = st._bootstrap_one_sample(positive, seed=1)
    assert positive_result["p_le_zero"] == pytest.approx(0.0)
    assert positive_result["win_rate"] == pytest.approx(1.0)

    negative = np.full(200, -0.01)
    negative_result = st._bootstrap_one_sample(negative, seed=2)
    assert negative_result["p_le_zero"] == pytest.approx(1.0)

    empty_result = st._bootstrap_one_sample(np.array([]), seed=3)
    assert empty_result["mean"] is None and empty_result["n"] == 0


# ---------------------------------------------------------------------------
# 12) 회귀: 무작위 대조군(수십만 표본)을 매 반복 통째로 리샘플하면 메모리가 터지던 버그.
# ---------------------------------------------------------------------------


def test_bootstrap_two_sample_handles_a_much_larger_second_population_without_blowing_up() -> None:
    a = np.full(50, 0.02)
    b = np.zeros(500_000)  # stand-in for "every valid (date,symbol) cell in a 200-symbol universe"

    result = st._bootstrap_two_sample(a, b, seed=1, paths=200)

    assert result["n_a"] == 50
    assert result["n_b"] == 500_000
    assert result["resample_n"] == 50  # bounded by a's size, NOT b's -- the actual fix
    assert result["mean_diff"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# 13) 동일자-타심볼 페어링 (time-matched paired 비교의 핵심 로직).
# ---------------------------------------------------------------------------


def test_return_comparison_pairs_each_event_with_a_same_day_different_symbol() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    columns = ["AAAUSDT", "BBBUSDT"]
    composite = pd.DataFrame(0.0, index=idx, columns=columns)
    composite.loc[idx[1], "AAAUSDT"] = 2.0  # -> event fires for AAA only, on day idx[2]
    frames = _minimal_frames(idx, columns, composite=composite)
    _signal_bool, event_bool = st.signal_frames(frames, entry_z=1.0)
    assert event_bool.loc[idx[2]].tolist() == [True, False]

    forward_return = pd.DataFrame(
        {"AAAUSDT": [np.nan, np.nan, 0.05, np.nan, np.nan], "BBBUSDT": [np.nan, np.nan, -0.02, np.nan, np.nan]}, index=idx
    )

    result = st.compute_return_comparison(frames, event_bool, forward_return, seed=42)

    assert result.n_events_with_return == 1
    assert result.n_events_dropped_no_paired_control == 0
    # exactly one possible control (BBBUSDT on idx[2]) -- deterministic regardless of RNG seed
    assert result.time_matched_paired_bootstrap["mean"] == pytest.approx(0.05 - (-0.02))

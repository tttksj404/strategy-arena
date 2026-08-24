from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import Final, TypeAlias, assert_never

from pulse.models import Stock

from .bitget_market import (
    BitgetContract,
    BitgetError,
    BitgetPublicClient,
    TextFetcher,
    fetch_completed_candles,
    list_rwa_contracts,
    select_stock_contracts,
)
from .intraday_long_forecast import (
    REQUESTED_LEVERAGE,
    FixedLeverageFeasibility,
    InstrumentProfile,
    IntradayCandleSet,
    IntradayLongPlan,
    STRATEGY_VERSION,
    evaluate_fixed_leverage_feasibility,
    plan_intraday_long,
)
from .intraday_signal_ledger import ForwardPromotion, promotion_evaluation_is_ready, promotion_for
from .market_timeframes import CandleGranularity


MAX_CONCURRENT_MARKET_REQUESTS: Final = 4
TRIPLE_LEVERAGED_ETF_CODES: Final = frozenset(
    {
        "DFEN",
        "KORU",
        "SOXL",
        "SOXS",
        "SPXL",
        "SPXS",
        "SPXU",
        "TNA",
        "TQQQ",
        "TZA",
        "SQQQ",
        "TMF",
        "UPRO",
    }
)
TRIPLE_LEVERAGE_NAME_PATTERN: Final = re.compile(r"(?:\b3x\b|\b3×\b|\b3배\b|ultrapro|ultra pro|triple)", re.IGNORECASE)
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class LocalInstrument:
    code: str
    name: str
    profile: InstrumentProfile


@dataclass(frozen=True, slots=True)
class RawCandidate:
    contract: BitgetContract
    instrument: LocalInstrument
    plan: IntradayLongPlan
    feasibility: FixedLeverageFeasibility


@dataclass(frozen=True, slots=True)
class ScreenCandidate:
    raw: RawCandidate
    forward: ForwardPromotion

    @property
    def five_x_eligible(self) -> bool:
        return promotion_evaluation_is_ready(self.forward)

    def payload(self) -> JsonObject:
        plan = self.raw.plan
        feasibility = self.raw.feasibility
        return {
            "symbol": self.raw.contract.symbol,
            "underlying_symbol": self.raw.contract.underlying_symbol,
            "profile": self.raw.instrument.profile.value,
            "strategy_version": STRATEGY_VERSION,
            "status": "5x_eligible" if self.five_x_eligible else "collecting_forward_evidence",
            "required_closed_timeframes": _timeframe_labels(self.raw.instrument.profile),
            "entry_price": round(plan.entry_price, 6),
            "target_price": round(plan.target_price, 6),
            "stop_price": round(plan.stop_price, 6),
            "score": plan.score,
            "window": {
                "starts_at": plan.starts_at.isoformat(),
                "expires_at": plan.expires_at.isoformat(),
                "label": "다음 완료 1분봉 15개" if plan.profile is InstrumentProfile.TRIPLE_LEVERAGED_ETF else "다음 완료 5분봉 12개",
            },
            "five_x_feasibility": {
                "requested_leverage": REQUESTED_LEVERAGE,
                "contract_max_leverage": self.raw.contract.max_leverage,
                "supports_requested_leverage": self.raw.contract.max_leverage >= REQUESTED_LEVERAGE,
                "estimated_margin_loss_at_stop_pct_before_fees": round(
                    feasibility.estimated_margin_loss_at_stop_pct_before_fees,
                    2,
                ),
                "estimated_all_in_margin_loss_at_stop_pct": round(
                    feasibility.estimated_all_in_margin_loss_at_stop_pct,
                    2,
                ),
                "estimated_round_trip_margin_cost_pct": round(feasibility.estimated_round_trip_margin_cost_pct, 2),
                "estimated_net_margin_gain_at_target_pct": round(
                    feasibility.estimated_net_margin_gain_at_target_pct,
                    2,
                ),
            },
            "forward_evidence": _promotion_payload(self.forward),
            "risk_note": "공개 시세 기반 추정치입니다. 수수료 등급·펀딩·슬리피지·유지증거금·청산가는 계정별로 달라 주문 수익이나 청산 회피를 보장하지 않습니다.",
        }


@dataclass(frozen=True, slots=True)
class IntradayScreenReport:
    as_of: datetime
    eligible_contract_count: int
    scanned_contract_count: int
    candidates: tuple[ScreenCandidate, ...]
    rejection_counts: tuple[tuple[str, int], ...]

    def payload(self) -> JsonObject:
        eligible = [candidate.payload() for candidate in self.candidates if candidate.five_x_eligible]
        return {
            "mode": "research_only",
            "execution": {
                "enabled": False,
                "reason": "공개 시세·폐봉·전진 검증만 수행합니다. Bitget 계정, 레버리지 설정, 주문 API는 호출하지 않습니다.",
            },
            "as_of": self.as_of.isoformat(),
            "data_policy": {
                "closed_candles_only": True,
                "standard_profile": "15분봉 추세 + 5분봉 확인/진입",
                "triple_leveraged_etf_profile": "15분봉 추세 + 5분봉 확인 + 1분봉 진입",
            },
            "universe": {
                "eligible_contract_count": self.eligible_contract_count,
                "scanned_contract_count": self.scanned_contract_count,
                "intersection": "Bitget 정상 RWA 계약 ∩ 로컬 US 주식/ETF 유니버스",
            },
            "promotion_rule": "종목·프로필별 전진 정산 100개 이상 및 마지막 시간순 20% 보류구간의 목표선 선도달률·Wilson 하한·사후비용 기대수익·5배 고정 손실한도를 모두 통과해야 5x_eligible입니다.",
            "candidates": [candidate.payload() for candidate in self.candidates],
            "five_x_eligible_candidates": eligible,
            "rejections": [{"reason": reason, "count": count} for reason, count in self.rejection_counts],
        }


def classify_instrument(stock: Stock) -> InstrumentProfile:
    """Choose the stricter 1m trigger profile only for identifiable US triple-leveraged ETFs."""

    is_triple_leveraged = stock.asset_type == "etf" and (
        stock.code in TRIPLE_LEVERAGED_ETF_CODES or bool(TRIPLE_LEVERAGE_NAME_PATTERN.search(stock.name))
    )
    return InstrumentProfile.TRIPLE_LEVERAGED_ETF if is_triple_leveraged else InstrumentProfile.STANDARD


def eligible_local_us_instruments() -> tuple[LocalInstrument, ...]:
    """Load the narrow local stock/ETF universe with its immutable screening profile."""

    stocks = Stock.objects.filter(market="US", asset_type__in=("stock", "etf")).only("code", "name", "asset_type")
    return tuple(LocalInstrument(stock.code, stock.name, classify_instrument(stock)) for stock in stocks)


def _timeframe_labels(profile: InstrumentProfile) -> list[str]:
    match profile:
        case InstrumentProfile.STANDARD:
            return [CandleGranularity.FIFTEEN_MINUTES.value, CandleGranularity.FIVE_MINUTES.value]
        case InstrumentProfile.TRIPLE_LEVERAGED_ETF:
            return [
                CandleGranularity.FIFTEEN_MINUTES.value,
                CandleGranularity.FIVE_MINUTES.value,
                CandleGranularity.ONE_MINUTE.value,
            ]
        case unreachable:
            assert_never(unreachable)


def _promotion_payload(evaluation: ForwardPromotion) -> JsonObject:
    return {
        "settled_signal_count": evaluation.total.settled_signal_count,
        "target_first_rate_pct": round(evaluation.total.target_first_rate_pct, 2),
        "wilson_lower_bound_pct": round(evaluation.total.wilson_lower_bound_pct, 2),
        "post_cost_expected_margin_return_pct": round(evaluation.total.post_cost_expected_margin_return_pct, 2),
        "post_cost_margin_return_lower_bound_pct": round(
            evaluation.total.post_cost_margin_return_lower_bound_pct,
            2,
        ),
        "validation_signal_count": evaluation.validation.settled_signal_count,
        "held_out_signal_count": evaluation.held_out.settled_signal_count,
    }


def _load_market(contract: BitgetContract, profile: InstrumentProfile, fetch_text: TextFetcher, now: datetime) -> IntradayCandleSet:
    context = fetch_completed_candles(contract, CandleGranularity.FIFTEEN_MINUTES, fetch_text, now)
    confirmation = fetch_completed_candles(contract, CandleGranularity.FIVE_MINUTES, fetch_text, now)
    trigger = confirmation
    if profile is InstrumentProfile.TRIPLE_LEVERAGED_ETF:
        trigger = fetch_completed_candles(contract, CandleGranularity.ONE_MINUTE, fetch_text, now)
    return IntradayCandleSet(context_15m=context, confirmation_5m=confirmation, trigger_1m=trigger)


def _screen_contract(contract: BitgetContract, instrument: LocalInstrument, fetch_text: TextFetcher, now: datetime) -> tuple[RawCandidate | None, str]:
    try:
        market = _load_market(contract, instrument.profile, fetch_text, now)
    except BitgetError:
        return None, "market_data_error"
    plan = plan_intraday_long(instrument.profile, market)
    if plan is None:
        return None, "no_multitimeframe_signal"
    feasibility = evaluate_fixed_leverage_feasibility(plan, contract.taker_fee_rate)
    # The plan-time cost floor is useful context, but never discard a chart setup
    # before the current public bid/ask can price its actual 5x execution cost.
    return RawCandidate(contract, instrument, plan, feasibility), ""


def screen_bitget_intraday_longs(now: datetime | None = None) -> IntradayScreenReport:
    """Screen public Bitget stock/ETF contracts on closed intraday data without account access."""

    reference_time = now or datetime.now(UTC)
    client = BitgetPublicClient()
    instruments = eligible_local_us_instruments()
    instruments_by_code = {instrument.code: instrument for instrument in instruments}
    contracts = select_stock_contracts(list_rwa_contracts(client.get_text), frozenset(instruments_by_code))
    evaluator = partial(_screen_contract, fetch_text=client.get_text, now=reference_time)
    inputs = tuple((contract, instruments_by_code[contract.underlying_symbol]) for contract in contracts)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MARKET_REQUESTS) as executor:
        outcomes = tuple(executor.map(lambda input_pair: evaluator(*input_pair), inputs))
    raw_candidates: list[RawCandidate] = []
    rejections: Counter[str] = Counter()
    for candidate, rejection in outcomes:
        if candidate is None:
            rejections[rejection] += 1
        else:
            raw_candidates.append(candidate)
    candidates = tuple(
        sorted(
            (
                ScreenCandidate(raw, promotion_for(raw.instrument.profile, raw.contract.symbol, STRATEGY_VERSION))
                for raw in raw_candidates
            ),
            key=lambda candidate: (
                candidate.five_x_eligible,
                candidate.raw.plan.score,
                candidate.forward.total.wilson_lower_bound_pct,
            ),
            reverse=True,
        )
    )
    return IntradayScreenReport(
        as_of=reference_time,
        eligible_contract_count=len(contracts),
        scanned_contract_count=len(contracts),
        candidates=candidates,
        rejection_counts=tuple(sorted(rejections.items())),
    )

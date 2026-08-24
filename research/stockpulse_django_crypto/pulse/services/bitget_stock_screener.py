from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import TypeAlias

from pulse.models import Stock

from .bitget_market import (
    BitgetContract,
    BitgetError,
    BitgetPublicClient,
    TextFetcher,
    fetch_completed_4h_candles,
    list_rwa_contracts,
    select_stock_contracts,
)
from .intraday_long_forecast import REQUESTED_LEVERAGE
from .long_forecast import EvaluationMetrics, Forecast, PromotionEvaluation, SplitEvaluation, evaluate_promotion, promotion_is_reliable, trend_atr_forecast


MAX_CONCURRENT_MARKET_REQUESTS: int = 4
JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ScreenCandidate:
    contract: BitgetContract
    forecast: Forecast
    promotion: PromotionEvaluation

    def payload(self) -> JsonObject:
        loss_at_stop = min(abs(self.forecast.stop_return_pct) * REQUESTED_LEVERAGE, 100.0)
        return {
            "symbol": self.contract.symbol,
            "underlying_symbol": self.contract.underlying_symbol,
            "entry_price": round(self.forecast.entry_price, 4),
            "target_price": round(self.forecast.target_price, 4),
            "stop_price": round(self.forecast.stop_price, 4),
            "target_return_pct": round(self.forecast.target_return_pct, 2),
            "stop_return_pct": round(self.forecast.stop_return_pct, 2),
            "score": self.forecast.score,
            "window": {
                "starts_at": self.forecast.starts_at.isoformat(),
                "expires_at": self.forecast.expires_at.isoformat(),
                "bars": 12,
                "label": "다음 완료 4시간봉 12개(최대 48시간)",
            },
            "validation": _split_payload(self.promotion.validation),
            "held_out": _split_payload(self.promotion.held_out),
            "risk": {
                "requested_leverage": REQUESTED_LEVERAGE,
                "contract_max_leverage": self.contract.max_leverage,
                "supports_requested_leverage": self.contract.max_leverage >= REQUESTED_LEVERAGE,
                "estimated_margin_loss_at_stop_pct_before_fees": round(loss_at_stop, 1),
                "note": "5배 고정도 손절가 도달 전 수수료·펀딩·슬리피지·유지증거금 때문에 강제청산을 피한다고 보장하지 않습니다. 이 서비스는 레버리지를 설정하거나 주문하지 않습니다.",
            },
        }


@dataclass(frozen=True, slots=True)
class ScreenReport:
    as_of: datetime
    eligible_contract_count: int
    scanned_contract_count: int
    candidates: tuple[ScreenCandidate, ...]
    rejection_counts: tuple[tuple[str, int], ...]

    def payload(self) -> JsonObject:
        return {
            "mode": "research_only",
            "execution": {
                "enabled": False,
                "reason": "공개 시세 조회·후보 검증만 수행합니다. 주문·계정·레버리지 API는 호출하지 않습니다.",
            },
            "as_of": self.as_of.isoformat(),
            "universe": {
                "eligible_contract_count": self.eligible_contract_count,
                "scanned_contract_count": self.scanned_contract_count,
                "intersection": "Bitget 정상 RWA 계약 ∩ 로컬 US 주식/ETF 유니버스",
            },
            "promotion_rule": {
                "validation_and_holdout": "각 구간 신호 10개 이상, 목표선 선도달률 55% 이상, 무조건 롱 기준선 대비 5%p 이상",
                "holdout_touched_for_selection": False,
            },
            "candidates": [candidate.payload() for candidate in self.candidates],
            "rejections": [{"reason": reason, "count": count} for reason, count in self.rejection_counts],
        }


def _metrics_payload(metrics: EvaluationMetrics) -> JsonObject:
    return {
        "signal_count": metrics.signal_count,
        "target_hits": metrics.target_hits,
        "stop_hits": metrics.stop_hits,
        "unresolved_count": metrics.unresolved_count,
        "ambiguous_count": metrics.ambiguous_count,
        "target_hit_rate_pct": round(metrics.target_hit_rate_pct, 2),
        "mean_terminal_return_pct": round(metrics.mean_terminal_return_pct, 2),
    }


def _split_payload(evaluation: SplitEvaluation) -> JsonObject:
    return {
        "trend_atr_v1": _metrics_payload(evaluation.trend),
        "unconditional_long_v1": _metrics_payload(evaluation.baseline),
        "uplift_pct_points": round(
            evaluation.trend.target_hit_rate_pct - evaluation.baseline.target_hit_rate_pct,
            2,
        ),
    }


def eligible_local_us_symbols() -> frozenset[str]:
    """Return only local assets that match the stock/ETF product boundary."""

    return frozenset(
        Stock.objects.filter(market="US", asset_type__in=("stock", "etf")).values_list("code", flat=True)
    )


def _screen_contract(
    contract: BitgetContract,
    fetch_text: TextFetcher,
    reference_time: datetime,
) -> tuple[ScreenCandidate | None, str]:
    try:
        candles = fetch_completed_4h_candles(contract, fetch_text, reference_time)
    except BitgetError:
        return None, "market_data_error"
    promotion = evaluate_promotion(candles)
    forecast = trend_atr_forecast(candles)
    if not promotion_is_reliable(promotion.validation, promotion.held_out):
        return None, "no_reliable_winner"
    if forecast is None:
        return None, "no_current_signal"
    return ScreenCandidate(contract=contract, forecast=forecast, promotion=promotion), ""


def screen_bitget_stock_longs(now: datetime | None = None) -> ScreenReport:
    """Screen every currently eligible stock/ETF contract without private Bitget access."""

    reference_time = now or datetime.now(UTC)
    client = BitgetPublicClient()
    local_us_symbols = eligible_local_us_symbols()
    contracts = select_stock_contracts(list_rwa_contracts(client.get_text), local_us_symbols)
    candidates: list[ScreenCandidate] = []
    rejections: Counter[str] = Counter()
    evaluator = partial(_screen_contract, fetch_text=client.get_text, reference_time=reference_time)
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_MARKET_REQUESTS) as executor:
        results = tuple(executor.map(evaluator, contracts))
    for candidate, rejection in results:
        if candidate is None:
            rejections[rejection] += 1
        else:
            candidates.append(candidate)
    ranked = tuple(
        sorted(
            candidates,
            key=lambda candidate: (candidate.forecast.score, candidate.promotion.held_out.trend.target_hit_rate_pct),
            reverse=True,
        )
    )
    return ScreenReport(
        as_of=reference_time,
        eligible_contract_count=len(contracts),
        scanned_contract_count=len(contracts),
        candidates=ranked,
        rejection_counts=tuple(sorted(rejections.items())),
    )

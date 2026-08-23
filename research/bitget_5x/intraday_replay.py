from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import sqrt
from typing import Final, Mapping, assert_never

from .signal_outcomes import SignalOutcome


MIN_REPLAY_VALIDATION_SIGNALS: Final = 80
MIN_REPLAY_HELD_OUT_SIGNALS: Final = 20
MIN_TARGET_FIRST_RATE_PCT: Final = 55.0
MIN_WILSON_LOWER_BOUND_PCT: Final = 50.0
MAX_MARGIN_LOSS_AT_STOP_PCT: Final = 8.0
CONSERVATIVE_MEAN_RETURN_T_CRITICAL: Final = 2.093


class ReplayEntryFilter(StrEnum):
    """A research-only causal entry filter layered on the shared 5x breakout plan."""

    BREAKOUT = "breakout"
    TECHNICAL_SCORE = "technical_score_v1"
    BOLLINGER_SQUEEZE = "bollinger_squeeze_v1"


@dataclass(frozen=True, slots=True)
class ReplayRule:
    """One immutable research filter that may be compared only before its holdout is read."""

    name: str
    minimum_volume_ratio: float
    entry_filter: ReplayEntryFilter = ReplayEntryFilter.BREAKOUT


@dataclass(frozen=True, slots=True)
class ReplayTrade:
    """One causally replayed fixed-5x outcome with its rule and completion-derived return."""

    rule: ReplayRule
    contract_symbol: str
    signal_opened_at: datetime
    outcome: SignalOutcome
    post_cost_margin_return_pct: float
    margin_loss_at_stop_pct: float


@dataclass(frozen=True, slots=True)
class ReplayPerformance:
    """Fixed replay metrics for one chronological segment."""

    settled_signal_count: int
    target_first_count: int
    stop_first_count: int
    unresolved_count: int
    ambiguous_count: int
    post_cost_expected_margin_return_pct: float
    post_cost_margin_return_lower_bound_pct: float
    worst_margin_loss_at_stop_pct: float

    @property
    def target_first_rate_pct(self) -> float:
        if self.settled_signal_count == 0:
            return 0.0
        return self.target_first_count / self.settled_signal_count * 100

    @property
    def wilson_lower_bound_pct(self) -> float:
        if self.settled_signal_count == 0:
            return 0.0
        z_squared = 1.96**2
        proportion = self.target_first_count / self.settled_signal_count
        centre = proportion + z_squared / (2 * self.settled_signal_count)
        variance = (
            proportion * (1 - proportion) / self.settled_signal_count
            + z_squared / (4 * self.settled_signal_count**2)
        )
        lower = (centre - 1.96 * sqrt(variance)) / (1 + z_squared / self.settled_signal_count)
        return max(lower * 100, 0.0)


@dataclass(frozen=True, slots=True)
class ReplaySelectionResult:
    """A validation-selected candidate and its untouched-holdout verdict."""

    selected_rule_name: str | None
    validation: ReplayPerformance
    held_out: ReplayPerformance
    is_reliable: bool


@dataclass(frozen=True, slots=True)
class ReplayCandidateValidation:
    """Validation-only diagnostics for one precommitted candidate rule."""

    rule_name: str
    validation: ReplayPerformance
    validation_passes: bool


def _empty_performance() -> ReplayPerformance:
    return ReplayPerformance(0, 0, 0, 0, 0, 0.0, 0.0, 0.0)


def _lower_mean_return_bound(returns: tuple[float, ...]) -> float:
    """Use a conservative lower mean bound after a nontrivial sample has accumulated."""

    if len(returns) < MIN_REPLAY_HELD_OUT_SIGNALS:
        return 0.0
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    return mean_return - CONSERVATIVE_MEAN_RETURN_T_CRITICAL * sqrt(variance / len(returns))


def _performance(trades: tuple[ReplayTrade, ...]) -> ReplayPerformance:
    """Aggregate fixed replay outcomes without reusing any future result as a feature."""

    target_first = 0
    stop_first = 0
    unresolved = 0
    ambiguous = 0
    returns: list[float] = []
    worst_loss = 0.0
    for trade in trades:
        match trade.outcome:
            case SignalOutcome.TARGET:
                target_first += 1
            case SignalOutcome.STOP:
                stop_first += 1
            case SignalOutcome.UNRESOLVED:
                unresolved += 1
            case SignalOutcome.AMBIGUOUS:
                ambiguous += 1
            case SignalOutcome.PENDING:
                raise AssertionError("replay trade cannot remain pending")
            case unreachable:
                assert_never(unreachable)
        returns.append(trade.post_cost_margin_return_pct)
        worst_loss = max(worst_loss, trade.margin_loss_at_stop_pct)
    mean_return = sum(returns) / len(returns) if returns else 0.0
    return ReplayPerformance(
        settled_signal_count=len(trades),
        target_first_count=target_first,
        stop_first_count=stop_first,
        unresolved_count=unresolved,
        ambiguous_count=ambiguous,
        post_cost_expected_margin_return_pct=mean_return,
        post_cost_margin_return_lower_bound_pct=_lower_mean_return_bound(tuple(returns)),
        worst_margin_loss_at_stop_pct=worst_loss,
    )


def _clears(performance: ReplayPerformance, minimum_count: int) -> bool:
    return (
        performance.settled_signal_count >= minimum_count
        and performance.target_first_rate_pct >= MIN_TARGET_FIRST_RATE_PCT
        and performance.wilson_lower_bound_pct >= MIN_WILSON_LOWER_BOUND_PCT
        and performance.post_cost_expected_margin_return_pct > 0
        and performance.post_cost_margin_return_lower_bound_pct > 0
        and performance.worst_margin_loss_at_stop_pct <= MAX_MARGIN_LOSS_AT_STOP_PCT
    )


def _split_chronologically(trades: tuple[ReplayTrade, ...]) -> tuple[tuple[ReplayTrade, ...], tuple[ReplayTrade, ...]]:
    ordered = tuple(sorted(trades, key=lambda trade: trade.signal_opened_at))
    boundary = len(ordered) * 4 // 5
    return ordered[:boundary], ordered[boundary:]


def evaluate_replay_candidates(
    trades_by_rule: Mapping[str, tuple[ReplayTrade, ...]],
) -> dict[str, ReplayCandidateValidation]:
    """Expose every candidate's validation gates without reading its holdout into selection."""

    diagnostics: dict[str, ReplayCandidateValidation] = {}
    for rule_name, trades in sorted(trades_by_rule.items()):
        validation_trades, _ = _split_chronologically(trades)
        validation = _performance(validation_trades)
        diagnostics[rule_name] = ReplayCandidateValidation(
            rule_name=rule_name,
            validation=validation,
            validation_passes=_clears(validation, MIN_REPLAY_VALIDATION_SIGNALS),
        )
    return diagnostics


def select_replay_candidate(trades_by_rule: Mapping[str, tuple[ReplayTrade, ...]]) -> ReplaySelectionResult:
    """Choose solely on chronological validation, then issue one holdout verdict for that frozen choice."""

    diagnostics = evaluate_replay_candidates(trades_by_rule)
    candidates: list[tuple[str, ReplayPerformance]] = [
        (rule_name, diagnostic.validation)
        for rule_name, diagnostic in diagnostics.items()
        if diagnostic.validation_passes
    ]
    if not candidates:
        empty = _empty_performance()
        return ReplaySelectionResult(None, empty, empty, False)
    selected_rule_name, validation = min(
        candidates,
        key=lambda candidate: (
            -candidate[1].post_cost_margin_return_lower_bound_pct,
            -candidate[1].target_first_rate_pct,
            candidate[0],
        ),
    )
    _, selected_hold_out_trades = _split_chronologically(trades_by_rule[selected_rule_name])
    held_out = _performance(selected_hold_out_trades)
    return ReplaySelectionResult(
        selected_rule_name=selected_rule_name,
        validation=validation,
        held_out=held_out,
        is_reliable=_clears(held_out, MIN_REPLAY_HELD_OUT_SIGNALS),
    )

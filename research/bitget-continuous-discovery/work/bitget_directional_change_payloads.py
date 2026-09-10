"""Typed JSON payload builders for directional-change paper-search reports."""

from __future__ import annotations

from selection_bias_stats import DeflatedSharpeResult, PboResult

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def number_payload(values: dict[str, float | int]) -> dict[str, JsonValue]:
    payload: dict[str, JsonValue] = {}
    for key, value in values.items():
        payload[key] = value
    return payload


def pbo_payload(pbo: PboResult) -> dict[str, JsonValue]:
    return {
        "pbo": pbo.pbo,
        "splits": pbo.splits,
        "candidates": pbo.candidates,
        "median_logit": pbo.median_logit,
    }


def dsr_payload(dsr: DeflatedSharpeResult) -> dict[str, JsonValue]:
    return {
        "observed_sharpe": dsr.observed_sharpe,
        "benchmark_sharpe": dsr.benchmark_sharpe,
        "probability": dsr.probability,
        "trials": dsr.trials,
        "observations": dsr.observations,
    }

# Provides cache, timestamp, JSON, and market-data integrity utilities.

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Final, TypeAlias

import pandas as pd  # noqa: PANDAS_OK
import requests


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

BASE_DIR: Final = Path(__file__).resolve().parent
CACHE_DIR: Final = BASE_DIR / "cache"
RESULTS_DIR: Final = BASE_DIR / "results"
REPORT_DIR: Final = BASE_DIR / "report"
SYMBOL_PATTERN: Final = re.compile(r"^[A-Z0-9]{1,32}$")


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    monotonic: bool
    duplicate_count: int
    gap_count: int
    positive_volume_ratio: float

    @property
    def valid(self) -> bool:
        return self.monotonic and self.duplicate_count == 0 and self.positive_volume_ratio > 0.0


@dataclass(frozen=True, slots=True)
class PipelineError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class StrategyResult:
    candidate_id: str
    family: str
    equity: pd.Series
    trade_returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    stress_total_return: float
    metadata: dict[str, JsonValue]


def validate_symbol(symbol: str) -> str:
    if SYMBOL_PATTERN.fullmatch(symbol) is None:
        raise PipelineError(f"invalid market symbol: {symbol!r}")
    return symbol


def ensure_output_dirs() -> None:
    for directory in (CACHE_DIR, RESULTS_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def utc_timestamp(value: str | int | float | pd.Timestamp) -> pd.Timestamp:
    if isinstance(value, (int, float)):
        return pd.to_datetime(value, unit="ms", utc=True)
    return pd.to_datetime(value, utc=True)


def normalize_market_frame(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    normalized = frame.copy()
    if timestamp_column in normalized.columns:
        # format="ISO8601": cached CSVs mix second and sub-second precision; pandas 3 no longer infers mixed formats.
        normalized[timestamp_column] = pd.to_datetime(normalized[timestamp_column], utc=True, format="ISO8601")
        normalized = normalized.set_index(timestamp_column)
    normalized.index = pd.to_datetime(normalized.index, utc=True, format="ISO8601")
    normalized.index.name = "timestamp"
    return normalized


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    # Atomic write: a killed process must never leave a partially written cache file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    normalize_market_frame(frame).to_csv(tmp_path, compression="gzip", encoding="utf-8")
    os.replace(tmp_path, path)


def load_frame(path: Path) -> pd.DataFrame:
    return normalize_market_frame(pd.read_csv(path, encoding="utf-8", compression="gzip"))


def save_json(path: Path, payload: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def json_safe(value: JsonValue) -> JsonValue:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def load_json(path: Path) -> JsonValue:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(
    session: requests.Session,
    url: str,
    params: dict[str, str | int],
) -> JsonValue:
    session.headers.update({"User-Agent": "strategy-arena-wave1/1.0"})
    last_error: requests.RequestException | json.JSONDecodeError | None = None
    for attempt in range(3):
        try:
            response = session.get(url, params=params, timeout=(5.0, 30.0))
            response.raise_for_status()
            if len(response.text) > 16_000_000:
                raise PipelineError("response exceeded 16 MB")
            time.sleep(0.15)
            return json.loads(response.text)
        except (requests.RequestException, json.JSONDecodeError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    if last_error is None:
        raise PipelineError("request retry loop ended without a response")
    raise last_error


def integrity_report(frame: pd.DataFrame, expected_interval: timedelta) -> IntegrityReport:
    index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    duplicate_count = int(index.duplicated().sum())
    gaps = index.to_series().diff().dropna() > expected_interval * 1.5
    positive_volume_ratio = 1.0
    if "volume" in frame.columns and len(frame) > 0:
        positive_volume_ratio = float((frame["volume"] > 0).mean())
    return IntegrityReport(
        monotonic=index.is_monotonic_increasing,
        duplicate_count=duplicate_count,
        gap_count=int(gaps.sum()),
        positive_volume_ratio=positive_volume_ratio,
    )


def close_correlation(left: pd.DataFrame, right: pd.DataFrame) -> float:
    # Daily bars are stamped at different UTC hours per exchange (Binance 00:00, Bitget 16:00);
    # align on calendar date so the overlap is non-empty.
    left_close = left["close"].copy()
    right_close = right["close"].copy()
    left_close.index = pd.DatetimeIndex(left_close.index).normalize()
    right_close.index = pd.DatetimeIndex(right_close.index).normalize()
    left_close = left_close[~left_close.index.duplicated(keep="last")]
    right_close = right_close[~right_close.index.duplicated(keep="last")]
    aligned = pd.concat(
        [left_close.rename("left"), right_close.rename("right")],
        axis=1,
        join="inner",
    ).dropna()
    return float(aligned["left"].corr(aligned["right"])) if len(aligned) >= 2 else float("nan")


def report_payload(report: IntegrityReport) -> dict[str, JsonValue]:
    return {key: value for key, value in asdict(report).items()}


def strategy_payload(result: StrategyResult) -> dict[str, JsonValue]:
    equity = [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": float(value)}
        for timestamp, value in result.equity.items()
    ]
    positions = [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": float(value)}
        for timestamp, value in result.positions.items()
    ]
    turnover = [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": float(value)}
        for timestamp, value in result.turnover.items()
    ]
    trade_returns = [
        {"timestamp": pd.Timestamp(timestamp).isoformat(), "value": float(value)}
        for timestamp, value in result.trade_returns.items()
    ]
    return {
        "candidate_id": result.candidate_id,
        "family": result.family,
        "equity": equity,
        "trade_returns": trade_returns,
        "positions": positions,
        "turnover": turnover,
        "stress_total_return": float(result.stress_total_return),
        "metadata": json_safe(result.metadata),
    }

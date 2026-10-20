# Cross-venue funding-rate validation: is the funding signal a market-common
# phenomenon or a Binance artifact?
#
# Reference venue: Binance (wave1 cache). Challengers: Bybit and OKX via public
# read-only REST (no keys), Bitget from the wave1 cache as a reference-only
# fourth venue. Produces research/validation/CROSS_VENUE_REPORT.md.

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd  # noqa: PANDAS_OK
import requests


BASE_DIR: Final = Path(__file__).resolve().parent
VENUE_CACHE_DIR: Final = BASE_DIR / "cache_venue"
WAVE1_CACHE_DIR: Final = BASE_DIR.parent / "wave1" / "cache"
REPORT_PATH: Final = BASE_DIR / "CROSS_VENUE_REPORT.md"
RESULTS_JSON_PATH: Final = BASE_DIR / "results" / "cross_venue.json"
UNIVERSE_PATH: Final = WAVE1_CACHE_DIR / "universe.json"

START_MS: Final = 1_704_067_200_000  # 2024-01-01T00:00:00Z
SLEEP_SECONDS: Final = 0.12
RETRIES: Final = 3
USER_AGENT: Final = "strategy-arena-cross-venue/1.0"
BYBIT_URL: Final = "https://api.bybit.com/v5/market/funding/history"
OKX_URL: Final = "https://www.okx.com/api/v5/public/funding-rate-history"
MAX_PAGES: Final = 120

BUCKET_FREQ: Final = "8h"
EVENTS_7D: Final = 21  # 8h buckets in 7 days
APR_PER_7D_SUM: Final = 365.0 / 7.0  # annualise a rolling 7-day funding sum
ENTRY_APR: Final = 0.15  # W2c/F1f carry entry threshold
LOW_APR: Final = 0.05  # low-funding noise band: |7d APR| < 5%
HIGH_APR: Final = 0.15  # high-funding band: |7d APR| > 15%
MIN_HIGH_N: Final = 30  # minimum pooled high-band sample for a verdict
TOP_COUNT: Final = 12
MULTIPLIER_PREFIX: Final = re.compile(r"^(?:10{3,}|1M)")
VENUES: Final = ("bybit", "okx", "bitget")


@dataclass(frozen=True, slots=True)
class VenueError(Exception):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class PairMetrics:
    symbol: str
    raw_n: int
    raw_sign: float | None
    score_n: int
    score_sign: float | None
    pearson: float | None
    entry_agree: float | None
    entry_active_n: int
    entry_agree_active: float | None
    low_n: int
    low_sign: float | None
    high_n: int
    high_sign: float | None
    span_days: int


def _empty_metrics(symbol: str) -> PairMetrics:
    return PairMetrics(symbol, 0, None, 0, None, None, None, 0, None, 0, None, 0, None, 0)


def _get_json(session: requests.Session, url: str, params: dict[str, str | int]) -> dict[str, object]:
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = session.get(url, params=params, timeout=(5.0, 30.0))
            time.sleep(SLEEP_SECONDS)
            response.raise_for_status()
            payload = json.loads(response.text)
            if not isinstance(payload, dict):
                raise VenueError(f"non-object response from {url}")
            return payload
        except (requests.RequestException, json.JSONDecodeError, VenueError) as error:
            last_error = error
            if attempt < RETRIES - 1:
                time.sleep(0.5 * (2**attempt))
    raise VenueError(f"request failed after {RETRIES} attempts: {url} ({last_error})")


def _event_frame(rows: list[tuple[int, float]], symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["timestamp", "funding_rate"])
    if frame.empty:
        return pd.DataFrame(columns=["symbol", "funding_rate"], index=pd.DatetimeIndex([], name="timestamp", tz="UTC"))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["symbol"] = symbol
    frame = frame.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp").set_index("timestamp")
    frame = frame[~frame.index.duplicated(keep="first")]
    start = pd.to_datetime(START_MS, unit="ms", utc=True)
    return frame.loc[frame.index >= start, ["symbol", "funding_rate"]]


def fetch_bybit(session: requests.Session, symbol: str) -> pd.DataFrame:
    rows: list[tuple[int, float]] = []
    cursor: int | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, str | int] = {"category": "linear", "symbol": symbol, "limit": 200}
        if cursor is not None:
            params["endTime"] = cursor
        payload = _get_json(session, BYBIT_URL, params)
        if payload.get("retCode") != 0:
            raise VenueError(f"bybit {symbol}: retCode={payload.get('retCode')} {payload.get('retMsg')}")
        result = payload.get("result")
        entries = result.get("list") if isinstance(result, dict) else None
        if not isinstance(entries, list):
            raise VenueError(f"bybit {symbol}: result.list missing")
        page = [
            (int(entry["fundingRateTimestamp"]), float(entry["fundingRate"]))
            for entry in entries
            if isinstance(entry, dict) and "fundingRateTimestamp" in entry and "fundingRate" in entry
        ]
        if not page:
            break
        rows.extend(page)
        oldest = min(timestamp for timestamp, _ in page)
        if oldest <= START_MS:
            break
        next_cursor = oldest - 1
        if cursor is not None and next_cursor >= cursor:
            raise VenueError(f"bybit {symbol}: pagination did not advance")
        cursor = next_cursor
    else:
        raise VenueError(f"bybit {symbol}: exceeded {MAX_PAGES} pages")
    return _event_frame(rows, symbol)


def okx_inst_id(symbol: str) -> str:
    # OKX has no 1000x-multiplier contract naming (Binance's "1000PEPEUSDT" is
    # OKX's "PEPE-USDT-SWAP" with a different contract face value) -- strip a
    # leading 1000-style multiplier before building the instId. If OKX still
    # doesn't list the stripped base, the request itself fails and the caller
    # records that as a skip rather than guessing further.
    base = symbol.removesuffix("USDT")
    stripped = MULTIPLIER_PREFIX.sub("", base)
    return f"{stripped}-USDT-SWAP"


def fetch_okx(session: requests.Session, symbol: str) -> pd.DataFrame:
    inst_id = okx_inst_id(symbol)
    rows: list[tuple[int, float]] = []
    cursor: int | None = None
    for _ in range(MAX_PAGES):
        params: dict[str, str | int] = {"instId": inst_id, "limit": 100}
        if cursor is not None:
            params["after"] = cursor
        payload = _get_json(session, OKX_URL, params)
        if payload.get("code") != "0":
            raise VenueError(f"okx {inst_id}: code={payload.get('code')} {payload.get('msg')}")
        entries = payload.get("data")
        if not isinstance(entries, list):
            raise VenueError(f"okx {inst_id}: data missing")
        page = [
            (int(entry["fundingTime"]), float(entry["fundingRate"]))
            for entry in entries
            if isinstance(entry, dict) and "fundingTime" in entry and "fundingRate" in entry
        ]
        if not page:
            break  # OKX public endpoint retains roughly the last 3 months only.
        rows.extend(page)
        oldest = min(timestamp for timestamp, _ in page)
        if oldest <= START_MS:
            break
        if cursor is not None and oldest >= cursor:
            raise VenueError(f"okx {inst_id}: pagination did not advance")
        cursor = oldest
    else:
        raise VenueError(f"okx {inst_id}: exceeded {MAX_PAGES} pages")
    return _event_frame(rows, symbol)


def _read_cache(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", encoding="utf-8")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, format="ISO8601")
    frame = frame.dropna(subset=["timestamp", "funding_rate"]).sort_values("timestamp").set_index("timestamp")
    return frame[~frame.index.duplicated(keep="first")]


def _trim(frame: pd.DataFrame) -> pd.DataFrame:
    start = pd.to_datetime(START_MS, unit="ms", utc=True)
    return frame.loc[frame.index >= start]


def _save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
    frame.to_csv(tmp_path, compression="gzip", encoding="utf-8")
    os.replace(tmp_path, path)


def load_or_fetch(venue: str, symbol: str, fetch: Callable[[], pd.DataFrame], notes: list[str]) -> pd.DataFrame | None:
    path = VENUE_CACHE_DIR / f"{venue}_{symbol}.csv.gz"
    if path.is_file():
        return _read_cache(path)
    try:
        frame = fetch()
    except (VenueError, KeyError, TypeError, ValueError, OverflowError) as error:
        notes.append(f"{venue} {symbol}: fetch failed -> N/A ({error})")
        return None
    if frame.empty:
        notes.append(f"{venue} {symbol}: no data in window -> N/A")
        return None
    _save_cache(path, frame)
    return frame


def bucket_series(frame: pd.DataFrame) -> pd.Series:
    # Sum funding within each UTC-anchored 8h bucket: venues that pay 4h/1h
    # funding accrue additively, so the bucket sum stays comparable to an 8h venue.
    # round() (not floor()) because Binance cache timestamps carry a few ms of
    # positive settlement jitter (e.g. 16:00:00.001) and a round-to-nearest is
    # robust to jitter on either side of the 00/08/16 UTC mark; floor() would
    # mis-bucket any venue whose jitter lands a moment before the mark.
    if frame.empty:
        return pd.Series(dtype=float)
    return frame["funding_rate"].groupby(frame.index.round(BUCKET_FREQ)).sum()


def apr_score(buckets: pd.Series) -> pd.Series:
    # 7d rolling funding total annualised; NaN-gapped windows drop out via min_periods,
    # which enforces the same contiguity requirement as the strategy signal.
    if buckets.empty:
        return pd.Series(dtype=float)
    grid = pd.date_range(buckets.index.min(), buckets.index.max(), freq=BUCKET_FREQ, tz="UTC")
    filled = buckets.reindex(grid)
    return (filled.rolling(EVENTS_7D, min_periods=EVENTS_7D).sum() * APR_PER_7D_SUM).dropna()


def _sign_agreement(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size == 0:
        return None
    return float(np.mean(np.sign(left) == np.sign(right)))


def _metrics_from_joined(symbol: str, raw: pd.DataFrame, score: pd.DataFrame) -> PairMetrics:
    raw_sign = _sign_agreement(raw["ref"].to_numpy(), raw["other"].to_numpy())
    ref = score["ref"].to_numpy()
    other = score["other"].to_numpy()
    score_sign = _sign_agreement(ref, other)
    pearson: float | None = None
    if ref.size >= 3 and float(np.std(ref)) > 0.0 and float(np.std(other)) > 0.0:
        pearson = float(np.corrcoef(ref, other)[0, 1])
    entry_agree = float(np.mean((ref > ENTRY_APR) == (other > ENTRY_APR))) if ref.size else None
    active = (ref > ENTRY_APR) | (other > ENTRY_APR)
    entry_active_n = int(active.sum())
    entry_agree_active = (
        float(np.mean((ref[active] > ENTRY_APR) == (other[active] > ENTRY_APR))) if entry_active_n else None
    )
    low_mask = np.abs(ref) < LOW_APR
    high_mask = np.abs(ref) > HIGH_APR
    span = int((score.index.max() - score.index.min()).days) if len(score) else 0
    return PairMetrics(
        symbol=symbol,
        raw_n=len(raw),
        raw_sign=raw_sign,
        score_n=len(score),
        score_sign=score_sign,
        pearson=pearson,
        entry_agree=entry_agree,
        entry_active_n=entry_active_n,
        entry_agree_active=entry_agree_active,
        low_n=int(low_mask.sum()),
        low_sign=_sign_agreement(ref[low_mask], other[low_mask]),
        high_n=int(high_mask.sum()),
        high_sign=_sign_agreement(ref[high_mask], other[high_mask]),
        span_days=span,
    )


def pair_metrics(symbol: str, ref_buckets: pd.Series, other_buckets: pd.Series) -> tuple[PairMetrics, pd.DataFrame, pd.DataFrame]:
    raw = pd.concat([ref_buckets.rename("ref"), other_buckets.rename("other")], axis=1, join="inner").dropna()
    score = pd.concat([apr_score(ref_buckets).rename("ref"), apr_score(other_buckets).rename("other")], axis=1, join="inner").dropna()
    return _metrics_from_joined(symbol, raw, score), raw, score


def select_symbols() -> list[str]:
    payload = json.loads(UNIVERSE_PATH.read_text(encoding="utf-8"))
    raw_symbols = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(raw_symbols, list):
        raise VenueError("universe.json has no symbols list")
    symbols = [item for item in raw_symbols if isinstance(item, str)][:TOP_COUNT]
    for required in ("BTCUSDT", "ETHUSDT"):
        if required not in symbols:
            symbols.append(required)
    return symbols


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100.0:.1f}%"


def _corr(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _pair_table(rows: list[PairMetrics], aggregate: PairMetrics) -> list[str]:
    header = (
        "| 심볼 | N(8h) | 원시 부호일치 | N(7d) | 7d 부호일치 | Pearson | 진입일치(전체) | 진입일치(활성/N) "
        "| 저펀딩 N/일치 | 고펀딩 N/일치 | 기간(일) |"
    )
    divider = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, divider]
    for row in [*rows, aggregate]:
        if row.score_n == 0 and row.raw_n == 0:
            lines.append(f"| {row.symbol} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            f"| {row.symbol} | {row.raw_n} | {_pct(row.raw_sign)} | {row.score_n} | {_pct(row.score_sign)} "
            f"| {_corr(row.pearson)} | {_pct(row.entry_agree)} | {_pct(row.entry_agree_active)} ({row.entry_active_n}) "
            f"| {row.low_n} / {_pct(row.low_sign)} | {row.high_n} / {_pct(row.high_sign)} | {row.span_days} |"
        )
    return lines


SUPPLEMENTARY_LABELS: Final = {"okx": "OKX (단기, ~90일 API 한도)", "bitget": "Bitget (참고, wave1 캐시 ~133일)"}


def _verdict(aggregates: dict[str, PairMetrics]) -> tuple[list[str], bool]:
    # Primary verdict: Binance vs Bybit only. Bybit's endTime walkback reaches
    # 2024-01-01, so it is the only challenger with a long-horizon (2+ year)
    # overlap against the Binance reference. OKX's public history API caps out
    # around ~93 days (confirmed empirically) and Bitget's wave1 cache is a
    # ~133-day snapshot, so both are reported as supplementary corroboration
    # only and never gate the pass/fail call.
    lines: list[str] = []
    primary = aggregates["bybit"]
    if primary.high_n < MIN_HIGH_N or primary.high_sign is None:
        lines.append(f"- [주 판정] Binance vs Bybit (2024-01-01~현재, 장기): 고펀딩 표본 {primary.high_n}건(<{MIN_HIGH_N}) -> 표본 부족, 판정 보류")
        primary_pass = False
    else:
        primary_pass = primary.high_sign >= 0.90
        lines.append(
            f"- [주 판정] Binance vs Bybit (2024-01-01~현재, 장기, 겹침 {primary.span_days}일): "
            f"고펀딩 부호 일치 {_pct(primary.high_sign)} (N={primary.high_n}) -> "
            + ("기준(>=90%) 충족" if primary_pass else "기준(>=90%) 미달")
        )
    for venue in ("okx", "bitget"):
        aggregate = aggregates[venue]
        label = SUPPLEMENTARY_LABELS[venue]
        if aggregate.high_n < MIN_HIGH_N or aggregate.high_sign is None:
            lines.append(f"- [보조 확인] Binance vs {label}: 고펀딩 표본 {aggregate.high_n}건(<{MIN_HIGH_N}) -> 표본 부족, 참고 불가")
            continue
        supportive = aggregate.high_sign >= 0.90
        lines.append(
            f"- [보조 확인] Binance vs {label}: 고펀딩 부호 일치 {_pct(aggregate.high_sign)} (N={aggregate.high_n}, 겹침 {aggregate.span_days}일) -> "
            + ("주 판정과 부합" if supportive else "주 판정과 불일치(단기 표본 특성일 수 있음)")
        )
    return lines, primary_pass


def write_report(
    symbols: list[str],
    per_pair: dict[str, list[PairMetrics]],
    aggregates: dict[str, PairMetrics],
    notes: list[str],
    started: datetime,
    verdict_lines: list[str],
    verdict_pass: bool,
) -> None:
    pair_titles = {
        "bybit": "Binance vs Bybit (주 판정, 장기 2024-01~)",
        "okx": "Binance vs OKX (보조 확인, 단기 ~90일)",
        "bitget": "Binance vs Bitget (보조 확인/참고, ~133일 캐시)",
    }
    lines: list[str] = [
        "# 교차 거래소 펀딩레이트 검증 리포트 (CROSS_VENUE_REPORT)",
        "",
        f"- 생성: {started.strftime('%Y-%m-%d %H:%M UTC')} / 스크립트: `research/validation/cross_venue_funding.py`",
        f"- 목적: 펀딩 신호가 Binance 고유 산출물이 아니라 시장 공통 현상인지 3사(Bybit·OKX, 참고 Bitget) 교차검증",
        f"- 대상: universe.json 상위 {len(symbols)}개 심볼 ({', '.join(symbols)})",
        "- 수집 구간 목표: 2024-01-01 ~ 현재 (UTC, 8h 펀딩 이벤트). 실제 확보 구간은 거래소별 API 한도에 따라 다름(아래 커버리지 제약 참조).",
        "",
        "## 방법",
        "",
        "- 각 거래소 펀딩 이벤트를 UTC 정시 8h 버킷(00/08/16시)으로 반올림 정렬하고 버킷 내 합산."
        " 4h/1h 주기 심볼도 합산 누적이라 8h 거래소와 비교 가능. (반올림을 쓰는 이유: Binance 캐시 자체에 수 ms 정산 지연"
        " 타임스탬프가 섞여 있어 내림 정렬 시 오분류 위험이 있음.)",
        f"- 7d APR 스코어 = 직전 {EVENTS_7D}버킷(7일) 펀딩 합 x 365/7 (= 기존 전략 코드의 rolling mean x 3 x 365와 동일값). 결측 버킷이 낀 창은 제외(전략 신호와 동일한 연속성 요건).",
        f"- 진입신호: 7d APR > {ENTRY_APR:.0%} (W2c/F1f 임계). 진입일치(활성)은 한쪽이라도 신호가 켜진 버킷만 대상.",
        f"- 구간 분해: Binance 스코어 기준 저펀딩 |7d APR| < {LOW_APR:.0%}, 고펀딩 |7d APR| > {HIGH_APR:.0%}.",
        f"- 판정 기준(주 판정만 적용): Binance vs Bybit 고펀딩 구간 부호 일치율 >= 90% (집계 N >= {MIN_HIGH_N})이면"
        " \"펀딩 = 시장 공통, 거래소 재현성 차단기 해제 근거\". OKX·Bitget은 단기/제한 커버리지라 보조 확인으로만 사용하고 판정을 좌우하지 않음.",
        "",
        "## 커버리지 제약",
        "",
        "- Bybit: 2024-01-01부터 현재까지 endTime 워크백으로 전 구간 수집 (장기, 주 판정 대상).",
        "- OKX: 공개 funding-rate-history 엔드포인트가 최근 약 3개월만 반환(실측: after 90일 전 데이터 있음, 100일 전 빈 응답)."
        " after 워크백이 자연스럽게 그 지점에서 빈 페이지를 받고 종료되므로 2024-01-01 목표는 도달 불가 -> 단기 보조 확인으로 강등."
        " 표의 기간(일) 열이 실제 겹침 구간.",
        "- Bitget: wave1 캐시(BTC/ETH/SOL만, 약 133일). 그 외 심볼은 N/A. 보조 확인/참고용.",
        "",
    ]
    for venue in VENUES:
        lines.append(f"## {pair_titles[venue]}")
        lines.append("")
        lines.extend(_pair_table(per_pair[venue], aggregates[venue]))
        lines.append("")
    lines.append("## 집계 요약")
    lines.append("")
    lines.append("| 거래소쌍 | N(7d) | 7d 부호일치 | Pearson | 진입일치(활성/N) | 저펀딩 일치 | 고펀딩 일치 |")
    lines.append("|---|---|---|---|---|---|---|")
    for venue in VENUES:
        aggregate = aggregates[venue]
        lines.append(
            f"| {pair_titles[venue]} | {aggregate.score_n} | {_pct(aggregate.score_sign)} | {_corr(aggregate.pearson)} "
            f"| {_pct(aggregate.entry_agree_active)} ({aggregate.entry_active_n}) | {_pct(aggregate.low_sign)} (N={aggregate.low_n}) "
            f"| {_pct(aggregate.high_sign)} (N={aggregate.high_n}) |"
        )
    lines.append("")
    lines.append("## 저펀딩 노이즈 가설 검정")
    lines.append("")
    lines.append("저펀딩 구간에서만 불일치가 몰려 있고 고펀딩 구간 일치율이 높으면, 불일치는 신호가 아니라 0 근처 노이즈라는 가설을 지지한다.")
    lines.append("")
    lines.append("| 거래소쌍 | 저펀딩 N | 저펀딩 부호일치 | 고펀딩 N | 고펀딩 부호일치 | 차이(고-저) |")
    lines.append("|---|---|---|---|---|---|")
    for venue in VENUES:
        aggregate = aggregates[venue]
        gap = (
            f"{(aggregate.high_sign - aggregate.low_sign) * 100.0:+.1f}%p"
            if aggregate.high_sign is not None and aggregate.low_sign is not None
            else "N/A"
        )
        lines.append(
            f"| {pair_titles[venue]} | {aggregate.low_n} | {_pct(aggregate.low_sign)} "
            f"| {aggregate.high_n} | {_pct(aggregate.high_sign)} | {gap} |"
        )
    lines.append("")
    lines.append("## 판정")
    lines.append("")
    lines.extend(verdict_lines)
    lines.append("")
    if verdict_pass:
        lines.append(
            "**결론: 주 판정(Binance vs Bybit, 장기) 고펀딩 구간 신호 일치율 >= 90% 충족 — "
            "펀딩은 시장 공통 현상이며, 거래소 재현성 차단기 해제 근거가 된다. OKX·Bitget 단기 확인 결과는 위 표 참조(참고용, 판정 비좌우).**"
        )
    else:
        lines.append("**결론: 주 판정(Binance vs Bybit, 장기) 기준 미충족 또는 표본 부족 — 차단기 해제 근거로 채택하지 않는다. 위 판정 항목 참조.**")
    if notes:
        lines.append("")
        lines.append("## 수집 로그 (스킵/실패)")
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
    lines.append("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


PAIR_LABELS_JSON: Final = {
    "bybit": "Binance vs Bybit (primary, long-horizon 2024-01-01~present)",
    "okx": "Binance vs OKX (supplementary, short-horizon ~90d API limit)",
    "bitget": "Binance vs Bitget (supplementary/reference, ~133d wave1 cache)",
}


def build_results_payload(
    symbols: list[str],
    per_pair: dict[str, list[PairMetrics]],
    aggregates: dict[str, PairMetrics],
    verdict_lines: list[str],
    verdict_pass: bool,
    notes: list[str],
    started: datetime,
) -> dict[str, object]:
    return {
        "generated_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "research/validation/cross_venue_funding.py",
        "symbols": symbols,
        "methodology": {
            "bucket_freq": BUCKET_FREQ,
            "bucket_alignment": "round-to-nearest 8h UTC grid (00/08/16)",
            "score_window_events": EVENTS_7D,
            "score_formula": "rolling_sum(21 buckets) * 365/7  (identical to rolling_mean*3*365 used by fam_funding.funding_score)",
            "entry_threshold_apr": ENTRY_APR,
            "low_apr_band_abs": LOW_APR,
            "high_apr_band_abs": HIGH_APR,
            "min_high_sample_for_verdict": MIN_HIGH_N,
            "collection_target_start": "2024-01-01T00:00:00Z",
        },
        "pairs": {
            venue: {
                "label": PAIR_LABELS_JSON[venue],
                "symbols": [asdict(metrics) for metrics in per_pair[venue]],
                "aggregate": asdict(aggregates[venue]),
            }
            for venue in VENUES
        },
        "verdict": {
            "primary_venue": "bybit",
            "primary_pass": verdict_pass,
            "high_funding_sign_agreement_threshold": 0.90,
            "lines": verdict_lines,
        },
        "collection_notes": notes,
    }


def write_results_json(payload: dict[str, object]) -> None:
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main() -> int:
    started = datetime.now(timezone.utc)
    symbols = select_symbols()
    notes: list[str] = []
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    per_pair: dict[str, list[PairMetrics]] = {venue: [] for venue in VENUES}
    pooled_raw: dict[str, list[pd.DataFrame]] = {venue: [] for venue in VENUES}
    pooled_score: dict[str, list[pd.DataFrame]] = {venue: [] for venue in VENUES}
    for symbol in symbols:
        binance_path = WAVE1_CACHE_DIR / f"binance_funding_{symbol}.csv.gz"
        if not binance_path.is_file():
            notes.append(f"binance {symbol}: wave1 cache missing -> symbol skipped")
            for venue in VENUES:
                per_pair[venue].append(_empty_metrics(symbol))
            continue
        ref_buckets = bucket_series(_trim(_read_cache(binance_path)))
        frames: dict[str, pd.DataFrame | None] = {}
        frames["bybit"] = load_or_fetch("bybit", symbol, lambda sym=symbol: fetch_bybit(session, sym), notes)
        if MULTIPLIER_PREFIX.match(symbol.removesuffix("USDT")):
            notes.append(f"okx {symbol}: multiplier symbol -> trying stripped instId {okx_inst_id(symbol)}")
        frames["okx"] = load_or_fetch("okx", symbol, lambda sym=symbol: fetch_okx(session, sym), notes)
        bitget_path = WAVE1_CACHE_DIR / f"bitget_funding_{symbol}.csv.gz"
        if bitget_path.is_file():
            frames["bitget"] = _trim(_read_cache(bitget_path))
        else:
            frames["bitget"] = None
            notes.append(f"bitget {symbol}: no wave1 cache -> N/A")
        for venue in VENUES:
            frame = frames[venue]
            if frame is None or frame.empty:
                per_pair[venue].append(_empty_metrics(symbol))
                continue
            metrics, raw_joined, score_joined = pair_metrics(symbol, ref_buckets, bucket_series(frame))
            per_pair[venue].append(metrics)
            if len(raw_joined):
                pooled_raw[venue].append(raw_joined)
            if len(score_joined):
                pooled_score[venue].append(score_joined)
        print(f"[{symbol}] collected", flush=True)
    aggregates: dict[str, PairMetrics] = {}
    for venue in VENUES:
        if pooled_raw[venue] or pooled_score[venue]:
            raw = pd.concat(pooled_raw[venue]) if pooled_raw[venue] else pd.DataFrame(columns=["ref", "other"])
            score = pd.concat(pooled_score[venue]) if pooled_score[venue] else pd.DataFrame(columns=["ref", "other"])
            aggregates[venue] = _metrics_from_joined("AGGREGATE", raw, score)
        else:
            aggregates[venue] = _empty_metrics("AGGREGATE")
    verdict_lines, verdict_pass = _verdict(aggregates)
    write_report(symbols, per_pair, aggregates, notes, started, verdict_lines, verdict_pass)
    payload = build_results_payload(symbols, per_pair, aggregates, verdict_lines, verdict_pass, notes, started)
    write_results_json(payload)
    print(f"report: {REPORT_PATH}")
    print(f"json: {RESULTS_JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

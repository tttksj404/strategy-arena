from __future__ import annotations

import html
import json
import logging
import re
import ssl
from dataclasses import asdict, dataclass
from datetime import timezone, datetime
from pathlib import Path
from typing import Callable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

_log = logging.getLogger(__name__)
SSL_CONTEXT = ssl._create_unverified_context()
EASTERN = ZoneInfo("America/New_York")
HTTP_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class OfficialMacroEvent:
    name: str
    start: str
    end: str = ""
    impact: str = "medium"
    source: str = ""
    summary_ko: str = ""
    strategy_hint_ko: str = ""
    assets: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _fetch_text(url: str) -> str:
    with urlopen(url, timeout=HTTP_TIMEOUT_SECONDS, context=SSL_CONTEXT) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean_html_text(raw: str) -> str:
    text = html.unescape(raw)
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text


def _parse_bls_release_datetime(date_text: str, time_text: str) -> datetime:
    normalized = time_text.replace("A.M.", "AM").replace("P.M.", "PM").replace("a.m.", "AM").replace("p.m.", "PM")
    parsed = datetime.strptime(f"{date_text} {normalized}", "%B %d, %Y %I:%M %p")
    return parsed.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def _build_bls_event(
    *,
    name: str,
    url: str,
    impact: str,
    summary_ko: str,
    strategy_hint_ko: str,
    assets: tuple[str, ...],
    fetcher: Callable[[str], str],
) -> OfficialMacroEvent | None:
    raw = fetcher(url)
    text = _clean_html_text(raw)
    match = re.search(
        r"Next Release\s+.*?scheduled to be released on ([A-Za-z]+ \d{1,2}, \d{4}), at ([0-9:]+ ?[AP]\.?M\.?)",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        return None
    release_at = _parse_bls_release_datetime(match.group(1), match.group(2))
    return OfficialMacroEvent(
        name=name,
        start=release_at.isoformat(),
        impact=impact,
        source=url,
        summary_ko=summary_ko,
        strategy_hint_ko=strategy_hint_ko,
        assets=assets,
    )


def fetch_bls_macro_events(fetcher: Callable[[str], str] = _fetch_text) -> tuple[OfficialMacroEvent, ...]:
    specs = (
        (
            "미국 CPI",
            "https://www.bls.gov/cpi/",
            "high",
            "소비자물가 발표는 금리 기대와 달러 방향, 메이저 코인 변동성에 직접 영향을 줄 수 있습니다.",
            "발표 전후에는 메이저 신규 진입과 레버리지를 보수적으로 보고, 달러와 금리 반응 확인 후 추세 재진입을 검토합니다.",
            ("BTC", "ETH"),
        ),
        (
            "미국 PPI",
            "https://www.bls.gov/ppi/",
            "medium",
            "생산자물가는 인플레이션 선행 신호로 해석되며 CPI와 함께 금리 기대를 흔들 수 있습니다.",
            "CPI 직전/직후와 겹치는 경우 인플레이션 리스크 구간으로 묶어 메이저 진입을 축소합니다.",
            ("BTC", "ETH"),
        ),
        (
            "미국 고용지표",
            "https://www.bls.gov/cps/",
            "high",
            "고용지표는 경기 방향과 금리 기대를 동시에 흔들어 메이저 코인의 방향성과 변동성을 크게 바꿀 수 있습니다.",
            "발표 당일과 직후 반응 구간은 추세 확정 전까지 메이저 레버리지를 보수적으로 운용합니다.",
            ("BTC", "ETH"),
        ),
    )
    events: list[OfficialMacroEvent] = []
    for name, url, impact, summary_ko, strategy_hint_ko, assets in specs:
        try:
            event = _build_bls_event(
                name=name,
                url=url,
                impact=impact,
                summary_ko=summary_ko,
                strategy_hint_ko=strategy_hint_ko,
                assets=assets,
                fetcher=fetcher,
            )
        except Exception:
            _log.warning("BLS macro event fetch failed for %s — skipping", name, exc_info=True)
            event = None
        if event is not None:
            events.append(event)
    return tuple(events)


def _parse_bea_release_lines(text: str) -> list[tuple[str, str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    rows: list[tuple[str, str, str]] = []
    months = {
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    }
    idx = 0
    while idx < len(lines) - 2:
        line = lines[idx]
        next_line = lines[idx + 1]
        next_next_line = lines[idx + 2]
        if next_line.split(" ", 1)[0] in months and re.match(r"^[0-9]{1,2}:[0-9]{2} [AP]M$", next_next_line):
            rows.append((line, next_line, next_next_line))
            idx += 3
            continue
        idx += 1
    return rows


def _parse_bea_release_datetime(date_text: str, time_text: str) -> datetime:
    parsed = datetime.strptime(f"{date_text} 2026 {time_text}", "%B %d %Y %I:%M %p")
    return parsed.replace(tzinfo=EASTERN).astimezone(timezone.utc)


def fetch_bea_macro_events(fetcher: Callable[[str], str] = _fetch_text) -> tuple[OfficialMacroEvent, ...]:
    url = "https://www.bea.gov/news/schedule/full"
    try:
        raw = fetcher(url)
    except Exception:
        _log.warning("BEA macro events fetch failed — returning empty", exc_info=True)
        return ()
    rows = _parse_bea_release_lines(_clean_html_text(raw))
    events: list[OfficialMacroEvent] = []
    for title, date_text, time_text in rows:
        title_lower = title.lower()
        if (
            "gdp" not in title_lower
            and "gross domestic product" not in title_lower
            and "personal income and outlays" not in title_lower
        ):
            continue
        if "gdp" in title_lower or "gross domestic product" in title_lower:
            summary_ko = "GDP 발표는 성장 기대와 위험자산 선호를 바꾸는 핵심 이벤트로 메이저 코인 심리에 직접 영향을 줄 수 있습니다."
            strategy_hint_ko = "GDP 발표 전후에는 추세 추종 진입을 서두르지 말고, 금리/달러와 함께 해석한 뒤 원래 전략 복귀 시점을 판단합니다."
            name = "미국 GDP"
        else:
            summary_ko = "개인소득·소비와 PCE 관련 발표는 소비 둔화와 인플레이션 완화/재가속 기대를 동시에 움직일 수 있습니다."
            strategy_hint_ko = "메이저 코인 비중은 발표 결과가 금리·달러 방향과 같은 쪽으로 정렬되는지 확인한 뒤 확대 여부를 정합니다."
            name = "미국 개인소득/소비(PCE)"
        try:
            release_at = _parse_bea_release_datetime(date_text, time_text)
        except ValueError:
            continue
        events.append(
            OfficialMacroEvent(
                name=name,
                start=release_at.isoformat(),
                impact="high" if name == "미국 GDP" else "medium",
                source=url,
                summary_ko=summary_ko,
                strategy_hint_ko=strategy_hint_ko,
                assets=("BTC", "ETH"),
            )
        )
    return tuple(events)


def fetch_fomc_events(fetcher: Callable[[str], str] = _fetch_text) -> tuple[OfficialMacroEvent, ...]:
    url = "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm"
    try:
        raw = fetcher(url)
    except Exception:
        _log.warning("FOMC events fetch failed — returning empty", exc_info=True)
        return ()
    text = _clean_html_text(raw)
    block_match = re.search(r"For 2026:(.*?)The Committee releases a policy statement", text, flags=re.DOTALL)
    if block_match is None:
        return ()
    block = block_match.group(1)
    events: list[OfficialMacroEvent] = []
    line_pattern = re.findall(
        r"Tuesday, ([A-Za-z]+) (\d{1,2}), and Wednesday, ([A-Za-z]+) (\d{1,2})",
        block,
    )
    for month_start, day_start, month_end, day_end in line_pattern:
        start_dt = datetime.strptime(f"{month_start} {day_start} 2026 14:00", "%B %d %Y %H:%M").replace(tzinfo=EASTERN)
        end_dt = datetime.strptime(f"{month_end} {day_end} 2026 14:30", "%B %d %Y %H:%M").replace(tzinfo=EASTERN)
        events.append(
            OfficialMacroEvent(
                name="FOMC",
                start=start_dt.astimezone(timezone.utc).isoformat(),
                end=end_dt.astimezone(timezone.utc).isoformat(),
                impact="high",
                source=url,
                summary_ko="FOMC는 금리·유동성 기대를 직접 바꾸는 메이저 코인 최상위 거시 이벤트입니다.",
                strategy_hint_ko="회의 전후와 기자회견까지는 메이저 신규 진입과 레버리지를 보수적으로 보고, 결과 해석이 끝난 뒤 원래 전략 복귀 여부를 판단합니다.",
                assets=("BTC", "ETH"),
            )
        )
    return tuple(events)


def fetch_official_macro_events(fetcher: Callable[[str], str] = _fetch_text) -> tuple[OfficialMacroEvent, ...]:
    rows = list(fetch_fomc_events(fetcher=fetcher))
    rows.extend(fetch_bls_macro_events(fetcher=fetcher))
    rows.extend(fetch_bea_macro_events(fetcher=fetcher))
    rows.sort(key=lambda item: item.start)
    deduped: list[OfficialMacroEvent] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row.name, row.start)
        if key in seen:
            continue
        deduped.append(row)
        seen.add(key)
    return tuple(deduped)


def write_official_macro_events(
    *,
    output_path: str | Path,
    fetcher: Callable[[str], str] = _fetch_text,
) -> Path:
    rows = fetch_official_macro_events(fetcher=fetcher)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"generated_at": datetime.now(timezone.utc).isoformat(), "events": [row.as_dict() for row in rows]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return target

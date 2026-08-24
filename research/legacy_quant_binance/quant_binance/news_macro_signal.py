from __future__ import annotations

import hashlib
import json
import logging
import ssl
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import timezone, datetime, timedelta, time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote
from urllib.request import urlopen
from zoneinfo import ZoneInfo


_log = logging.getLogger(__name__)
SSL_CONTEXT = ssl._create_unverified_context()
HTTP_TIMEOUT_SECONDS = 20
FEAR_GREED_API_URL = "https://api.alternative.me/fng/?limit=1&format=json"
SEOUL = ZoneInfo("Asia/Seoul")
GOOGLE_NEWS_TEMPLATE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
DEFAULT_QUERIES: tuple[tuple[str, str], ...] = (
    ("bitcoin OR btc OR \"crypto market\" when:1d", "crypto_market"),
    ("bitcoin ETF inflows OR bitcoin etf when:1d", "etf_flow"),
    ("Federal Reserve OR FOMC OR CPI OR inflation when:1d", "macro"),
    ("oil OR middle east OR war market when:1d", "geopolitics"),
    ("crypto regulation OR SEC OR exchange hack when:1d", "regulation"),
)
BREAKING_KEYWORDS = (
    "breaking",
    "emergency",
    "hack",
    "liquidation",
    "war",
    "missile",
    "tariff",
    "sec",
    "etf",
    "cpi",
    "fomc",
    "pce",
    "nfp",
)
BULLISH_KEYWORDS = {
    "etf inflow": 1.3,
    "inflow": 0.8,
    "approval": 0.8,
    "adoption": 0.7,
    "reserve": 0.7,
    "buyback": 0.4,
    "purchase": 0.5,
    "accumulation": 0.6,
    "rate cut": 1.0,
    "cuts": 0.7,
    "dovish": 0.9,
    "cooling inflation": 0.9,
    "soft landing": 0.5,
    "liquidity": 0.5,
    "breakout": 0.5,
    "surge": 0.3,
}
BEARISH_KEYWORDS = {
    "war": 1.0,
    "missile": 1.0,
    "oil": 0.7,
    "inflation": 0.7,
    "hot cpi": 1.0,
    "tariff": 0.8,
    "hawkish": 1.0,
    "rate hike": 1.1,
    "yields rise": 0.8,
    "lawsuit": 0.8,
    "sec": 0.6,
    "hack": 1.3,
    "outflow": 0.9,
    "liquidation": 1.0,
    "crash": 1.0,
    "sell-off": 0.9,
    "recession": 0.7,
}
CATEGORY_KEYWORDS = {
    "macro": ("fed", "fomc", "cpi", "ppi", "pce", "inflation", "rate", "yield", "payroll", "jobs", "gdp"),
    "etf_flow": ("etf", "inflow", "outflow"),
    "geopolitics": ("war", "middle east", "oil", "missile", "attack", "sanction"),
    "regulation": ("sec", "regulation", "lawsuit", "policy", "ban"),
    "exchange_risk": ("hack", "exploit", "exchange", "liquidation", "insolvency"),
    "liquidity": ("liquidity", "rate cut", "balance sheet", "stimulus"),
}


@dataclass(frozen=True)
class NewsHeadline:
    title: str
    published_at: str
    source: str
    query_label: str
    url: str = ""
    categories: tuple[str, ...] = ()
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    uncertainty_score: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True)
class NewsMacroSignal:
    generated_at: str
    refreshed_at: str
    refresh_reason: str
    next_scheduled_refresh_at: str
    trigger_reasons: tuple[str, ...]
    bullish_score: float
    bearish_score: float
    uncertainty_score: float
    dominant_bias: str
    event_types: tuple[str, ...]
    majors_bias: str
    leverage_cap: int
    size_multiplier: float
    entry_policy_bias: str
    macro_inputs: dict[str, float | int]
    headlines: tuple[NewsHeadline, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["headlines"] = [headline.as_dict() for headline in self.headlines]
        return payload


@dataclass(frozen=True)
class RefreshDecision:
    should_refresh: bool
    reason: str
    trigger_reasons: tuple[str, ...]
    next_scheduled_refresh_at: datetime


def _fetch_text(url: str) -> str:
    try:
        with urlopen(url, timeout=HTTP_TIMEOUT_SECONDS, context=SSL_CONTEXT) as response:
            return response.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def fetch_fear_greed_index(*, fetcher=_fetch_text) -> tuple[int, str]:
    """Fetch Crypto Fear & Greed Index. Returns (value 0-100, classification).
    Defaults to (50, "Neutral") on any failure."""
    try:
        raw = fetcher(FEAR_GREED_API_URL)
        payload = json.loads(raw)
        entry = payload["data"][0]
        return int(entry["value"]), str(entry["value_classification"])
    except Exception:
        _log.warning("Fear & Greed index fetch failed — using neutral default (50)", exc_info=True)
        return 50, "Neutral"


def _parse_pub_date(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

def _headline_signature(title: str, source: str, published_at: datetime | None) -> str:
    base = "|".join((title.strip().lower(), source.strip().lower(), published_at.isoformat() if published_at else ""))
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def _classify_categories(text: str) -> tuple[str, ...]:
    text_lower = text.lower()
    categories: list[str] = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text_lower for keyword in keywords):
            categories.append(category)
    deduped: list[str] = []
    seen: set[str] = set()
    for category in categories:
        if category not in seen:
            deduped.append(category)
            seen.add(category)
    return tuple(deduped)


def _keyword_score(text: str, mapping: dict[str, float]) -> float:
    text_lower = text.lower()
    return round(sum(weight for keyword, weight in mapping.items() if keyword in text_lower), 6)


def fetch_google_news_headlines(
    *,
    queries: Iterable[tuple[str, str]] = DEFAULT_QUERIES,
    fetcher=_fetch_text,
    now: datetime | None = None,
) -> tuple[NewsHeadline, ...]:
    now = now or datetime.now(tz=timezone.utc)
    cutoff = now - timedelta(hours=36)
    rows: list[NewsHeadline] = []
    seen_signatures: set[str] = set()
    for query, label in queries:
        try:
            xml_text = fetcher(GOOGLE_NEWS_TEMPLATE.format(query=quote(query)))
            if not xml_text:
                continue
            root = ET.fromstring(xml_text)
        except (ET.ParseError, Exception):
            continue
        for item in root.findall("./channel/item"):
            title = (item.findtext("title") or "").strip()
            if not title:
                continue
            source = (item.findtext("source") or "").strip()
            link = (item.findtext("link") or "").strip()
            published_at = _parse_pub_date((item.findtext("pubDate") or "").strip())
            if published_at is not None and published_at < cutoff:
                continue
            signature = _headline_signature(title, source, published_at)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            combined = f"{title} {source} {label}"
            bullish = _keyword_score(combined, BULLISH_KEYWORDS)
            bearish = _keyword_score(combined, BEARISH_KEYWORDS)
            uncertainty = 0.25 if bullish > 0.0 and bearish > 0.0 else 0.0
            if any(keyword in combined.lower() for keyword in BREAKING_KEYWORDS):
                uncertainty += 0.25
            rows.append(
                NewsHeadline(
                    title=title,
                    published_at=published_at.isoformat() if published_at else "",
                    source=source,
                    query_label=label,
                    url=link,
                    categories=_classify_categories(combined),
                    bullish_score=bullish,
                    bearish_score=bearish,
                    uncertainty_score=round(min(uncertainty, 1.0), 6),
                )
            )
    rows.sort(key=lambda item: item.published_at, reverse=True)
    return tuple(rows[:30])

def _next_schedule(now: datetime) -> datetime:
    local = now.astimezone(SEOUL)
    today = local.date()
    windows = [
        datetime.combine(today, time(hour=6, minute=0), tzinfo=SEOUL),
        datetime.combine(today, time(hour=18, minute=0), tzinfo=SEOUL),
        datetime.combine(today + timedelta(days=1), time(hour=6, minute=0), tzinfo=SEOUL),
    ]
    for candidate in windows:
        if candidate > local:
            return candidate.astimezone(timezone.utc)
    return windows[-1].astimezone(timezone.utc)


def _current_schedule_label(now: datetime) -> str:
    local = now.astimezone(SEOUL)
    return f"{local.date().isoformat()}-am" if local.hour < 18 else f"{local.date().isoformat()}-pm"


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        _log.warning("failed to read news signal cache %s", path, exc_info=True)
        return None


def _load_official_events(path: Path | None) -> tuple[dict[str, object], ...]:
    if path is None or not path.exists():
        return ()
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return ()
    events = payload.get("events")
    if not isinstance(events, list):
        return ()
    return tuple(item for item in events if isinstance(item, dict))


def _minutes_until(target_iso: str, *, now: datetime) -> float | None:
    try:
        target = datetime.fromisoformat(target_iso)
    except Exception:
        return None
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return (target.astimezone(timezone.utc) - now).total_seconds() / 60.0

def decide_refresh(
    *,
    now: datetime,
    existing_payload: dict[str, object] | None,
    official_events: tuple[dict[str, object], ...],
    headlines: tuple[NewsHeadline, ...],
) -> RefreshDecision:
    next_scheduled = _next_schedule(now)
    trigger_reasons: list[str] = []
    if existing_payload is None:
        return RefreshDecision(True, "bootstrap", ("BOOTSTRAP",), next_scheduled)

    last_schedule_label = str(existing_payload.get("last_schedule_label") or "")
    current_schedule_label = _current_schedule_label(now)
    latest_headline_hash = hashlib.sha256(
        json.dumps([headline.as_dict() for headline in headlines[:8]], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    previous_headline_hash = str(existing_payload.get("headline_hash") or "")

    if last_schedule_label != current_schedule_label and now.astimezone(SEOUL).hour in {6, 7, 18, 19}:
        trigger_reasons.append("SCHEDULED_WINDOW")

    for event in official_events:
        minutes_until = _minutes_until(str(event.get("start") or ""), now=now)
        if minutes_until is None:
            continue
        impact = str(event.get("impact") or "medium").lower()
        if impact == "high" and -30.0 <= minutes_until <= 120.0:
            trigger_reasons.append(f"HIGH_IMPACT_EVENT:{event.get('name', 'event')}")
            break

    recent_breaking = any(
        headline.published_at
        and any(keyword in headline.title.lower() for keyword in BREAKING_KEYWORDS)
        for headline in headlines[:10]
    )
    if recent_breaking and latest_headline_hash != previous_headline_hash:
        trigger_reasons.append("BREAKING_HEADLINE_SHIFT")

    if trigger_reasons:
        return RefreshDecision(True, trigger_reasons[0].split(":", 1)[0].lower(), tuple(trigger_reasons), next_scheduled)

    refreshed_at_raw = existing_payload.get("refreshed_at")
    if refreshed_at_raw:
        try:
            refreshed_at = datetime.fromisoformat(str(refreshed_at_raw))
            if refreshed_at.tzinfo is None:
                refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
        except Exception:
            refreshed_at = now - timedelta(hours=24)
        if now - refreshed_at >= timedelta(hours=18):
            return RefreshDecision(True, "stale_refresh", ("STALE_REFRESH",), next_scheduled)

    return RefreshDecision(False, "not_due", (), next_scheduled)

def build_signal(
    *,
    now: datetime,
    refresh_reason: str,
    trigger_reasons: tuple[str, ...],
    next_scheduled_refresh_at: datetime,
    headlines: tuple[NewsHeadline, ...],
    official_events: tuple[dict[str, object], ...],
    fear_greed: tuple[int, str] = (50, "Neutral"),
) -> NewsMacroSignal:
    bullish = sum(item.bullish_score for item in headlines)
    bearish = sum(item.bearish_score for item in headlines)
    uncertainty = sum(item.uncertainty_score for item in headlines)
    category_counts: dict[str, int] = {}
    for headline in headlines:
        for category in headline.categories:
            category_counts[category] = category_counts.get(category, 0) + 1

    high_impact_window = False
    for event in official_events:
        minutes_until = _minutes_until(str(event.get("start") or ""), now=now)
        if minutes_until is None:
            continue
        impact = str(event.get("impact") or "medium").lower()
        if impact == "high" and -60.0 <= minutes_until <= 180.0:
            high_impact_window = True
            uncertainty += 3.5
            category_counts["macro"] = category_counts.get("macro", 0) + 2
        elif impact == "medium" and 0.0 <= minutes_until <= 180.0:
            uncertainty += 1.5

    bullish_score = round(min(bullish / 8.0, 1.0), 6)
    bearish_score = round(min(bearish / 8.0, 1.0), 6)
    uncertainty_score = round(min(uncertainty / 8.0, 1.0), 6)
    net = bullish_score - bearish_score
    if net >= 0.18:
        dominant_bias = "bullish"
    elif net <= -0.18:
        dominant_bias = "bearish"
    else:
        dominant_bias = "neutral"

    event_types = tuple(sorted(category for category, count in category_counts.items() if count > 0))
    execution_risk_categories = {"exchange_risk", "geopolitics"}
    directional_bearish_categories = {"macro", "etf_flow", "regulation", "liquidity"}
    execution_risk_score = min(1.0, (0.45 * uncertainty_score) + (0.3 if high_impact_window else 0.0) + (0.25 if execution_risk_categories.intersection(event_types) else 0.0))
    directional_bearish_score = min(1.0, max(0.0, bearish_score - (0.35 * uncertainty_score)) + (0.15 if directional_bearish_categories.intersection(event_types) else 0.0))
    majors_bias = "majors_only" if (uncertainty_score >= 0.55 or bearish_score >= 0.6) else "neutral"
    if uncertainty_score >= 0.82:
        leverage_cap = 1
        size_multiplier = 0.0
        entry_policy_bias = "halt_high_impact_window"
    elif high_impact_window or uncertainty_score >= 0.6 or bearish_score >= 0.68:
        leverage_cap = 2
        size_multiplier = 0.5
        entry_policy_bias = "pre_event_reduce"
    elif bearish_score > bullish_score + 0.12:
        leverage_cap = 3
        size_multiplier = 0.72
        entry_policy_bias = "risk_off_reduce"
    elif bullish_score > bearish_score + 0.18 and uncertainty_score <= 0.42:
        leverage_cap = 0
        size_multiplier = 1.15
        entry_policy_bias = "supportive_majors"
    else:
        leverage_cap = 0
        size_multiplier = 1.0
        entry_policy_bias = "neutral"

    macro_inputs = {
        "official_high_impact_window": 1.0 if high_impact_window else 0.0,
        "truflation_yoy": 2.9 if bearish_score >= 0.72 and "macro" in event_types else 2.2,
        "us10y_yield": 4.8 if bearish_score >= 0.72 and "macro" in event_types else 4.25,
        "oil_momentum_pct": 13.0 if "geopolitics" in event_types and bearish_score >= bullish_score else 3.0,
        "tga_drain_score": 0.45,
        "fed_balance_sheet_30d_pct": 0.1 if bullish_score > bearish_score else -0.05,
        "mmf_30d_pct": -0.05 if bullish_score > bearish_score else 0.05,
        "labor_stress_score": 0.72 if bearish_score >= 0.72 and "macro" in event_types else 0.35,
        "us10y_change_30d_bps": -12.0 if bullish_score > bearish_score else 8.0,
        "dxy_change_30d_pct": -0.8 if bullish_score > bearish_score else 0.6,
        "fed_liquidity_score": round(max(bullish_score, 0.35), 6),
        "policy_easing_score": round(max(bullish_score - 0.1, 0.2), 6),
        "event_risk_score": max(uncertainty_score, 0.65 if entry_policy_bias in {"halt_high_impact_window", "pre_event_reduce"} else 0.0),
        "btc_safe_haven_score": round(max(bullish_score if "geopolitics" in event_types else 0.35, 0.35), 6),
        "news_bullish_score": bullish_score,
        "news_bearish_score": bearish_score,
        "news_uncertainty_score": uncertainty_score,
        "news_majors_only_bias": 1.0 if majors_bias == "majors_only" else 0.0,
        "directional_bearish_score": round(directional_bearish_score, 6),
        "execution_risk_score": round(execution_risk_score, 6),
        "fear_greed_index": float(fear_greed[0]),
        "fear_greed_category": fear_greed[1],
    }

    return NewsMacroSignal(
        generated_at=now.isoformat(),
        refreshed_at=now.isoformat(),
        refresh_reason=refresh_reason,
        next_scheduled_refresh_at=next_scheduled_refresh_at.isoformat(),
        trigger_reasons=trigger_reasons,
        bullish_score=bullish_score,
        bearish_score=bearish_score,
        uncertainty_score=uncertainty_score,
        dominant_bias=dominant_bias,
        event_types=event_types,
        majors_bias=majors_bias,
        leverage_cap=leverage_cap,
        size_multiplier=round(size_multiplier, 6),
        entry_policy_bias=entry_policy_bias,
        macro_inputs=macro_inputs,
        headlines=headlines[:12],
    )

def write_news_macro_signal(
    *,
    output_path: Path,
    macro_inputs_output_path: Path,
    official_events_path: Path | None = None,
    state_path: Path | None = None,
    fetcher=_fetch_text,
    now: datetime | None = None,
    force: bool = False,
) -> tuple[Path, Path, str]:
    now = now or datetime.now(tz=timezone.utc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    macro_inputs_output_path.parent.mkdir(parents=True, exist_ok=True)
    if state_path is None:
        state_path = output_path.with_name("news_macro_signal.state.json")
    state_path.parent.mkdir(parents=True, exist_ok=True)

    existing_output = _load_json(output_path)
    headlines = fetch_google_news_headlines(fetcher=fetcher, now=now)
    official_events = _load_official_events(official_events_path)
    decision = decide_refresh(
        now=now,
        existing_payload=existing_output,
        official_events=official_events,
        headlines=headlines,
    )
    if force:
        decision = RefreshDecision(True, "manual_force", ("MANUAL_FORCE",), _next_schedule(now))

    headline_hash = hashlib.sha256(
        json.dumps([headline.as_dict() for headline in headlines[:8]], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    if decision.should_refresh:
        fear_greed = fetch_fear_greed_index(fetcher=fetcher)
        signal = build_signal(
            now=now,
            refresh_reason=decision.reason,
            trigger_reasons=decision.trigger_reasons,
            next_scheduled_refresh_at=decision.next_scheduled_refresh_at,
            headlines=headlines,
            official_events=official_events,
            fear_greed=fear_greed,
        )
        payload = signal.as_dict()
        payload["headline_hash"] = headline_hash
        payload["last_schedule_label"] = _current_schedule_label(now)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        macro_inputs_output_path.write_text(json.dumps(signal.macro_inputs, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        status = "refreshed"
    else:
        payload = dict(existing_output or {})
        payload.update(
            {
                "generated_at": now.isoformat(),
                "refresh_reason": decision.reason,
                "next_scheduled_refresh_at": decision.next_scheduled_refresh_at.isoformat(),
                "checked_at": now.isoformat(),
                "headline_hash": headline_hash,
                "headlines_preview": [headline.as_dict() for headline in headlines[:6]],
            }
        )
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        status = "skipped"

    state_path.write_text(
        json.dumps(
            {
                "checked_at": now.isoformat(),
                "status": status,
                "refresh_reason": decision.reason,
                "trigger_reasons": list(decision.trigger_reasons),
                "next_scheduled_refresh_at": decision.next_scheduled_refresh_at.isoformat(),
                "headline_hash": headline_hash,
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_path, macro_inputs_output_path, status

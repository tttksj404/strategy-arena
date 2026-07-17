import contextlib
import datetime as dt
import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlparse


LOGICAL_SCHEMAS: Final[dict[str, tuple[str, ...]]] = {
    "race_data": ("market_odds_snapshots",),
    "prediction": ("predictions",),
    "user_account": ("users", "devices", "analysis_usage"),
    "billing": ("subscriptions", "rewarded_ad_transactions"),
    "analytics": ("user_view_events",),
}

DEFAULT_SQLITE_PATH: Final[str] = os.path.join(os.path.dirname(__file__), "data", "strategy_arena.sqlite")
DEFAULT_FREE_DAILY_ANALYSIS_LIMIT: Final[int] = 3
DEFAULT_REWARDED_AD_MAX_CREDITS: Final[int] = 1
DEFAULT_REWARDED_AD_DAILY_CAP: Final[int] = 20
DEFAULT_REWARDED_AD_IP_PER_MIN_CAP: Final[int] = 20
PREVIEW_PRO_FLAG_PATH: Final[str] = os.path.join(os.path.dirname(__file__), ".runtime", "racelens-preview", "force_pro")
_FORCE_PRO_PRODUCTION_WARNING_EMITTED = False


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    storage: str
    url: str
    sqlite_path: str | None


def database_config() -> DatabaseConfig:
    url = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}").strip()
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        return DatabaseConfig(storage="postgresql", url=url, sqlite_path=None)
    if url.startswith("sqlite:///"):
        return DatabaseConfig(storage="sqlite", url=url, sqlite_path=url.replace("sqlite:///", "", 1))
    return DatabaseConfig(storage="sqlite", url=f"sqlite:///{url}", sqlite_path=url)


def _sqlite_table(schema: str, table: str) -> str:
    return f"{schema}__{table}"


@contextlib.contextmanager
def _connect_sqlite(config: DatabaseConfig):
    path = config.sqlite_path or DEFAULT_SQLITE_PATH
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextlib.contextmanager
def _connect_postgres(config: DatabaseConfig):
    import psycopg

    with psycopg.connect(config.url) as conn:
        yield conn


def _connection(config: DatabaseConfig):
    if config.storage == "postgresql":
        return _connect_postgres(config)
    return _connect_sqlite(config)


def _execute(conn, sql: str, params: tuple = ()) -> None:
    conn.execute(sql, params)


def _sqlite_column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(str(row["name"] if not isinstance(row, tuple) else row[1]) == column for row in rows)


def _table_ref(config: DatabaseConfig, schema: str, table: str) -> str:
    if config.storage == "postgresql":
        return f"{schema}.{table}"
    return _sqlite_table(schema, table)


def init_app_database() -> None:
    config = database_config()
    with _connection(config) as conn:
        if config.storage == "postgresql":
            _init_postgres(conn)
        else:
            _init_sqlite(conn)


def _init_sqlite(conn) -> None:
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("race_data", "market_odds_snapshots")} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sport TEXT NOT NULL, race_date TEXT NOT NULL, meet TEXT NOT NULL, race_no TEXT NOT NULL,
        payload_json TEXT NOT NULL, captured_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("prediction", "predictions")} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sport TEXT NOT NULL, race_date TEXT NOT NULL, meet TEXT NOT NULL, race_no TEXT NOT NULL,
        status TEXT NOT NULL, market_used INTEGER NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("user_account", "users")} (
        id TEXT PRIMARY KEY, provider TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("user_account", "devices")} (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL, platform TEXT NOT NULL, last_seen_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("user_account", "analysis_usage")} (
        user_id TEXT NOT NULL, usage_date TEXT NOT NULL, count INTEGER NOT NULL DEFAULT 0,
        rewarded_credits INTEGER NOT NULL DEFAULT 0, rewarded_claims INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL,
        PRIMARY KEY (user_id, usage_date)
    )""")
    usage_table = _sqlite_table("user_account", "analysis_usage")
    if not _sqlite_column_exists(conn, usage_table, "rewarded_credits"):
        _execute(conn, f"ALTER TABLE {usage_table} ADD COLUMN rewarded_credits INTEGER NOT NULL DEFAULT 0")
    if not _sqlite_column_exists(conn, usage_table, "rewarded_claims"):
        _execute(conn, f"ALTER TABLE {usage_table} ADD COLUMN rewarded_claims INTEGER NOT NULL DEFAULT 0")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("billing", "subscriptions")} (
        user_id TEXT PRIMARY KEY, platform TEXT NOT NULL, product_id TEXT NOT NULL,
        status TEXT NOT NULL, expires_at TEXT, updated_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("billing", "rewarded_ad_transactions")} (
        transaction_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, device_id TEXT NOT NULL,
        ad_unit TEXT NOT NULL, reward_amount INTEGER NOT NULL, reward_item TEXT NOT NULL,
        rewarded_at TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("analytics", "user_view_events")} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL, event_name TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("analytics", "ip_user_creations")} (
        ip_address TEXT NOT NULL, usage_date TEXT NOT NULL, user_id TEXT NOT NULL,
        device_id TEXT NOT NULL, created_at TEXT NOT NULL
    )""")
    _execute(conn, f"""CREATE INDEX IF NOT EXISTS {_sqlite_table("analytics", "ip_user_creations")}_idx
        ON {_sqlite_table("analytics", "ip_user_creations")} (ip_address, usage_date, created_at)""")
    _execute(conn, f"""CREATE TABLE IF NOT EXISTS {_sqlite_table("analytics", "rate_limits")} (
        scope TEXT NOT NULL, key TEXT NOT NULL, window_start TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
        PRIMARY KEY (scope, key, window_start)
    )""")


def _init_postgres(conn) -> None:
    for schema in LOGICAL_SCHEMAS:
        _execute(conn, f"CREATE SCHEMA IF NOT EXISTS {schema}")
    _execute(conn, """CREATE TABLE IF NOT EXISTS race_data.market_odds_snapshots (
        id BIGSERIAL PRIMARY KEY,
        sport TEXT NOT NULL, race_date TEXT NOT NULL, meet TEXT NOT NULL, race_no TEXT NOT NULL,
        payload_json JSONB NOT NULL, captured_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS prediction.predictions (
        id BIGSERIAL PRIMARY KEY,
        sport TEXT NOT NULL, race_date TEXT NOT NULL, meet TEXT NOT NULL, race_no TEXT NOT NULL,
        status TEXT NOT NULL, market_used BOOLEAN NOT NULL, payload_json JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS user_account.users (
        id TEXT PRIMARY KEY, provider TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS user_account.devices (
        id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES user_account.users(id),
        platform TEXT NOT NULL, last_seen_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS user_account.analysis_usage (
        user_id TEXT NOT NULL REFERENCES user_account.users(id), usage_date DATE NOT NULL,
        count INTEGER NOT NULL DEFAULT 0, rewarded_credits INTEGER NOT NULL DEFAULT 0,
        rewarded_claims INTEGER NOT NULL DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (user_id, usage_date)
    )""")
    _execute(conn, "ALTER TABLE user_account.analysis_usage ADD COLUMN IF NOT EXISTS rewarded_credits INTEGER NOT NULL DEFAULT 0")
    _execute(conn, "ALTER TABLE user_account.analysis_usage ADD COLUMN IF NOT EXISTS rewarded_claims INTEGER NOT NULL DEFAULT 0")
    _execute(conn, """CREATE TABLE IF NOT EXISTS billing.subscriptions (
        user_id TEXT PRIMARY KEY REFERENCES user_account.users(id), platform TEXT NOT NULL,
        product_id TEXT NOT NULL, status TEXT NOT NULL, expires_at TIMESTAMPTZ, updated_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS billing.rewarded_ad_transactions (
        transaction_id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES user_account.users(id),
        device_id TEXT NOT NULL, ad_unit TEXT NOT NULL, reward_amount INTEGER NOT NULL,
        reward_item TEXT NOT NULL, rewarded_at TIMESTAMPTZ NOT NULL, created_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS analytics.user_view_events (
        id BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL REFERENCES user_account.users(id),
        event_name TEXT NOT NULL, payload_json JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS analytics.ip_user_creations (
        ip_address TEXT NOT NULL, usage_date DATE NOT NULL, user_id TEXT NOT NULL,
        device_id TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL
    )""")
    _execute(conn, """CREATE INDEX IF NOT EXISTS ip_user_creations_ip_day_idx
        ON analytics.ip_user_creations (ip_address, usage_date, created_at)""")
    _execute(conn, """CREATE TABLE IF NOT EXISTS analytics.rate_limits (
        scope TEXT NOT NULL, key TEXT NOT NULL, window_start TEXT NOT NULL,
        count INTEGER NOT NULL DEFAULT 0, updated_at TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (scope, key, window_start)
    )""")


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _user_id_for_device(device_id: str) -> str:
    digest = hashlib.sha256(device_id.encode("utf-8")).hexdigest()[:24]
    return f"usr_{digest}"


def _usage_date() -> str:
    return dt.datetime.now(dt.UTC).date().isoformat()


def _request_minute() -> str:
    return dt.datetime.now(dt.UTC).replace(second=0, microsecond=0).isoformat(timespec="minutes")


def _ip_new_user_cap() -> int:
    try:
        return max(0, int(os.environ.get("RACELENS_IP_NEW_USER_CAP", "5")))
    except ValueError:
        return 5


def _live_decision_ip_minute_cap() -> int:
    try:
        return max(0, int(os.environ.get("RACELENS_LIVE_DECISION_IP_PER_MIN_CAP", "30")))
    except ValueError:
        return 30


def _is_active_subscription(row: Any) -> bool:
    if not row:
        return False
    status = str(row["status"] if not isinstance(row, tuple) else row[0]).lower()
    expires_at = row["expires_at"] if not isinstance(row, tuple) else row[1]
    if status not in {"active", "trialing", "grace_period"}:
        return False
    if not expires_at:
        return True
    try:
        normalized = str(expires_at).replace("Z", "+00:00")
        expires = dt.datetime.fromisoformat(normalized)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=dt.UTC)
        return expires > dt.datetime.now(dt.UTC)
    except ValueError:
        return False


def _env_flag(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def entitlement_mode() -> str:
    return "production" if (os.environ.get("RACELENS_ENV") or "").strip().lower() == "production" else "preview"


def _configured_pro_ids() -> set[str]:
    return {
        item.strip()
        for item in (os.environ.get("RACELENS_PRO_DEVICE_IDS") or "").split(",")
        if item.strip()
    }


def _forced_pro_enabled(device_id: str, user_id: str, platform: str) -> bool:
    global _FORCE_PRO_PRODUCTION_WARNING_EMITTED
    configured = _configured_pro_ids()
    if entitlement_mode() == "production":
        # fail-closed: 전면 무료 pro는 env로 명시할 때만 켠다. 기본 off.
        public_pro = (os.environ.get("RACELENS_PUBLIC_PRO", "0").strip().lower()
                      in {"1", "true", "yes", "on"})
        if public_pro:
            return True
        if (
            (_env_flag("RACELENS_FORCE_PRO") or os.path.exists(PREVIEW_PRO_FLAG_PATH))
            and not _FORCE_PRO_PRODUCTION_WARNING_EMITTED
        ):
            logging.getLogger(__name__).warning("RACELENS_FORCE_PRO/preview flag ignored in production")
            _FORCE_PRO_PRODUCTION_WARNING_EMITTED = True
        return device_id in configured or user_id in configured
    configured_url = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}").strip()
    uses_default_local_db = configured_url in {"", f"sqlite:///{DEFAULT_SQLITE_PATH}", DEFAULT_SQLITE_PATH}
    if _env_flag("RACELENS_FORCE_PRO") or (uses_default_local_db and os.path.exists(PREVIEW_PRO_FLAG_PATH)):
        return True
    return device_id in configured or user_id in configured


def free_daily_analysis_limit() -> int:
    """무료 일일 분석 한도. RACELENS_FREE_DAILY_ANALYSIS_LIMIT env로 조정
    (IAP 출시 전 임시 무제한 운영 등). 미설정/비정상 값이면 기본 3."""
    raw = (os.environ.get("RACELENS_FREE_DAILY_ANALYSIS_LIMIT") or "").strip()
    if not raw:
        return DEFAULT_FREE_DAILY_ANALYSIS_LIMIT
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_FREE_DAILY_ANALYSIS_LIMIT


def rewarded_ads_enabled() -> bool:
    return _env_flag("RACELENS_REWARDED_ADS_ENABLED")


def rewarded_ad_max_credits() -> int:
    raw = (os.environ.get("RACELENS_REWARDED_AD_MAX_CREDITS") or "").strip()
    if not raw:
        return DEFAULT_REWARDED_AD_MAX_CREDITS
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_REWARDED_AD_MAX_CREDITS


def rewarded_ad_daily_cap() -> int:
    raw = (os.environ.get("RACELENS_REWARDED_AD_DAILY_CAP") or "").strip()
    if not raw:
        return DEFAULT_REWARDED_AD_DAILY_CAP
    try:
        return max(0, int(raw))
    except ValueError:
        return DEFAULT_REWARDED_AD_DAILY_CAP


def _session_payload(
    user_id: str,
    device_id: str,
    entitlement: str,
    used: int,
    rewarded_credits: int = 0,
    claim_source: str | None = None,
) -> dict[str, Any]:
    limit = free_daily_analysis_limit()
    safe_used = max(0, min(limit, int(used)))
    safe_credits = max(0, min(99, int(rewarded_credits)))
    if entitlement == "pro":
        remaining = limit
        safe_used = 0
        safe_credits = 0
    else:
        remaining = max(0, limit - safe_used)
    payload = {
        "user_id": user_id,
        "device_id": device_id,
        "entitlement": entitlement,
        "free_analysis_limit": limit,
        "free_analysis_used": safe_used,
        "free_analysis_remaining": remaining,
        "rewarded_analysis_credits": safe_credits,
    }
    if claim_source:
        payload["analysis_claim_source"] = claim_source
    return payload


def _subscription_entitlement(conn, config: DatabaseConfig, user_id: str) -> str:
    subscriptions = _table_ref(config, "billing", "subscriptions")
    placeholder = "%s" if config.storage == "postgresql" else "?"
    row = conn.execute(
        f"SELECT status, expires_at FROM {subscriptions} WHERE user_id = {placeholder}",
        (user_id,),
    ).fetchone()
    return "pro" if _is_active_subscription(row) else "free"


def upsert_subscription(
    user_id: str,
    platform: str,
    product_id: str,
    status: str,
    expires_at: str | None,
) -> None:
    init_app_database()
    config = database_config()
    now = _now()
    table = _table_ref(config, "billing", "subscriptions")
    with _connection(config) as conn:
        if config.storage == "postgresql":
            _execute(conn, f"""INSERT INTO {table} (user_id, platform, product_id, status, expires_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id)
                DO UPDATE SET platform = EXCLUDED.platform, product_id = EXCLUDED.product_id,
                    status = EXCLUDED.status, expires_at = EXCLUDED.expires_at, updated_at = EXCLUDED.updated_at""",
                     (user_id, platform, product_id, status, expires_at, now))
        else:
            _execute(conn, f"""INSERT INTO {table} (user_id, platform, product_id, status, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id)
                DO UPDATE SET platform = excluded.platform, product_id = excluded.product_id,
                    status = excluded.status, expires_at = excluded.expires_at, updated_at = excluded.updated_at""",
                     (user_id, platform, product_id, status, expires_at, now))


def _free_usage_count(conn, config: DatabaseConfig, user_id: str) -> int:
    return _analysis_usage_state(conn, config, user_id)[0]


def _analysis_usage_state(conn, config: DatabaseConfig, user_id: str) -> tuple[int, int]:
    count, credits, _claims = _analysis_reward_state(conn, config, user_id)
    return count, credits


def _analysis_reward_state(conn, config: DatabaseConfig, user_id: str) -> tuple[int, int, int]:
    usage = _table_ref(config, "user_account", "analysis_usage")
    placeholder = "%s" if config.storage == "postgresql" else "?"
    row = conn.execute(
        f"SELECT count, rewarded_credits, rewarded_claims FROM {usage} WHERE user_id = {placeholder} AND usage_date = {placeholder}",
        (user_id, _usage_date()),
    ).fetchone()
    if not row:
        return 0, 0, 0
    count = int(row["count"] if not isinstance(row, tuple) else row[0])
    credits = int(row["rewarded_credits"] if not isinstance(row, tuple) else row[1])
    claims = int(row["rewarded_claims"] if not isinstance(row, tuple) else row[2])
    return count, credits, claims


def _upsert_free_usage(conn, config: DatabaseConfig, user_id: str, count: int, now: str) -> None:
    _existing_count, rewarded_credits = _analysis_usage_state(conn, config, user_id)
    _upsert_analysis_usage(conn, config, user_id, count, rewarded_credits, now)


def _upsert_analysis_usage(conn, config: DatabaseConfig, user_id: str, count: int, rewarded_credits: int, now: str) -> None:
    _current_count, _current_credits, rewarded_claims = _analysis_reward_state(conn, config, user_id)
    _upsert_rewarded_usage(conn, config, user_id, count, rewarded_credits, rewarded_claims, now)


def _upsert_rewarded_usage(
    conn,
    config: DatabaseConfig,
    user_id: str,
    count: int,
    rewarded_credits: int,
    rewarded_claims: int,
    now: str,
) -> None:
    usage = _table_ref(config, "user_account", "analysis_usage")
    safe_count = max(0, int(count))
    safe_credits = max(0, min(rewarded_ad_max_credits(), int(rewarded_credits)))
    safe_claims = max(0, int(rewarded_claims))
    if config.storage == "postgresql":
        _execute(conn, f"""INSERT INTO {usage} (user_id, usage_date, count, rewarded_credits, rewarded_claims, updated_at) VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, usage_date)
            DO UPDATE SET count = EXCLUDED.count, rewarded_credits = EXCLUDED.rewarded_credits,
                rewarded_claims = EXCLUDED.rewarded_claims, updated_at = EXCLUDED.updated_at""",
                 (user_id, _usage_date(), safe_count, safe_credits, safe_claims, now))
    else:
        _execute(conn, f"""INSERT INTO {usage} (user_id, usage_date, count, rewarded_credits, rewarded_claims, updated_at) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, usage_date)
            DO UPDATE SET count = excluded.count, rewarded_credits = excluded.rewarded_credits,
                rewarded_claims = excluded.rewarded_claims, updated_at = excluded.updated_at""",
                 (user_id, _usage_date(), safe_count, safe_credits, safe_claims, now))


def _existing_device_user(conn, config: DatabaseConfig, device_id: str) -> str | None:
    devices = _table_ref(config, "user_account", "devices")
    placeholder = "%s" if config.storage == "postgresql" else "?"
    row = conn.execute(f"SELECT user_id FROM {devices} WHERE id = {placeholder}", (device_id,)).fetchone()
    if not row:
        return None
    return str(row["user_id"] if not isinstance(row, tuple) else row[0])


def _recent_ip_user(conn, config: DatabaseConfig, ip_address: str) -> str | None:
    table = _table_ref(config, "analytics", "ip_user_creations")
    placeholder = "%s" if config.storage == "postgresql" else "?"
    row = conn.execute(
        f"""SELECT user_id FROM {table}
            WHERE ip_address = {placeholder} AND usage_date = {placeholder}
            ORDER BY created_at DESC LIMIT 1""",
        (ip_address, _usage_date()),
    ).fetchone()
    if not row:
        return None
    return str(row["user_id"] if not isinstance(row, tuple) else row[0])


def _ip_creation_count(conn, config: DatabaseConfig, ip_address: str) -> int:
    table = _table_ref(config, "analytics", "ip_user_creations")
    placeholder = "%s" if config.storage == "postgresql" else "?"
    row = conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE ip_address = {placeholder} AND usage_date = {placeholder}",
        (ip_address, _usage_date()),
    ).fetchone()
    return int(row[0])


def _record_ip_creation(conn, config: DatabaseConfig, ip_address: str, user_id: str, device_id: str, now: str) -> None:
    table = _table_ref(config, "analytics", "ip_user_creations")
    if config.storage == "postgresql":
        _execute(conn, f"INSERT INTO {table} (ip_address, usage_date, user_id, device_id, created_at) VALUES (%s, %s, %s, %s, %s)",
                 (ip_address, _usage_date(), user_id, device_id, now))
    else:
        _execute(conn, f"INSERT INTO {table} (ip_address, usage_date, user_id, device_id, created_at) VALUES (?, ?, ?, ?, ?)",
                 (ip_address, _usage_date(), user_id, device_id, now))


def _ensure_device_user(conn, config: DatabaseConfig, device_id: str, platform: str, now: str, ip_address: str | None) -> str:
    user_id = _existing_device_user(conn, config, device_id)
    if user_id is None:
        user_id = _user_id_for_device(device_id)
        if ip_address:
            cap = _ip_new_user_cap()
            count = _ip_creation_count(conn, config, ip_address)
            recent_user = _recent_ip_user(conn, config, ip_address)
            if cap and count >= cap and recent_user:
                user_id = recent_user
            else:
                _record_ip_creation(conn, config, ip_address, user_id, device_id, now)
    users = _table_ref(config, "user_account", "users")
    devices = _table_ref(config, "user_account", "devices")
    if config.storage == "postgresql":
        _execute(conn, f"INSERT INTO {users} (id, provider, created_at) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING", (user_id, "anonymous", now))
        _execute(conn, f"""INSERT INTO {devices} (id, user_id, platform, last_seen_at) VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET platform = EXCLUDED.platform, last_seen_at = EXCLUDED.last_seen_at""", (device_id, user_id, platform, now))
    else:
        _execute(conn, f"INSERT OR IGNORE INTO {users} (id, provider, created_at) VALUES (?, ?, ?)", (user_id, "anonymous", now))
        _execute(conn, f"""INSERT INTO {devices} (id, user_id, platform, last_seen_at) VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET platform = excluded.platform, last_seen_at = excluded.last_seen_at""", (device_id, user_id, platform, now))
    return user_id


def ensure_app_session(device_id: str, platform: str, ip_address: str | None = None) -> dict[str, Any]:
    init_app_database()
    config = database_config()
    now = _now()
    with _connection(config) as conn:
        user_id = _ensure_device_user(conn, config, device_id, platform, now, ip_address)
        entitlement = "pro" if _forced_pro_enabled(device_id, user_id, platform) else _subscription_entitlement(conn, config, user_id)
        used, rewarded_credits = _analysis_usage_state(conn, config, user_id)
    return _session_payload(user_id, device_id, entitlement, used, rewarded_credits)


def claim_live_decision_session(device_id: str, platform: str, ip_address: str | None = None) -> tuple[dict[str, Any], bool]:
    init_app_database()
    config = database_config()
    now = _now()
    with _connection(config) as conn:
        user_id = _ensure_device_user(conn, config, device_id, platform, now, ip_address)
        entitlement = "pro" if _forced_pro_enabled(device_id, user_id, platform) else _subscription_entitlement(conn, config, user_id)
        used, rewarded_credits = _analysis_usage_state(conn, config, user_id)
        if entitlement == "pro":
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits, "pro"), True
        if used >= free_daily_analysis_limit():
            if rewarded_credits <= 0:
                return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False
            rewarded_credits -= 1
            _upsert_analysis_usage(conn, config, user_id, used, rewarded_credits, now)
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits, "rewarded_ad"), True
        used += 1
        _upsert_analysis_usage(conn, config, user_id, used, rewarded_credits, now)
    return _session_payload(user_id, device_id, "free", used, rewarded_credits, "free_daily"), True


def release_live_decision_session(device_id: str, platform: str, claimed_session: dict[str, Any]) -> dict[str, Any]:
    init_app_database()
    config = database_config()
    now = _now()
    claimed_user_id = str(claimed_session.get("user_id") or "")
    with _connection(config) as conn:
        user_id = claimed_user_id or _ensure_device_user(conn, config, device_id, platform, now, None)
        entitlement = "pro" if _forced_pro_enabled(device_id, user_id, platform) else _subscription_entitlement(conn, config, user_id)
        used, rewarded_credits = _analysis_usage_state(conn, config, user_id)
        claim_source = str(claimed_session.get("analysis_claim_source") or "free_daily")
        if entitlement != "pro" and claim_source == "rewarded_ad":
            rewarded_credits = min(rewarded_ad_max_credits(), rewarded_credits + 1)
            _upsert_analysis_usage(conn, config, user_id, used, rewarded_credits, now)
        elif entitlement != "pro" and used > 0:
            used -= 1
            _upsert_analysis_usage(conn, config, user_id, used, rewarded_credits, now)
    return _session_payload(user_id, device_id, entitlement, used, rewarded_credits)


def claim_rewarded_ad_credit(device_id: str, platform: str, ip_address: str | None = None) -> tuple[dict[str, Any], bool]:
    init_app_database()
    config = database_config()
    now = _now()
    with _connection(config) as conn:
        user_id = _ensure_device_user(conn, config, device_id, platform, now, ip_address)
        entitlement = "pro" if _forced_pro_enabled(device_id, user_id, platform) else _subscription_entitlement(conn, config, user_id)
        used, rewarded_credits, rewarded_claims = _analysis_reward_state(conn, config, user_id)
        if entitlement == "pro":
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False
        daily_cap = rewarded_ad_daily_cap()
        if daily_cap <= 0 or rewarded_claims >= daily_cap:
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False
        max_credits = rewarded_ad_max_credits()
        if max_credits <= 0 or rewarded_credits >= max_credits:
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False
        rewarded_credits += 1
        rewarded_claims += 1
        _upsert_rewarded_usage(conn, config, user_id, used, rewarded_credits, rewarded_claims, now)
    return _session_payload(user_id, device_id, "free", used, rewarded_credits), True


def claim_verified_rewarded_ad_credit(
    device_id: str,
    platform: str,
    transaction_id: str,
    ad_unit: str,
    reward_amount: int,
    reward_item: str,
    rewarded_at: str,
) -> tuple[dict[str, Any], bool, bool]:
    """Persist one verified Google transaction and grant at most one credit."""
    init_app_database()
    config = database_config()
    now = _now()
    transactions = _table_ref(config, "billing", "rewarded_ad_transactions")
    with _connection(config) as conn:
        user_id = _ensure_device_user(conn, config, device_id, platform, now, None)
        if config.storage == "postgresql":
            cursor = conn.execute(
                f"""INSERT INTO {transactions}
                    (transaction_id, user_id, device_id, ad_unit, reward_amount, reward_item, rewarded_at, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (transaction_id) DO NOTHING""",
                (transaction_id, user_id, device_id, ad_unit, reward_amount, reward_item, rewarded_at, now),
            )
        else:
            cursor = conn.execute(
                f"""INSERT OR IGNORE INTO {transactions}
                    (transaction_id, user_id, device_id, ad_unit, reward_amount, reward_item, rewarded_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (transaction_id, user_id, device_id, ad_unit, reward_amount, reward_item, rewarded_at, now),
            )
        duplicate = cursor.rowcount == 0
        entitlement = "pro" if _forced_pro_enabled(device_id, user_id, platform) else _subscription_entitlement(conn, config, user_id)
        used, rewarded_credits, rewarded_claims = _analysis_reward_state(conn, config, user_id)
        if duplicate or entitlement == "pro":
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False, duplicate
        daily_cap = rewarded_ad_daily_cap()
        max_credits = rewarded_ad_max_credits()
        if daily_cap <= 0 or rewarded_claims >= daily_cap or max_credits <= 0 or rewarded_credits >= max_credits:
            return _session_payload(user_id, device_id, entitlement, used, rewarded_credits), False, False
        rewarded_credits += 1
        rewarded_claims += 1
        _upsert_rewarded_usage(conn, config, user_id, used, rewarded_credits, rewarded_claims, now)
    return _session_payload(user_id, device_id, "free", used, rewarded_credits), True, False


def _rewarded_ad_ip_minute_cap() -> int:
    try:
        return max(0, int(os.environ.get("RACELENS_REWARDED_AD_IP_PER_MIN_CAP", str(DEFAULT_REWARDED_AD_IP_PER_MIN_CAP))))
    except ValueError:
        return DEFAULT_REWARDED_AD_IP_PER_MIN_CAP


def check_rewarded_ad_ip_rate_limit(ip_address: str) -> tuple[bool, int]:
    cap = _rewarded_ad_ip_minute_cap()
    if cap <= 0:
        return True, cap
    init_app_database()
    config = database_config()
    table = _table_ref(config, "analytics", "rate_limits")
    now = _now()
    window = _request_minute()
    scope = "rewarded_ad_claim"
    with _connection(config) as conn:
        if config.storage == "postgresql":
            _execute(conn, f"""INSERT INTO {table} (scope, key, window_start, count, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (scope, key, window_start)
                DO UPDATE SET count = {table}.count + 1, updated_at = EXCLUDED.updated_at""",
                     (scope, ip_address, window, 1, now))
        else:
            _execute(conn, f"""INSERT INTO {table} (scope, key, window_start, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, key, window_start)
                DO UPDATE SET count = count + 1, updated_at = excluded.updated_at""",
                     (scope, ip_address, window, 1, now))
        placeholder = "%s" if config.storage == "postgresql" else "?"
        row = conn.execute(
            f"SELECT count FROM {table} WHERE scope = {placeholder} AND key = {placeholder} AND window_start = {placeholder}",
            (scope, ip_address, window),
        ).fetchone()
    count = int(row["count"] if not isinstance(row, tuple) else row[0])
    return count <= cap, cap


def check_live_decision_ip_rate_limit(ip_address: str) -> tuple[bool, int]:
    cap = _live_decision_ip_minute_cap()
    if cap <= 0:
        return True, cap
    init_app_database()
    config = database_config()
    table = _table_ref(config, "analytics", "rate_limits")
    now = _now()
    window = _request_minute()
    scope = "live_decision"
    with _connection(config) as conn:
        if config.storage == "postgresql":
            _execute(conn, f"""INSERT INTO {table} (scope, key, window_start, count, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (scope, key, window_start)
                DO UPDATE SET count = {table}.count + 1, updated_at = EXCLUDED.updated_at""",
                     (scope, ip_address, window, 1, now))
        else:
            _execute(conn, f"""INSERT INTO {table} (scope, key, window_start, count, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(scope, key, window_start)
                DO UPDATE SET count = count + 1, updated_at = excluded.updated_at""",
                     (scope, ip_address, window, 1, now))
        placeholder = "%s" if config.storage == "postgresql" else "?"
        row = conn.execute(
            f"SELECT count FROM {table} WHERE scope = {placeholder} AND key = {placeholder} AND window_start = {placeholder}",
            (scope, ip_address, window),
        ).fetchone()
    count = int(row["count"] if not isinstance(row, tuple) else row[0])
    return count <= cap, cap


def record_live_decision(device_id: str, platform: str, context: dict[str, str], result: dict) -> dict[str, Any]:
    session = ensure_app_session(device_id, platform)
    config = database_config()
    now = _now()
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    event_payload = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    with _connection(config) as conn:
        prediction = _table_ref(config, "prediction", "predictions")
        analytics = _table_ref(config, "analytics", "user_view_events")
        odds = _table_ref(config, "race_data", "market_odds_snapshots")
        if config.storage == "postgresql":
            _execute(conn, f"INSERT INTO {prediction} (sport, race_date, meet, race_no, status, market_used, payload_json, created_at) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)",
                     (context["sport"], context["date"], context["meet"], context["race_no"], str(result.get("status", "hold")), bool(result.get("market_used")), payload, now))
            _execute(conn, f"INSERT INTO {analytics} (user_id, event_name, payload_json, created_at) VALUES (%s, %s, %s::jsonb, %s)", (session["user_id"], "live_decision_view", event_payload, now))
            if result.get("market_odds"):
                _execute(conn, f"INSERT INTO {odds} (sport, race_date, meet, race_no, payload_json, captured_at) VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                         (context["sport"], context["date"], context["meet"], context["race_no"], json.dumps(result["market_odds"], ensure_ascii=False), now))
        else:
            _execute(conn, f"INSERT INTO {prediction} (sport, race_date, meet, race_no, status, market_used, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     (context["sport"], context["date"], context["meet"], context["race_no"], str(result.get("status", "hold")), 1 if result.get("market_used") else 0, payload, now))
            _execute(conn, f"INSERT INTO {analytics} (user_id, event_name, payload_json, created_at) VALUES (?, ?, ?, ?)", (session["user_id"], "live_decision_view", event_payload, now))
            if result.get("market_odds"):
                _execute(conn, f"INSERT INTO {odds} (sport, race_date, meet, race_no, payload_json, captured_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (context["sport"], context["date"], context["meet"], context["race_no"], json.dumps(result["market_odds"], ensure_ascii=False), now))
    return session


def record_market_odds_snapshot(
    sport: str,
    race_date: str,
    meet: str,
    race_no: str,
    market_odds: list[dict[str, Any]],
    captured_at: str,
) -> bool:
    config = database_config()
    init_app_database()
    payload = json.dumps(market_odds, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    with _connection(config) as conn:
        table = _table_ref(config, "race_data", "market_odds_snapshots")
        placeholder = "%s" if config.storage == "postgresql" else "?"
        previous = conn.execute(
            f"SELECT payload_json, captured_at FROM {table} "
            f"WHERE sport = {placeholder} AND race_date = {placeholder} "
            f"AND meet = {placeholder} AND race_no = {placeholder} "
            "ORDER BY captured_at DESC LIMIT 1",
            (sport, race_date, meet, race_no),
        ).fetchone()
        if previous:
            previous_payload = previous["payload_json"] if not isinstance(previous, tuple) else previous[0]
            if not isinstance(previous_payload, str):
                previous_payload = json.dumps(
                    previous_payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            previous_at = previous["captured_at"] if not isinstance(previous, tuple) else previous[1]
            try:
                elapsed = (
                    dt.datetime.fromisoformat(captured_at)
                    - dt.datetime.fromisoformat(str(previous_at))
                ).total_seconds()
            except (TypeError, ValueError):
                elapsed = 60.0
            if previous_payload == payload and 0.0 <= elapsed < 60.0:
                return False
        if config.storage == "postgresql":
            _execute(
                conn,
                f"INSERT INTO {table} (sport, race_date, meet, race_no, payload_json, captured_at) "
                "VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
                (sport, race_date, meet, race_no, payload, captured_at),
            )
        else:
            _execute(
                conn,
                f"INSERT INTO {table} (sport, race_date, meet, race_no, payload_json, captured_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sport, race_date, meet, race_no, payload, captured_at),
            )
    return True


def record_market_odds_snapshot_safely(
    sport: str,
    race_date: str,
    meet: str,
    race_no: str,
    market_odds: list[dict[str, Any]],
    captured_at: str,
) -> bool:
    try:
        return record_market_odds_snapshot(
            sport, race_date, meet, race_no, market_odds, captured_at
        )
    except Exception:  # noqa: BLE001 — telemetry must never break prediction
        logging.exception("market odds snapshot persistence failed")
        return False


def record_ux_event(device_id: str, platform: str, event_name: str, payload: dict[str, Any]) -> dict[str, str]:
    session = ensure_app_session(device_id, platform)
    config = database_config()
    now = _now()
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with _connection(config) as conn:
        analytics = _table_ref(config, "analytics", "user_view_events")
        if config.storage == "postgresql":
            _execute(conn, f"INSERT INTO {analytics} (user_id, event_name, payload_json, created_at) VALUES (%s, %s, %s::jsonb, %s)",
                     (session["user_id"], event_name, payload_json, now))
        else:
            _execute(conn, f"INSERT INTO {analytics} (user_id, event_name, payload_json, created_at) VALUES (?, ?, ?, ?)",
                     (session["user_id"], event_name, payload_json, now))
    return session


def record_ux_event_safely(
    device_id: str,
    platform: str,
    event_name: str,
    payload: dict[str, Any],
) -> tuple[dict[str, str], dict]:
    config = database_config()
    try:
        session = record_ux_event(device_id, platform, event_name, payload)
        return session, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def ensure_app_session_safely(device_id: str, platform: str, ip_address: str | None = None) -> tuple[dict[str, Any], dict]:
    config = database_config()
    try:
        session = ensure_app_session(device_id, platform, ip_address=ip_address)
        return session, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def _row_count(conn, config: DatabaseConfig, schema: str) -> int:
    total = 0
    for table in LOGICAL_SCHEMAS[schema]:
        table_name = _table_ref(config, schema, table)
        total += int(conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}").fetchone()[0])
    return total


def app_data_layer_status() -> dict:
    config = database_config()
    try:
        init_app_database()
        with _connection(config) as conn:
            schemas = [
                {"name": schema, "tables": list(tables), "row_count": _row_count(conn, config, schema)}
                for schema, tables in LOGICAL_SCHEMAS.items()
            ]
        return {"ready": True, "storage": config.storage, "schemas": schemas}
    except (OSError, sqlite3.Error, ImportError) as exc:
        return {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []}


def fallback_app_session(device_id: str) -> dict[str, Any]:
    return _session_payload("anonymous", device_id, "free", free_daily_analysis_limit())


def _database_error_types() -> tuple[type[BaseException], ...]:
    errors: tuple[type[BaseException], ...] = (OSError, sqlite3.Error, ImportError)
    try:
        import psycopg
    except ImportError:
        return errors
    return errors + (psycopg.Error,)


def record_live_decision_safely(
    device_id: str,
    platform: str,
    context: dict[str, str],
    result: dict,
) -> tuple[dict[str, Any], dict]:
    config = database_config()
    try:
        session = record_live_decision(device_id, platform, context, result)
        return session, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def claim_live_decision_session_safely(device_id: str, platform: str, ip_address: str | None = None) -> tuple[dict[str, Any], bool, dict]:
    config = database_config()
    try:
        session, allowed = claim_live_decision_session(device_id, platform, ip_address=ip_address)
        return session, allowed, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            False,
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def claim_rewarded_ad_credit_safely(device_id: str, platform: str, ip_address: str | None = None) -> tuple[dict[str, Any], bool, dict]:
    config = database_config()
    try:
        session, granted = claim_rewarded_ad_credit(device_id, platform, ip_address=ip_address)
        return session, granted, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            False,
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def claim_verified_rewarded_ad_credit_safely(
    device_id: str,
    platform: str,
    transaction_id: str,
    ad_unit: str,
    reward_amount: int,
    reward_item: str,
    rewarded_at: str,
) -> tuple[dict[str, Any], bool, bool, dict]:
    config = database_config()
    try:
        session, granted, duplicate = claim_verified_rewarded_ad_credit(
            device_id,
            platform,
            transaction_id,
            ad_unit,
            reward_amount,
            reward_item,
            rewarded_at,
        )
        return session, granted, duplicate, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            False,
            False,
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )


def release_live_decision_session_safely(device_id: str, platform: str, claimed_session: dict[str, Any]) -> tuple[dict[str, Any], dict]:
    config = database_config()
    try:
        session = release_live_decision_session(device_id, platform, claimed_session)
        return session, app_data_layer_status()
    except _database_error_types() as exc:
        return (
            fallback_app_session(device_id),
            {"ready": False, "storage": config.storage, "error": f"{type(exc).__name__}: {exc}", "schemas": []},
        )

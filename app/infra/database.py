import hmac
import sqlite3
from contextlib import contextmanager
from typing import Optional

from app.core.config import settings
# auth_utils imports no app modules, so this does not create an import cycle.
from app.web.auth_utils import generate_feed_token

def init_db():
    """Initialize the database with the schema."""
    conn = sqlite3.connect(settings.DB_PATH)
    cursor = conn.cursor()
    
    # Subscriptions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        feed_url TEXT UNIQUE NOT NULL,
        title TEXT,
        description TEXT,
        slug TEXT UNIQUE,
        image_url TEXT,
        is_active BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_checked_at TIMESTAMP,
        remove_ads BOOLEAN DEFAULT 1,
        remove_promos BOOLEAN DEFAULT 1,
        remove_intros BOOLEAN DEFAULT 0,
        remove_outros BOOLEAN DEFAULT 0,
        custom_instructions TEXT,
        append_summary BOOLEAN DEFAULT 0,
        append_title_intro BOOLEAN DEFAULT 0,
        ai_rewrite_description BOOLEAN DEFAULT 0,
        ai_audio_summary BOOLEAN DEFAULT 0
    )
    """)
    
    # Simple migration attempts (ignore if exists)
    try:
        cursor.execute("ALTER TABLE subscriptions ADD COLUMN ai_rewrite_description BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass
        
    try:
        cursor.execute("ALTER TABLE subscriptions ADD COLUMN ai_audio_summary BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass


    # App Settings Singleton Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS app_settings (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        whisper_model TEXT DEFAULT 'base',
        ai_model_cascade TEXT DEFAULT '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash-lite"]',
        piper_model TEXT DEFAULT 'en_GB-cori-high.onnx',
        concurrent_downloads INTEGER DEFAULT 2,
        retention_days INTEGER DEFAULT 30,
        check_interval_minutes INTEGER DEFAULT 60,
        daily_download_limit INTEGER DEFAULT 0,
        
        ad_prompt_base TEXT,
        ad_target_sponsor TEXT,
        ad_target_promo TEXT,
        ad_target_intro TEXT,
        ad_target_outro TEXT,
        summary_prompt_template TEXT,
        
        active_ai_provider TEXT DEFAULT 'gemini',
        openai_api_key TEXT,
        anthropic_api_key TEXT,
        openrouter_api_key TEXT,
        openai_model TEXT DEFAULT 'gpt-4o',
        anthropic_model TEXT DEFAULT 'claude-3-5-sonnet',
        openrouter_model TEXT DEFAULT 'google/gemini-2.0-flash-001',
        app_external_url TEXT,
        
        enable_feed_auth INTEGER DEFAULT 0,
        feed_auth_username TEXT,
        feed_auth_password TEXT,
        -- Standalone/global feed bearer token. Replaces reliance on
        -- feed_auth_password for feed URLs. Stored retrievably on purpose:
        -- see the note on users.feed_token below.
        feed_auth_token TEXT,

        auth_enabled INTEGER DEFAULT 0,
        require_password_change INTEGER DEFAULT 0,
        -- NOTE: `initial_password` is deliberately absent. It used to hold the
        -- generated admin password in PLAINTEXT and the unauthenticated /login
        -- page rendered it. The write and the read are gone; the column and any
        -- value left in it are removed on upgrade below. Do not re-add it.
        ip_allowlist TEXT,
        
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    
    # Ensure default settings exist
    cursor.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
    
    # Set default summary prompt template if not set
    cursor.execute("""
        UPDATE app_settings 
        SET summary_prompt_template = ?
        WHERE id = 1 AND (summary_prompt_template IS NULL OR summary_prompt_template = '')
    """, ("""You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
The summary must:
1. NOT mention the podcast name, episode title, or date.
2. Start immediately with "This episode includes".
3. Briefly summarize key topics.
Transcript Context: {transcript_context}""",))

    # Discovery Cache Table
    #
    # Read-through cache for Podcast Index discovery payloads (trending, the
    # category list, per-category listings). Deliberately its own table:
    # discovery results are not-yet-subscribed podcasts, and `subscriptions`
    # is read unfiltered by the dashboard and by the processor's feed-check
    # loop, so parking them there would show them as subscriptions and poll
    # them. Rows here are disposable - dropping the table only costs one
    # upstream fetch.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS discovery_cache (
        cache_key TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        -- Per-user random bearer credential for read-only RSS feed access.
        --
        -- DELIBERATELY STORED RETRIEVABLE (plaintext), NOT HASHED. Do not
        -- "improve" this into a hash: the dashboard has to re-display the
        -- full feed URL, containing this token, at any time. Hashing it makes
        -- the URL unrecoverable after creation and breaks feed display.
        --
        -- The security win here is NOT storage secrecy. It is that the feed
        -- credential is decoupled from the account password (which previously
        -- leaked into podcast-client logs, proxies, and browser history via
        -- the base64 ?auth= parameter) and that it is revocable per user
        -- without touching the login password. This is how tokenized podcast
        -- feeds normally work.
        feed_token TEXT
    )
    """)
    
    # Access Requests Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS access_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        email TEXT,
        reason TEXT,
        requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'pending',
        ip_address TEXT,
        reviewed_by TEXT,
        reviewed_at TIMESTAMP
    )
    """)
    
    # Login Attempts Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS login_attempts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        ip_address TEXT,
        success INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        user_agent TEXT
    )
    """)

    # Episodes Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subscription_id INTEGER NOT NULL,
        guid TEXT NOT NULL,
        title TEXT NOT NULL,
        pub_date TIMESTAMP,
        original_url TEXT NOT NULL,
        duration INTEGER,
        status TEXT DEFAULT 'pending',
        processed_at TIMESTAMP,
        error_message TEXT,
        local_filename TEXT,
        transcript_path TEXT,
        ad_report_path TEXT,
        processing_step TEXT,
        progress INTEGER DEFAULT 0,
        description TEXT,
        ai_summary TEXT,
        report_path TEXT,
        file_size INTEGER,
        FOREIGN KEY (subscription_id) REFERENCES subscriptions (id),
        UNIQUE(subscription_id, guid)
    )
    """)
    
    # Migrations for existing databases
    migrations = [
        "ALTER TABLE episodes ADD COLUMN transcript_path TEXT",
        "ALTER TABLE episodes ADD COLUMN ad_report_path TEXT",
        "ALTER TABLE episodes ADD COLUMN processing_step TEXT",
        "ALTER TABLE episodes ADD COLUMN progress INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN description TEXT",
        "ALTER TABLE episodes ADD COLUMN report_path TEXT",
        "ALTER TABLE subscriptions ADD COLUMN image_url TEXT",
        "ALTER TABLE episodes ADD COLUMN file_size INTEGER",
        "ALTER TABLE episodes ADD COLUMN retry_count INTEGER DEFAULT 0",
        "ALTER TABLE episodes ADD COLUMN next_retry_at TIMESTAMP",
        "ALTER TABLE subscriptions ADD COLUMN remove_ads BOOLEAN DEFAULT 1",
        "ALTER TABLE subscriptions ADD COLUMN remove_promos BOOLEAN DEFAULT 1",
        "ALTER TABLE subscriptions ADD COLUMN remove_intros BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN remove_outros BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN custom_instructions TEXT",
        "ALTER TABLE subscriptions ADD COLUMN append_summary BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN append_title_intro BOOLEAN DEFAULT 0",
        
        # New prompt migrations
        "ALTER TABLE app_settings ADD COLUMN ad_prompt_base TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_sponsor TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_promo TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_intro TEXT",
        "ALTER TABLE app_settings ADD COLUMN ad_target_outro TEXT",
        "ALTER TABLE app_settings ADD COLUMN summary_prompt_template TEXT",
        
        # Multi-Provider AI migrations
        "ALTER TABLE app_settings ADD COLUMN active_ai_provider TEXT DEFAULT 'gemini'",
        "ALTER TABLE app_settings ADD COLUMN openai_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN anthropic_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openrouter_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN openai_model TEXT DEFAULT 'gpt-4o'",
        "ALTER TABLE app_settings ADD COLUMN anthropic_model TEXT DEFAULT 'claude-3-5-sonnet'",
        "ALTER TABLE app_settings ADD COLUMN openrouter_model TEXT DEFAULT 'google/gemini-2.0-flash-001'",
        "ALTER TABLE episodes ADD COLUMN processing_flags TEXT",
        "ALTER TABLE app_settings ADD COLUMN gemini_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN app_external_url TEXT",
        "ALTER TABLE app_settings ADD COLUMN enable_feed_auth INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_username TEXT",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_password TEXT",
        "ALTER TABLE app_settings ADD COLUMN auth_enabled INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN require_password_change INTEGER DEFAULT 0",
        # `ALTER TABLE app_settings ADD COLUMN initial_password TEXT` used to
        # live here. Removed: the column stored a plaintext admin password and
        # is now purged below. Re-adding it would recreate it on every upgrade.
        "ALTER TABLE app_settings ADD COLUMN ip_allowlist TEXT",
        "ALTER TABLE subscriptions ADD COLUMN description TEXT",
        "ALTER TABLE episodes ADD COLUMN ai_summary TEXT",
        "ALTER TABLE episodes ADD COLUMN is_manual_download BOOLEAN DEFAULT 0",
        "ALTER TABLE subscriptions ADD COLUMN retention_days INTEGER DEFAULT 30",
        "ALTER TABLE subscriptions ADD COLUMN manual_retention_days INTEGER DEFAULT 14",
        "ALTER TABLE subscriptions ADD COLUMN retention_limit INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN check_interval_minutes INTEGER DEFAULT 60",
        
        # Global Subscription Defaults
        "ALTER TABLE app_settings ADD COLUMN default_remove_ads INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_remove_promos INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_remove_intros INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_remove_outros INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_ai_rewrite_description INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_ai_audio_summary INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_append_title_intro INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN default_retention_limit INTEGER DEFAULT 1",
        "ALTER TABLE app_settings ADD COLUMN default_retention_days INTEGER DEFAULT 30",
        "ALTER TABLE app_settings ADD COLUMN default_manual_retention_days INTEGER DEFAULT 14",
        "ALTER TABLE app_settings ADD COLUMN default_custom_instructions TEXT",
        "ALTER TABLE episodes ADD COLUMN listen_count INTEGER DEFAULT 0",
        "ALTER TABLE app_settings ADD COLUMN gemini_api_keys TEXT",

        # Feed tokens: replace the password-derived ?auth= feed credential.
        # feed_auth_password is intentionally NOT dropped here - removing it
        # is a separate change.
        "ALTER TABLE users ADD COLUMN feed_token TEXT",
        "ALTER TABLE app_settings ADD COLUMN feed_auth_token TEXT",

        # Podcast Index (discovery) credentials. Operator-editable from
        # Admin - System; `.env` (PODCAST_INDEX_API_KEY / _SECRET) seeds an
        # install and is the fallback when these are empty, mirroring how the
        # AI provider keys resolve.
        "ALTER TABLE app_settings ADD COLUMN podcast_index_api_key TEXT",
        "ALTER TABLE app_settings ADD COLUMN podcast_index_api_secret TEXT"
    ]

    for sql in migrations:
        try:
            cursor.execute(sql)
        except sqlite3.OperationalError:
            pass # Column likely exists

    # Purge the plaintext admin password left behind by older installs.
    #
    # `app_settings.initial_password` stored the generated admin password in
    # clear text, and the /login page - which is exempt from auth - rendered
    # it. Both code paths are gone, but removing the code does not remove the
    # data: an install upgraded from before the fix still holds a live
    # password in that column. This migration is the only code guaranteed to
    # run on upgrade, so the destruction happens here.
    #
    # Two steps, in this order, because they can fail independently:
    #   1. Overwrite the value unconditionally. This is the step that actually
    #      destroys the secret and it must not depend on the SQLite version.
    #   2. Drop the column, so the secret cannot be re-populated by anything
    #      that still knows the name. DROP COLUMN needs SQLite >= 3.35, and it
    #      also refuses when the column is referenced by an index, view, or
    #      trigger - so it is attempted, not assumed. If it does not happen the
    #      column survives cleared, which is the safe outcome; startup must
    #      never crash over cosmetics.
    #
    # Both installs therefore exist in the wild - column present-but-empty and
    # column absent - and every read here is guarded by a table_info check
    # rather than by an assumption about which one this is.
    settings_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(app_settings)")
    }
    if "initial_password" in settings_columns:
        # Unconditional: no WHERE, no "only if non-empty". Cheap, and it cannot
        # be fooled by a value this code failed to anticipate.
        cursor.execute("UPDATE app_settings SET initial_password = NULL")
        if sqlite3.sqlite_version_info >= (3, 35, 0):
            try:
                cursor.execute(
                    "ALTER TABLE app_settings DROP COLUMN initial_password"
                )
            except sqlite3.OperationalError:
                # Older/odd SQLite builds, or the column is referenced by
                # something. Already cleared above; leave it.
                pass

    # Backfill: every existing user must end up with a working feed token,
    # otherwise their feeds break the moment password-based auth is removed.
    # Generated one row at a time so each user gets a distinct token.
    cursor.execute(
        "SELECT id FROM users WHERE feed_token IS NULL OR feed_token = ''"
    )
    for (user_id,) in cursor.fetchall():
        cursor.execute(
            "UPDATE users SET feed_token = ? WHERE id = ?",
            (generate_feed_token(), user_id),
        )

    # A feed token is a bearer credential; two users must never share one, and
    # a duplicate would resolve to the wrong account. Partial so the many rows
    # that legitimately hold NULL/'' (users predating the backfill, mid-upgrade)
    # do not collide with each other.
    try:
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_feed_token_unique "
            "ON users(feed_token) "
            "WHERE feed_token IS NOT NULL AND feed_token != ''"
        )
    except sqlite3.OperationalError:
        # Pre-existing duplicates would block the index. Leave the data alone
        # and let find_user_by_feed_token's ambiguity check hold the line.
        pass

    conn.commit()
    conn.close()

    # Backfill the standalone/global feed token too.
    #
    # This is the counterpart of the per-user backfill above, and it is here
    # for the same reason: the migration is the only code guaranteed to run on
    # upgrade. Without it a populated standalone install (auth_enabled=0,
    # enable_feed_auth=1) comes up with feed_auth_token still NULL, the
    # middleware correctly fails closed on "enabled but unconfigured", and
    # every feed and audio request 401s indefinitely. It self-healed only when
    # a human opened the dashboard, since that was the sole caller of
    # ensure_global_feed_token - a headless container nobody logs into served
    # nothing, forever, with no error anywhere.
    #
    # Called after the connection above is committed and closed so it does not
    # contend with this function's own write transaction.
    ensure_global_feed_token()

@contextmanager
def get_db_connection():
    """Get a database connection."""
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    # Set a busy timeout to avoid 'database is locked' errors during heavy processing
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Feed tokens
#
# These are bearer credentials for read-only RSS feed access. They are stored
# retrievably (not hashed) on purpose - see the comment on users.feed_token in
# init_db(). Treat them like API keys: never log them.
# ---------------------------------------------------------------------------

def get_discovery_cache(cache_key: str, ttl_seconds: int):
    """Return a cached discovery payload, or None when absent or stale.

    Staleness is computed in SQL against the row's own `fetched_at` so the
    comparison uses the same clock that wrote it. A row that fails to parse as
    JSON is treated as a miss rather than an error: the cache is disposable,
    and a corrupt entry must never take the feature down.
    """
    import json

    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM discovery_cache "
                "WHERE cache_key = ? "
                "AND fetched_at >= datetime('now', ?)",
                (cache_key, f"-{int(ttl_seconds)} seconds"),
            ).fetchone()
    except sqlite3.Error:
        return None

    if not row:
        return None
    try:
        return json.loads(row["payload"])
    except (ValueError, TypeError):
        return None


def set_discovery_cache(cache_key: str, payload) -> None:
    """Store (or refresh) a discovery payload under `cache_key`."""
    import json

    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO discovery_cache (cache_key, payload, fetched_at) "
            "VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(cache_key) DO UPDATE SET "
            "payload = excluded.payload, fetched_at = CURRENT_TIMESTAMP",
            (cache_key, json.dumps(payload)),
        )
        conn.commit()


def clear_discovery_cache() -> None:
    """Drop every cached discovery payload.

    Called when the Podcast Index credentials change: entries fetched with the
    old key would otherwise stay served for up to the full TTL, which reads as
    "the new key did nothing".
    """
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM discovery_cache")
            conn.commit()
    except sqlite3.Error:
        pass


def get_feed_token(user_id: int) -> Optional[str]:
    """Return the user's feed token, or None if they have none."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT feed_token FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    token = row["feed_token"]
    return token if token else None


def ensure_feed_token(user_id: int) -> str:
    """Return the user's feed token, generating and persisting one if absent."""
    existing = get_feed_token(user_id)
    if existing:
        return existing
    token = generate_feed_token()
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET feed_token = ? WHERE id = ?", (token, user_id)
        )
        conn.commit()
    return token


def rotate_feed_token(user_id: int) -> str:
    """Generate a new feed token for the user, persist it, and return it.

    This is the revocation path: the previous token stops resolving
    immediately, and no other credential (including the account password) is
    affected.
    """
    token = generate_feed_token()
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET feed_token = ? WHERE id = ?", (token, user_id)
        )
        conn.commit()
    return token


def find_user_by_feed_token(token: Optional[str]) -> Optional[dict]:
    """Resolve a feed token to a user row, or None.

    Two things this deliberately does NOT do:

    1. It never matches on an empty or missing token. A NULL/empty stored
       column must not be authenticated by a NULL/empty presented token, so
       both sides are rejected up front and empty stored tokens are excluded
       from the candidate set by the SQL.
    2. It never compares tokens with `==` in SQL or in Python. Candidate rows
       are compared with `hmac.compare_digest` so a match cannot be recovered
       byte-by-byte from response timing. The comparison is done on the UTF-8
       encodings: `compare_digest` raises TypeError on str arguments holding
       non-ASCII characters, which let a junk credential turn an
       unauthenticated request into a 500 instead of a 401.

    The select is deliberately narrow. It used to be `SELECT *`, which handed
    every caller the row's `password_hash` for no reason; callers need only the
    identity fields.
    """
    if not token or not isinstance(token, str):
        return None

    try:
        presented = token.encode('utf-8')
    except UnicodeEncodeError:
        return None

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id, username, feed_token FROM users "
            "WHERE feed_token IS NOT NULL AND feed_token != ''"
        ).fetchall()

    match = None
    matches = 0
    for row in rows:
        # Compare every candidate; do not short-circuit on the first hit.
        stored = str(row["feed_token"]).encode('utf-8', 'surrogatepass')
        if hmac.compare_digest(stored, presented):
            match = row
            matches += 1

    if matches != 1:
        # Zero matches is the ordinary rejection. More than one means two
        # users share a token, which a unique index should now prevent; if it
        # ever happens, "last row wins" would authenticate the wrong account.
        # Refuse instead of guessing.
        return None
    return dict(match)


def get_global_feed_token() -> Optional[str]:
    """Return the standalone/global feed token, or None if unset."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT feed_auth_token FROM app_settings WHERE id = 1"
        ).fetchone()
    if not row:
        return None
    token = row["feed_auth_token"]
    return token if token else None


def ensure_global_feed_token() -> str:
    """Return the global feed token, generating and persisting one if absent."""
    existing = get_global_feed_token()
    if existing:
        return existing
    token = generate_feed_token()
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            "UPDATE app_settings SET feed_auth_token = ? WHERE id = 1", (token,)
        )
        conn.commit()
    return token

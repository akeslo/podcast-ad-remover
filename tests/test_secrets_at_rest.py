"""Secrets at rest: session key resolution, API key encryption + migration,
no secret echoed into admin HTML, no password in the session cookie."""
import base64
import json
import os
import sqlite3

import pytest

from app.core import secrets_store
from app.core.config import settings as app_settings
from app.core.secrets_store import (
    ENC_PREFIX,
    SAVED_SENTINEL,
    decrypt_secret,
    encrypt_secret,
    resolve_session_secret,
)
from app.infra.database import get_db_connection, init_db

SAME_ORIGIN = {"Origin": "http://testserver"}
OPENAI_KEY = "sk-test-openai-PLAINTEXT-1234567890"
GEMINI_KEYS = ["AIza-gemini-one-PLAINTEXT", "AIza-gemini-two-PLAINTEXT"]
PI_SECRET = "podcast-index-secret-PLAINTEXT"


def _raw_settings():
    conn = sqlite3.connect(app_settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
    conn.close()
    return dict(row)


def _plant_plaintext():
    conn = sqlite3.connect(app_settings.DB_PATH)
    conn.execute(
        "UPDATE app_settings SET openai_api_key = ?, gemini_api_keys = ?, "
        "podcast_index_api_secret = ? WHERE id = 1",
        (OPENAI_KEY, json.dumps(GEMINI_KEYS), PI_SECRET),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def clean_keys():
    yield
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET openai_api_key = NULL, gemini_api_keys = NULL, "
            "podcast_index_api_key = NULL, podcast_index_api_secret = NULL WHERE id = 1"
        )
        conn.commit()


# --- session key ------------------------------------------------------------

def test_session_secret_prefers_env():
    assert resolve_session_secret("from-env") == "from-env"


def test_session_secret_reads_file(tmp_path, monkeypatch):
    f = tmp_path / "key"
    f.write_text("from-file\n")
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    monkeypatch.setenv("SESSION_SECRET_KEY_FILE", str(f))
    assert resolve_session_secret("") == "from-file"


def test_session_secret_generated_random_and_persisted(tmp_path, monkeypatch):
    monkeypatch.delenv("SESSION_SECRET_KEY", raising=False)
    monkeypatch.delenv("SESSION_SECRET_KEY_FILE", raising=False)
    monkeypatch.setattr(app_settings, "DATA_DIR", str(tmp_path))
    first = resolve_session_secret("")
    assert len(first) >= 64
    assert resolve_session_secret("") == first  # persisted, stable
    path = tmp_path / "secrets" / "session_secret_key"
    assert path.read_text() == first
    assert oct(path.stat().st_mode & 0o777) == "0o600"

    other = tmp_path / "other"
    monkeypatch.setattr(app_settings, "DATA_DIR", str(other))
    assert resolve_session_secret("") != first  # random, not a constant


def test_no_session_default_in_source():
    root = os.path.join(os.path.dirname(__file__), "..", "app")
    for dirpath, _, files in os.walk(root):
        for name in files:
            if name.endswith(".py"):
                with open(os.path.join(dirpath, name), encoding="utf-8") as f:
                    assert "change-me" not in f.read(), name


# --- encryption ---------------------------------------------------------------

def test_encrypt_roundtrip_and_passthrough():
    enc = encrypt_secret("abc")
    assert enc.startswith(ENC_PREFIX) and "abc" not in enc
    assert decrypt_secret(enc) == "abc"
    assert decrypt_secret("legacy-plain") == "legacy-plain"
    assert encrypt_secret(None) is None and encrypt_secret("") == ""
    assert encrypt_secret(enc) == enc


def test_migration_encrypts_existing_plaintext(clean_keys):
    _plant_plaintext()
    init_db()
    raw = _raw_settings()
    assert raw["openai_api_key"].startswith(ENC_PREFIX)
    assert raw["podcast_index_api_secret"].startswith(ENC_PREFIX)
    for k in json.loads(raw["gemini_api_keys"]):
        assert k.startswith(ENC_PREFIX)
    blob = json.dumps(raw)
    for secret in (OPENAI_KEY, PI_SECRET, *GEMINI_KEYS):
        assert secret not in blob

    init_db()  # idempotent, no double encryption
    from app.core.utils import get_global_settings
    s = get_global_settings()
    assert s["openai_api_key"] == OPENAI_KEY
    assert json.loads(s["gemini_api_keys"]) == GEMINI_KEYS
    from app.core.discovery import get_credentials
    assert get_credentials()[1] == PI_SECRET


def test_feed_auth_password_column_purged():
    conn = sqlite3.connect(app_settings.DB_PATH)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(app_settings)")}
    if "feed_auth_password" not in cols:
        conn.execute("ALTER TABLE app_settings ADD COLUMN feed_auth_password TEXT")
    conn.execute("UPDATE app_settings SET feed_auth_password = 'Feed-Pl4in-Pw' WHERE id = 1")
    conn.commit()
    conn.close()
    init_db()
    conn = sqlite3.connect(app_settings.DB_PATH)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(app_settings)")}
    conn.close()
    if sqlite3.sqlite_version_info >= (3, 35, 0):
        assert "feed_auth_password" not in cols
    else:
        assert _raw_settings().get("feed_auth_password") is None


# --- admin UI -----------------------------------------------------------------

def test_admin_pages_never_render_saved_keys(client, clean_keys):
    _plant_plaintext()
    init_db()
    ai = client.get("/admin/ai").text
    system = client.get("/admin/system").text
    for secret in (OPENAI_KEY, PI_SECRET, *GEMINI_KEYS):
        assert secret not in ai and secret not in system
    assert SAVED_SENTINEL in ai and SAVED_SENTINEL in system


def test_ai_save_with_sentinel_keeps_and_encrypts(client, clean_keys):
    _plant_plaintext()
    init_db()
    r = client.post(
        "/admin/ai/update",
        data={
            "ai_model_cascade": '["gemini-2.5-flash"]',
            "openai_api_key": SAVED_SENTINEL,
            "anthropic_api_key": "sk-ant-new-key",
            "openrouter_api_key": "",
            "gemini_api_keys": json.dumps([f"{SAVED_SENTINEL}:1", "AIza-new"]),
        },
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )
    assert r.status_code == 303
    raw = _raw_settings()
    assert "sk-ant-new-key" not in json.dumps(raw)
    assert raw["openrouter_api_key"] is None
    from app.core.utils import get_global_settings
    s = get_global_settings()
    assert s["openai_api_key"] == OPENAI_KEY
    assert s["anthropic_api_key"] == "sk-ant-new-key"
    assert json.loads(s["gemini_api_keys"]) == [GEMINI_KEYS[1], "AIza-new"]


def test_system_save_with_sentinel_keeps_podcast_index_secret(client, clean_keys):
    _plant_plaintext()
    init_db()
    client.post(
        "/admin/system/update",
        data={
            "concurrent_downloads": 2,
            "retention_days": 30,
            "check_interval_minutes": 60,
            "podcast_index_api_key": "pi-key",
            "podcast_index_api_secret": SAVED_SENTINEL,
        },
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )
    raw = _raw_settings()
    assert raw["podcast_index_api_key"].startswith(ENC_PREFIX)
    from app.core.discovery import get_credentials
    assert get_credentials() == ("pi-key", PI_SECRET)


# --- session cookie -----------------------------------------------------------

def test_session_cookie_holds_user_id_not_password(client):
    from app.web.auth_utils import hash_password

    password = "C00kie-Pl4in-Passw0rd!"
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = 'cookieuser'")
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("cookieuser", hash_password(password)),
        )
        conn.commit()
    try:
        client.post(
            "/login",
            data={"username": "cookieuser", "password": password},
            headers=SAME_ORIGIN,
            follow_redirects=False,
        )
        cookie = client.cookies.get("session")
        assert cookie, "login did not set a session cookie"
        payload = base64.b64decode(cookie.split(".")[0] + "==").decode()
        data = json.loads(payload)
        assert password not in payload
        assert not any("password" in k for k in data)
    finally:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM users WHERE username = 'cookieuser'")
            conn.commit()

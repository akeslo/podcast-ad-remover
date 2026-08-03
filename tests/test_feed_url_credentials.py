"""Web-layer tests for the feed-token change.

Three negative properties are under test - the account password must not be
retained in the session, must not be handed out in a redirect URL, and must
not appear in any generated feed URL. Because they are negative properties,
the assertions search whole strings for the password value rather than
checking one named key: a refactor that moves the password into a
differently-named field should still fail these.
"""
import base64
import json

import pytest

from app.infra.database import (
    ensure_feed_token,
    get_db_connection,
    rotate_feed_token,
)
from app.web.auth_utils import hash_password


TEST_PASSWORD = "S3cret-Passw0rd-Do-Not-Leak"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _set_settings(**columns):
    """Set columns on the single app_settings row."""
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


def _create_user(username, password, is_admin=1):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, hash_password(password), is_admin),
        )
        conn.commit()
        return cursor.lastrowid


def _decode_session_cookie(client):
    """Return the decoded Starlette session dict from the client's cookie jar.

    SessionMiddleware stores base64(json) with an itsdangerous signature after
    a '.'; the signature is irrelevant here because we only care about what
    the server chose to put inside.
    """
    raw = client.cookies.get("session")
    if not raw:
        return {}
    payload = raw.split(".")[0]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload))


def _decode_auth_param(url):
    assert "auth=" in url, url
    encoded = url.split("auth=", 1)[1].split("&", 1)[0]
    return base64.b64decode(encoded).decode("utf-8")


class _FakeUser:
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username


class _FakeSub:
    slug = "some-show"


@pytest.fixture(autouse=True)
def _reset_settings():
    """Leave app_settings in a known-disabled state around every test."""
    _set_settings(auth_enabled=0, enable_feed_auth=0, feed_auth_username=None)
    yield
    _set_settings(auth_enabled=0, enable_feed_auth=0, feed_auth_username=None)


# --------------------------------------------------------------------------
# 1. the session must never carry the account password
# --------------------------------------------------------------------------

def test_login_does_not_store_password_in_session(client):
    user_id = _create_user("session-probe", TEST_PASSWORD)
    _set_settings(auth_enabled=1)

    response = client.post(
        "/login",
        data={"username": "session-probe", "password": TEST_PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code in (302, 303), response.text

    session = _decode_session_cookie(client)
    assert "user_pass" not in session
    assert TEST_PASSWORD not in json.dumps(session)
    # Sanity check that we are actually inspecting a logged-in session.
    assert user_id in [v for v in session.values() if isinstance(v, int)]


def test_router_source_has_no_user_pass_session_key():
    """Regression guard: the key must not come back under any code path."""
    from pathlib import Path

    import app.web.router as router_module

    source = Path(router_module.__file__).read_text(encoding="utf-8")
    assert '"user_pass"' not in source
    assert "'user_pass'" not in source


# --------------------------------------------------------------------------
# 2. the approval redirect must not carry a credential
# --------------------------------------------------------------------------

def test_approval_redirect_contains_no_credential(client):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", ("approved-probe",))
        cursor = conn.execute(
            "INSERT INTO access_requests (username, status) VALUES (?, 'pending')",
            ("approved-probe",),
        )
        request_id = cursor.lastrowid
        conn.commit()

    response = client.post(
        f"/admin/access-requests/{request_id}/approve", follow_redirects=False
    )
    assert response.status_code == 303

    location = response.headers["location"]
    assert location == "/admin/access"
    assert "password" not in location.lower()
    assert "?" not in location  # no query string at all

    # The password went into the session flash instead.
    session = _decode_session_cookie(client)
    temp_password = session.get("_flash", {}).get("approved_temp_password")
    assert temp_password, "temporary password was not handed over via flash"

    # It renders once...
    page = client.get("/admin/access")
    assert temp_password in page.text

    # ...and is cleared, so it does not linger in the session.
    assert "_flash" not in _decode_session_cookie(client)
    second_page = client.get("/admin/access")
    assert temp_password not in second_page.text


# --------------------------------------------------------------------------
# 3. generated feed URLs carry the token, never the password
# --------------------------------------------------------------------------

def test_feed_credential_is_the_token_not_the_password():
    from app.web.router import build_feed_auth_token

    user_id = _create_user("feed-probe", TEST_PASSWORD, is_admin=0)
    token = ensure_feed_token(user_id)

    decoded = base64.b64decode(
        build_feed_auth_token(
            {"enable_feed_auth": 1, "auth_enabled": 1},
            _FakeUser(user_id, "feed-probe"),
        )
    ).decode("utf-8")

    assert decoded == f"feed-probe:{token}"
    assert TEST_PASSWORD not in decoded


def test_generate_rss_links_embeds_token(client):
    from app.web.router import generate_rss_links

    user_id = _create_user("links-probe", TEST_PASSWORD, is_admin=0)
    token = ensure_feed_token(user_id)

    global_settings = {
        "enable_feed_auth": 1,
        "auth_enabled": 1,
        "app_external_url": "https://example.test",
    }

    request = client.build_request("GET", "/")
    links = generate_rss_links(
        request, _FakeSub(), global_settings, _FakeUser(user_id, "links-probe")
    )

    assert token in _decode_auth_param(links["rss"])
    assert TEST_PASSWORD not in links["rss"]
    assert TEST_PASSWORD not in _decode_auth_param(links["rss"])


def test_feed_url_is_plain_when_feed_auth_disabled(client):
    from app.web.router import generate_rss_links

    user_id = _create_user("plain-probe", TEST_PASSWORD, is_admin=0)
    request = client.build_request("GET", "/")
    links = generate_rss_links(
        request,
        _FakeSub(),
        {"enable_feed_auth": 0, "auth_enabled": 1, "app_external_url": "https://example.test"},
        _FakeUser(user_id, "plain-probe"),
    )
    assert "auth=" not in links["rss"]


def test_feed_url_fails_loudly_without_a_user():
    """No silent fallback to a password, and no bare 'admin' guess."""
    from app.web.router import build_feed_auth_token

    with pytest.raises(RuntimeError):
        build_feed_auth_token({"enable_feed_auth": 1, "auth_enabled": 1}, None)


def test_standalone_feed_url_uses_the_global_token():
    from app.infra.database import ensure_global_feed_token
    from app.web.router import build_feed_auth_token

    _set_settings(feed_auth_username="feeduser")
    global_token = ensure_global_feed_token()

    decoded = base64.b64decode(
        build_feed_auth_token({"enable_feed_auth": 1, "auth_enabled": 0})
    ).decode("utf-8")

    assert decoded == f"feeduser:{global_token}"


def test_settings_save_never_stores_a_feed_password(client):
    """The submitted feed password is ignored, not persisted."""
    _set_settings(feed_auth_password=None)

    client.post(
        "/admin/system/update",
        data={
            "concurrent_downloads": 2,
            "retention_days": 30,
            "check_interval_minutes": 60,
            "enable_feed_auth": "on",
            "feed_auth_username": "feeduser",
            "feed_auth_password": TEST_PASSWORD,
        },
        follow_redirects=False,
    )

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT feed_auth_password FROM app_settings WHERE id = 1"
        ).fetchone()
    assert row["feed_auth_password"] != TEST_PASSWORD


# --------------------------------------------------------------------------
# 4. rotation actually revokes
# --------------------------------------------------------------------------

def test_rotation_changes_the_feed_url_credential():
    from app.web.router import build_feed_auth_token

    user_id = _create_user("rotate-probe", TEST_PASSWORD, is_admin=0)
    before = ensure_feed_token(user_id)
    after = rotate_feed_token(user_id)
    assert before != after

    decoded = base64.b64decode(
        build_feed_auth_token(
            {"enable_feed_auth": 1, "auth_enabled": 1},
            _FakeUser(user_id, "rotate-probe"),
        )
    ).decode("utf-8")
    assert after in decoded
    assert before not in decoded


def test_rotate_route_rotates_the_global_token(client):
    from app.infra.database import ensure_global_feed_token, get_global_feed_token

    _set_settings(enable_feed_auth=1, feed_auth_username="feeduser")
    before = ensure_global_feed_token()

    response = client.post("/feed-token/rotate", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert get_global_feed_token() != before

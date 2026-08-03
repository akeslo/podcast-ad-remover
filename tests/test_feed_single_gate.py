"""One gate for feed authorization, and a credential the audio URLs can reuse.

`/feed/unified` used to authorise itself, on top of `feed_auth_middleware`
already authorising it. The two disagreed:

* the middleware treats the credential as an opaque token and ignores any
  username half, while the route additionally required
  `username == feed_auth_username`;
* the middleware accepts a bare token, which the route could not parse at all.

So a bare token passed the gate and was then 401'd by the route it had just
passed. These tests run through the real ASGI stack and pin the behaviour to
a single gate.
"""
import base64
import os

import pytest

from app.core.config import settings as app_settings
from app.infra.database import get_db_connection


UNIFIED_PATHS = ("/feed/unified", "/feed/unified.xml")


def _set_settings(**columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


@pytest.fixture
def unified_feed():
    """Standalone feed auth, with a real unified.xml on disk."""
    from app.infra.database import ensure_global_feed_token

    _set_settings(auth_enabled=0, enable_feed_auth=1, feed_auth_username="feeduser")
    token = ensure_global_feed_token()

    os.makedirs(app_settings.FEEDS_DIR, exist_ok=True)
    path = os.path.join(app_settings.FEEDS_DIR, "unified.xml")
    existing = None
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            existing = handle.read()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            '<?xml version="1.0"?><rss><channel><item>'
            '<enclosure url="https://example.test/audio/ep1.mp3" />'
            "</item></channel></rss>"
        )

    yield token

    if existing is None:
        os.remove(path)
    else:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(existing)
    _set_settings(auth_enabled=0, enable_feed_auth=0, feed_auth_username=None)


@pytest.mark.parametrize("path", UNIFIED_PATHS)
def test_bare_token_is_accepted_end_to_end(unified_feed, client, path):
    """The regression: this passed the middleware and was 401'd by the route."""
    response = client.get(path, params={"auth": unified_feed})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", UNIFIED_PATHS)
def test_envelope_token_is_still_accepted(unified_feed, client, path):
    """Generated URLs carry base64("user:token"); they must keep working."""
    envelope = base64.b64encode(f"feeduser:{unified_feed}".encode()).decode()
    response = client.get(path, params={"auth": envelope})
    assert response.status_code == 200, response.text


@pytest.mark.parametrize("path", UNIFIED_PATHS)
def test_wrong_token_is_still_rejected(unified_feed, client, path):
    """Deleting the route's gate must not have opened the feed."""
    response = client.get(path, params={"auth": "not-the-token"})
    assert response.status_code == 401


@pytest.mark.parametrize("path", UNIFIED_PATHS)
def test_no_credential_is_still_rejected(unified_feed, client, path):
    response = client.get(path)
    assert response.status_code == 401


def test_username_half_no_longer_decides_authorization(unified_feed, client):
    """The middleware ignores the username; the route must not reinstate it.

    Previously the route demanded username == feed_auth_username, so this
    request 401'd even though the token was correct.
    """
    envelope = base64.b64encode(f"anyone:{unified_feed}".encode()).decode()
    response = client.get("/feed/unified", params={"auth": envelope})
    assert response.status_code == 200, response.text


def test_audio_urls_carry_the_callers_own_credential(unified_feed, client):
    """A bare token must produce audio URLs that are themselves authorised.

    The old code re-encoded base64(username:token) from the presented
    credential, so a bare token produced enclosure URLs with no auth at all
    and every download 401'd.
    """
    response = client.get("/feed/unified", params={"auth": unified_feed})
    assert response.status_code == 200
    assert f"auth={unified_feed}" in response.text


def test_individual_feed_audio_urls_carry_the_callers_own_credential(
    unified_feed, client
):
    slug = "single-gate-probe"
    with get_db_connection() as conn:
        conn.execute("DELETE FROM subscriptions WHERE slug = ?", (slug,))
        conn.execute(
            "INSERT INTO subscriptions (title, feed_url, slug) VALUES (?, ?, ?)",
            ("Single Gate Probe", "https://example.test/p.xml", slug),
        )
        conn.commit()

    path = os.path.join(app_settings.FEEDS_DIR, f"{slug}.xml")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            '<?xml version="1.0"?><rss><channel><item>'
            '<enclosure url="https://example.test/audio/ep1.mp3" />'
            "</item></channel></rss>"
        )
    try:
        response = client.get(f"/feeds/{slug}.xml", params={"auth": unified_feed})
        assert response.status_code == 200, response.text
        assert f"auth={unified_feed}" in response.text
    finally:
        os.remove(path)
        with get_db_connection() as conn:
            conn.execute("DELETE FROM subscriptions WHERE slug = ?", (slug,))
            conn.commit()

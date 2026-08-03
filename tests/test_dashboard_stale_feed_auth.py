"""The dashboard must not 500 on a stale-but-real feed-auth config.

`build_feed_auth_token` raises `RuntimeError` in standalone mode when
`enable_feed_auth = 1` but `feed_auth_username` is NULL. That combination is
not reachable from any current save path, but it is reachable from history:
the old code tolerated a missing username with an `or 'feed'` fallback, so a
database written before that fallback was removed can genuinely hold it. On
such an install the dashboard - the app's front page - raised straight out of
`_render_index`.

The fallback is deliberately NOT restored. `'feed'` produced a URL the
unified-feed validator rejects with 401, i.e. a silently broken feed. Instead
the dashboard renders, the feed links are emitted without the (unbuildable)
credential, and an inline warning names the setting to fix.
"""
import pytest

from app.infra.database import get_db_connection


FEED_URL = "https://example.invalid/stale-feed-auth.xml"


@pytest.fixture
def stale_feed_auth(client):
    """Standalone install with feed auth on and no feed_auth_username."""
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        before = conn.execute(
            "SELECT auth_enabled, enable_feed_auth, feed_auth_username"
            " FROM app_settings WHERE id = 1"
        ).fetchone()
        conn.execute(
            "UPDATE app_settings SET auth_enabled = 0, enable_feed_auth = 1,"
            " feed_auth_username = NULL WHERE id = 1"
        )
        cur = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            (FEED_URL, "Stale Show", "stale-show"),
        )
        sub_id = cur.lastrowid
        conn.commit()

    yield client

    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET auth_enabled = ?, enable_feed_auth = ?,"
            " feed_auth_username = ? WHERE id = 1",
            (before["auth_enabled"], before["enable_feed_auth"],
             before["feed_auth_username"]),
        )
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
        conn.commit()


def test_dashboard_renders_instead_of_crashing(stale_feed_auth):
    response = stale_feed_auth.get("/", follow_redirects=False)

    assert response.status_code == 200, response.status_code


def test_dashboard_warns_and_names_the_setting(stale_feed_auth):
    """A 500 is the wrong way to say "your config is stale"."""
    body = stale_feed_auth.get("/", follow_redirects=False).text

    assert "Feed Auth Username" in body, "the warning does not name the setting"
    assert "/admin/system" in body, "the warning does not link to the fix"


def test_no_broken_credential_is_smuggled_into_the_feed_urls(stale_feed_auth):
    """The removed `or 'feed'` fallback must not come back in any form.

    It built base64('feed:<token>'), a URL the app itself answers with 401.
    Emitting no ?auth= at all is the honest outcome; the inline warning is
    what tells the operator the links are not usable yet.
    """
    import base64

    body = stale_feed_auth.get("/", follow_redirects=False).text

    assert "?auth=" not in body and "&auth=" not in body
    assert base64.b64encode(b"feed:").decode()[:6] not in body


def test_admin_system_still_renders_so_the_state_is_fixable(stale_feed_auth):
    response = stale_feed_auth.get("/admin/system", follow_redirects=False)

    assert response.status_code == 200, response.status_code


def test_a_healthy_standalone_install_still_gets_authenticated_links(client):
    """The warning path must not swallow the normal case."""
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        before = conn.execute(
            "SELECT auth_enabled, enable_feed_auth, feed_auth_username"
            " FROM app_settings WHERE id = 1"
        ).fetchone()
        conn.execute(
            "UPDATE app_settings SET auth_enabled = 0, enable_feed_auth = 1,"
            " feed_auth_username = 'feeduser' WHERE id = 1"
        )
        cur = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            (FEED_URL + "?healthy", "Healthy Show", "healthy-show"),
        )
        sub_id = cur.lastrowid
        conn.commit()
    try:
        body = client.get("/", follow_redirects=False).text
        assert "auth=" in body, "feed links lost their credential"
        assert "Feed Auth Username" not in body, "spurious stale-config warning"
    finally:
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE app_settings SET auth_enabled = ?, enable_feed_auth = ?,"
                " feed_auth_username = ? WHERE id = 1",
                (before["auth_enabled"], before["enable_feed_auth"],
                 before["feed_auth_username"]),
            )
            conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
            conn.commit()

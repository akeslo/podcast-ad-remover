"""The one-time "your feed URLs changed" banner.

Old feed URLs embedded the account password; new ones embed a feed token and
the feed routes have no password fallback. Existing subscribers' URLs
therefore start returning 401 on upgrade - and podcast clients do not surface
a 401 on a background refresh, they just stop downloading. Without a banner
the user's podcasts stop on a random day with no signal at all.
"""
import os

import pytest

from app.infra.database import get_db_connection
from app.web import router as router_module


SAME_ORIGIN = {"Origin": "http://testserver"}


def _set_settings(**columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


@pytest.fixture(autouse=True)
def _clean_marker():
    def _remove():
        path = router_module._feed_url_notice_marker_path()
        if os.path.exists(path):
            os.remove(path)

    _remove()
    _set_settings(auth_enabled=0, enable_feed_auth=1, feed_auth_username="feeduser")
    yield
    _remove()
    _set_settings(auth_enabled=0, enable_feed_auth=0, feed_auth_username=None)


FEED_AUTH_ON = {"enable_feed_auth": 1, "auth_enabled": 0}


def test_upgraded_install_is_told_its_feed_urls_changed():
    """Subscriptions already exist, so shared URLs are already broken."""
    assert router_module.feed_url_notice_pending(FEED_AUTH_ON, subscription_count=3)


def test_fresh_install_is_never_nagged():
    """No subscriptions means no already-shared URL to invalidate."""
    assert not router_module.feed_url_notice_pending(FEED_AUTH_ON, subscription_count=0)
    # ...and it is acknowledged permanently, so adding a podcast later does
    # not resurrect a notice about a change this install never lived through.
    assert not router_module.feed_url_notice_pending(FEED_AUTH_ON, subscription_count=5)


def test_no_notice_when_feed_auth_is_off():
    """Without feed auth the URLs never carried a credential."""
    assert not router_module.feed_url_notice_pending(
        {"enable_feed_auth": 0, "auth_enabled": 0}, subscription_count=3
    )


def test_dismissal_persists():
    assert router_module.feed_url_notice_pending(FEED_AUTH_ON, subscription_count=3)
    router_module.acknowledge_feed_url_notice()
    assert not router_module.feed_url_notice_pending(FEED_AUTH_ON, subscription_count=3)


def test_dismiss_route_is_same_origin_guarded(client):
    """It is a state change; an anonymous script must not be able to fire it."""
    response = client.post("/feed-url-notice/dismiss", follow_redirects=False)
    assert response.status_code == 403
    assert not os.path.exists(router_module._feed_url_notice_marker_path())


def test_dismiss_route_works_for_the_operator(client):
    response = client.post(
        "/feed-url-notice/dismiss", headers=SAME_ORIGIN, follow_redirects=False
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/"
    assert os.path.exists(router_module._feed_url_notice_marker_path())


def test_banner_renders_on_the_dashboard_of_an_upgraded_install(client):
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (title, feed_url, slug) VALUES (?, ?, ?)",
            ("Notice Probe", "https://example.test/probe.xml", "notice-probe"),
        )
        conn.commit()
    try:
        page = client.get("/")
        assert page.status_code == 200
        assert "your feed URLs have changed" in page.text
        assert "/feed-url-notice/dismiss" in page.text

        client.post(
            "/feed-url-notice/dismiss", headers=SAME_ORIGIN, follow_redirects=False
        )

        page = client.get("/")
        assert "your feed URLs have changed" not in page.text
    finally:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM subscriptions WHERE slug = ?", ("notice-probe",))
            conn.commit()

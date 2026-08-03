"""Authorization tests for the administrative surface.

Every assertion here is driven through the real ASGI stack (TestClient over
the actual `app.main:app`, so `auth_middleware`, `feed_auth_middleware` and
the route dependencies all run). A unit call to `require_admin` would prove
nothing: the whole class of bug under test is that in standalone mode
(`auth_enabled = 0`) `require_admin` *deliberately* returns a dummy admin and
`auth_middleware` skips its `/admin` privilege check, so the only thing
standing between an anonymous POST and an install-wide state change is what
the route itself declares.

An "anonymous request" here means what an attacker's script actually sends:
no session cookie and no `Origin`/`Referer` header. A browser posting the
app's own form always sends `Origin`.
"""
import re

import pytest

from app.infra.database import get_db_connection


SAME_ORIGIN = {"Origin": "http://testserver"}
CROSS_ORIGIN = {"Origin": "http://attacker.example"}


def _set_settings(**columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


def _get_setting(name):
    with get_db_connection() as conn:
        return conn.execute(
            f"SELECT {name} FROM app_settings WHERE id = 1"
        ).fetchone()[name]


@pytest.fixture(autouse=True)
def _standalone_mode():
    """Standalone mode is the vulnerable configuration - test in it."""
    _set_settings(auth_enabled=0, enable_feed_auth=1, feed_auth_username="feeduser")
    yield
    _set_settings(auth_enabled=0, enable_feed_auth=0, feed_auth_username=None)


# --------------------------------------------------------------------------
# FIX 1 - unauthenticated feed-token rotation
# --------------------------------------------------------------------------

def test_anonymous_post_cannot_rotate_the_feed_token(client):
    """The denial-of-service that killed every subscription, permanently.

    Rotation revokes every feed URL in existence and there is no self-healing
    path, so an attacker looping this endpoint permanently breaks the install.
    """
    from app.infra.database import ensure_global_feed_token, get_global_feed_token

    before = ensure_global_feed_token()

    response = client.post("/feed-token/rotate", follow_redirects=False)

    assert response.status_code == 403, response.text
    assert get_global_feed_token() == before, "the feed token was rotated anyway"


def test_cross_site_post_cannot_rotate_the_feed_token(client):
    from app.infra.database import ensure_global_feed_token, get_global_feed_token

    before = ensure_global_feed_token()

    response = client.post(
        "/feed-token/rotate", headers=CROSS_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 403, response.text
    assert get_global_feed_token() == before


def test_operator_can_still_rotate_the_feed_token(client):
    """The gate must not break the legitimate path."""
    from app.infra.database import ensure_global_feed_token, get_global_feed_token

    before = ensure_global_feed_token()

    response = client.post(
        "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 303
    assert get_global_feed_token() != before


# --------------------------------------------------------------------------
# FIX 2 - unauthenticated settings write (the gate's own off-switch)
# --------------------------------------------------------------------------

def test_anonymous_post_cannot_disable_feed_auth(client):
    """The exploit that nullified the feed-token work entirely.

    With feed auth switched off, every feed and every audio file serves to
    anyone. That switch must not sit on the public side of the door.
    """
    assert _get_setting("enable_feed_auth") == 1

    response = client.post(
        "/admin/system/update",
        data={
            "concurrent_downloads": 2,
            "retention_days": 30,
            "check_interval_minutes": 60,
            # enable_feed_auth omitted == submitted false
        },
        follow_redirects=False,
    )

    assert response.status_code == 403, response.text
    assert _get_setting("enable_feed_auth") == 1, "feed auth was disabled anyway"


def test_operator_can_still_save_system_settings(client):
    response = client.post(
        "/admin/system/update",
        data={
            "concurrent_downloads": 4,
            "retention_days": 30,
            "check_interval_minutes": 60,
            "enable_feed_auth": "on",
            "feed_auth_username": "feeduser",
        },
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert _get_setting("concurrent_downloads") == 4


# --------------------------------------------------------------------------
# FIX 2 (audit) - no state-changing /admin route may be left unguarded
# --------------------------------------------------------------------------

def _admin_write_routes():
    """Every non-safe route under /admin declared by app/web/router.py."""
    from app.web.router import router as web_router

    routes = []
    for route in web_router.routes:
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", set()) or set()
        writes = methods - {"GET", "HEAD", "OPTIONS"}
        if path.startswith("/admin") and writes:
            routes.append((sorted(writes)[0], path))
    return sorted(routes)


def test_the_audit_actually_found_routes():
    """Guard against the sweep below silently passing on an empty list."""
    assert len(_admin_write_routes()) >= 10


@pytest.mark.parametrize("method,path", _admin_write_routes())
def test_no_admin_write_route_accepts_an_anonymous_request(client, method, path):
    """A blanket sweep, so the next route added cannot quietly reopen this."""
    concrete = re.sub(r"\{[^}]+\}", "1", path)

    response = client.request(method, concrete, follow_redirects=False)

    assert response.status_code == 403, (
        f"{method} {concrete} accepted an anonymous request "
        f"({response.status_code})"
    )


# --------------------------------------------------------------------------
# Safe methods must not be broken by the origin check
# --------------------------------------------------------------------------

def test_admin_pages_still_render_for_the_operator(client):
    for path in ("/admin/system", "/admin/ai", "/admin/access"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"

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
# FIX 2 (audit) - no state-changing route may be left unguarded
# --------------------------------------------------------------------------
#
# Widened from "/admin only" to *every* state-changing route on the router.
# The original scope was the reason five ordinary-user write routes (`/add`,
# `/episodes/{id}/download`, `/api/episodes/{id}/reprocess`,
# `/api/episodes/{id}/ignore`, `/subscriptions/{id}/settings`) passed a green
# authorization audit while completely unguarded: they are the same class of
# hole, they just do not live under `/admin`. An audit whose scope is a path
# prefix only ever proves something about that prefix.

#: Routes that must stay reachable by an anonymous, origin-less caller, each
#: for a stated reason. This list is the audit's only escape hatch - adding to
#: it is a deliberate, reviewable act, which is the point.
PUBLIC_WRITE_ROUTES = {
    # The unauthenticated entry point, by definition. It is guarded instead by
    # per-IP rate limiting plus constant-time password verification; requiring
    # an Origin here would break scripted and non-browser logins for no gain,
    # since login CSRF cannot change any existing state.
    ("POST", "/login"),
    # The "let me in" form, shown to people who by construction have no
    # account and no session. Its only effect is to queue a request an admin
    # must then approve.
    ("POST", "/submit-access-request"),
}


def _declared_routes():
    """(method, mounted path) for every route on the *application*.

    Deliberately `app.main.app.routes`, not `app.web.router.routes`. Scoping
    the sweep to one router is the same mistake as scoping it to one path
    prefix, one level up: `app/api/subscriptions.py` declared eight
    state-changing routes - including a DELETE that destroys a subscription,
    all of its episodes and every artifact file on disk - and this audit
    certified the app as safe for as long as it structurally could not see
    that file. Sweeping the mounted app means a router added tomorrow is
    covered the day it is mounted, with no one having to remember this test.

    Paths here are the mounted paths (`/api/...` for the API router), because
    that is what a caller can actually reach.
    """
    from app.main import app as fastapi_app

    def walk(routes, prefix=""):
        for route in routes:
            # FastAPI versions differ in how `include_router` is represented:
            # older ones copy the sub-router's routes onto the app with the
            # prefix already applied, newer ones keep a wrapper that holds the
            # original router and its prefix. Handle both, or this sweep goes
            # quietly back to seeing nothing outside app/web/router.py - which
            # is the exact failure mode it exists to end.
            context = getattr(route, "include_context", None)
            included = getattr(context, "included_router", None)
            if included is not None:
                yield from walk(included.routes, prefix + (context.prefix or ""))
                continue
            path = prefix + getattr(route, "path", "")
            for method in (getattr(route, "methods", set()) or set()):
                yield method, path

    yield from walk(fastapi_app.routes)


def _state_changing_routes():
    """Every non-safe (method, path) on the mounted app, minus the public ones.

    Every write method on a route is returned, not just one: taking
    `sorted(writes)[0]` meant a route declaring both POST and DELETE was only
    ever probed on DELETE, and the other method rode along untested.
    """
    routes = {
        (method, path)
        for method, path in _declared_routes()
        if method not in ("GET", "HEAD", "OPTIONS")
        and (method, path) not in PUBLIC_WRITE_ROUTES
    }
    return sorted(routes)


def test_the_audit_actually_found_routes():
    """Guard against the sweep below silently passing on an empty list."""
    found = _state_changing_routes()
    assert len(found) >= 20
    # The sweep must reach past /admin, or it is the old prefix-scoped audit
    # wearing a new name.
    assert any(not path.startswith("/admin") for _, path in found)
    # ...and past app/web/router.py, or it is the old single-router audit
    # wearing a new name. The /api router is the one that was invisible.
    assert any(path.startswith("/api/") for _, path in found), (
        "the sweep is not seeing the mounted /api router"
    )


def test_every_public_write_route_still_exists():
    """An exemption for a route that no longer exists is a silent hole."""
    declared = set(_declared_routes())
    for entry in PUBLIC_WRITE_ROUTES:
        assert entry in declared, f"stale exemption: {entry}"


@pytest.mark.parametrize("method,path", _state_changing_routes())
def test_no_write_route_accepts_an_anonymous_request(client, method, path):
    """A blanket sweep, so the next route added cannot quietly reopen this."""
    concrete = re.sub(r"\{[^}]+\}", "1", path)

    response = client.request(method, concrete, follow_redirects=False)

    assert response.status_code == 403, (
        f"{method} {concrete} accepted an anonymous request "
        f"({response.status_code})"
    )


# --------------------------------------------------------------------------
# FIX 3 - the /api router, which no audit could previously see
# --------------------------------------------------------------------------
#
# app/api/subscriptions.py is mounted at /api and declared eight state-changing
# routes with no guard at all. Proven against a real database in standalone
# mode: an anonymous, origin-less DELETE /api/subscriptions/1 returned 200 and
# took the subscription, its episodes and its artifact files with it, while the
# guarded twin of the same action (POST /add) correctly returned 403.

API_WRITE_ROUTES = [
    ("POST", "/api/subscriptions"),
    ("DELETE", "/api/subscriptions/1"),
    ("DELETE", "/api/episodes/1"),
    ("POST", "/api/subscriptions/1/check"),
    ("POST", "/api/episodes/1/process"),
    ("POST", "/api/episodes/1/cancel"),
    ("POST", "/api/search"),
    ("POST", "/api/episodes/1/track-listen"),
]


@pytest.mark.parametrize("method,path", API_WRITE_ROUTES)
def test_anonymous_request_to_api_router_is_rejected(client, method, path):
    response = client.request(method, path, follow_redirects=False)
    assert response.status_code == 403, (
        f"{method} {path} accepted an anonymous request ({response.status_code})"
    )


@pytest.mark.parametrize("method,path", API_WRITE_ROUTES)
def test_cross_site_request_to_api_router_is_rejected(client, method, path):
    response = client.request(
        method, path, headers=CROSS_ORIGIN, follow_redirects=False
    )
    assert response.status_code == 403, response.text


ADMIN_PASSWORD = "guard-test-password-123"


def _create_admin_user(username, password):
    from app.web.auth_utils import hash_password

    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
            (username, hash_password(password)),
        )
        conn.commit()


def _seed_subscription_and_episode():
    """One real row of each, so the legitimate-path tests hit real handlers."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("http://example.invalid/guard-test.xml", "Guard Test", "guard-test"),
        )
        sub_id = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
        conn.execute(
            "INSERT INTO episodes (subscription_id, guid, title, original_url,"
            " status) VALUES (?, ?, ?, ?, ?)",
            (
                sub_id,
                "guard-test-guid",
                "Guard Test Episode",
                "http://example.invalid/guard-test.mp3",
                "completed",
            ),
        )
        ep_id = conn.execute("SELECT last_insert_rowid() AS i").fetchone()["i"]
        conn.commit()
    return sub_id, ep_id


@pytest.mark.parametrize("auth_enabled", [0, 1])
def test_same_origin_api_calls_still_work_in_both_auth_modes(client, auth_enabled):
    """The gate must not break ordinary use - the half that actually matters.

    Run in both deployments. With auth_enabled = 0 `require_auth` is a no-op
    and the origin check is the whole boundary; with auth_enabled = 1 it is a
    real one, and the TestClient session fixture is what carries the login.
    """
    _set_settings(auth_enabled=auth_enabled)
    if auth_enabled:
        _create_admin_user("guard-test-admin", ADMIN_PASSWORD)
        login = client.post(
            "/login",
            data={"username": "guard-test-admin", "password": ADMIN_PASSWORD},
            headers=SAME_ORIGIN,
            follow_redirects=False,
        )
        assert login.status_code in (302, 303), login.text
    sub_id, ep_id = _seed_subscription_and_episode()

    ok = {200, 303}

    # Ordinary user actions.
    for method, path in (
        ("POST", f"/api/subscriptions/{sub_id}/check"),
        ("POST", f"/api/episodes/{ep_id}/process"),
        ("POST", f"/api/episodes/{ep_id}/track-listen"),
    ):
        response = client.request(method, path, headers=SAME_ORIGIN)
        assert response.status_code in ok, (
            f"{method} {path} -> {response.status_code}: {response.text}"
        )

    # The admin action: destroys the subscription, its episodes and its files.
    response = client.delete(f"/api/subscriptions/{sub_id}", headers=SAME_ORIGIN)
    assert response.status_code in ok, response.text
    with get_db_connection() as conn:
        remaining = conn.execute(
            "SELECT COUNT(*) AS n FROM subscriptions WHERE id = ?", (sub_id,)
        ).fetchone()["n"]
    assert remaining == 0, "the legitimate delete no longer works"


# --------------------------------------------------------------------------
# Safe methods must not be broken by the origin check
# --------------------------------------------------------------------------

def test_admin_pages_still_render_for_the_operator(client):
    for path in ("/admin/system", "/admin/ai", "/admin/access"):
        response = client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"

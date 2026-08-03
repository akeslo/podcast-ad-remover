"""Authorization tests for the ordinary-user write surface.

Five state-changing routes were left completely unguarded when the admin
surface was locked down, because they are not administrative and so did not
match a `/admin` prefix:

    POST /add
    POST /episodes/{id}/download
    POST /api/episodes/{id}/reprocess
    POST /api/episodes/{id}/ignore
    POST /subscriptions/{id}/settings

They now carry `require_user_action` (= `require_auth` + `require_same_origin`).
`require_admin` would have been wrong: these are things any signed-in user is
supposed to be able to do, and in the standalone deployment there is no user
to be an admin at all.

The half of this file that matters most is the "still works" half. In
standalone mode (`auth_enabled = 0`) `require_auth` is a no-op by design, and
the whole guard reduces to the same-origin/CSRF check. Locking the owner of a
single-user install out of adding a podcast would be a strictly worse outcome
than the drive-by POST being closed, so every route is proved to still work
for a legitimate browser request in *both* auth modes.

Everything runs through the real ASGI stack (`app.main:app`), so
`auth_middleware`, `feed_auth_middleware` and the route dependencies all
execute. Background tasks that would touch the network are stubbed; the
authorization decision has already been made by the time they run.
"""
import pytest

from app.infra.database import get_db_connection
from app.web.auth_utils import hash_password


SAME_ORIGIN = {"Origin": "http://testserver"}
CROSS_ORIGIN = {"Origin": "http://attacker.example"}

FEED_URL = "https://example.invalid/user-route-authorization.xml"
PASSWORD = "correct-horse-battery-staple"


def _set_auth_enabled(value):
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute("UPDATE app_settings SET auth_enabled = ? WHERE id = 1", (value,))
        conn.commit()


@pytest.fixture
def fixtures():
    """One subscription and one episode to act on, cleaned up afterwards."""
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            (FEED_URL, "Fixture Show", "fixture-show"),
        )
        sub_id = cur.lastrowid
        cur = conn.execute(
            "INSERT INTO episodes (subscription_id, guid, title, original_url, status)"
            " VALUES (?, ?, ?, ?, ?)",
            (sub_id, "fixture-guid", "Fixture Episode",
             "https://example.invalid/ep.mp3", "completed"),
        )
        ep_id = cur.lastrowid
        conn.commit()
    yield {"sub_id": sub_id, "ep_id": ep_id}
    with get_db_connection() as conn:
        conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (sub_id,))
        # `_requests` posts /add with a FEED_URL-derived value, so a passing
        # success case leaves a real subscription behind. Sweep the whole
        # family, not just the seeded row.
        conn.execute("DELETE FROM subscriptions WHERE feed_url LIKE ?", (FEED_URL + "%",))
        conn.commit()


@pytest.fixture(autouse=True)
def _no_background_work(monkeypatch):
    """Neuter every background task these routes queue.

    They fetch feeds and run the AI pipeline over real network calls. None of
    them participate in the authorization decision under test - they only run
    *after* a request was allowed - so stubbing them keeps the suite hermetic
    without weakening what is being proved.
    """
    from app.core.processor import Processor
    from app.core.feed import FeedManager

    async def _anoop(*args, **kwargs):
        return None

    for name in ("check_feeds", "process_queue", "cleanup_old_episodes",
                 "version_episode", "delete_episode"):
        if hasattr(Processor, name):
            monkeypatch.setattr(Processor, name, _anoop)

    monkeypatch.setattr(
        FeedManager, "parse_feed",
        staticmethod(lambda url: ("Stub Show", "stub-show", None, "stubbed")),
    )


@pytest.fixture
def standalone(client):
    """The common deployment: no users at all, `require_auth` a no-op."""
    _set_auth_enabled(0)
    yield client
    _set_auth_enabled(0)


@pytest.fixture
def multi_user(client):
    """Auth enabled, with a genuinely logged-in non-admin session."""
    _set_auth_enabled(1)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", ("routetester",))
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("routetester", hash_password(PASSWORD)),
        )
        conn.commit()

    response = client.post(
        "/login",
        data={"username": "routetester", "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text

    yield client

    client.cookies.clear()
    _set_auth_enabled(0)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", ("routetester",))
        conn.commit()


def _requests(f):
    """(name, method, path, body) for each guarded route, given fixture ids."""
    return [
        ("add", "POST", "/add",
         {"feed_url": FEED_URL + "?new", "initial_count": 1}),
        ("download", "POST", f"/episodes/{f['ep_id']}/download", None),
        ("reprocess", "POST", f"/api/episodes/{f['ep_id']}/reprocess", None),
        ("ignore", "POST", f"/api/episodes/{f['ep_id']}/ignore", None),
        ("settings", "POST", f"/subscriptions/{f['sub_id']}/settings",
         {"retention_days": 30, "manual_retention_days": 14, "retention_limit": 1}),
    ]


def _ids(f):
    return [case[0] for case in _requests(f)]


# --------------------------------------------------------------------------
# Rejected: what an attacker's script and a hostile page actually send
# --------------------------------------------------------------------------

@pytest.mark.parametrize("index", range(5))
def test_originless_post_is_rejected_in_standalone_mode(standalone, fixtures, index):
    """curl and every blind scripted loop: no session, no Origin, no Referer."""
    name, method, path, body = _requests(fixtures)[index]

    response = standalone.request(method, path, data=body, follow_redirects=False)

    assert response.status_code == 403, f"{name}: {response.status_code}"


@pytest.mark.parametrize("index", range(5))
def test_cross_site_post_is_rejected_in_standalone_mode(standalone, fixtures, index):
    """A hostile page auto-submitting a form at the victim's install."""
    name, method, path, body = _requests(fixtures)[index]

    response = standalone.request(
        method, path, data=body, headers=CROSS_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 403, f"{name}: {response.status_code}"


@pytest.mark.parametrize("index", range(5))
def test_anonymous_post_is_rejected_in_multi_user_mode(client, fixtures, index):
    """With auth on and no session, the request must not reach the handler.

    Rejection may come from `auth_middleware` (a 302 to /login) or from
    `require_user_action` itself - either is a refusal; what must never happen
    is a 2xx or a state change.
    """
    _set_auth_enabled(1)
    try:
        name, method, path, body = _requests(fixtures)[index]
        response = client.request(method, path, data=body, follow_redirects=False)
        assert response.status_code in (302, 401, 403), \
            f"{name}: {response.status_code}"
    finally:
        _set_auth_enabled(0)


def test_cross_site_post_is_rejected_even_with_a_valid_session(multi_user, fixtures):
    """Authentication alone is not CSRF protection - the origin check still runs."""
    for name, method, path, body in _requests(fixtures):
        response = multi_user.request(
            method, path, data=body, headers=CROSS_ORIGIN, follow_redirects=False
        )
        assert response.status_code == 403, f"{name}: {response.status_code}"


# --------------------------------------------------------------------------
# Still works: the half that matters more
# --------------------------------------------------------------------------

def _assert_all_succeed(http, fixtures):
    for name, method, path, body in _requests(fixtures):
        response = http.request(
            method, path, data=body, headers=SAME_ORIGIN, follow_redirects=False
        )
        assert response.status_code in (200, 303), \
            f"{name}: legitimate request rejected with {response.status_code}"


def test_standalone_owner_can_still_use_every_route(standalone, fixtures):
    """The regression that would be worse than the hole.

    A single-user install has nobody to log in as. If the guard demanded a
    real session here, the owner could no longer add a podcast.
    """
    _assert_all_succeed(standalone, fixtures)


def test_signed_in_user_can_still_use_every_route(multi_user, fixtures):
    _assert_all_succeed(multi_user, fixtures)


def test_settings_write_actually_lands_for_a_legitimate_request(standalone, fixtures):
    """Proof the success cases above are real work, not just a passed gate."""
    sub_id = fixtures["sub_id"]

    response = standalone.post(
        f"/subscriptions/{sub_id}/settings",
        data={"remove_ads": "on", "retention_days": 7,
              "manual_retention_days": 3, "retention_limit": 5},
        headers=SAME_ORIGIN,
        follow_redirects=False,
    )
    assert response.status_code == 303, response.text

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT remove_ads, retention_days FROM subscriptions WHERE id = ?",
            (sub_id,),
        ).fetchone()
    assert row["remove_ads"] == 1
    assert row["retention_days"] == 7


def test_add_actually_creates_a_subscription_for_a_legitimate_request(standalone):
    """`/add` is the route a broken guard would most visibly kill."""
    url = FEED_URL + "?created"
    try:
        response = standalone.post(
            "/add",
            data={"feed_url": url, "initial_count": 1},
            headers=SAME_ORIGIN,
            follow_redirects=False,
        )
        assert response.status_code == 303, response.text

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT id FROM subscriptions WHERE feed_url = ?", (url,)
            ).fetchone()
        assert row is not None, "the subscription was not created"
    finally:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM subscriptions WHERE feed_url = ?", (url,))
            conn.commit()

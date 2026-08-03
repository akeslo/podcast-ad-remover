"""Authorization tests for POST /feed-token/rotate.

The per-user feed token exists so that a user can revoke their own feed URLs
without anyone else's help and without touching their account password. That
capability was unreachable: the dashboard renders the "Rotate Feed Token"
button for every signed-in user whenever feed auth is on, but the route was
gated by `require_admin_action`, so a non-admin got a 403 on click and could
never revoke anything.

The route is now `require_user_action` (= `require_auth` + `require_same_origin`),
matching its own docstring ("the current user's token") and matching the other
ordinary-user write routes in `tests/test_user_route_authorization.py`.

Everything runs through the real ASGI stack (`app.main:app`), so
`auth_middleware` and the route dependency both execute.
"""
import pytest

from app.infra.database import get_db_connection
from app.web.auth_utils import hash_password


SAME_ORIGIN = {"Origin": "http://testserver"}
CROSS_ORIGIN = {"Origin": "http://attacker.example"}

PASSWORD = "correct-horse-battery-staple"


def _set_auth_enabled(value):
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute("UPDATE app_settings SET auth_enabled = ? WHERE id = 1", (value,))
        conn.commit()


def _make_user(username, is_admin):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, hash_password(PASSWORD), 1 if is_admin else 0),
        )
        conn.commit()


def _drop_user(username):
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()


def _feed_token(username):
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT feed_token FROM users WHERE username = ?", (username,)
        ).fetchone()
    return row["feed_token"] if row else None


def _global_feed_token():
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT feed_auth_token FROM app_settings WHERE id = 1"
        ).fetchone()
    return row["feed_auth_token"] if row else None


@pytest.fixture
def non_admin(client):
    """Auth enabled, with a genuinely logged-in NON-admin session."""
    _set_auth_enabled(1)
    _make_user("rotatetester", is_admin=False)

    response = client.post(
        "/login",
        data={"username": "rotatetester", "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text

    yield client

    client.cookies.clear()
    _set_auth_enabled(0)
    _drop_user("rotatetester")


@pytest.fixture
def admin(client):
    """Auth enabled, with a genuinely logged-in ADMIN session."""
    _set_auth_enabled(1)
    _make_user("rotateadmin", is_admin=True)

    response = client.post(
        "/login",
        data={"username": "rotateadmin", "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text

    yield client

    client.cookies.clear()
    _set_auth_enabled(0)
    _drop_user("rotateadmin")


@pytest.fixture
def standalone(client):
    """The common deployment: no users at all, `require_auth` a no-op."""
    _set_auth_enabled(0)
    yield client
    _set_auth_enabled(0)


# --------------------------------------------------------------------------
# The regression: the button the non-admin can see must be a button that works
# --------------------------------------------------------------------------

def test_non_admin_can_rotate_their_own_feed_token(non_admin):
    """A non-admin must be able to revoke their own feed URLs.

    This is the entire point of a per-user feed token. Under
    `require_admin_action` this returned 403 and the token never changed.
    """
    before = _feed_token("rotatetester")

    response = non_admin.post(
        "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 303, response.status_code
    after = _feed_token("rotatetester")
    assert after, "no feed token was issued"
    assert after != before, "the feed token did not actually rotate"


def test_admin_can_still_rotate_their_own_feed_token(admin):
    """Widening the guard must not have narrowed it for admins."""
    before = _feed_token("rotateadmin")

    response = admin.post(
        "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 303, response.status_code
    after = _feed_token("rotateadmin")
    assert after and after != before


def test_rotation_touches_only_the_calling_users_token(non_admin):
    """The route is per-user: nobody else's feed URLs may be revoked."""
    _make_user("rotatebystander", is_admin=False)
    try:
        # Materialise the bystander's token so there is something to compare.
        from app.infra.database import ensure_feed_token

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE username = ?", ("rotatebystander",)
            ).fetchone()
        bystander_before = ensure_feed_token(row["id"])

        response = non_admin.post(
            "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
        )
        assert response.status_code == 303

        assert _feed_token("rotatebystander") == bystander_before
    finally:
        _drop_user("rotatebystander")


# --------------------------------------------------------------------------
# Guards that must not have been weakened
# --------------------------------------------------------------------------

def test_cross_site_rotate_is_rejected_even_with_a_valid_session(non_admin):
    """`require_same_origin` still applies: a hostile page cannot revoke feeds."""
    before = _feed_token("rotatetester")

    response = non_admin.post(
        "/feed-token/rotate", headers=CROSS_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 403, response.status_code
    assert _feed_token("rotatetester") == before


def test_anonymous_rotate_is_rejected_in_multi_user_mode(client):
    """No session, auth on: refused by the middleware or the dependency."""
    _set_auth_enabled(1)
    try:
        response = client.post(
            "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
        )
        assert response.status_code in (302, 401, 403), response.status_code
    finally:
        _set_auth_enabled(0)


def test_originless_rotate_is_rejected_in_standalone_mode(standalone):
    """curl against a standalone install: no Origin, no Referer, no rotation.

    In standalone mode there are no users, so `require_admin` degraded to a
    dummy admin and never rejected anything - the same-origin check was always
    the whole boundary here. Swapping to `require_user_action` therefore
    changes nothing about what this branch allows, which is exactly what this
    test pins.
    """
    before = _global_feed_token()

    response = standalone.post("/feed-token/rotate", follow_redirects=False)

    assert response.status_code == 403, response.status_code
    assert _global_feed_token() == before


def test_cross_site_rotate_is_rejected_in_standalone_mode(standalone):
    before = _global_feed_token()

    response = standalone.post(
        "/feed-token/rotate", headers=CROSS_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 403, response.status_code
    assert _global_feed_token() == before


def test_same_origin_rotate_still_works_in_standalone_mode(standalone):
    """The single-user owner must keep being able to revoke the shared token."""
    before = _global_feed_token()

    response = standalone.post(
        "/feed-token/rotate", headers=SAME_ORIGIN, follow_redirects=False
    )

    assert response.status_code == 303, response.status_code
    after = _global_feed_token()
    assert after, "no global feed token was issued"
    assert after != before, "the global feed token did not actually rotate"

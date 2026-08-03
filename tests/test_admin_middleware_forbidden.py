"""A logged-in non-admin hitting /admin/* must get 403, not 500.

`auth_middleware` is registered with `@app.middleware("http")`. Starlette's
exception handlers do not run for exceptions raised inside a pure ASGI
middleware, so the `HTTPException(403)` it raised for the admin-privilege
check was never translated into a response - `ServerErrorMiddleware` caught it
and returned a 500 instead. The middleware now returns a real response.

Why nothing caught this before: every existing admin-authorization test runs
with `auth_enabled = 0`, and the whole `if settings['auth_enabled']:` block -
including the admin check - is unreachable there. This file is deliberately
the opposite case: `auth_enabled = 1` with a real non-admin session.
"""
import pytest

from app.infra.database import get_db_connection
from app.web.auth_utils import hash_password


PASSWORD = "correct-horse-battery-staple"


def _set_auth_enabled(value):
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute("UPDATE app_settings SET auth_enabled = ? WHERE id = 1", (value,))
        conn.commit()


@pytest.fixture
def non_admin(client):
    _set_auth_enabled(1)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", ("mwtester",))
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            ("mwtester", hash_password(PASSWORD)),
        )
        conn.commit()

    response = client.post(
        "/login",
        data={"username": "mwtester", "password": PASSWORD},
        follow_redirects=False,
    )
    assert response.status_code == 302, response.text

    yield client

    client.cookies.clear()
    _set_auth_enabled(0)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = ?", ("mwtester",))
        conn.commit()


ADMIN_PATHS = [
    "/admin",
    "/admin/system",
    "/admin/ai",
    "/admin/access",
    "/admin/queue",
    "/admin/logs",
]


@pytest.mark.parametrize("path", ADMIN_PATHS)
def test_non_admin_gets_403_not_500_on_admin_pages(non_admin, path):
    response = non_admin.get(path, follow_redirects=False)

    assert response.status_code == 403, \
        f"{path}: expected 403, got {response.status_code}"


def test_non_admin_admin_post_is_forbidden_not_a_server_error(non_admin):
    """The state-changing side of the same check."""
    response = non_admin.post(
        "/admin/system/update",
        headers={"Origin": "http://testserver"},
        data={"auth_enabled": "0"},
        follow_redirects=False,
    )

    assert response.status_code == 403, response.status_code


def test_non_admin_keeps_access_to_the_ordinary_dashboard(non_admin):
    """The 403 must be scoped to /admin, not applied to the whole app."""
    response = non_admin.get("/", follow_redirects=False)

    assert response.status_code == 200, response.status_code

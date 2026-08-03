"""The login page must not hand out the admin password.

`/login` is exempt from `auth_middleware`, so everything it renders is public
to anyone who can reach the app. The generated first-run admin password used
to be stored in `app_settings.initial_password` and printed on this page, in
a `<code>` block, until the operator happened to change it.
"""
from pathlib import Path

import pytest

from app.infra.database import get_db_connection
from app.web.auth_utils import generate_secure_password


PROBE_PASSWORD = "Initial-Admin-Passw0rd-Probe"


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
def _column_exists():
    """Skip cleanly once the column itself is dropped from the schema."""
    with get_db_connection() as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(app_settings)").fetchall()
        }
    return "initial_password" in columns


def test_login_page_does_not_render_a_stored_initial_password(client, _column_exists):
    """Even with the column populated, the page must not leak it."""
    if not _column_exists:
        pytest.skip("initial_password column already dropped")

    _set_settings(auth_enabled=1, initial_password=PROBE_PASSWORD)
    try:
        page = client.get("/login")
        assert page.status_code == 200
        assert PROBE_PASSWORD not in page.text
    finally:
        _set_settings(auth_enabled=0, initial_password=None)


def test_login_page_does_not_leak_on_first_launch(client, _column_exists):
    """The first-launch branch is the one that used to print the password."""
    if not _column_exists:
        pytest.skip("initial_password column already dropped")

    with get_db_connection() as conn:
        existing = conn.execute("SELECT id, username FROM users").fetchall()
        conn.execute("DELETE FROM users")
        conn.commit()

    _set_settings(auth_enabled=1, initial_password=PROBE_PASSWORD)
    try:
        page = client.get("/login")
        assert page.status_code == 200
        assert PROBE_PASSWORD not in page.text
    finally:
        _set_settings(auth_enabled=0, initial_password=None)
        with get_db_connection() as conn:
            for row in existing:
                conn.execute(
                    "INSERT OR IGNORE INTO users (id, username, password_hash) "
                    "VALUES (?, ?, '')",
                    (row["id"], row["username"]),
                )
            conn.commit()


def test_router_never_reads_or_writes_the_plaintext_password_column():
    """Regression guard: no code path may put it back on a rendered page."""
    import app.web.router as router_module

    source = Path(router_module.__file__).read_text(encoding="utf-8")
    statements = [
        line
        for line in source.splitlines()
        if "initial_password" in line and not line.lstrip().startswith("#")
    ]
    # The only surviving mention is the local variable holding the generated
    # password long enough to hash and log it - never a SELECT or an UPDATE.
    for line in statements:
        assert "SELECT" not in line.upper(), line
        assert "UPDATE app_settings" not in line, line


def test_login_template_has_no_password_placeholder():
    template = Path(__file__).resolve().parents[1] / "app/web/templates/login.html"
    body = template.read_text(encoding="utf-8")
    # A Jinja comment mentioning it is fine; an interpolation is not.
    assert "{{ initial_password }}" not in body
    assert "{% if initial_password %}" not in body

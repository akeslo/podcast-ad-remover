"""Tests for per-user feed tokens (the replacement for password-in-URL feeds).

These exercise the data layer only: schema/migration idempotency, the backfill,
and the token accessors in app.infra.database.
"""
import pytest

from app.infra.database import (
    ensure_feed_token,
    ensure_global_feed_token,
    find_user_by_feed_token,
    get_db_connection,
    get_feed_token,
    get_global_feed_token,
    init_db,
    rotate_feed_token,
)
from app.web.auth_utils import generate_feed_token


def _make_user(username: str) -> int:
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, "not-a-real-hash"),
        )
        conn.commit()
        return cur.lastrowid


def _clear_token(user_id: int) -> None:
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET feed_token = NULL WHERE id = ?", (user_id,))
        conn.commit()


@pytest.fixture
def user_id():
    uid = _make_user(f"feedtoken-user-{generate_feed_token()[:8]}")
    yield uid
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
        conn.commit()


def _columns(table: str) -> set:
    with get_db_connection() as conn:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_generate_feed_token_is_url_safe_and_unique():
    a, b = generate_feed_token(), generate_feed_token()
    assert a != b
    assert len(a) >= 32
    assert all(c.isalnum() or c in "-_" for c in a)


def test_schema_has_feed_token_columns():
    assert "feed_token" in _columns("users")
    assert "feed_auth_token" in _columns("app_settings")
    # The old column is deliberately retained; removing it is a separate change.
    assert "feed_auth_password" in _columns("app_settings")


def test_migration_is_idempotent(user_id):
    token = ensure_feed_token(user_id)
    for _ in range(3):
        init_db()  # must not raise against an already-migrated, populated DB
    assert "feed_token" in _columns("users")
    assert "feed_auth_token" in _columns("app_settings")
    # Re-running must not clobber an existing token.
    assert get_feed_token(user_id) == token


def test_backfill_gives_every_user_a_token(user_id):
    other = _make_user(f"feedtoken-backfill-{generate_feed_token()[:8]}")
    try:
        _clear_token(user_id)
        _clear_token(other)

        init_db()

        t1, t2 = get_feed_token(user_id), get_feed_token(other)
        assert t1 and t2
        assert t1 != t2  # tokens are per-user, not shared

        with get_db_connection() as conn:
            missing = conn.execute(
                "SELECT COUNT(*) AS n FROM users "
                "WHERE feed_token IS NULL OR feed_token = ''"
            ).fetchone()["n"]
        assert missing == 0
    finally:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (other,))
            conn.commit()


def test_ensure_feed_token_is_stable(user_id):
    first = ensure_feed_token(user_id)
    assert first
    assert ensure_feed_token(user_id) == first


def test_ensure_feed_token_generates_when_missing(user_id):
    _clear_token(user_id)
    assert get_feed_token(user_id) is None
    token = ensure_feed_token(user_id)
    assert token
    assert get_feed_token(user_id) == token


def test_find_user_by_feed_token_resolves(user_id):
    token = ensure_feed_token(user_id)
    found = find_user_by_feed_token(token)
    assert found is not None
    assert found["id"] == user_id


def test_find_user_by_feed_token_rejects_empty_and_none(user_id):
    # A user with a NULL token must not be reachable by a NULL/empty token.
    _clear_token(user_id)
    assert find_user_by_feed_token(None) is None
    assert find_user_by_feed_token("") is None
    ensure_feed_token(user_id)
    assert find_user_by_feed_token(None) is None
    assert find_user_by_feed_token("") is None


def test_find_user_by_feed_token_rejects_unknown_token(user_id):
    ensure_feed_token(user_id)
    assert find_user_by_feed_token(generate_feed_token()) is None


def test_rotated_token_no_longer_resolves(user_id):
    old = ensure_feed_token(user_id)
    new = rotate_feed_token(user_id)
    assert new != old
    assert find_user_by_feed_token(old) is None
    found = find_user_by_feed_token(new)
    assert found is not None and found["id"] == user_id
    assert get_feed_token(user_id) == new


def test_global_feed_token_ensure_and_get():
    with get_db_connection() as conn:
        conn.execute("UPDATE app_settings SET feed_auth_token = NULL WHERE id = 1")
        conn.commit()
    assert get_global_feed_token() is None
    token = ensure_global_feed_token()
    assert token
    assert get_global_feed_token() == token
    assert ensure_global_feed_token() == token


def test_find_user_by_feed_token_rejects_non_ascii_without_raising(user_id):
    """`hmac.compare_digest` raises TypeError on non-ASCII str arguments; the
    per-user lookup is reachable with an attacker-controlled token, so this
    used to be a free unauthenticated 500."""
    ensure_feed_token(user_id)
    assert find_user_by_feed_token("café") is None
    assert find_user_by_feed_token("☃" * 40) is None


def test_find_user_by_feed_token_does_not_return_the_password_hash(user_id):
    ensure_feed_token(user_id)
    found = find_user_by_feed_token(get_feed_token(user_id))
    assert found is not None
    assert "password_hash" not in found
    assert set(found) == {"id", "username", "feed_token"}

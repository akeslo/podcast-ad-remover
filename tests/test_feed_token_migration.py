"""Upgrade-path tests for feed tokens.

A fresh database proves nothing here: the failure mode is specifically a
*populated pre-upgrade* install, where the migration is the only code that
runs. The pre-upgrade shape is built by taking the current schema and removing
the two columns the feed-token change introduced.
"""
import base64
import os
import sqlite3

import pytest

from app.core.config import settings as app_settings
from app.infra.database import (
    find_user_by_feed_token,
    get_db_connection,
    get_global_feed_token,
    init_db,
)


def _use_data_dir(monkeypatch, data_dir):
    """Point the app at a throwaway DATA_DIR.

    `DB_PATH` is a derived property with no setter, so the data directory is
    the only redirection point.
    """
    for sub in ("db", "feeds", "audio"):
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
    monkeypatch.setattr(app_settings, "DATA_DIR", str(data_dir))


def _make_pre_upgrade_db(monkeypatch, data_dir, *, users=("kes",), enable_feed_auth=1, auth_enabled=0):
    """A populated install from before the feed-token change.

    Built by materialising the current schema and then stripping
    `users.feed_token` and `app_settings.feed_auth_token`, so every other
    column an old install would have is present and realistic.
    """
    _use_data_dir(monkeypatch, data_dir)
    init_db()
    conn = sqlite3.connect(app_settings.DB_PATH)

    conn.execute("DROP INDEX IF EXISTS idx_users_feed_token_unique")
    conn.execute("ALTER TABLE users DROP COLUMN feed_token")
    conn.execute("ALTER TABLE app_settings DROP COLUMN feed_auth_token")
    conn.execute(
        "UPDATE app_settings SET enable_feed_auth = ?, auth_enabled = ?, "
        "feed_auth_username = 'feeds' WHERE id = 1",
        (enable_feed_auth, auth_enabled),
    )
    for name in users:
        conn.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (name, "not-a-real-hash"),
        )
    conn.commit()

    cols = {r[1] for r in conn.execute("PRAGMA table_info(app_settings)")}
    assert "feed_auth_token" not in cols, "fixture failed to build a pre-upgrade DB"
    user_cols = {r[1] for r in conn.execute("PRAGMA table_info(users)")}
    assert "feed_token" not in user_cols
    conn.close()


@pytest.fixture
def upgraded_db(tmp_path, monkeypatch):
    """Yield a DB that was populated pre-upgrade and then migrated."""
    data_dir = tmp_path / "install"
    _make_pre_upgrade_db(monkeypatch, data_dir)
    init_db()  # the upgrade
    return data_dir


def test_upgrade_populates_the_global_feed_token(upgraded_db):
    """A standalone install must not come up with a NULL global token.

    Before the fix, `feed_auth_token` stayed NULL after init_db(); the
    middleware then correctly failed closed and every feed and audio request
    401'd until a human opened the dashboard, which was the only caller of
    ensure_global_feed_token().
    """
    token = get_global_feed_token()
    assert token, "global feed token is still unset after the upgrade migration"


def test_upgrade_populates_per_user_feed_tokens(upgraded_db):
    with get_db_connection() as conn:
        rows = conn.execute("SELECT id, feed_token FROM users").fetchall()
    assert rows
    for row in rows:
        assert row["feed_token"]
        assert find_user_by_feed_token(row["feed_token"])["id"] == row["id"]


def test_upgrade_is_idempotent(upgraded_db):
    first = get_global_feed_token()
    init_db()
    assert get_global_feed_token() == first


def test_feed_request_authenticates_after_upgrade_with_no_dashboard_visit(
    upgraded_db, client
):
    """End to end: the standalone install serves feeds straight after upgrade.

    Nothing here touches the dashboard - the only thing that ran is init_db().
    """
    token = get_global_feed_token()
    assert client.get("/feeds/show.xml").status_code == 401
    assert client.get("/feeds/show.xml", params={"auth": "wrong"}).status_code == 401
    # A correct token is no longer rejected by the enabled-but-unconfigured
    # branch. 404/200 both mean "authenticated"; only 401 is the regression.
    assert client.get("/feeds/show.xml", params={"auth": token}).status_code != 401
    assert client.get("/audio/ep.mp3", params={"auth": token}).status_code != 401
    # The unified feed is now behind the same middleware, but router.py still
    # runs its own inline check, and that one accepts only the legacy
    # base64("user:token") envelope. Use the form both gates accept, so this
    # test keeps passing whether or not the inline check is removed.
    envelope = base64.b64encode(f"feeds:{token}".encode()).decode()
    assert client.get("/feed/unified.xml", params={"auth": envelope}).status_code != 401
    assert client.get("/feed/unified.xml").status_code == 401


def test_unique_index_on_feed_token_exists(upgraded_db):
    with get_db_connection() as conn:
        names = {
            r["name"] for r in conn.execute("PRAGMA index_list('users')")
        }
    assert "idx_users_feed_token_unique" in names


def test_duplicate_feed_token_is_rejected_by_the_index(upgraded_db):
    with get_db_connection() as conn:
        conn.execute("INSERT INTO users (username, password_hash) VALUES ('two', 'x')")
        conn.commit()
        token = conn.execute(
            "SELECT feed_token FROM users WHERE username = 'kes'"
        ).fetchone()["feed_token"]
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE users SET feed_token = ? WHERE username = 'two'", (token,)
            )
            conn.commit()


def test_ambiguous_token_resolves_to_nobody(tmp_path, monkeypatch):
    """If duplicates ever exist, refuse rather than pick the last row.

    The old loop kept the *last* match, so a duplicate authenticated the wrong
    account.
    """
    _make_pre_upgrade_db(monkeypatch, tmp_path / "dupes", users=("a", "b"))
    init_db()

    with get_db_connection() as conn:
        token = conn.execute(
            "SELECT feed_token FROM users WHERE username = 'a'"
        ).fetchone()["feed_token"]
        # Force the duplicate in past the index to exercise the runtime guard.
        conn.execute("DROP INDEX IF EXISTS idx_users_feed_token_unique")
        conn.execute("UPDATE users SET feed_token = ? WHERE username = 'b'", (token,))
        conn.commit()

    assert find_user_by_feed_token(token) is None

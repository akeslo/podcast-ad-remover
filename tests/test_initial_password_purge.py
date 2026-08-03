"""Upgrade-path tests for the plaintext `app_settings.initial_password` column.

A test on a fresh database proves nothing here. The whole failure mode is a
*populated pre-upgrade* install: the column exists and still contains a live
admin password that the unauthenticated /login page used to render. Removing
the code that wrote and read it does not remove the data, so every fixture
below plants a real password in the column first and then runs the migration.
"""
import os
import sqlite3

import pytest

from app.core.config import settings as app_settings
from app.infra.database import get_db_connection, init_db

STORED_PASSWORD = "Pl4inText-Adm1n-Passw0rd"


def _use_data_dir(monkeypatch, data_dir):
    """Point the app at a throwaway DATA_DIR.

    `DB_PATH` is a derived property with no setter, so the data directory is
    the only redirection point.
    """
    for sub in ("db", "feeds", "audio"):
        os.makedirs(os.path.join(data_dir, sub), exist_ok=True)
    monkeypatch.setattr(app_settings, "DATA_DIR", str(data_dir))


def _make_pre_upgrade_db(monkeypatch, data_dir):
    """A populated install from before the fix, holding a real password.

    Built by materialising the current schema and then re-adding the column
    the fix removes, so every other column a real install would have is
    present.
    """
    _use_data_dir(monkeypatch, data_dir)
    init_db()

    conn = sqlite3.connect(app_settings.DB_PATH)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(app_settings)")}
    if "initial_password" not in columns:
        conn.execute("ALTER TABLE app_settings ADD COLUMN initial_password TEXT")
    conn.execute(
        "UPDATE app_settings SET initial_password = ?, auth_enabled = 1 WHERE id = 1",
        (STORED_PASSWORD,),
    )
    conn.commit()

    # The fixture is worthless if it did not actually store the secret.
    stored = conn.execute(
        "SELECT initial_password FROM app_settings WHERE id = 1"
    ).fetchone()[0]
    assert stored == STORED_PASSWORD, "fixture failed to build a pre-upgrade DB"
    conn.close()


def _column_names():
    with get_db_connection() as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info(app_settings)")}


def _raw_db_bytes():
    """Every byte of the database file, WAL included.

    A cleared column can still leave the old value in a free page or in the
    write-ahead log, which is where a plaintext password would keep sitting
    for an attacker with file access.
    """
    blob = b""
    for suffix in ("", "-wal", "-shm"):
        path = app_settings.DB_PATH + suffix
        if os.path.exists(path):
            with open(path, "rb") as handle:
                blob += handle.read()
    return blob


@pytest.fixture
def upgraded_db(tmp_path, monkeypatch):
    """Yield a DB that held a plaintext password and was then migrated."""
    data_dir = tmp_path / "install"
    _make_pre_upgrade_db(monkeypatch, data_dir)
    init_db()  # the upgrade
    return data_dir


def test_upgrade_destroys_the_stored_plaintext_password(upgraded_db):
    """The value must be unreadable through SQL after the migration."""
    columns = _column_names()
    if "initial_password" not in columns:
        # Dropped outright - the strongest outcome, nothing left to read.
        return
    with get_db_connection() as conn:
        value = conn.execute(
            "SELECT initial_password FROM app_settings WHERE id = 1"
        ).fetchone()["initial_password"]
    assert value is None, "the pre-upgrade plaintext password survived init_db()"


def test_upgrade_drops_the_column_on_a_modern_sqlite(upgraded_db):
    """DROP COLUMN landed on 3.35. Below that, clearing is the correct answer."""
    columns = _column_names()
    if sqlite3.sqlite_version_info >= (3, 35, 0):
        assert "initial_password" not in columns
    else:
        assert "initial_password" in columns


def test_password_is_not_recoverable_by_a_raw_query_on_either_shape(upgraded_db):
    """Whichever shape the install ends in, no SQL path returns the secret."""
    columns = _column_names()
    with get_db_connection() as conn:
        row = dict(conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone())
    assert STORED_PASSWORD not in [v for v in row.values() if isinstance(v, str)]
    if "initial_password" in columns:
        assert row["initial_password"] is None


def test_upgrade_is_idempotent(upgraded_db):
    """init_db() runs on every boot; a second pass must not fail or resurrect."""
    before = _column_names()
    init_db()
    init_db()
    assert _column_names() == before
    if "initial_password" in before:
        with get_db_connection() as conn:
            assert (
                conn.execute(
                    "SELECT initial_password FROM app_settings WHERE id = 1"
                ).fetchone()["initial_password"]
                is None
            )


def test_a_fresh_install_never_creates_the_column(tmp_path, monkeypatch):
    """No new install should carry the plaintext column at all."""
    _use_data_dir(monkeypatch, tmp_path / "fresh")
    init_db()
    assert "initial_password" not in _column_names()


def test_other_settings_survive_the_purge(upgraded_db):
    """The migration must destroy the password and nothing else."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT auth_enabled, feed_auth_token FROM app_settings WHERE id = 1"
        ).fetchone()
    assert row["auth_enabled"] == 1
    assert row["feed_auth_token"]


def test_password_bytes_are_gone_from_the_database_file(upgraded_db):
    """Belt and braces: the string should not survive anywhere on disk.

    VACUUM is not run by the migration, so if this ever fails it is a real
    finding about residue rather than a nit - the point of the change is that
    an attacker with the data volume cannot read the admin password.
    """
    assert STORED_PASSWORD.encode() not in _raw_db_bytes()

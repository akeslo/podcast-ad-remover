"""
Tests for app/core/orphan_cleanup.py - the filesystem-vs-database
reconciliation sweep.

This code deletes the operator's media, so the safety properties are the
deliverable: a referenced directory is never touched, a fresh path survives,
a database error sweeps nothing, and dry-run deletes nothing while reporting
honestly.

Everything runs against a throwaway podcasts tree and a throwaway sqlite file;
no real /data volume and no network.
"""
import os
import sqlite3
import time
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.orphan_cleanup import episode_slug_for, sweep_orphans

HOUR = 3600
OLD = 72 * HOUR  # comfortably past every default threshold


def _age(path: str, seconds: float):
    """Backdate mtime/atime so the age gate sees the path as old."""
    past = time.time() - seconds
    os.utime(path, (past, past))


def _make_file(path: str, size: int = 16, age: float = OLD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(b"x" * size)
    _age(path, age)


def _make_episode_dir(root: str, sub_slug: str, ep_slug: str, size: int = 16, age: float = OLD) -> str:
    path = os.path.join(root, sub_slug, ep_slug)
    os.makedirs(path, exist_ok=True)
    _make_file(os.path.join(path, "processed.mp4"), size=size, age=age)
    _age(path, age)
    _age(os.path.join(root, sub_slug), age)
    return path


@pytest.fixture
def fake_db(tmp_path, monkeypatch):
    """A real sqlite database with the two tables the sweep reads."""
    db_path = str(tmp_path / "podcasts.db")
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE subscriptions (id INTEGER PRIMARY KEY, slug TEXT)")
    conn.execute("CREATE TABLE episodes (id INTEGER PRIMARY KEY, subscription_id INTEGER, guid TEXT)")
    conn.commit()
    conn.close()

    from contextlib import contextmanager

    @contextmanager
    def _get_db_connection():
        c = sqlite3.connect(db_path)
        c.row_factory = sqlite3.Row
        try:
            yield c
        finally:
            c.close()

    monkeypatch.setattr("app.infra.database.get_db_connection", _get_db_connection)
    return db_path


def _seed(db_path, subs, episodes=()):
    conn = sqlite3.connect(db_path)
    conn.executemany("INSERT INTO subscriptions (id, slug) VALUES (?, ?)", subs)
    conn.executemany("INSERT INTO episodes (subscription_id, guid) VALUES (?, ?)", episodes)
    conn.commit()
    conn.close()


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "podcasts"
    root.mkdir()
    return str(root)


# --------------------------------------------------------------------------
# Orphans are removed
# --------------------------------------------------------------------------

def test_unreferenced_subscription_directory_is_removed(fake_db, tree):
    """The `loading-*` placeholder case: a directory with no subscription row."""
    _seed(fake_db, [(1, "real-show")], [(1, "guid-1")])
    _make_episode_dir(tree, "real-show", episode_slug_for("guid-1"))
    orphan = os.path.join(tree, "loading-1770511947")
    _make_episode_dir(tree, "loading-1770511947", "some-episode", size=1024)

    res = sweep_orphans(podcasts_dir=tree)

    assert res.ran
    assert res.orphan_dirs == [orphan]
    assert not os.path.exists(orphan)
    assert os.path.exists(os.path.join(tree, "real-show"))
    assert res.bytes_reclaimed == 1024


def test_unreferenced_episode_directory_is_removed(fake_db, tree):
    """An episode directory whose row was hard-deleted."""
    _seed(fake_db, [(1, "real-show")], [(1, "kept-guid")])
    kept = _make_episode_dir(tree, "real-show", episode_slug_for("kept-guid"))
    gone = _make_episode_dir(tree, "real-show", "dropped-guid", size=2048)

    res = sweep_orphans(podcasts_dir=tree)

    assert res.orphan_dirs == [gone]
    assert not os.path.exists(gone)
    assert os.path.exists(kept)
    assert res.bytes_reclaimed == 2048


def test_guid_with_slash_and_space_maps_to_its_directory(fake_db, tree):
    """Referenced-ness uses the same slug transform the writer uses."""
    guid = "http://example.com/ep 1"
    _seed(fake_db, [(1, "show")], [(1, guid)])
    kept = _make_episode_dir(tree, "show", episode_slug_for(guid))

    res = sweep_orphans(podcasts_dir=tree)

    assert res.orphan_dirs == []
    assert os.path.exists(kept)


def test_every_episode_status_counts_as_referenced(fake_db, tree):
    """
    The sweep must not care about status: a queued/in-flight/soft-deleted row
    still owns its directory.
    """
    _seed(fake_db, [(1, "show")], [(1, "queued-guid"), (1, "ignored-guid")])
    a = _make_episode_dir(tree, "show", episode_slug_for("queued-guid"))
    b = _make_episode_dir(tree, "show", episode_slug_for("ignored-guid"))

    res = sweep_orphans(podcasts_dir=tree)

    assert res.orphan_dirs == []
    assert os.path.exists(a) and os.path.exists(b)


# --------------------------------------------------------------------------
# Age gating
# --------------------------------------------------------------------------

def test_fresh_unreferenced_directory_survives(fake_db, tree):
    """
    A `loading-*` directory can be legitimately in use right now - the
    subscription is renamed only after the feed fetch succeeds. Name alone is
    never the gate.
    """
    _seed(fake_db, [(1, "show")])
    fresh = os.path.join(tree, "loading-9999999999-abcd1234")
    _make_episode_dir(tree, "loading-9999999999-abcd1234", "ep", age=0)

    res = sweep_orphans(podcasts_dir=tree)

    assert res.orphan_dirs == []
    assert os.path.exists(fresh)


def test_old_part_file_is_removed_and_fresh_one_survives(fake_db, tree):
    _seed(fake_db, [(1, "show")], [(1, "g1")])
    ep = _make_episode_dir(tree, "show", episode_slug_for("g1"))
    stale = os.path.join(ep, "download.mp4.part")
    fresh = os.path.join(ep, "inflight.mp4.part")
    _make_file(stale, size=512, age=OLD)
    _make_file(fresh, size=512, age=60)

    res = sweep_orphans(podcasts_dir=tree)

    assert res.part_files == [stale]
    assert not os.path.exists(stale)
    assert os.path.exists(fresh)
    assert res.bytes_reclaimed == 512


def test_part_threshold_is_a_setting(fake_db, tree, monkeypatch):
    _seed(fake_db, [(1, "show")], [(1, "g1")])
    ep = _make_episode_dir(tree, "show", episode_slug_for("g1"))
    part = os.path.join(ep, "download.mp4.part")
    _make_file(part, size=8, age=6 * HOUR)

    monkeypatch.setattr(settings, "PART_FILE_MAX_AGE_HOURS", 48)
    assert sweep_orphans(podcasts_dir=tree).part_files == []
    assert os.path.exists(part)

    monkeypatch.setattr(settings, "PART_FILE_MAX_AGE_HOURS", 1)
    assert sweep_orphans(podcasts_dir=tree).part_files == [part]
    assert not os.path.exists(part)


def test_default_part_threshold_cannot_kill_a_long_running_download(fake_db, tree):
    """A multi-GB video download must never be inside the default window."""
    assert settings.PART_FILE_MAX_AGE_HOURS >= 12
    assert settings.ORPHAN_MIN_AGE_HOURS >= 12


# --------------------------------------------------------------------------
# Fail-safe behaviour
# --------------------------------------------------------------------------

def test_database_error_sweeps_nothing(fake_db, tree):
    _seed(fake_db, [(1, "show")])
    orphan = os.path.join(tree, "loading-1")
    _make_episode_dir(tree, "loading-1", "ep")

    def _boom():
        raise sqlite3.OperationalError("no such table: subscriptions")

    with patch("app.core.orphan_cleanup._load_reference_index", side_effect=_boom):
        res = sweep_orphans(podcasts_dir=tree)

    assert not res.ran
    assert res.removed_count == 0
    assert res.bytes_reclaimed == 0
    assert "could not build database reference index" in res.skipped_reason
    assert os.path.exists(orphan)


def test_empty_subscriptions_table_sweeps_nothing(fake_db, tree):
    """A lost/reset database must not read as 'the whole tree is orphaned'."""
    _make_episode_dir(tree, "show", "ep")

    res = sweep_orphans(podcasts_dir=tree)

    assert not res.ran
    assert res.removed_count == 0
    assert "no subscriptions" in res.skipped_reason
    assert os.path.exists(os.path.join(tree, "show", "ep"))


def test_episode_row_with_unknown_subscription_sweeps_nothing(fake_db, tree):
    """Unexpected layout -> fail closed."""
    _seed(fake_db, [(1, "show")], [(99, "orphan-row-guid")])
    _make_episode_dir(tree, "show", "whatever")

    res = sweep_orphans(podcasts_dir=tree)

    assert not res.ran
    assert res.removed_count == 0
    assert os.path.exists(os.path.join(tree, "show", "whatever"))


def test_subscription_without_slug_sweeps_nothing(fake_db, tree):
    _seed(fake_db, [(1, "show"), (2, None)])
    _make_episode_dir(tree, "show", "whatever")

    res = sweep_orphans(podcasts_dir=tree)

    assert not res.ran
    assert os.path.exists(os.path.join(tree, "show", "whatever"))


def test_missing_podcasts_dir_is_a_no_op(fake_db, tmp_path):
    res = sweep_orphans(podcasts_dir=str(tmp_path / "nope"))
    assert not res.ran
    assert res.removed_count == 0


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------

def test_dry_run_deletes_nothing_and_reports_honestly(fake_db, tree):
    _seed(fake_db, [(1, "show")], [(1, "g1")])
    kept = _make_episode_dir(tree, "show", episode_slug_for("g1"))
    orphan_dir = os.path.join(tree, "loading-1770511947")
    _make_episode_dir(tree, "loading-1770511947", "ep", size=4096)
    stale_part = os.path.join(kept, "download.mp4.part")
    _make_file(stale_part, size=1024, age=OLD)

    res = sweep_orphans(dry_run=True, podcasts_dir=tree)

    assert res.ran and res.dry_run
    assert res.orphan_dirs == [orphan_dir]
    assert res.part_files == [stale_part]
    assert res.bytes_reclaimed == 4096 + 1024
    # Nothing actually gone.
    assert os.path.exists(orphan_dir)
    assert os.path.exists(stale_part)
    assert os.path.exists(kept)
    assert "would reclaim" in res.summary()

    # And the real run removes exactly what the dry run promised.
    real = sweep_orphans(podcasts_dir=tree)
    assert real.orphan_dirs == res.orphan_dirs
    assert real.part_files == res.part_files
    assert real.bytes_reclaimed == res.bytes_reclaimed


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_processor_runs_the_sweep_and_can_be_disabled(monkeypatch):
    from app.core.processor import Processor

    proc = Processor.__new__(Processor)

    calls = []

    def _fake_sweep(*a, **kw):
        from app.core.orphan_cleanup import SweepResult
        calls.append(kw)
        return SweepResult(ran=True)

    monkeypatch.setattr("app.core.orphan_cleanup.sweep_orphans", _fake_sweep)

    monkeypatch.setattr(settings, "ORPHAN_CLEANUP_ENABLED", False)
    assert await proc.cleanup_orphans() is None
    assert calls == []

    monkeypatch.setattr(settings, "ORPHAN_CLEANUP_ENABLED", True)
    res = await proc.cleanup_orphans()
    assert res is not None and res.ran
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_processor_swallows_sweep_failure(monkeypatch):
    from app.core.processor import Processor

    proc = Processor.__new__(Processor)
    monkeypatch.setattr(settings, "ORPHAN_CLEANUP_ENABLED", True)

    def _boom(*a, **kw):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("app.core.orphan_cleanup.sweep_orphans", _boom)
    assert await proc.cleanup_orphans() is None

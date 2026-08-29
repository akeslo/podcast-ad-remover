"""
Tests for retention: app/core/retention.py (selection) and
Processor.cleanup_old_episodes (the only thing that deletes).

This code deletes the operator's media, so the safety properties are the
deliverable, not the happy path:

  * retention keeps exactly `retention_limit` per subscription and reclaims
    the rest;
  * an in-flight episode (pending / processing / rate_limited / a failed row
    with a retry scheduled) is NEVER selected;
  * a database error deletes nothing;
  * the dry-run plan matches what a real run deletes, exactly;
  * an episode whose subscription is gone is protected, not reclaimed.

Everything runs against a throwaway sqlite file carrying the REAL schema (via
`init_db`, which conftest already ran into a temp DATA_DIR) and a throwaway
podcasts tree. No real /data volume, no network.
"""
import os
import sqlite3
from datetime import datetime, timedelta

import pytest

from app.core.config import settings
from app.core.processor import Processor
from app.core.retention import (
    REASON_FAILED_ABANDONED,
    REASON_MANUAL_EXPIRED,
    REASON_OVER_LIMIT,
    select_expired_episodes,
)

# Every in-flight status, and why it must survive. `EpisodeRepository.get_queue`
# treats exactly these as the live queue.
IN_FLIGHT_STATUSES = ["pending", "processing", "rate_limited"]


def _days_ago(n: float) -> str:
    return (datetime.now() - timedelta(days=n)).strftime("%Y-%m-%d %H:%M:%S")


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A sqlite database carrying the real episodes/subscriptions schema."""
    db_path = str(tmp_path / "podcasts.db")

    # Copy the real schema out of the database the conftest already initialised,
    # so these tests cannot drift from production DDL.
    src = sqlite3.connect(settings.DB_PATH)
    ddl = [
        row[0] for row in src.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name IN ('episodes', 'subscriptions') AND sql IS NOT NULL"
        ).fetchall()
    ]
    src.close()
    assert len(ddl) == 2, "expected the real episodes and subscriptions schema"

    conn = sqlite3.connect(db_path)
    for stmt in ddl:
        conn.execute(stmt)
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
    # `app/infra/repository.py` does `from app.infra.database import get_db_connection`
    # at module scope, binding the function into its own namespace at import
    # time, so patching only the source module leaves every repository method
    # talking to the real database. Both names must be redirected.
    monkeypatch.setattr("app.infra.repository.get_db_connection", _get_db_connection)
    return db_path


def _add_sub(db_path, id, slug, retention_limit=1, manual_retention_days=7):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO subscriptions (id, title, feed_url, slug, retention_limit, manual_retention_days) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (id, slug, f"http://example.invalid/{slug}", slug, retention_limit, manual_retention_days),
    )
    conn.commit()
    conn.close()


def _add_ep(db_path, id, sub_id, guid, status="completed", pub_date=None,
            processed_at=None, is_manual=0, next_retry_at=None):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO episodes (id, subscription_id, guid, title, original_url, status, "
        "pub_date, processed_at, is_manual_download, next_retry_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (id, sub_id, guid, f"title-{guid}", f"http://example.invalid/{guid}.mp3",
         status, pub_date, processed_at, is_manual, next_retry_at),
    )
    conn.commit()
    conn.close()


def _selected(db_path):
    """(ids, {id: reason}) that retention would reclaim."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        items = select_expired_episodes(conn)
    finally:
        conn.close()
    return [i.id for i in items], {i.id: i.reason for i in items}


# --------------------------------------------------------------------------
# The core rule: keep exactly retention_limit, reclaim the rest
# --------------------------------------------------------------------------

def test_keeps_exactly_retention_limit_and_reclaims_the_rest(db):
    _add_sub(db, 1, "show", retention_limit=1)
    # Newest first by pub_date; only the newest may survive.
    _add_ep(db, 10, 1, "newest", pub_date=_days_ago(1), processed_at=_days_ago(1))
    _add_ep(db, 11, 1, "middle", pub_date=_days_ago(5), processed_at=_days_ago(5))
    _add_ep(db, 12, 1, "oldest", pub_date=_days_ago(9), processed_at=_days_ago(9))

    ids, reasons = _selected(db)

    assert ids == [11, 12], "the newest completed episode is the keep set at limit 1"
    assert reasons[11] == REASON_OVER_LIMIT
    assert 10 not in ids


def test_retention_limit_is_per_subscription_not_global(db):
    _add_sub(db, 1, "show-a", retention_limit=1)
    _add_sub(db, 2, "show-b", retention_limit=2)
    _add_ep(db, 10, 1, "a-new", pub_date=_days_ago(1))
    _add_ep(db, 11, 1, "a-old", pub_date=_days_ago(2))
    _add_ep(db, 20, 2, "b-new", pub_date=_days_ago(1))
    _add_ep(db, 21, 2, "b-mid", pub_date=_days_ago(2))
    _add_ep(db, 22, 2, "b-old", pub_date=_days_ago(3))

    ids, _ = _selected(db)

    # show-a keeps 1 (drops 11), show-b keeps 2 (drops 22).
    assert ids == [11, 22]


def test_a_subscription_within_its_limit_loses_nothing(db):
    _add_sub(db, 1, "show", retention_limit=3)
    _add_ep(db, 10, 1, "one", pub_date=_days_ago(1))
    _add_ep(db, 11, 1, "two", pub_date=_days_ago(2))

    assert _selected(db)[0] == []


# --------------------------------------------------------------------------
# The bug this change fixes: non-completed rows were invisible forever
# --------------------------------------------------------------------------

def test_abandoned_failed_episode_is_reclaimed(db):
    """
    THE REGRESSION TEST FOR THE ACTUAL BUG.

    The count-based selector ranks over `status='completed'` only, so on prod
    (one completed episode per subscription, limit 1) it matched nothing while
    non-completed episode directories accumulated forever - invisible to the
    orphan sweep too, which protects every episodes row. Without selector 3
    this returns [] and the disk never drains.
    """
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1), processed_at=_days_ago(1))
    _add_ep(db, 11, 1, "died", status="failed",
            pub_date=_days_ago(30), processed_at=_days_ago(30))

    ids, reasons = _selected(db)

    assert ids == [11]
    assert reasons[11] == REASON_FAILED_ABANDONED


def test_recently_failed_episode_survives_the_grace_window(db):
    """
    `requeue_stuck` marks every interrupted episode failed on each restart.
    Reaping on sight would silently discard work moments after a routine
    restart, so a fresh failure must survive.
    """
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "just-died", status="failed",
            pub_date=_days_ago(30), processed_at=_days_ago(0.5))

    assert _selected(db)[0] == []


def test_failed_episode_with_a_scheduled_retry_is_in_flight(db):
    """A failed row with next_retry_at set is in the live queue. Never touch it."""
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "retrying", status="failed",
            pub_date=_days_ago(60), processed_at=_days_ago(60),
            next_retry_at=_days_ago(-1))

    assert _selected(db)[0] == []


def test_failed_episode_with_no_usable_date_is_not_reclaimed(db):
    """Undateable means ungated, and ungated means protected."""
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "undated", status="failed", pub_date=None, processed_at=None)

    assert _selected(db)[0] == []


# --------------------------------------------------------------------------
# In-flight episodes are never deleted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("status", IN_FLIGHT_STATUSES)
def test_in_flight_episode_is_never_selected(db, status):
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1), processed_at=_days_ago(1))
    # Old enough to trip every age gate, and beyond the retention limit by
    # count. Only its status protects it.
    _add_ep(db, 11, 1, "inflight", status=status,
            pub_date=_days_ago(90), processed_at=_days_ago(90))

    assert 11 not in _selected(db)[0], f"'{status}' is in flight and must survive"


def test_unknown_future_status_is_protected_by_default(db):
    """Reclaimable statuses are an allowlist, so a status added later is safe."""
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "novel", status="some_status_invented_later",
            pub_date=_days_ago(90), processed_at=_days_ago(90))

    assert _selected(db)[0] == []


def test_ignored_soft_deleted_episode_is_not_reselected(db):
    """Its files are already gone; its row exists to stop the guid re-downloading."""
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "gone", status="ignored",
            pub_date=_days_ago(90), processed_at=_days_ago(90))

    assert _selected(db)[0] == []


# --------------------------------------------------------------------------
# Orphaned rows stay protected (the 64331cb rule)
# --------------------------------------------------------------------------

def test_episode_whose_subscription_is_gone_is_protected(db):
    """
    Deleting a subscription does not delete its episodes; prod carries 21 such
    rows. We cannot tell which directory their files live in, so retention must
    not act on them - matching the orphan sweep, which protects them too.
    """
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1))
    _add_ep(db, 90, 1008, "orphan-completed", pub_date=_days_ago(90))
    _add_ep(db, 91, 1008, "orphan-failed", status="failed",
            pub_date=_days_ago(90), processed_at=_days_ago(90))

    ids, _ = _selected(db)

    assert 90 not in ids and 91 not in ids


# --------------------------------------------------------------------------
# Manual downloads keep their own time-based policy
# --------------------------------------------------------------------------

def test_expired_manual_download_is_reclaimed(db):
    _add_sub(db, 1, "show", retention_limit=1, manual_retention_days=7)
    _add_ep(db, 11, 1, "manual-old", is_manual=1,
            pub_date=_days_ago(30), processed_at=_days_ago(30))

    ids, reasons = _selected(db)

    assert ids == [11]
    assert reasons[11] == REASON_MANUAL_EXPIRED


def test_fresh_manual_download_survives_and_ignores_retention_limit(db):
    """Manual downloads are governed by time, not by the auto count."""
    _add_sub(db, 1, "show", retention_limit=1, manual_retention_days=7)
    _add_ep(db, 10, 1, "auto-keeper", pub_date=_days_ago(1), processed_at=_days_ago(1))
    _add_ep(db, 11, 1, "manual-new", is_manual=1,
            pub_date=_days_ago(2), processed_at=_days_ago(2))

    assert _selected(db)[0] == []


# --------------------------------------------------------------------------
# Fail closed
# --------------------------------------------------------------------------

def test_database_error_deletes_nothing(db, monkeypatch, tmp_path):
    """An unreadable database must be a no-op, never a purge."""
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1))
    _add_ep(db, 11, 1, "expired", pub_date=_days_ago(9))

    from contextlib import contextmanager

    @contextmanager
    def _broken():
        raise sqlite3.OperationalError("database is locked")
        yield  # pragma: no cover

    monkeypatch.setattr("app.infra.database.get_db_connection", _broken)

    proc = object.__new__(Processor)
    deleted = []
    proc.delete_episode = lambda i: deleted.append(i)

    import asyncio
    asyncio.run(proc.cleanup_old_episodes())

    assert deleted == []


def test_one_selector_failing_does_not_delete_more(db, monkeypatch):
    """
    A broken selector contributes nothing; it must never widen the result.
    Failing to select is always safe, because it deletes less.
    """
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1))
    _add_ep(db, 11, 1, "expired", pub_date=_days_ago(9))

    def _boom(conn):
        raise sqlite3.OperationalError("no such function: ROW_NUMBER")

    monkeypatch.setattr("app.core.retention._rows_over_limit", _boom)

    assert _selected(db)[0] == []


# --------------------------------------------------------------------------
# Dry run reports exactly what a real run deletes
# --------------------------------------------------------------------------

def test_dry_run_plan_matches_the_real_run_and_deletes_nothing(db, tmp_path, monkeypatch):
    """
    The plan the operator reviews must be the plan that runs. If these two ever
    diverge, the approval means nothing.
    """
    root = tmp_path / "podcasts"
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1), processed_at=_days_ago(1))
    _add_ep(db, 11, 1, "expired", pub_date=_days_ago(9), processed_at=_days_ago(9))
    _add_ep(db, 12, 1, "died", status="failed",
            pub_date=_days_ago(40), processed_at=_days_ago(40))

    for guid, size in (("keeper", 5000), ("expired", 3000), ("died", 1000)):
        d = root / "show" / guid
        d.mkdir(parents=True)
        (d / "processed.mp4").write_bytes(b"x" * size)

    from app.core.retention import plan

    expired, planned_bytes = plan(podcasts_dir=str(root))

    assert [e.id for e in expired] == [11, 12]
    assert planned_bytes == 4000
    # Dry run touched nothing.
    assert (root / "show" / "expired" / "processed.mp4").exists()
    assert (root / "show" / "died" / "processed.mp4").exists()
    assert (root / "show" / "keeper" / "processed.mp4").exists()

    # Now the real run, over the same database, must delete exactly that set.
    proc = object.__new__(Processor)
    deleted = []

    async def _delete(episode_id):
        deleted.append(episode_id)
        return True

    proc.delete_episode = _delete

    import asyncio
    asyncio.run(proc.cleanup_old_episodes())

    assert deleted == [e.id for e in expired], "the dry-run plan and the real run must agree"


def test_real_run_removes_the_directory_and_keeps_the_keeper(db, tmp_path, monkeypatch):
    """End to end through the real delete_episode: files actually go."""
    # Use the real settings.get_episode_dir against conftest's throwaway
    # DATA_DIR rather than patching it: `settings` is a pydantic model and
    # refuses attribute injection, and going through the real path resolver is
    # the more faithful test anyway. The slug is unique to this test so it
    # cannot collide with another one's tree.
    slug = "retention-real-run-show"
    _add_sub(db, 1, slug, retention_limit=1)
    _add_ep(db, 10, 1, "keeper", pub_date=_days_ago(1), processed_at=_days_ago(1))
    _add_ep(db, 11, 1, "expired", pub_date=_days_ago(9), processed_at=_days_ago(9))

    dirs = {}
    for guid in ("keeper", "expired"):
        d = settings.get_episode_dir(slug, guid)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "processed.mp4"), "wb") as fh:
            fh.write(b"x" * 128)
        dirs[guid] = d

    proc = object.__new__(Processor)
    from app.infra.repository import EpisodeRepository, SubscriptionRepository
    proc.ep_repo = EpisodeRepository()
    proc.sub_repo = SubscriptionRepository()
    # The feed regeneration is a side effect unrelated to what is under test.
    proc.rss_gen = type("_NoFeeds", (), {
        "generate_feed": lambda self, sid: None,
        "generate_unified_feed": lambda self: None,
    })()

    import asyncio
    asyncio.run(proc.cleanup_old_episodes())

    assert not os.path.exists(dirs["expired"]), "the expired episode's files must be gone"
    assert os.path.exists(os.path.join(dirs["keeper"], "processed.mp4")), "the keep set must survive"


# --------------------------------------------------------------------------
# requeue_stuck must leave a dateable row
# --------------------------------------------------------------------------

def test_requeue_stuck_stamps_processed_at(db):
    """
    Without this stamp a restart-interrupted episode is undateable, so the
    grace window cannot be measured from when it actually failed.
    """
    _add_sub(db, 1, "show", retention_limit=1)
    _add_ep(db, 11, 1, "running", status="processing", pub_date=_days_ago(30))

    from app.infra.repository import EpisodeRepository
    EpisodeRepository().requeue_stuck()

    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT status, processed_at FROM episodes WHERE id = 11").fetchone()
    conn.close()

    assert row["status"] == "failed"
    assert row["processed_at"] is not None

    # And having just failed, it is inside the grace window, so nothing reclaims it.
    assert _selected(db)[0] == []

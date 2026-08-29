"""
Which episodes retention may reclaim, and why.

This module is SELECTION ONLY. It opens no files, deletes nothing, and has no
side effects, which is what lets `Processor.cleanup_old_episodes` and the
`--dry-run` report share one definition of the keep set instead of each
carrying its own. Two mechanisms deciding what to keep is how they drift apart,
and this codebase has already paid for a duplicated map once.

THE OPERATOR'S RULE THIS IMPLEMENTS:

    "The pod should align to retention for the active episodes as long as it
     does all else can be deleted"

So the keep set is whatever retention says to keep for an ACTIVE subscription,
and everything else on disk is reclaimable - by retention here when a row names
it, and by `app/core/orphan_cleanup.py` when no row does.

WHY A THIRD SELECTOR EXISTS (this is the bug that let 350G accumulate):

The count-based auto retention computes ROW_NUMBER() over a set that has
ALREADY been filtered to `status='completed'`. A non-completed episode is
therefore never counted against `retention_limit` and never ranked beyond it,
so it is never selected. Measured on the live install: 55 episode rows, of
which 8 were completed - exactly one per subscription against a limit of 1 - so
`rn > 1` matched zero rows and retention reported success on every pass while
47 episode directories sat on disk permanently.

Those same directories were invisible to the orphan sweep too, because the
sweep protects every episodes row whatever its status (deliberately, so a
retention-deleted episode is never reaped twice). A directory named by a
non-completed row was thus unreachable by BOTH mechanisms, forever. Selector 3
is the fix, and it lives in retention rather than in the sweep because "this
row is expired" is a retention question and "nothing references this" is the
sweep's.

RECLAIMABLE STATUSES ARE AN ALLOWLIST, NEVER A DENYLIST:

  reclaimable   'completed'  beyond the retention limit / past manual expiry
  reclaimable   'failed'     with next_retry_at IS NULL, past the grace window
  PROTECTED     'pending'    queued, in flight
  PROTECTED     'processing' running right now
  PROTECTED     'rate_limited' awaiting an API quota reset, in flight
  PROTECTED     'failed'     with next_retry_at SET - a scheduled retry is in
                             flight (EpisodeRepository.get_queue lists exactly
                             these four states as the live queue)
  PROTECTED     'ignored'    already file-free from a prior soft delete; its
                             row survives to stop the guid re-downloading
  PROTECTED     anything else, including a status invented after this was
                             written, because it is not on the list

An allowlist is the whole safety property here. A denylist would silently start
deleting any status added later.
"""
import logging
from dataclasses import dataclass

from app.core.config import settings

logger = logging.getLogger(__name__)

# Reasons, used in logs and in the dry-run report.
REASON_MANUAL_EXPIRED = "manual download past manual_retention_days"
REASON_OVER_LIMIT = "auto download beyond retention_limit"
REASON_FAILED_ABANDONED = "failed with no scheduled retry, past FAILED_RETENTION_DAYS"


@dataclass(frozen=True)
class ExpiredEpisode:
    """One episode retention would reclaim."""
    id: int
    title: str
    reason: str
    subscription_slug: str | None = None
    guid: str | None = None


def _rows_manual_expired(conn) -> list[ExpiredEpisode]:
    """Manual downloads older than their subscription's manual_retention_days."""
    cursor = conn.execute("""
        SELECT e.id, e.title, e.guid, s.slug
        FROM episodes e
        LEFT JOIN subscriptions s ON e.subscription_id = s.id
        WHERE e.status = 'completed'
          AND e.is_manual_download = 1
          AND e.processed_at IS NOT NULL
          AND datetime(e.processed_at)
              < datetime('now', '-' || COALESCE(s.manual_retention_days, 14) || ' days')
    """)
    return [
        ExpiredEpisode(row["id"], row["title"], REASON_MANUAL_EXPIRED, row["slug"], row["guid"])
        for row in cursor.fetchall()
    ]


def _rows_over_limit(conn) -> list[ExpiredEpisode]:
    """
    Auto downloads ranked beyond their subscription's retention_limit.

    The JOIN to subscriptions is what keeps an episode row whose subscription
    was deleted out of the result. Those rows are protected, exactly as the
    orphan sweep protects them: we cannot tell which directory their files live
    in, so we never act on them.
    """
    cursor = conn.execute("""
        SELECT t.id, t.title, t.guid, s.slug
        FROM (
            SELECT id, title, guid, subscription_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY subscription_id
                       -- id DESC is a tiebreak, not decoration: pub_date is not
                       -- unique, and was historically NULL for every YouTube
                       -- episode. Without a stable second key the keep slot
                       -- among tied rows is arbitrary and can differ between
                       -- runs, deleting a different episode on each pass.
                       ORDER BY pub_date DESC, id DESC
                   ) as rn
            FROM episodes
            WHERE status = 'completed'
              AND (is_manual_download IS NULL OR is_manual_download = 0)
        ) t
        JOIN subscriptions s ON t.subscription_id = s.id
        WHERE t.rn > COALESCE(s.retention_limit, 1)
    """)
    return [
        ExpiredEpisode(row["id"], row["title"], REASON_OVER_LIMIT, row["slug"], row["guid"])
        for row in cursor.fetchall()
    ]


def _rows_failed_abandoned(conn) -> list[ExpiredEpisode]:
    """
    Failed episodes with no scheduled retry, past the grace window.

    All three conditions are load-bearing:

      status = 'failed'       terminal, and on the reclaimable allowlist.
      next_retry_at IS NULL   no retry scheduled, so NOT in flight. A failed row
                              with next_retry_at set is in the live queue and
                              must never be touched.
      past the grace window   `requeue_stuck` turns every interrupted
                              'processing' episode into 'failed' on each
                              restart, so reaping on sight would silently
                              discard work moments after a routine restart.

    COALESCE(processed_at, pub_date) dates the row. Every current path into
    'failed' stamps processed_at (`update_status`, and `requeue_stuck` since
    this change), so the pub_date fallback only reaches legacy rows written
    before that was true - which are old by definition. A row that cannot be
    dated at all is skipped, not reclaimed.
    """
    grace_days = max(0, int(getattr(settings, "FAILED_RETENTION_DAYS", 7)))
    cursor = conn.execute("""
        SELECT e.id, e.title, e.guid, s.slug
        FROM episodes e
        JOIN subscriptions s ON e.subscription_id = s.id
        WHERE e.status = 'failed'
          AND e.next_retry_at IS NULL
          AND COALESCE(e.processed_at, e.pub_date) IS NOT NULL
          AND datetime(COALESCE(e.processed_at, e.pub_date))
              < datetime('now', '-' || ? || ' days')
    """, (grace_days,))
    return [
        ExpiredEpisode(row["id"], row["title"], REASON_FAILED_ABANDONED, row["slug"], row["guid"])
        for row in cursor.fetchall()
    ]


def select_expired_episodes(conn) -> list[ExpiredEpisode]:
    """
    Every episode retention would reclaim, deduplicated by id.

    Each selector is independently guarded: one failing (an old SQLite without
    window functions, say) must not take the others down with it, but it also
    must not be silent, so the failure is logged at error level and that
    selector contributes nothing. Failing to select is always safe - it deletes
    less, never more.
    """
    found: dict[int, ExpiredEpisode] = {}
    for selector in (_rows_manual_expired, _rows_over_limit, _rows_failed_abandoned):
        try:
            for item in selector(conn):
                found.setdefault(item.id, item)
        except Exception as e:
            logger.error(f"Retention selector {selector.__name__} failed: {e}")
    return sorted(found.values(), key=lambda item: item.id)


# ---------------------------------------------------------------------------
# Dry-run reporting
# ---------------------------------------------------------------------------

def plan(podcasts_dir: str | None = None) -> tuple[list[ExpiredEpisode], int]:
    """
    What retention would reclaim right now, and how many bytes that frees.

    Read-only. Opens its own connection, deletes nothing, and is the exact same
    selection `cleanup_old_episodes` acts on - that shared definition is the
    point of this module.
    """
    import os

    from app.core.orphan_cleanup import _dir_size, episode_slug_for
    from app.infra.database import get_db_connection

    root = podcasts_dir or settings.PODCASTS_DIR
    with get_db_connection() as conn:
        expired = select_expired_episodes(conn)

    total = 0
    for item in expired:
        if not item.subscription_slug or not item.guid:
            continue
        path = os.path.join(root, item.subscription_slug, episode_slug_for(item.guid))
        if os.path.isdir(path):
            total += _dir_size(path)
    return expired, total


def _main() -> int:
    import argparse

    from app.core.orphan_cleanup import _human_bytes, sweep_orphans

    parser = argparse.ArgumentParser(
        description="Report what retention (and optionally the orphan sweep) would reclaim.",
    )
    # Dry run is the DEFAULT and the only mode this entry point offers. Actually
    # deleting is the background processor's job, so there is no --apply here to
    # fat-finger.
    parser.add_argument("--podcasts-dir", default=None, help="Override the podcasts directory (testing).")
    parser.add_argument("--include-orphans", action="store_true",
                        help="Also run the orphan sweep in dry-run and report a combined total.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    expired, retention_bytes = plan(podcasts_dir=args.podcasts_dir)
    print("RETENTION (dry run - nothing deleted)")
    for item in expired:
        print(f"  expire  [{item.id}] {item.subscription_slug}/{item.title}  <- {item.reason}")
    print(f"  {len(expired)} episode(s), {_human_bytes(retention_bytes)}")

    combined = retention_bytes
    if args.include_orphans:
        res = sweep_orphans(dry_run=True, podcasts_dir=args.podcasts_dir)
        print("\nORPHAN SWEEP (dry run - nothing deleted)")
        for path in res.orphan_dirs:
            print(f"  orphan dir   {path}")
        for path in res.part_files:
            print(f"  stale .part  {path}")
        for err in res.errors:
            print(f"  ERROR        {err}")
        print(f"  {res.summary()}")
        combined += res.bytes_reclaimed
        print(f"\nCOMBINED would reclaim {_human_bytes(combined)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

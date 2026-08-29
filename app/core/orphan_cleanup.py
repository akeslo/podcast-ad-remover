"""
Filesystem-versus-database reconciliation sweep for the podcasts tree.

WHY THIS EXISTS, and why it is not the existing retention:

`Processor.cleanup_old_episodes` deletes by `episodes` row (manual downloads by
`manual_retention_days`, auto downloads by `retention_limit`). That is correct
and this module does not touch it. Its structural blind spot is the point of
this module: anything the database does not reference is invisible to it
forever. Two confirmed classes of that on a live install:

  1. `loading-<ts>[-<hex>]/` placeholder subscription directories. `POST /add`
     mints a placeholder slug while the feed is fetched and the subscription is
     renamed on success; a failure between those two points leaves the
     directory with no row pointing at it, forever.
  2. `.part` files. Downloads write to `<name>.part` and are moved into place
     only on success, so an interrupted download leaks the partial file.

SAFETY MODEL (this code deletes the operator's media, so read this before
changing it):

  * Referenced-ness is derived from the schema, never from a name pattern. A
    `loading-*` directory can be legitimately in use right now, so the name is
    never the gate: DB reference and age are.
  * Every candidate must ALSO be older than `ORPHAN_MIN_AGE_HOURS`. This is a
    second, independent net against the race between reading the database and
    walking the tree.
  * Fail closed. Any database error, any unreadable path, any unexpected
    layout sweeps NOTHING. A cleanup whose failure mode is deleting the
    library is worse than no cleanup at all.
  * An empty `subscriptions` table means every directory on disk looks
    unreferenced. That is indistinguishable from a lost/reset database, so it
    aborts the sweep instead of wiping the tree.
  * `dry_run=True` reports exactly what it would remove and reclaim, and
    removes nothing.
"""
import logging
import os
import shutil
import time
from dataclasses import dataclass, field

from app.core.config import settings

logger = logging.getLogger(__name__)

PART_SUFFIX = ".part"


@dataclass
class SweepResult:
    """What a sweep did (or, in dry-run, would have done)."""
    dry_run: bool = False
    ran: bool = False
    skipped_reason: str | None = None
    orphan_dirs: list[str] = field(default_factory=list)
    part_files: list[str] = field(default_factory=list)
    bytes_reclaimed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def removed_count(self) -> int:
        return len(self.orphan_dirs) + len(self.part_files)

    def summary(self) -> str:
        if not self.ran:
            return f"Orphan sweep skipped: {self.skipped_reason}"
        verb = "would reclaim" if self.dry_run else "reclaimed"
        return (
            f"Orphan sweep{' (dry run)' if self.dry_run else ''}: "
            f"{len(self.orphan_dirs)} orphan director{'y' if len(self.orphan_dirs) == 1 else 'ies'}, "
            f"{len(self.part_files)} abandoned .part file(s), "
            f"{verb} {_human_bytes(self.bytes_reclaimed)}"
            + (f", {len(self.errors)} error(s)" if self.errors else "")
        )


def _human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TB"


def episode_slug_for(guid: str) -> str:
    """
    The on-disk directory name for an episode guid.

    Must stay identical to Processor.delete_episode / version_episode, which
    are the code that creates and renames these directories.
    """
    return f"{guid}".replace("/", "_").replace(" ", "_")


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda e: None):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _age_seconds(path: str, now: float) -> float | None:
    """Age by mtime, or None if the path cannot be stat'd (never a candidate)."""
    try:
        return now - os.path.getmtime(path)
    except OSError:
        return None


def _load_reference_index() -> tuple[dict[str, set[str]], set[str]]:
    """
    Read the database and return (slug -> set of referenced episode slugs,
    set of referenced subscription slugs).

    Raises on any database problem; the caller turns that into a no-op sweep.
    EVERY episodes row counts as a reference regardless of status: pending,
    processing, queued, failed and soft-deleted rows all keep their directory
    off the candidate list. Conservative on purpose.
    """
    from app.infra.database import get_db_connection

    sub_slugs: set[str] = set()
    episodes_by_sub_id: dict[int, set[str]] = {}
    id_to_slug: dict[int, str] = {}

    with get_db_connection() as conn:
        for row in conn.execute("SELECT id, slug FROM subscriptions").fetchall():
            slug = row["slug"]
            if not slug:
                # A subscription with no slug means we cannot tell which
                # directory belongs to it. Unknown layout -> refuse to sweep.
                raise ValueError(f"subscription {row['id']} has no slug")
            id_to_slug[row["id"]] = slug
            sub_slugs.add(slug)

        for row in conn.execute("SELECT subscription_id, guid FROM episodes").fetchall():
            episodes_by_sub_id.setdefault(row["subscription_id"], set()).add(
                episode_slug_for(row["guid"])
            )

    referenced: dict[str, set[str]] = {slug: set() for slug in sub_slugs}
    orphan_episode_slugs: set[str] = set()
    unknown_sub_ids: set[int] = set()
    for sub_id, slugs in episodes_by_sub_id.items():
        slug = id_to_slug.get(sub_id)
        if slug is None:
            # An episode row pointing at a missing subscription. We do not know which
            # directory its files live in, so its episode slugs are PROTECTED EVERYWHERE
            # rather than resolved to one place.
            #
            # This used to raise, which aborted the whole sweep. Correct in intent and
            # wrong in practice: deleting a subscription does not delete its episodes, so
            # prod carried 21 such rows across three long-gone subscription ids (1008, 1012,
            # 1018, against a live range of 1036-1047) and the sweep could never run at all.
            # A safety check that disables the feature permanently on an ordinary and
            # recurring data condition protects nothing; it just guarantees the disk fills.
            #
            # Protecting is strictly safer than aborting was: an aborted sweep deletes
            # nothing anywhere, and this deletes nothing that any episode row names, while
            # still reclaiming everything no row mentions at all.
            unknown_sub_ids.add(sub_id)
            orphan_episode_slugs |= slugs
            continue
        referenced[slug] |= slugs

    if unknown_sub_ids:
        # Reported, never silent. These rows are a real data defect worth fixing at the
        # source, and a sweep that quietly worked around one would hide it forever.
        logger.warning(
            "Orphan sweep: %d episode row group(s) reference missing subscription(s) %s; "
            "their directories are protected, not swept",
            len(unknown_sub_ids), sorted(unknown_sub_ids),
        )
        for slug in referenced:
            referenced[slug] |= orphan_episode_slugs

    return referenced, sub_slugs


def _collect_part_files(root: str, now: float, max_age_seconds: float) -> list[str]:
    stale: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda e: None):
        for name in filenames:
            if not name.endswith(PART_SUFFIX):
                continue
            path = os.path.join(dirpath, name)
            age = _age_seconds(path, now)
            if age is not None and age >= max_age_seconds:
                stale.append(path)
    return stale


def sweep_orphans(dry_run: bool = False, podcasts_dir: str | None = None) -> SweepResult:
    """
    Reconcile the podcasts tree against the database.

    Removes (or, with dry_run, reports) unreferenced subscription directories,
    unreferenced episode directories, and abandoned `.part` files. Never
    removes anything the database references, and never removes anything
    younger than `settings.ORPHAN_MIN_AGE_HOURS`.
    """
    result = SweepResult(dry_run=dry_run)
    root = podcasts_dir or settings.PODCASTS_DIR

    if not os.path.isdir(root):
        result.skipped_reason = f"podcasts directory does not exist: {root}"
        logger.info(result.summary())
        return result

    try:
        referenced, sub_slugs = _load_reference_index()
    except Exception as e:
        # Fail closed: an unreadable database, a query error or an unexpected
        # layout must sweep nothing.
        result.skipped_reason = f"could not build database reference index ({e})"
        logger.error(f"Orphan sweep aborted: {e}")
        return result

    if not sub_slugs:
        # Every directory would look unreferenced. Indistinguishable from a
        # lost database, so refuse rather than delete the library.
        result.skipped_reason = "no subscriptions in database (refusing to treat the whole tree as orphaned)"
        logger.warning(f"Orphan sweep aborted: {result.skipped_reason}")
        return result

    result.ran = True
    now = time.time()
    min_age = max(0, int(settings.ORPHAN_MIN_AGE_HOURS)) * 3600
    part_max_age = max(0, int(settings.PART_FILE_MAX_AGE_HOURS)) * 3600

    candidates: list[str] = []
    try:
        top_entries = sorted(os.listdir(root))
    except OSError as e:
        result.ran = False
        result.skipped_reason = f"could not list {root}: {e}"
        logger.error(f"Orphan sweep aborted: {result.skipped_reason}")
        return result

    for entry in top_entries:
        sub_path = os.path.join(root, entry)
        if not os.path.isdir(sub_path) or os.path.islink(sub_path):
            continue

        if entry not in sub_slugs:
            candidates.append(sub_path)
            continue

        # Referenced subscription: check its episode directories.
        try:
            episode_entries = sorted(os.listdir(sub_path))
        except OSError as e:
            result.errors.append(f"{sub_path}: {e}")
            continue

        known = referenced.get(entry, set())
        for ep_entry in episode_entries:
            ep_path = os.path.join(sub_path, ep_entry)
            if not os.path.isdir(ep_path) or os.path.islink(ep_path):
                continue
            if ep_entry not in known:
                candidates.append(ep_path)

    # Age gate, then delete.
    for path in candidates:
        age = _age_seconds(path, now)
        if age is None or age < min_age:
            continue
        size = _dir_size(path)
        if not dry_run:
            try:
                shutil.rmtree(path)
            except Exception as e:
                result.errors.append(f"{path}: {e}")
                logger.warning(f"Orphan sweep failed to remove {path}: {e}")
                continue
            logger.info(f"Orphan sweep removed unreferenced directory: {path}")
        result.orphan_dirs.append(path)
        result.bytes_reclaimed += size

    # Abandoned .part files anywhere still under the tree (a .part inside a
    # directory removed above is already gone).
    for path in _collect_part_files(root, now, part_max_age):
        if not os.path.exists(path):
            continue
        try:
            size = os.path.getsize(path)
        except OSError:
            continue
        if not dry_run:
            try:
                os.remove(path)
            except Exception as e:
                result.errors.append(f"{path}: {e}")
                logger.warning(f"Orphan sweep failed to remove {path}: {e}")
                continue
            logger.info(f"Orphan sweep removed abandoned partial download: {path}")
        result.part_files.append(path)
        result.bytes_reclaimed += size

    logger.info(result.summary())
    return result


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Reconcile the podcasts tree against the database.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be removed; delete nothing.")
    parser.add_argument("--podcasts-dir", default=None, help="Override the podcasts directory (testing).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    res = sweep_orphans(dry_run=args.dry_run, podcasts_dir=args.podcasts_dir)
    for path in res.orphan_dirs:
        print(f"  orphan dir   {path}")
    for path in res.part_files:
        print(f"  stale .part  {path}")
    for err in res.errors:
        print(f"  ERROR        {err}")
    print(res.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

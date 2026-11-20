"""process_queue must start failed/rate_limited episodes whose retry is due.

get_pending() returns them, but the claim used to require status 'pending',
so every scheduled retry was skipped forever and sat in the Retry Queue.
"""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from app.core.processor import Processor
from app.infra.database import get_db_connection

# test_queue_retry.py stubs Processor.process_queue at class level; keep the
# real one from import time.
REAL_PROCESS_QUEUE = Processor.process_queue

PAST = "2000-01-01 00:00:00"
FUTURE = "2999-01-01 00:00:00"


@pytest.fixture
def queue_rows():
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET status = 'completed' WHERE status = 'processing'")
        sub_id = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            ("http://example.invalid/claim.xml", "Claim Show", "claim-show"),
        ).lastrowid
        spec = [
            ("failed-due", "failed", PAST),
            ("rate-limited-due", "rate_limited", PAST),
            ("failed-later", "failed", FUTURE),
            ("failed-terminal", "failed", None),
        ]
        for guid, status, next_retry_at in spec:
            conn.execute(
                "INSERT INTO episodes (subscription_id, guid, title, original_url,"
                " status, next_retry_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sub_id, guid, guid, f"http://example.invalid/{guid}.mp3", status, next_retry_at),
            )
        conn.commit()
    yield sub_id
    with get_db_connection() as conn:
        conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (sub_id,))
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
        conn.commit()


def _statuses(sub_id):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT guid, status FROM episodes WHERE subscription_id = ?", (sub_id,)
        ).fetchall()
    return {r["guid"]: r["status"] for r in rows}


def test_due_retries_are_claimed_and_started(queue_rows):
    proc = Processor.__new__(Processor)
    from app.infra.repository import EpisodeRepository, SubscriptionRepository
    proc.ep_repo = EpisodeRepository()
    proc.sub_repo = SubscriptionRepository()
    proc._process_single_episode_task = AsyncMock()

    with patch("app.web.router.get_global_settings", return_value={"concurrent_downloads": 5}), \
         patch("app.core.processor.host_mem_used_fraction", return_value=0.1):
        async def run():
            await REAL_PROCESS_QUEUE(proc)
            await asyncio.sleep(0)
        asyncio.run(run())

    started = {c.args[0]["guid"] for c in proc._process_single_episode_task.call_args_list}
    assert started == {"failed-due", "rate-limited-due"}
    assert _statuses(queue_rows) == {
        "failed-due": "processing",
        "rate-limited-due": "processing",
        "failed-later": "failed",
        "failed-terminal": "failed",
    }
    Processor._active_task_ids.clear()

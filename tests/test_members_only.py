"""Members-only YouTube videos: skipped at refresh, ignored on failure, never left 'downloading'."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.youtube_feed import YouTubeFeedManager, is_members_only_error
from app.infra.database import get_db_connection
from app.infra.repository import EpisodeRepository
from tests.test_processor import make_episode, make_mocked_processor, make_subscription

MEMBERS_MSG = ("ERROR: [youtube] abcdefghijk: This video is available to this "
               "channel's members on level: Friends With Benefits")


def _mock_ydl(entries):
    ydl = MagicMock()
    ydl.__enter__.return_value = ydl
    ydl.extract_info.return_value = {"entries": entries}
    return ydl


def test_refresh_skips_members_only_entries():
    entries = [
        {"id": "aaaaaaaaaaa", "title": "Free", "duration": 5},
        {"id": "bbbbbbbbbbb", "title": "NEW BONUS EPISODE", "duration": 5,
         "availability": "subscriber_only"},
        {"id": "ccccccccccc", "title": "Premium", "duration": 5,
         "availability": "premium_only"},
    ]
    with patch("app.core.youtube_feed.yt_dlp.YoutubeDL", return_value=_mock_ydl(entries)):
        videos = YouTubeFeedManager.parse_videos("https://youtube.com/@x")
    assert [v["guid"] for v in videos] == ["aaaaaaaaaaa"]


def test_error_detection():
    assert is_members_only_error(MEMBERS_MSG)
    assert is_members_only_error(MEMBERS_MSG.replace("'", "’"))
    assert is_members_only_error("Join this channel to get access to members-only content")
    assert not is_members_only_error("HTTP Error 403")


@pytest.mark.asyncio
async def test_members_only_download_failure_is_ignored_without_retry():
    processor = make_mocked_processor()
    sub = make_subscription(source_type="youtube", slug="yt-members-test")
    ep = make_episode(guid="members-guid", original_url="https://www.youtube.com/watch?v=abcdefghijk")
    ydl = MagicMock()
    ydl.download.side_effect = RuntimeError(MEMBERS_MSG)
    with patch("app.core.processor.yt_dlp.YoutubeDL", return_value=ydl):
        await processor._process_episode_inner(ep, sub, {"retry_count": 0})
    processor.ep_repo.update_status.assert_any_call(ep.id, "ignored", error="Members-only video")
    processor.ep_repo.update_retry.assert_not_called()


def _insert_stuck(status):
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO subscriptions (id, title, slug, feed_url, source_type) "
                     "VALUES (990, 't', 'yt-t990', 'u', 'youtube')")
        cur = conn.execute(
            "INSERT INTO episodes (subscription_id, guid, title, original_url, status, processing_step, progress) "
            "VALUES (990, ?, 't', 'u', 'processing', 'Downloading video from YouTube...', 5)", (status,))
        conn.commit()
        return cur.lastrowid


@pytest.mark.parametrize("status", ["ignored", "failed"])
def test_terminal_status_clears_stale_step_and_progress(status):
    ep_id = _insert_stuck(status)
    EpisodeRepository().update_status(ep_id, status, error="x")
    with get_db_connection() as conn:
        row = conn.execute("SELECT status, processing_step, progress FROM episodes WHERE id = ?", (ep_id,)).fetchone()
    assert row["status"] == status
    assert row["processing_step"] is None
    assert row["progress"] == 0

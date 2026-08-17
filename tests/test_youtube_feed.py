"""Tests for youtube_feed.py's pub_date fallback.

`extract_flat` mode never populates `upload_date`, so a YouTube episode's
pub_date must never be left null - some podcast clients (Pocket Casts
confirmed) refuse to play an episode whose RSS item has no <pubDate>.
Nothing here touches the network - yt_dlp.YoutubeDL.extract_info is patched.
"""
from datetime import datetime
from unittest.mock import MagicMock, patch

from app.core.youtube_feed import YouTubeFeedManager


def _mock_ydl(entries):
    mock_ydl = MagicMock()
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {"entries": entries}
    return mock_ydl


def test_parse_videos_uses_upload_date_when_present():
    entries = [{"id": "abcdefghijk", "title": "Ep 1", "upload_date": "20260101",
                "duration": 100, "description": ""}]
    with patch("app.core.youtube_feed.yt_dlp.YoutubeDL", return_value=_mock_ydl(entries)):
        videos = YouTubeFeedManager.parse_videos("https://youtube.com/@x")

    assert videos[0]["pub_date"] == datetime(2026, 1, 1)


def test_parse_videos_falls_back_to_timestamp_field():
    entries = [{"id": "abcdefghijk", "title": "Ep 1", "timestamp": 1700000000,
                "duration": 100, "description": ""}]
    with patch("app.core.youtube_feed.yt_dlp.YoutubeDL", return_value=_mock_ydl(entries)):
        videos = YouTubeFeedManager.parse_videos("https://youtube.com/@x")

    assert videos[0]["pub_date"] == datetime.fromtimestamp(1700000000)


def test_parse_videos_never_leaves_pub_date_null():
    """No upload_date, no timestamp fields at all - the extract_flat case
    that broke Pocket Casts playback for every YouTube episode."""
    entries = [{"id": "abcdefghijk", "title": "Ep 1",
                "duration": 100, "description": ""}]
    with patch("app.core.youtube_feed.yt_dlp.YoutubeDL", return_value=_mock_ydl(entries)):
        videos = YouTubeFeedManager.parse_videos("https://youtube.com/@x")

    assert videos[0]["pub_date"] is not None
    assert isinstance(videos[0]["pub_date"], datetime)

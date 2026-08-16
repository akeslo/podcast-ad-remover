"""Tests for search.py's YouTube-channel blending.

Nothing here touches the network - yt_dlp.YoutubeDL.extract_info and the
Podcast Index / iTunes search paths are all patched out.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.search import PodcastSearcher


def _entry(channel, channel_id, thumb_url="https://img.example/x.jpg"):
    return {
        "channel": channel,
        "channel_id": channel_id,
        "channel_url": f"https://www.youtube.com/channel/{channel_id}",
        "thumbnails": [{"url": thumb_url}],
    }


@pytest.mark.asyncio
async def test_search_youtube_dedupes_by_channel():
    entries = [
        _entry("Show A", "UC_A"),
        _entry("Show A", "UC_A"),
        _entry("Show B", "UC_B"),
    ]
    mock_ydl = MagicMock()
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {"entries": entries}

    with patch("app.core.search.yt_dlp.YoutubeDL", return_value=mock_ydl):
        results = await PodcastSearcher.search_youtube("test", limit=5)

    assert [r["title"] for r in results] == ["YT: Show A", "YT: Show B"]
    assert results[0]["feed_url"] == "https://www.youtube.com/channel/UC_A/videos"


@pytest.mark.asyncio
async def test_search_youtube_respects_limit():
    entries = [_entry(f"Show {i}", f"UC_{i}") for i in range(10)]
    mock_ydl = MagicMock()
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.return_value = {"entries": entries}

    with patch("app.core.search.yt_dlp.YoutubeDL", return_value=mock_ydl):
        results = await PodcastSearcher.search_youtube("test", limit=3)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_search_youtube_returns_empty_on_extraction_failure():
    mock_ydl = MagicMock()
    mock_ydl.__enter__.return_value = mock_ydl
    mock_ydl.extract_info.side_effect = RuntimeError("network down")

    with patch("app.core.search.yt_dlp.YoutubeDL", return_value=mock_ydl):
        results = await PodcastSearcher.search_youtube("test", limit=5)

    assert results == []


@pytest.mark.asyncio
async def test_search_blends_podcast_and_youtube_results():
    podcast_results = [{"title": "A Podcast", "feed_url": "https://feeds.example/a.xml",
                         "image": "", "description": ""}]
    youtube_results = [{"title": "YT: A Channel", "feed_url": "https://www.youtube.com/channel/UC_X/videos",
                         "image": "", "description": "YouTube channel"}]

    with patch("app.core.discovery.search", new=AsyncMock(return_value=podcast_results)), \
         patch.object(PodcastSearcher, "search_youtube", new=AsyncMock(return_value=youtube_results)):
        results = await PodcastSearcher.search("test")

    assert results == podcast_results + youtube_results


@pytest.mark.asyncio
async def test_search_blends_youtube_even_when_podcast_index_falls_back_to_itunes():
    youtube_results = [{"title": "YT: A Channel", "feed_url": "https://www.youtube.com/channel/UC_X/videos",
                         "image": "", "description": "YouTube channel"}]

    with patch("app.core.discovery.search", new=AsyncMock(side_effect=RuntimeError("not configured"))), \
         patch.object(PodcastSearcher, "search_itunes", new=AsyncMock(return_value=[])), \
         patch.object(PodcastSearcher, "search_youtube", new=AsyncMock(return_value=youtube_results)):
        results = await PodcastSearcher.search("test")

    assert results == youtube_results

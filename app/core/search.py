"""Podcast search.

Podcast Index is the primary source (it returns the publisher's real show
description; iTunes' search endpoint returns only the artist name). iTunes
remains as an automatic fallback for two cases that both leave a working
search box rather than a broken one:

* Podcast Index is not configured - a fresh install with no API key still has
  a usable search box.
* Podcast Index errors, or returns nothing at all.

The iTunes path is deliberately kept, not deleted. It is the degraded mode,
and it is the only source that needs no credentials.
"""

import asyncio
import logging
from typing import Dict, List

import httpx
import yt_dlp

logger = logging.getLogger(__name__)


class PodcastSearcher:
    BASE_URL = "https://itunes.apple.com/search"

    # Cap on YouTube channel results blended into a text search. Kept small -
    # this is a supplementary source alongside the podcast directory, not the
    # primary one, and each result costs a yt_dlp extraction.
    YOUTUBE_RESULT_LIMIT = 5

    @staticmethod
    async def search(term: str, limit: int = 10) -> List[Dict]:
        """Search for podcasts, preferring Podcast Index over iTunes, with
        YouTube channels blended in so a name search (not just a pasted
        channel URL) can surface a YouTube result to add."""
        if not term:
            return []

        try:
            from app.core import discovery

            results = await discovery.search(term, limit=limit)
            if not results:
                logger.info(
                    "Podcast Index returned no results for a search; "
                    "falling back to iTunes."
                )
                results = await PodcastSearcher.search_itunes(term, limit=limit)
        except Exception as e:
            # Includes PodcastIndexNotConfigured. Degrading here is the point:
            # search must keep working without an API key. Discovery's own
            # trending/browse routes do NOT degrade - they surface the error.
            logger.warning(
                "Podcast Index search unavailable (%s); falling back to iTunes.",
                e,
            )
            results = await PodcastSearcher.search_itunes(term, limit=limit)

        youtube_results = await PodcastSearcher.search_youtube(
            term, limit=PodcastSearcher.YOUTUBE_RESULT_LIMIT
        )
        return results + youtube_results

    @staticmethod
    async def search_itunes(term: str, limit: int = 10) -> List[Dict]:
        """Search for podcasts using the iTunes API (fallback source)."""
        params = {
            "term": term,
            "media": "podcast",
            "entity": "podcast",
            "limit": limit,
            # Locale: English, USA - matches the discovery source's lang=en.
            "country": "US",
            "lang": "en_us",
        }

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(
                    PodcastSearcher.BASE_URL, params=params, timeout=10.0
                )
                resp.raise_for_status()
                data = resp.json()

                results = []
                for item in data.get("results", []):
                    feed_url = item.get("feedUrl")
                    title = item.get("collectionName")
                    if not feed_url or not title:
                        continue
                    results.append({
                        "title": title,
                        "feed_url": feed_url,
                        "image": item.get("artworkUrl600") or "",
                        # iTunes' search endpoint carries no show description.
                        "description": item.get("artistName") or "",
                    })
                return results
            except Exception as e:
                logger.warning("iTunes search failed: %s", e)
                return []

    @staticmethod
    async def search_youtube(term: str, limit: int = 5) -> List[Dict]:
        """Search YouTube by name and return the channels behind the top
        video results, deduped. yt_dlp has no "search channels" mode, so
        this searches videos and collapses to their uploading channel -
        same source of truth youtube_feed.py already relies on. Runs off the
        event loop thread since yt_dlp's extraction is a blocking network
        call, same as the interactive request path it's called from."""

        def _search() -> List[Dict]:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": "in_playlist",
                "skip_download": True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    # Over-fetch videos since multiple results often share a
                    # channel; the search is only "done" once `limit` distinct
                    # channels are collected, not `limit` videos.
                    info = ydl.extract_info(
                        f"ytsearch{limit * 4}:{term}", download=False
                    )
            except Exception as e:
                logger.warning("YouTube search failed: %s", e)
                return []

            entries = (info or {}).get("entries") or []
            seen_channels = set()
            results: List[Dict] = []
            for entry in entries:
                if not entry:
                    continue
                channel_id = entry.get("channel_id")
                channel_url = entry.get("channel_url") or (
                    f"https://www.youtube.com/channel/{channel_id}"
                    if channel_id
                    else None
                )
                if not channel_url or channel_url in seen_channels:
                    continue
                seen_channels.add(channel_url)

                thumbnails = entry.get("thumbnails") or []
                image = thumbnails[-1]["url"] if thumbnails else ""
                title = entry.get("channel") or entry.get("uploader") or "Unknown Channel"

                results.append({
                    "title": f"YT: {title}",
                    # /add's source_type detection matches on the URL text
                    # (app/web/router.py), so a plain channel URL is enough -
                    # no separate source_type field needs to travel with it.
                    "feed_url": channel_url.rstrip("/") + "/videos",
                    "image": image,
                    "description": "YouTube channel",
                })
                if len(results) >= limit:
                    break
            return results

        return await asyncio.to_thread(_search)

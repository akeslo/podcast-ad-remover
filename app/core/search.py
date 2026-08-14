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

import logging
from typing import Dict, List

import httpx

logger = logging.getLogger(__name__)


class PodcastSearcher:
    BASE_URL = "https://itunes.apple.com/search"

    @staticmethod
    async def search(term: str, limit: int = 10) -> List[Dict]:
        """Search for podcasts, preferring Podcast Index over iTunes."""
        if not term:
            return []

        try:
            from app.core import discovery

            results = await discovery.search(term, limit=limit)
            if results:
                return results
            logger.info(
                "Podcast Index returned no results for a search; "
                "falling back to iTunes."
            )
        except Exception as e:
            # Includes PodcastIndexNotConfigured. Degrading here is the point:
            # search must keep working without an API key. Discovery's own
            # trending/browse routes do NOT degrade - they surface the error.
            logger.warning(
                "Podcast Index search unavailable (%s); falling back to iTunes.",
                e,
            )

        return await PodcastSearcher.search_itunes(term, limit=limit)

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

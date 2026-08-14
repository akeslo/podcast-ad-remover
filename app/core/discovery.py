"""Podcast discovery via the Podcast Index API.

Sibling of `app/core/search.py`. Everything here returns the same card shape
the search path already produces - `{title, feed_url, image, description}` -
so the dashboard can render search results, trending feeds and category
listings through one renderer.

Two deliberate choices worth stating up front:

1. **Failures raise, they never degrade to `[]`.** An empty list is a valid
   answer ("nothing trending in this category"), so using it to signal an
   outage makes a dead API key indistinguishable from a quiet category and
   hides a real outage from the operator forever. The routes turn these
   exceptions into a 502 with a readable message.
2. **Descriptions are flattened to plain text here, server-side.** Podcast
   Index returns raw feed-supplied HTML (`<p>`, entities, and whatever else a
   third-party publisher put in their RSS). That string is rendered into the
   dashboard, so it is stripped at the boundary rather than trusted downstream.
"""

import hashlib
import html
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

# Podcast Index requires a non-empty, identifying User-Agent and rejects
# requests without one.
USER_AGENT = "PodcastAdRemover/1.0"
BASE_URL = "https://api.podcastindex.org/api/1.0"

# Locale for discovery results: English-language feeds.
DEFAULT_LANGUAGE = "en"

# How long a cached discovery payload stays fresh. Trending moves slowly and
# the category list barely moves at all; the API is a shared third-party
# resource and the dashboard auto-refreshes, so an hour is generous.
CACHE_TTL_SECONDS = 3600

DEFAULT_LIMIT = 40
MAX_LIMIT = 100

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


class PodcastIndexError(RuntimeError):
    """Any failure talking to the Podcast Index API."""


class PodcastIndexNotConfigured(PodcastIndexError):
    """No API key/secret is configured, in the database or the environment."""


def _plain_text(value: Optional[str]) -> str:
    """Flatten feed-supplied HTML to a single line of plain text."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def get_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Resolve the Podcast Index credentials.

    Operator-editable database settings win over the environment, mirroring
    how `app.core.ai_services.AdDetector.create_provider` resolves provider API
    keys: `.env` seeds an install, the admin UI overrides it afterwards.
    """
    db_key = db_secret = None
    try:
        from app.infra.database import get_db_connection

        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT podcast_index_api_key, podcast_index_api_secret "
                "FROM app_settings WHERE id = 1"
            ).fetchone()
        if row:
            db_key = row["podcast_index_api_key"]
            db_secret = row["podcast_index_api_secret"]
    except Exception as e:  # pragma: no cover - DB unavailable/pre-migration
        logger.warning("Could not read Podcast Index settings from DB: %s", e)

    key = db_key.strip() if isinstance(db_key, str) and db_key.strip() else None
    secret = (
        db_secret.strip()
        if isinstance(db_secret, str) and db_secret.strip()
        else None
    )

    if not key:
        key = settings.PODCAST_INDEX_API_KEY or None
    if not secret:
        secret = settings.PODCAST_INDEX_API_SECRET or None

    return key, secret


def build_auth_headers(
    api_key: str, api_secret: str, unix_time: Optional[int] = None
) -> Dict[str, str]:
    """Build the four headers every Podcast Index request needs.

    The scheme is `sha1(key + secret + unixTime)`, lowercase hex, sent as the
    `Authorization` header alongside the key and the exact same timestamp used
    to compute it. Their side allows five minutes of clock skew, so the
    timestamp must be seconds (not milliseconds) and must be the *same* value
    that went into the hash - computing it twice is how this drifts.
    """
    if not api_key or not api_secret:
        raise PodcastIndexNotConfigured(
            "Podcast Index API key and secret are not configured."
        )

    stamp = str(int(unix_time if unix_time is not None else time.time()))
    digest = hashlib.sha1(
        f"{api_key}{api_secret}{stamp}".encode("utf-8")
    ).hexdigest()

    return {
        "X-Auth-Key": api_key,
        "X-Auth-Date": stamp,
        "Authorization": digest,
        "User-Agent": USER_AGENT,
    }


def _map_feed(item: Dict) -> Optional[Dict]:
    """Map a Podcast Index feed object onto the app's podcast-card shape.

    Field names are taken from live responses: a feed carries `url` (the RSS
    URL), `title`, `image` and `artwork` (artwork is the higher-resolution of
    the two when both are present), `description` and `author`.
    """
    feed_url = item.get("url") or item.get("originalUrl")
    title = item.get("title")
    if not feed_url or not title:
        # A feed with no usable RSS URL cannot be subscribed to, so it is not
        # a result - it is noise that would render an Add button that 400s.
        return None

    description = _plain_text(item.get("description")) or _plain_text(
        item.get("author")
    )

    return {
        "title": title,
        "feed_url": feed_url,
        "image": item.get("artwork") or item.get("image") or "",
        "description": description,
    }


def _map_feeds(payload: Dict) -> List[Dict]:
    feeds = payload.get("feeds")
    if not isinstance(feeds, list):
        return []
    mapped = [_map_feed(item) for item in feeds if isinstance(item, dict)]
    return [m for m in mapped if m]


def _clamp_limit(limit: int) -> int:
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(limit, MAX_LIMIT))


class PodcastIndexClient:
    """Thin async client for the handful of Podcast Index endpoints we use."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        timeout: float = 10.0,
    ):
        if api_key is None and api_secret is None:
            api_key, api_secret = get_credentials()
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def _headers(self) -> Dict[str, str]:
        return build_auth_headers(self.api_key, self.api_secret)

    async def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        if not self.is_configured:
            raise PodcastIndexNotConfigured(
                "Podcast Index API key and secret are not configured. "
                "Set them under Admin - System, or via PODCAST_INDEX_API_KEY "
                "and PODCAST_INDEX_API_SECRET."
            )

        url = f"{BASE_URL}/{path.lstrip('/')}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    url,
                    params=params or {},
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            # Never log the response body or the headers: the request carries
            # the API key, and the body can be large. Status + path only.
            raise PodcastIndexError(
                f"Podcast Index returned HTTP {e.response.status_code} for /{path}"
            ) from e
        except httpx.HTTPError as e:
            raise PodcastIndexError(
                f"Could not reach Podcast Index for /{path}: {e}"
            ) from e
        except ValueError as e:
            raise PodcastIndexError(
                f"Podcast Index returned a non-JSON response for /{path}"
            ) from e

        if not isinstance(data, dict):
            raise PodcastIndexError(
                f"Podcast Index returned an unexpected payload for /{path}"
            )

        # Their `status` is the string "true"/"false", not a bool.
        status_value = str(data.get("status", "true")).lower()
        if status_value in ("false", "0"):
            raise PodcastIndexError(
                f"Podcast Index reported an error for /{path}: "
                f"{data.get('description') or 'unknown error'}"
            )
        return data

    async def get_trending(
        self, limit: int = DEFAULT_LIMIT, category: Optional[str] = None
    ) -> List[Dict]:
        """Currently trending podcasts, newest-trend first."""
        params = {"max": _clamp_limit(limit), "lang": DEFAULT_LANGUAGE}
        if category:
            params["cat"] = category
        return _map_feeds(await self._get("podcasts/trending", params))

    async def get_categories(self) -> List[Dict]:
        """The full category taxonomy as `[{id, name}]`."""
        data = await self._get("categories/list")
        rows = data.get("feeds")
        if not isinstance(rows, list):
            return []
        categories = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            cid, name = row.get("id"), row.get("name")
            if cid is None or not name:
                continue
            categories.append({"id": cid, "name": str(name)})
        return categories

    async def get_by_category(
        self, category_id, limit: int = DEFAULT_LIMIT
    ) -> List[Dict]:
        """Podcasts within one category.

        The trending endpoint's `cat` filter accepts either a category name or
        a numeric id (both verified against the live API), so the id the
        category list handed the UI can be passed straight through.
        """
        return await self.get_trending(limit=limit, category=str(category_id))


# ---------------------------------------------------------------------------
# Read-through cache
#
# Discovery data is ephemeral and shared by every visitor, so it is cached in
# its own `discovery_cache` table. It is deliberately NOT written into
# `subscriptions`: rows there are things the operator subscribed to, and the
# dashboard and the processor's feed-check loop read that table unfiltered, so
# a not-yet-subscribed podcast parked there would show up as a subscription
# and get polled.
# ---------------------------------------------------------------------------


async def _cached(cache_key: str, fetcher, ttl: int = CACHE_TTL_SECONDS):
    from app.infra.database import (
        get_discovery_cache,
        set_discovery_cache,
    )

    cached = get_discovery_cache(cache_key, ttl)
    if cached is not None:
        return cached

    payload = await fetcher()
    try:
        set_discovery_cache(cache_key, payload)
    except Exception as e:  # pragma: no cover - cache write is best-effort
        logger.warning("Could not cache discovery payload %r: %s", cache_key, e)
    return payload


async def fetch_trending(limit: int = DEFAULT_LIMIT) -> List[Dict]:
    limit = _clamp_limit(limit)
    client = PodcastIndexClient()
    return await _cached(
        f"trending:{limit}", lambda: client.get_trending(limit=limit)
    )


async def fetch_categories() -> List[Dict]:
    client = PodcastIndexClient()
    return await _cached("categories", client.get_categories)


async def fetch_by_category(category_id, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    limit = _clamp_limit(limit)
    client = PodcastIndexClient()
    return await _cached(
        f"category:{category_id}:{limit}",
        lambda: client.get_by_category(category_id, limit=limit),
    )


async def search(term: str, limit: int = 10) -> List[Dict]:
    """Search Podcast Index by term. Not cached - the key space is unbounded."""
    client = PodcastIndexClient()
    data = await client._get(
        "search/byterm",
        {"q": term, "max": _clamp_limit(limit), "lang": DEFAULT_LANGUAGE},
    )
    return _map_feeds(data)


__all__ = [
    "CACHE_TTL_SECONDS",
    "PodcastIndexClient",
    "PodcastIndexError",
    "PodcastIndexNotConfigured",
    "build_auth_headers",
    "fetch_by_category",
    "fetch_categories",
    "fetch_trending",
    "get_credentials",
    "search",
]
"""Discovery routes: trending podcasts and category browsing.

Mounted at `/api` from `app/main.py`, alongside `app/api/subscriptions.py`.

These are read-only GETs, gated with the same `require_user_action` dependency
the existing `POST /api/search` uses. On a GET that dependency reduces to
`require_auth` - `require_same_origin` returns early for safe methods - so it
is a real boundary when `auth_enabled = 1` and a deliberate no-op in
standalone mode, where the IP allowlist is the boundary. The gate is not
decoration here even in standalone mode: every uncached call spends the
install's third-party API quota.

Nothing here writes to `subscriptions`. Results are cached in
`discovery_cache` and the operator subscribes through the existing `POST /add`
flow, which runs `validate_feed_url()`.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.discovery import (
    DEFAULT_LIMIT,
    PodcastIndexError,
    PodcastIndexNotConfigured,
    fetch_by_category,
    fetch_categories,
    fetch_trending,
)
from app.web.router import require_user_action

logger = logging.getLogger(__name__)

router = APIRouter()


def _handle(e: Exception, what: str):
    """Turn a discovery failure into an honest HTTP error.

    Never an empty list: a silent `[]` makes an expired key, a dead upstream
    and a genuinely empty category look identical, which is how an outage goes
    unnoticed for weeks.
    """
    if isinstance(e, PodcastIndexNotConfigured):
        logger.warning("Discovery request for %s with no credentials set", what)
        raise HTTPException(
            status_code=503,
            detail=(
                "Podcast discovery is not configured. Add a Podcast Index API "
                "key and secret under Admin - System."
            ),
        )
    logger.error("Discovery request for %s failed: %s", what, e)
    raise HTTPException(status_code=502, detail=f"Podcast discovery failed: {e}")


@router.get("/trending", dependencies=[Depends(require_user_action)])
async def trending(limit: int = Query(DEFAULT_LIMIT, ge=1, le=100)):
    """Currently trending podcasts, as podcast cards."""
    try:
        return await fetch_trending(limit=limit)
    except PodcastIndexError as e:
        _handle(e, "trending")


@router.get("/categories", dependencies=[Depends(require_user_action)])
async def categories():
    """The Podcast Index category taxonomy, as `[{id, name}]`."""
    try:
        return await fetch_categories()
    except PodcastIndexError as e:
        _handle(e, "categories")


@router.get("/categories/{category_id}", dependencies=[Depends(require_user_action)])
async def category_podcasts(
    category_id: int, limit: int = Query(DEFAULT_LIMIT, ge=1, le=100)
):
    """Podcasts within one category, as podcast cards.

    `category_id` is typed as `int` on purpose: it is interpolated into an
    upstream query parameter, and the path type is what stops an arbitrary
    string getting there.
    """
    try:
        return await fetch_by_category(category_id, limit=limit)
    except PodcastIndexError as e:
        _handle(e, f"category {category_id}")

"""
Dynamic video serving routes with view tracking.
"""
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import os
import time
import hashlib
import logging

from app.core.config import settings
from app.infra.repository import EpisodeRepository, SubscriptionRepository
from app.web.auth_utils import get_client_ip

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory cache for deduplication (IP+episode -> last_access_time)
_view_cache: dict[str, float] = {}
DEDUPE_WINDOW_SECONDS = 2 * 60 * 60  # 2 hours


def _get_cache_key(ip: str, episode_id: int) -> str:
    """Generate a cache key for deduplication."""
    return hashlib.md5(f"{ip}:{episode_id}".encode()).hexdigest()


def _should_count_view(ip: str, episode_id: int) -> bool:
    """Check if this request should count as a new view (deduplication)."""
    global _view_cache

    cache_key = _get_cache_key(ip, episode_id)
    now = time.time()

    # Clean old entries (prevent memory leak)
    expired = [k for k, v in _view_cache.items() if now - v > DEDUPE_WINDOW_SECONDS]
    for k in expired:
        del _view_cache[k]

    # Check if already counted recently
    if cache_key in _view_cache:
        last_access = _view_cache[cache_key]
        if now - last_access < DEDUPE_WINDOW_SECONDS:
            return False

    _view_cache[cache_key] = now
    return True


def _is_first_byte_request(request: Request) -> bool:
    """Check if this is the first request for the file (not a mid-stream Range request)."""
    range_header = request.headers.get("Range", "")
    if not range_header:
        return True  # No Range header = full file request

    # Parse Range: bytes=START-END
    if range_header.startswith("bytes="):
        range_spec = range_header[6:]
        if "-" in range_spec:
            start = range_spec.split("-")[0]
            # Count as first request if starting at 0 or very beginning
            if start == "" or start == "0" or int(start) < 1024:
                return True
    return False


@router.get("/video/{path:path}")
async def serve_video(path: str, request: Request):
    """
    Serve video files dynamically with view tracking.

    Path format: {subscription_slug}/{episode_guid}/{filename}
    """
    # Build full file path
    file_path = Path(settings.PODCASTS_DIR) / path

    # Security: ensure the resolved path is within PODCASTS_DIR - checked
    # before any filesystem probe (exists()/is_file()) touches the
    # unvalidated path, so a traversal attempt never gets to run against the
    # filesystem at all. Mirrors audio_routes.py's serve_audio.
    try:
        file_path.resolve().relative_to(Path(settings.PODCASTS_DIR).resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    if not file_path.exists():
        logger.warning(f"Video file not found: {file_path}")
        raise HTTPException(status_code=404, detail="Video file not found")

    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not a file")

    # Track view if this is a first-byte request
    if _is_first_byte_request(request):
        try:
            # Try to find episode by path
            ep_repo = EpisodeRepository()
            sub_repo = SubscriptionRepository()

            # Parse path to extract subscription slug and filename
            path_parts = path.split("/")
            if len(path_parts) >= 1:
                subscription_slug = path_parts[0]
                filename = path_parts[-1]

                # Find subscription
                sub = sub_repo.get_by_slug(subscription_slug)
                if sub:
                    # Find episode by filename
                    episode = ep_repo.get_by_subscription_and_filename(sub.id, filename)
                    if episode:
                        # Get client IP (proxy headers honored only when
                        # TRUST_PROXY_HEADERS is set - see auth_utils.get_client_ip)
                        client_ip = get_client_ip(request, settings.TRUST_PROXY_HEADERS)

                        # Deduplicated view count
                        if _should_count_view(client_ip, episode.id):
                            ep_repo.increment_listen_count(episode.id)
                            logger.info(f"Tracked video view: episode={episode.id} ({filename}), IP={client_ip}")
                        else:
                            logger.debug(f"Deduplicated view: episode={episode.id}, IP={client_ip}")
        except Exception as e:
            logger.error(f"Error tracking video view: {e}")
            # Don't fail the request if tracking fails

    # Determine media type
    media_type = "video/mp4"
    if file_path.suffix.lower() == ".webm":
        media_type = "video/webm"
    elif file_path.suffix.lower() == ".mkv":
        media_type = "video/x-matroska"

    # Use FileResponse which handles Range requests automatically
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=file_path.name
    )

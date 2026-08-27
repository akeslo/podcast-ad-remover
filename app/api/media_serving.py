"""
Shared plumbing for the /audio and /video streaming routes.

audio_routes.py and video_routes.py each carried their own copy of the
first-byte-request Range parser and the IP-dedup view/listen cache. The
duplication meant every parity bug had to be found and fixed twice — it
already happened once for the malformed-Range-header 500 (both files patched
identically in the same commit) and again for feed-token auth gating.
Consolidating here means the next such bug is fixed in one place.
"""
import time
import hashlib

from fastapi import Request


def get_cache_key(ip: str, episode_id: int) -> str:
    """Generate a cache key for view/listen deduplication."""
    return hashlib.md5(f"{ip}:{episode_id}".encode()).hexdigest()


def should_count_access(cache: dict[str, float], ip: str, episode_id: int, dedupe_window_seconds: int) -> bool:
    """Check if this request should count as a new view/listen (deduplication).

    Mutates `cache` in place (expiring stale entries, recording this access)
    so callers keep their own module-level cache dict.
    """
    cache_key = get_cache_key(ip, episode_id)
    now = time.time()

    # Clean old entries (prevent memory leak)
    expired = [k for k, v in cache.items() if now - v > dedupe_window_seconds]
    for k in expired:
        del cache[k]

    # Check if already counted recently
    if cache_key in cache:
        last_access = cache[cache_key]
        if now - last_access < dedupe_window_seconds:
            return False

    cache[cache_key] = now
    return True


def is_first_byte_request(request: Request) -> bool:
    """Check if this is the first request for the file (not a mid-stream Range request)."""
    range_header = request.headers.get("Range", "")
    if not range_header:
        return True  # No Range header = full file request

    # Parse Range: bytes=START-END
    if range_header.startswith("bytes="):
        range_spec = range_header[6:]
        if "-" in range_spec:
            start = range_spec.split("-")[0]
            # Count as first request if starting at 0 or very beginning.
            # A malformed start (e.g. "bytes=abc-100") is not a first-byte
            # request; never let it raise out of here into a 500.
            if start == "" or start == "0":
                return True
            try:
                return int(start) < 1024
            except ValueError:
                return False
    return False

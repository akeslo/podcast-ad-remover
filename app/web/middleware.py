"""Feed request verification.

Feed URLs are authenticated with a per-user (or, in standalone mode, a global)
random *feed token*. The token is a bearer credential dedicated to podcast
clients - it is never the operator's account password, so a leaked feed URL
does not leak a login.

Two transports are accepted, both carrying the same token:

* ``?auth=<token>`` on the URL (what generated feed URLs use), and
* HTTP ``Authorization: Basic`` with the token in the *password* field, so
  podcast clients that already stored Basic credentials keep working.

There is deliberately no fallback to the old password-based credential: that
path required reading a plaintext password out of ``app_settings``, which is
exactly what this change removes. Old feed URLs stop working and must be
regenerated.
"""

import base64
import binascii
import hmac

from fastapi import Request, status
from fastapi.responses import Response

_UNAUTHORIZED_HEADERS = {'WWW-Authenticate': 'Basic realm="Podcast Feeds"'}

_TRUTHY = {'1', 'true', 'yes', 'on', 't', 'y'}


def _unauthorized() -> Response:
    return Response(
        status_code=status.HTTP_401_UNAUTHORIZED,
        headers=dict(_UNAUTHORIZED_HEADERS),
    )


def _is_enabled(value) -> bool:
    """Truthiness for a setting that may be an int, a bool, or a string."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    return str(value).strip().lower() in _TRUTHY


def _candidate_tokens(request: Request) -> list:
    """Every string the caller could plausibly have meant as the feed token.

    Each candidate is verified with a constant-time comparison (or a
    constant-time lookup), so offering more than one costs nothing in secrecy;
    it only buys transport compatibility.
    """
    candidates = []

    def add(value):
        if isinstance(value, str) and value and value not in candidates:
            candidates.append(value)

    auth_header = request.headers.get('Authorization') or ''
    if auth_header.startswith('Basic '):
        encoded = auth_header[len('Basic '):].strip()
        decoded = _b64_decode(encoded)
        if decoded is not None and ':' in decoded:
            # Basic auth: the token lives in the password field.
            add(decoded.split(':', 1)[1])

    raw_query_value = request.query_params.get('auth')
    if raw_query_value:
        # Current generated URLs carry the bare token.
        add(raw_query_value)
        # Tolerate the legacy base64("user:secret") envelope shape, treating
        # the secret half as a token candidate. No plaintext password is ever
        # read from the database to satisfy it.
        decoded = _b64_decode(raw_query_value)
        if decoded is not None and ':' in decoded:
            add(decoded.split(':', 1)[1])

    return candidates


def _tokens_equal(presented, expected) -> bool:
    """Constant-time equality for two token strings.

    ``hmac.compare_digest`` raises ``TypeError`` when either str argument
    contains a non-ASCII character, which turned a bogus credential into an
    unauthenticated 500 with a stack trace. Comparing the UTF-8 encodings
    instead keeps the comparison constant-time (bytes are always accepted) and
    turns every bad credential into a plain, quiet mismatch.
    """
    if not isinstance(presented, str) or not isinstance(expected, str):
        return False
    try:
        return hmac.compare_digest(presented.encode('utf-8'), expected.encode('utf-8'))
    except (UnicodeEncodeError, TypeError):
        return False


def _b64_decode(value: str):
    if not value:
        return None
    try:
        return base64.b64decode(value, validate=True).decode('utf-8')
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None


#: Every prefix that serves feed or audio content. ``/feed/`` is included so
#: the unified feed (``/feed/unified``, ``/feed/unified.xml``) and any future
#: route under it are gated by this middleware rather than by a hand-rolled
#: check inside the route - two independently maintained gates had already
#: drifted apart on which transports they accept. ``/video/`` was missing
#: here entirely (2026-08-22): the *other* auth gate in app/web/auth.py used
#: to reject it outright (redirect loop, video never played), and adding it
#: to that gate's public-path exemption without also adding it here would
#: have left video files served to anyone with the URL, no token required -
#: the two gates must move together for every media prefix.
_PROTECTED_PREFIXES = ('/feeds/', '/audio/', '/feed/', '/video/')


async def feed_auth_middleware(request: Request, call_next):
    """Protect ``/feeds/*``, ``/feed/*`` and ``/audio/*`` with feed-token
    verification.

    * per-user auth enabled  -> the token must resolve to a user
    * per-user auth disabled -> the token must equal the global feed token

    Fails closed on every ambiguity, including "feed auth enabled but no token
    configured" and "the settings row could not be read at all".
    """
    path = request.url.path

    if not any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES):
        return await call_next(request)

    from app.web.router import get_global_settings

    # `get_global_settings()` returns {} both when the settings row is missing
    # and when the read blew up. An empty mapping therefore means "unknown",
    # not "auth is off" - resolving `.get('enable_feed_auth')` to None on it
    # would disable the gate on exactly the failure it should hold shut for.
    try:
        settings = get_global_settings()
    except Exception:
        settings = None

    if not isinstance(settings, dict) or 'enable_feed_auth' not in settings:
        # Settings unreadable: treat as enabled-and-unconfigured and deny.
        return _unauthorized()

    if not _is_enabled(settings.get('enable_feed_auth')):
        return await call_next(request)

    candidates = _candidate_tokens(request)
    if not candidates:
        # Missing, empty, or unparseable credential.
        return _unauthorized()

    from app.infra.database import find_user_by_feed_token, get_global_feed_token

    if _is_enabled(settings.get('auth_enabled')):
        for token in candidates:
            if find_user_by_feed_token(token):
                return await call_next(request)
        return _unauthorized()

    expected_token = get_global_feed_token()
    if not expected_token:
        # Feed auth is enabled but no global token was ever generated. That is
        # a misconfiguration of an access-control gate, and the old code failed
        # OPEN here - serving every feed and audio file to the internet with no
        # credential at all. Fail closed instead: an operator sees 401s and
        # fixes the config; nobody sees a silent exposure.
        return _unauthorized()

    for token in candidates:
        if _tokens_equal(token, expected_token):
            return await call_next(request)

    return _unauthorized()

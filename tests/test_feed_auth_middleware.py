"""Feed request verification tests.

These exercise the security boundary in `app.web.middleware`: only a valid feed
token may reach /feeds/* or /audio/*, and every ambiguous case fails closed.

The database layer is stubbed - the point here is the middleware's decision
logic, not sqlite.
"""
import asyncio
import base64

import pytest

from app.web import middleware as mw

FEED_PATH = "/feeds/example.xml"

GLOBAL_TOKEN = "global-feed-token-abcdefghijklmnop"
USER_TOKEN = "user-feed-token-qrstuvwxyz012345"


@pytest.fixture
def feed_auth(monkeypatch):
    """Configure the feed-auth gate and stub the database contract.

    Returns a callable: feed_auth(enable_feed_auth=..., auth_enabled=...,
    global_token=...).
    """
    import app.infra.database as db
    import app.web.router as router

    def configure(enable_feed_auth=True, auth_enabled=False, global_token=GLOBAL_TOKEN):
        monkeypatch.setattr(
            router,
            "get_global_settings",
            lambda: {
                "enable_feed_auth": 1 if enable_feed_auth else 0,
                "auth_enabled": 1 if auth_enabled else 0,
            },
        )
        monkeypatch.setattr(db, "get_global_feed_token", lambda: global_token)
        monkeypatch.setattr(
            db,
            "find_user_by_feed_token",
            lambda token: {"id": 1, "username": "kes"} if token == USER_TOKEN else None,
        )

    return configure


def basic(username, password):
    raw = f"{username}:{password}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}


ALLOWED = object()


def call_middleware(path=FEED_PATH, query="", headers=None):
    """Drive the middleware directly.

    Downstream feed/audio handlers need real subscriptions and files, so
    running an *allowed* request through TestClient would fail for reasons that
    have nothing to do with authentication. Here `call_next` is a sentinel:
    the result is either ALLOWED (the middleware let the request through) or
    the Response it produced instead.
    """
    from starlette.requests import Request

    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1"))
        for k, v in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "root_path": "",
        "scheme": "http",
        "query_string": query.encode("utf-8"),
        "headers": raw_headers,
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }

    async def call_next(_request):
        return ALLOWED

    return asyncio.run(mw.feed_auth_middleware(Request(scope), call_next))


# --- global (standalone) mode -------------------------------------------------

def test_valid_global_token_query_param_passes(feed_auth):
    feed_auth()
    assert call_middleware(query="auth=" + GLOBAL_TOKEN) is ALLOWED


def test_valid_global_token_as_basic_password_passes(feed_auth):
    feed_auth()
    assert call_middleware(headers=basic("feeds", GLOBAL_TOKEN)) is ALLOWED


def test_wrong_token_is_rejected(feed_auth):
    feed_auth()
    assert call_middleware(query="auth=not-the-right-token").status_code == 401


def test_empty_token_is_rejected(feed_auth):
    feed_auth()
    assert call_middleware(query="auth=").status_code == 401


def test_absent_auth_parameter_is_rejected(feed_auth):
    feed_auth()
    assert call_middleware().status_code == 401


def test_malformed_basic_header_is_rejected(feed_auth):
    feed_auth()
    assert call_middleware(headers={"Authorization": "Basic !!!not-base64!!!"}).status_code == 401


def test_basic_header_without_colon_is_rejected(feed_auth):
    feed_auth()
    encoded = base64.b64encode(GLOBAL_TOKEN.encode()).decode()
    assert call_middleware(headers={"Authorization": "Basic " + encoded}).status_code == 401


def test_rejection_carries_www_authenticate(feed_auth):
    feed_auth()
    assert call_middleware().headers["WWW-Authenticate"] == 'Basic realm="Podcast Feeds"'


def test_enabled_but_unconfigured_fails_closed(feed_auth):
    """Feed auth on, no global token generated: deny, never fall through."""
    feed_auth(global_token=None)
    assert call_middleware(query="auth=" + GLOBAL_TOKEN).status_code == 401
    assert call_middleware().status_code == 401
    assert call_middleware(query="auth=").status_code == 401


def test_enabled_but_unconfigured_empty_string_token_fails_closed(feed_auth):
    feed_auth(global_token="")
    assert call_middleware(query="auth=").status_code == 401
    assert call_middleware(query="auth=anything").status_code == 401


def test_disabled_feed_auth_lets_requests_through(feed_auth):
    feed_auth(enable_feed_auth=False)
    assert call_middleware() is ALLOWED


def test_unprotected_path_is_not_gated(feed_auth):
    feed_auth()
    assert call_middleware(path="/login") is ALLOWED


# --- per-user mode ------------------------------------------------------------

def test_valid_user_token_passes(feed_auth):
    feed_auth(auth_enabled=True)
    assert call_middleware(query="auth=" + USER_TOKEN) is ALLOWED


def test_unknown_user_token_is_rejected(feed_auth):
    feed_auth(auth_enabled=True)
    assert call_middleware(query="auth=some-other-token").status_code == 401


def test_global_token_does_not_authenticate_a_user(feed_auth):
    feed_auth(auth_enabled=True)
    assert call_middleware(query="auth=" + GLOBAL_TOKEN).status_code == 401


def test_user_token_via_basic_password_passes(feed_auth):
    feed_auth(auth_enabled=True)
    assert call_middleware(headers=basic("kes", USER_TOKEN)) is ALLOWED


def test_audio_path_is_also_gated(feed_auth):
    feed_auth()
    assert call_middleware(path="/audio/whatever.mp3").status_code == 401
    assert call_middleware(path="/audio/whatever.mp3", query="auth=" + GLOBAL_TOKEN) is ALLOWED


def test_end_to_end_unauthenticated_feed_request_401s(client, feed_auth):
    """Same denial, through the real ASGI stack rather than a bare call."""
    feed_auth()
    assert client.get(FEED_PATH).status_code == 401
    assert client.get(FEED_PATH, params={"auth": "wrong"}).status_code == 401


# --- unit-level checks --------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, False), (0, False), (1, True), (True, True), (False, False),
    ("0", False), ("1", True), ("true", True), ("False", False), ("", False),
])
def test_is_enabled(value, expected):
    assert mw._is_enabled(value) is expected


# --- non-ASCII credentials must 401, never 500 --------------------------------

def test_non_ascii_query_token_is_rejected_cleanly(feed_auth):
    """`hmac.compare_digest` raises TypeError on non-ASCII str arguments.

    Before the fix this raised out of the middleware and became an
    unauthenticated 500 with a stack trace.
    """
    feed_auth()
    assert call_middleware(query="auth=caf%C3%A9").status_code == 401


def test_non_ascii_basic_password_is_rejected_cleanly(feed_auth):
    feed_auth()
    assert call_middleware(headers=basic("feeds", "café")).status_code == 401


def test_non_ascii_token_is_rejected_cleanly_in_per_user_mode(feed_auth):
    feed_auth(auth_enabled=True)
    assert call_middleware(query="auth=caf%C3%A9").status_code == 401


def test_end_to_end_non_ascii_token_401s(client, feed_auth):
    feed_auth()
    assert client.get(FEED_PATH + "?auth=caf%C3%A9").status_code == 401


def test_tokens_equal_handles_non_ascii():
    assert mw._tokens_equal("café", "café") is True
    assert mw._tokens_equal("café", "cafe") is False
    assert mw._tokens_equal("café", None) is False
    assert mw._tokens_equal(None, "x") is False


# --- unreadable settings must fail closed -------------------------------------

def _stub_settings(monkeypatch, value):
    import app.web.router as router
    if isinstance(value, Exception):
        def boom():
            raise value
        monkeypatch.setattr(router, "get_global_settings", boom)
    else:
        monkeypatch.setattr(router, "get_global_settings", lambda: value)


def test_empty_settings_mapping_fails_closed(monkeypatch):
    """`get_global_settings()` returns {} when the row is missing or the read
    failed. That is 'unknown', not 'auth is off'."""
    _stub_settings(monkeypatch, {})
    assert call_middleware().status_code == 401
    assert call_middleware(query="auth=" + GLOBAL_TOKEN).status_code == 401


def test_settings_without_the_flag_fails_closed(monkeypatch):
    _stub_settings(monkeypatch, {"auth_enabled": 0})
    assert call_middleware().status_code == 401


def test_settings_read_raising_fails_closed(monkeypatch):
    _stub_settings(monkeypatch, RuntimeError("database is locked"))
    assert call_middleware().status_code == 401


def test_non_mapping_settings_fails_closed(monkeypatch):
    _stub_settings(monkeypatch, None)
    assert call_middleware().status_code == 401


# --- /feed/* is gated by the same middleware ----------------------------------

@pytest.mark.parametrize("path", ["/feed/unified", "/feed/unified.xml", "/feed/anything-new"])
def test_unified_feed_paths_are_gated(feed_auth, path):
    feed_auth()
    assert call_middleware(path=path).status_code == 401
    assert call_middleware(path=path, query="auth=wrong").status_code == 401
    assert call_middleware(path=path, query="auth=" + GLOBAL_TOKEN) is ALLOWED


def test_unified_feed_accepts_the_same_credential_forms_as_feeds(feed_auth):
    feed_auth()
    for path in (FEED_PATH, "/feed/unified.xml"):
        assert call_middleware(path=path, query="auth=" + GLOBAL_TOKEN) is ALLOWED
        assert call_middleware(path=path, headers=basic("feeds", GLOBAL_TOKEN)) is ALLOWED
        envelope = base64.b64encode(f"feeds:{GLOBAL_TOKEN}".encode()).decode()
        assert call_middleware(path=path, query="auth=" + envelope) is ALLOWED


def test_feed_prefix_does_not_swallow_unrelated_paths(feed_auth):
    feed_auth()
    assert call_middleware(path="/feedback") is ALLOWED
    assert call_middleware(path="/feed") is ALLOWED


def test_unified_feed_not_gated_when_feed_auth_disabled(feed_auth):
    feed_auth(enable_feed_auth=False)
    assert call_middleware(path="/feed/unified.xml") is ALLOWED

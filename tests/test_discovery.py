"""Tests for Podcast Index discovery: auth headers, cache TTL, routes, fallback.

Nothing here touches the network. The client's HTTP layer is never exercised
against the real API - the auth-header construction is tested directly (it is
pure), and every route test replaces the fetch functions.
"""
import hashlib
import time

import pytest

from app.core import discovery, search as search_module
from app.core.discovery import (
    PodcastIndexClient,
    PodcastIndexError,
    PodcastIndexNotConfigured,
    build_auth_headers,
)
from app.infra.database import (
    clear_discovery_cache,
    get_db_connection,
    get_discovery_cache,
    set_discovery_cache,
)


# ---------------------------------------------------------------------------
# Auth headers
# ---------------------------------------------------------------------------

# Podcast Index's documented scheme: sha1(key + secret + unixTime), lowercase
# hex. Fixed inputs so the expectation is a constant, not a re-derivation of
# the implementation.
_KEY = "ABCDEFGHIJKLMNOP"
_SECRET = "s3cr3t-podcast-index-secret"
_TIME = 1700000000
_EXPECTED_DIGEST = "182cb689b769293d9dab7143d9f4c4b7cc2d3d2b"


def test_expected_digest_constant_matches_the_documented_scheme():
    """Guard the constant above so a typo in it cannot mask a real bug."""
    assert _EXPECTED_DIGEST == hashlib.sha1(
        f"{_KEY}{_SECRET}{_TIME}".encode("utf-8")
    ).hexdigest()


def test_build_auth_headers_matches_sha1_of_key_secret_time():
    headers = build_auth_headers(_KEY, _SECRET, unix_time=_TIME)

    assert headers["Authorization"] == _EXPECTED_DIGEST
    assert headers["X-Auth-Key"] == _KEY
    assert headers["X-Auth-Date"] == str(_TIME)
    assert headers["User-Agent"]


def test_build_auth_headers_digest_is_lowercase_hex():
    digest = build_auth_headers(_KEY, _SECRET, unix_time=_TIME)["Authorization"]
    assert len(digest) == 40
    assert digest == digest.lower()
    assert all(c in "0123456789abcdef" for c in digest)


def test_build_auth_headers_stamp_is_whole_seconds_and_matches_the_digest():
    """The timestamp sent must be the exact one that was hashed.

    Podcast Index allows five minutes of skew, so a stamp in milliseconds, or
    one recomputed after hashing, fails with an opaque 401 rather than a
    useful error.
    """
    before = int(time.time())
    headers = build_auth_headers(_KEY, _SECRET)
    after = int(time.time())

    stamp = headers["X-Auth-Date"]
    assert stamp.isdigit()
    assert before <= int(stamp) <= after

    assert headers["Authorization"] == hashlib.sha1(
        f"{_KEY}{_SECRET}{stamp}".encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize(
    "key,secret", [("", _SECRET), (_KEY, ""), (None, None)]
)
def test_build_auth_headers_refuses_missing_credentials(key, secret):
    with pytest.raises(PodcastIndexNotConfigured):
        build_auth_headers(key, secret)



async def test_client_refuses_to_request_without_credentials():
    client = PodcastIndexClient(api_key="", api_secret="")
    with pytest.raises(PodcastIndexNotConfigured):
        await client.get_trending()


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------

def test_map_feed_produces_the_shared_card_shape_and_strips_html():
    card = discovery._map_feed({
        "url": "https://example.com/feed.xml",
        "title": "A Show",
        "description": "<p>Line one.</p>\n<p>Line &amp; two.</p>",
        "image": "https://example.com/small.jpg",
        "artwork": "https://example.com/big.jpg",
        "author": "Someone",
    })

    assert set(card) == {"title", "feed_url", "image", "description"}
    assert card["feed_url"] == "https://example.com/feed.xml"
    # Artwork wins over image when both are present.
    assert card["image"] == "https://example.com/big.jpg"
    assert "<p>" not in card["description"]
    assert card["description"] == "Line one. Line & two."


def test_map_feed_falls_back_to_author_when_there_is_no_description():
    card = discovery._map_feed({
        "url": "https://example.com/feed.xml",
        "title": "A Show",
        "author": "Someone",
    })
    assert card["description"] == "Someone"


def test_map_feeds_drops_entries_with_no_feed_url():
    mapped = discovery._map_feeds({"feeds": [
        {"title": "No URL"},
        {"url": "https://example.com/feed.xml", "title": "Good"},
        "not a dict",
    ]})
    assert [m["title"] for m in mapped] == ["Good"]


# ---------------------------------------------------------------------------
# Cache TTL
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_cache():
    clear_discovery_cache()
    yield
    clear_discovery_cache()


def test_cache_returns_a_stored_payload_within_the_ttl():
    payload = [{"title": "A Show", "feed_url": "https://example.com/feed.xml"}]
    set_discovery_cache("trending:40", payload)

    assert get_discovery_cache("trending:40", 3600) == payload


def test_cache_miss_returns_none_for_an_unknown_key():
    assert get_discovery_cache("never-stored", 3600) is None


def test_cache_entry_older_than_the_ttl_is_a_miss():
    set_discovery_cache("trending:40", [{"title": "Stale"}])

    # Age the row past the TTL rather than sleeping.
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE discovery_cache "
            "SET fetched_at = datetime('now', '-2 hours') "
            "WHERE cache_key = ?",
            ("trending:40",),
        )
        conn.commit()

    assert get_discovery_cache("trending:40", discovery.CACHE_TTL_SECONDS) is None
    # Still readable under a TTL long enough to cover its age.
    assert get_discovery_cache("trending:40", 24 * 3600) == [{"title": "Stale"}]


def test_cache_write_replaces_an_existing_key_rather_than_erroring():
    set_discovery_cache("categories", [{"id": 1, "name": "Arts"}])
    set_discovery_cache("categories", [{"id": 9, "name": "Business"}])

    assert get_discovery_cache("categories", 3600) == [{"id": 9, "name": "Business"}]


def test_corrupt_cache_payload_is_treated_as_a_miss():
    """A disposable cache must never take the feature down."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO discovery_cache (cache_key, payload) VALUES (?, ?)",
            ("trending:40", "{not json"),
        )
        conn.commit()

    assert get_discovery_cache("trending:40", 3600) is None


def test_clear_discovery_cache_empties_every_key():
    set_discovery_cache("trending:40", [{"title": "A"}])
    set_discovery_cache("categories", [{"id": 1, "name": "Arts"}])

    clear_discovery_cache()

    assert get_discovery_cache("trending:40", 3600) is None
    assert get_discovery_cache("categories", 3600) is None



async def test_read_through_cache_fetches_once_then_serves_from_cache():
    calls = []

    async def fetcher():
        calls.append(1)
        return [{"title": "Fetched"}]

    first = await discovery._cached("trending:7", fetcher)
    second = await discovery._cached("trending:7", fetcher)

    assert first == second == [{"title": "Fetched"}]
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Discovery is never written into `subscriptions`
# ---------------------------------------------------------------------------


async def test_discovery_never_writes_to_the_subscriptions_table():
    """Cached discovery results must not leak into the dashboard/feed loop."""
    with get_db_connection() as conn:
        before = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]

    async def fetcher():
        return [{"title": "A Show", "feed_url": "https://example.com/feed.xml"}]

    await discovery._cached("trending:99", fetcher)

    with get_db_connection() as conn:
        after = conn.execute("SELECT COUNT(*) FROM subscriptions").fetchone()[0]
    assert after == before


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

_CARDS = [{
    "title": "A Show",
    "feed_url": "https://example.com/feed.xml",
    "image": "https://example.com/art.jpg",
    "description": "About things.",
}]


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn


def _async_raise(exc):
    async def _fn(*args, **kwargs):
        raise exc
    return _fn


def test_trending_route_returns_cards(client, monkeypatch):
    import app.api.discovery as routes
    monkeypatch.setattr(routes, "fetch_trending", _async_return(_CARDS))

    resp = client.get("/api/trending")
    assert resp.status_code == 200
    assert resp.json() == _CARDS


def test_categories_route_returns_the_taxonomy(client, monkeypatch):
    import app.api.discovery as routes
    categories = [{"id": 1, "name": "Arts"}, {"id": 9, "name": "Business"}]
    monkeypatch.setattr(routes, "fetch_categories", _async_return(categories))

    resp = client.get("/api/categories")
    assert resp.status_code == 200
    assert resp.json() == categories


def test_category_route_passes_the_id_through(client, monkeypatch):
    import app.api.discovery as routes
    seen = {}

    async def _fetch(category_id, limit=40):
        seen["category_id"] = category_id
        seen["limit"] = limit
        return _CARDS

    monkeypatch.setattr(routes, "fetch_by_category", _fetch)

    resp = client.get("/api/categories/9?limit=5")
    assert resp.status_code == 200
    assert resp.json() == _CARDS
    assert seen == {"category_id": 9, "limit": 5}


def test_category_route_rejects_a_non_numeric_id(client):
    # The app maps validation errors to 400, not FastAPI's default 422.
    assert client.get("/api/categories/not-a-number").status_code == 400


@pytest.mark.parametrize("path", ["/api/trending", "/api/categories/9"])
def test_missing_credentials_return_503_not_an_empty_list(client, monkeypatch, path):
    """An outage must be visible, never disguised as 'nothing found'."""
    import app.api.discovery as routes
    exc = PodcastIndexNotConfigured("no credentials")
    monkeypatch.setattr(routes, "fetch_trending", _async_raise(exc))
    monkeypatch.setattr(routes, "fetch_by_category", _async_raise(exc))

    # The dashboard asks for JSON, so the operator-facing reason survives.
    resp = client.get(path, headers={"Accept": "application/json"})
    assert resp.status_code == 503
    body = resp.json()
    assert body != []
    assert "not configured" in body["message"].lower()


@pytest.mark.parametrize(
    "path,target",
    [
        ("/api/trending", "fetch_trending"),
        ("/api/categories", "fetch_categories"),
        ("/api/categories/9", "fetch_by_category"),
    ],
)
def test_upstream_failure_returns_502_not_an_empty_list(
    client, monkeypatch, path, target
):
    import app.api.discovery as routes
    monkeypatch.setattr(
        routes, target, _async_raise(PodcastIndexError("upstream exploded"))
    )

    resp = client.get(path, headers={"Accept": "application/json"})
    assert resp.status_code == 502
    body = resp.json()
    assert body != []
    assert "upstream exploded" in body["message"]


def test_discovery_routes_do_not_leak_the_api_key_in_an_error(client, monkeypatch):
    import app.api.discovery as routes
    monkeypatch.setattr(
        routes,
        "fetch_trending",
        _async_raise(PodcastIndexError("Podcast Index returned HTTP 401 for /podcasts/trending")),
    )

    body = client.get("/api/trending", headers={"Accept": "application/json"}).text
    assert "X-Auth-Key" not in body


# ---------------------------------------------------------------------------
# Search: Podcast Index primary, iTunes fallback
# ---------------------------------------------------------------------------

_ITUNES_RESULTS = [{
    "title": "iTunes Show",
    "feed_url": "https://example.com/itunes.xml",
    "image": "",
    "description": "Some Artist",
}]



async def test_search_prefers_podcast_index_and_skips_itunes(monkeypatch):
    called = {"itunes": False}

    monkeypatch.setattr(discovery, "search", _async_return(_CARDS))
    monkeypatch.setattr(
        search_module.PodcastSearcher, "search_youtube", staticmethod(_async_return([]))
    )

    async def _itunes(term, limit=10):
        called["itunes"] = True
        return _ITUNES_RESULTS

    monkeypatch.setattr(
        search_module.PodcastSearcher, "search_itunes", staticmethod(_itunes)
    )

    results = await search_module.PodcastSearcher.search("anything")
    assert results == _CARDS
    assert called["itunes"] is False



async def test_search_falls_back_to_itunes_when_podcast_index_errors(monkeypatch):
    monkeypatch.setattr(
        discovery, "search", _async_raise(PodcastIndexError("upstream exploded"))
    )
    monkeypatch.setattr(
        search_module.PodcastSearcher,
        "search_itunes",
        staticmethod(_async_return(_ITUNES_RESULTS)),
    )
    monkeypatch.setattr(
        search_module.PodcastSearcher, "search_youtube", staticmethod(_async_return([]))
    )

    assert await search_module.PodcastSearcher.search("anything") == _ITUNES_RESULTS



async def test_search_falls_back_to_itunes_when_podcast_index_is_unconfigured(
    monkeypatch,
):
    monkeypatch.setattr(
        discovery, "search", _async_raise(PodcastIndexNotConfigured("no key"))
    )
    monkeypatch.setattr(
        search_module.PodcastSearcher,
        "search_itunes",
        staticmethod(_async_return(_ITUNES_RESULTS)),
    )
    monkeypatch.setattr(
        search_module.PodcastSearcher, "search_youtube", staticmethod(_async_return([]))
    )

    assert await search_module.PodcastSearcher.search("anything") == _ITUNES_RESULTS



async def test_search_falls_back_to_itunes_when_podcast_index_is_empty(monkeypatch):
    monkeypatch.setattr(discovery, "search", _async_return([]))
    monkeypatch.setattr(
        search_module.PodcastSearcher,
        "search_itunes",
        staticmethod(_async_return(_ITUNES_RESULTS)),
    )
    monkeypatch.setattr(
        search_module.PodcastSearcher, "search_youtube", staticmethod(_async_return([]))
    )

    assert await search_module.PodcastSearcher.search("anything") == _ITUNES_RESULTS



async def test_search_returns_empty_for_an_empty_term_without_calling_anything(
    monkeypatch,
):
    monkeypatch.setattr(
        discovery, "search", _async_raise(AssertionError("must not be called"))
    )
    assert await search_module.PodcastSearcher.search("") == []


# ---------------------------------------------------------------------------
# Credential resolution: DB overrides .env
# ---------------------------------------------------------------------------

def _set_db_credentials(key, secret):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE app_settings SET podcast_index_api_key = ?, "
            "podcast_index_api_secret = ? WHERE id = 1",
            (key, secret),
        )
        conn.commit()


@pytest.fixture
def _restore_db_credentials():
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT podcast_index_api_key, podcast_index_api_secret "
            "FROM app_settings WHERE id = 1"
        ).fetchone()
    yield
    _set_db_credentials(row["podcast_index_api_key"], row["podcast_index_api_secret"])


def test_db_credentials_override_the_environment(monkeypatch, _restore_db_credentials):
    from app.core.config import settings as env_settings

    monkeypatch.setattr(env_settings, "PODCAST_INDEX_API_KEY", "env-key")
    monkeypatch.setattr(env_settings, "PODCAST_INDEX_API_SECRET", "env-secret")
    _set_db_credentials("db-key", "db-secret")

    assert discovery.get_credentials() == ("db-key", "db-secret")


def test_environment_is_the_fallback_when_the_db_values_are_blank(
    monkeypatch, _restore_db_credentials
):
    from app.core.config import settings as env_settings

    monkeypatch.setattr(env_settings, "PODCAST_INDEX_API_KEY", "env-key")
    monkeypatch.setattr(env_settings, "PODCAST_INDEX_API_SECRET", "env-secret")
    _set_db_credentials("   ", None)

    assert discovery.get_credentials() == ("env-key", "env-secret")

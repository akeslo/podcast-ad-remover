"""Tests for the "Retry Queue" control (POST /admin/queue/retry-all).

One button re-queues the whole live queue in a single bulk UPDATE: every
episode whose status is pending, failed or rate_limited goes back to 'pending'
with next_retry_at cleared, then the background processor is kicked via
FastAPI BackgroundTasks. 'processing' rows are left to the work already in
flight; 'ignored' (cancelled) and 'completed' rows are not resurrected.

Everything runs through the real ASGI stack (the conftest `client`, which is a
TestClient over the actual `app.main:app`), so `auth_middleware`,
`feed_auth_middleware` and the route's `require_admin_action` dependency all
execute. Two fixtures keep the suite hermetic and deterministic:

* the queue processor is stubbed out. Left real, process_queue would claim the
  just-re-queued 'pending' rows and drive them through transcription/detection
  over the network, moving them out of 'pending' before we could assert the
  route had put them there;
* the login rate limiter is cleared, so a login in one test can never be
  locked out by failed attempts recorded by an earlier module (all TestClient
  requests present the same client host).
"""
import pytest

from app.infra.database import get_db_connection
from app.web.auth_utils import hash_password


SAME_ORIGIN = {"Origin": "http://testserver"}
CROSS_ORIGIN = {"Origin": "http://attacker.example"}

# Far-future sentinel so a backoff window is unambiguously "still waiting" and
# a cleared window is unambiguously NULL, regardless of when the suite runs.
FUTURE = "2999-01-01 00:00:00"
NON_ADMIN_PASSWORD = "correct-horse-battery-staple"

RETRY_ALL_URL = "/admin/queue/retry-all"
EMPTY_NOTE = "/admin/queue?error=Nothing+in+the+queue+to+retry"


def _set_settings(**columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


def _empty_queue():
    """Drop every episode so the retry-all queue holds nothing eligible."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM episodes")
        conn.commit()


def _clear_login_rate_limit():
    from app.web.rate_limiter import login_rate_limiter

    login_rate_limiter.failed_attempts.clear()
    login_rate_limiter.locked_ips.clear()


def _fetch_rows(sub_id):
    """{guid: (status, next_retry_at)} for every episode in a subscription."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT guid, status, next_retry_at FROM episodes"
            " WHERE subscription_id = ? ORDER BY id",
            (sub_id,),
        ).fetchall()
    return {r["guid"]: (r["status"], r["next_retry_at"]) for r in rows}


@pytest.fixture(autouse=True)
def _hermetic():
    """Stub the queued processor and reset the login limiter for every test."""
    from app.core.processor import Processor

    async def _anoop(*args, **kwargs):
        return None

    Processor.process_queue = _anoop  # type: ignore[method-assign]
    _clear_login_rate_limit()
    yield


@pytest.fixture
def seeded_queue():
    """One subscription holding exactly one episode per relevant status.

    Each status maps to one row so the assertions read directly off the row the
    route did or did not touch. Cleaned up afterwards.
    """
    with get_db_connection() as conn:
        cur = conn.execute(
            "INSERT INTO subscriptions (feed_url, title, slug) VALUES (?, ?, ?)",
            (
                "http://example.invalid/retry-all-queue.xml",
                "Retry Queue Show",
                "retry-all-queue",
            ),
        )
        sub_id = cur.lastrowid
        # failed / rate_limited / processing carry a live backoff so "cleared"
        # is observable; pending/completed/ignored have none.
        spec = [
            ("pending", None),
            ("failed", FUTURE),
            ("rate_limited", FUTURE),
            ("processing", FUTURE),
            ("completed", None),
            ("ignored", None),
        ]
        for status, next_retry_at in spec:
            conn.execute(
                "INSERT INTO episodes (subscription_id, guid, title, original_url,"
                " status, next_retry_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    sub_id,
                    f"retry-all-{status}",
                    f"ep retry-all-{status}",
                    f"http://example.invalid/retry-all-{status}.mp3",
                    status,
                    next_retry_at,
                ),
            )
        conn.commit()

    yield sub_id

    with get_db_connection() as conn:
        conn.execute("DELETE FROM episodes WHERE subscription_id = ?", (sub_id,))
        conn.execute("DELETE FROM subscriptions WHERE id = ?", (sub_id,))
        conn.commit()


# --------------------------------------------------------------------------
# (a) every eligible row is reset to pending with its backoff cleared
# --------------------------------------------------------------------------

def test_retry_all_resets_failed_rate_limited_and_pending(client, seeded_queue):
    """POST (standalone, same-origin) re-queues all three eligible statuses."""
    before = _fetch_rows(seeded_queue)
    assert before["retry-all-pending"] == ("pending", None)
    assert before["retry-all-failed"] == ("failed", FUTURE)
    assert before["retry-all-rate_limited"] == ("rate_limited", FUTURE)

    response = client.post(RETRY_ALL_URL, headers=SAME_ORIGIN, follow_redirects=False)
    assert response.status_code == 303, response.text
    assert response.headers["location"].startswith("/admin/queue?success="), (
        response.headers["location"]
    )

    after = _fetch_rows(seeded_queue)
    # All three eligible rows are now 'pending' with no backoff remaining.
    assert after["retry-all-pending"] == ("pending", None)
    assert after["retry-all-failed"] == ("pending", None)
    assert after["retry-all-rate_limited"] == ("pending", None)


# --------------------------------------------------------------------------
# (b) rows that are not eligible are left exactly as they were
# --------------------------------------------------------------------------

def test_retry_all_leaves_processing_completed_ignored_untouched(
    client, seeded_queue
):
    """'processing' is in flight; 'completed'/'ignored' are terminal. None move."""
    before = _fetch_rows(seeded_queue)

    response = client.post(RETRY_ALL_URL, headers=SAME_ORIGIN, follow_redirects=False)
    assert response.status_code == 303, response.text

    after = _fetch_rows(seeded_queue)
    assert after["retry-all-processing"] == before["retry-all-processing"]
    assert after["retry-all-completed"] == before["retry-all-completed"]
    assert after["retry-all-ignored"] == before["retry-all-ignored"]


# --------------------------------------------------------------------------
# (c) the route requires admin: refusals ...
# --------------------------------------------------------------------------

def test_anonymous_originless_post_is_rejected(client, seeded_queue):
    """curl / blind scripted loop: no session, no Origin, no Referer."""
    before = _fetch_rows(seeded_queue)

    response = client.post(RETRY_ALL_URL, follow_redirects=False)

    assert response.status_code == 403, response.text
    assert _fetch_rows(seeded_queue) == before, "an anonymous request changed the queue"


def test_cross_site_post_is_rejected(client, seeded_queue):
    """A hostile page auto-submitting the form at the victim's install."""
    before = _fetch_rows(seeded_queue)

    response = client.post(RETRY_ALL_URL, headers=CROSS_ORIGIN, follow_redirects=False)

    assert response.status_code == 403, response.text
    assert _fetch_rows(seeded_queue) == before, "a cross-site request changed the queue"


def test_logged_in_non_admin_is_rejected(client, seeded_queue):
    """Multi-user mode: a real session that is not an admin still cannot trigger it."""
    before = _fetch_rows(seeded_queue)

    _set_settings(auth_enabled=1)
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM users WHERE username = ?", ("ra-tester",))
            conn.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
                ("ra-tester", hash_password(NON_ADMIN_PASSWORD)),
            )
            conn.commit()

        login = client.post(
            "/login",
            data={"username": "ra-tester", "password": NON_ADMIN_PASSWORD},
            headers=SAME_ORIGIN,
            follow_redirects=False,
        )
        assert login.status_code == 302, login.text

        response = client.post(
            RETRY_ALL_URL, headers=SAME_ORIGIN, follow_redirects=False
        )
        assert response.status_code == 403, response.text
        assert _fetch_rows(seeded_queue) == before, "a non-admin changed the queue"
    finally:
        _set_settings(auth_enabled=0)
        with get_db_connection() as conn:
            conn.execute("DELETE FROM users WHERE username = ?", ("ra-tester",))
            conn.commit()


# --------------------------------------------------------------------------
# (c) ... and the legitimate caller still gets through (the half that matters)
# --------------------------------------------------------------------------

def test_standalone_owner_can_trigger_retry_all(client, seeded_queue):
    """Standalone (auth_enabled = 0): same-origin is the whole boundary, and it works."""
    with get_db_connection() as conn:
        auth = conn.execute(
            "SELECT auth_enabled FROM app_settings WHERE id = 1"
        ).fetchone()["auth_enabled"]
    assert auth == 0, "standalone posture is what this case exercises"

    response = client.post(RETRY_ALL_URL, headers=SAME_ORIGIN, follow_redirects=False)

    assert response.status_code == 303, response.text
    after = _fetch_rows(seeded_queue)
    # Real work happened: the failed and rate_limited rows were re-queued.
    assert after["retry-all-failed"] == ("pending", None)
    assert after["retry-all-rate_limited"] == ("pending", None)
    # In-flight and terminal rows were not touched.
    assert after["retry-all-processing"] == ("processing", FUTURE)
    assert after["retry-all-completed"] == ("completed", None)
    assert after["retry-all-ignored"] == ("ignored", None)


def test_empty_queue_redirects_with_honest_note(client):
    """Nothing eligible: report it honestly, never a fake success."""
    _empty_queue()

    response = client.post(RETRY_ALL_URL, headers=SAME_ORIGIN, follow_redirects=False)

    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert location == EMPTY_NOTE, location
    assert "success" not in location, "an empty queue must not claim success"


# --------------------------------------------------------------------------
# Header/UI: the Retry Queue control is present regardless of queue emptiness
# --------------------------------------------------------------------------

def test_retry_queue_button_renders_even_on_empty_queue(client):
    """The button is a header control, not tied to the row list."""
    _empty_queue()

    response = client.get("/admin/queue")
    assert response.status_code == 200, response.status_code
    assert RETRY_ALL_URL in response.text
    assert "Retry Queue" in response.text
    # It must ask for confirmation before it will submit.
    assert "confirm(" in response.text

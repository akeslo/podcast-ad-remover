"""A rejected admin POST must be diagnosable without a live proxy.

The same-origin check on `require_admin_action` is correct and is deliberately
not weakened here. What is tested is the *other* half: when it fires because
the deployment is misconfigured (reverse proxy rewriting `Host`, Public
Application URL left at the auto-detected LAN address, proxy stripping
`Origin`), the operator must be able to tell that from a bare 403.

Everything runs through the real ASGI stack against `app.main:app`. No proxy is
started; the proxy's effect is reproduced by sending the headers a proxy would
send, which is all the app ever sees of it.
"""
import logging

import pytest

from app.infra.database import get_db_connection


ADMIN_POST = "/admin/system/update"


def _set_settings(**columns):
    assignments = ", ".join(f"{name} = ?" for name in columns)
    with get_db_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO app_settings (id) VALUES (1)")
        conn.execute(
            f"UPDATE app_settings SET {assignments} WHERE id = 1",
            tuple(columns.values()),
        )
        conn.commit()


@pytest.fixture(autouse=True)
def _standalone_with_external_url():
    """Standalone mode with a Public Application URL that does not match.

    This is exactly the misconfigured reverse-proxy install: the app was
    auto-configured for the LAN address on first boot and the operator now
    reaches it through a public hostname.
    """
    _set_settings(auth_enabled=0, app_external_url="http://192.168.1.50:8000")
    yield
    _set_settings(auth_enabled=0, app_external_url=None)


def test_rejection_body_states_the_reason_rather_than_a_generic_denial(client):
    response = client.post(
        ADMIN_POST,
        headers={"Origin": "https://podcasts.example.com"},
        follow_redirects=False,
    )

    assert response.status_code == 403, response.text
    assert "origin" in response.text.lower(), (
        "the 403 body says nothing about the origin check, which is the whole "
        "cause of the failure"
    )


def test_rejection_body_reason_survives_a_json_client(client):
    response = client.post(
        ADMIN_POST,
        headers={"Origin": "https://podcasts.example.com", "Accept": "application/json"},
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert "origin" in response.json()["message"].lower()


def test_rejection_logs_both_the_received_and_the_expected_origin(client, caplog):
    """The log must name both sides of the comparison, or it is not a diagnosis."""
    with caplog.at_level(logging.WARNING, logger="app.web.auth"):
        response = client.post(
            ADMIN_POST,
            headers={
                # What the browser sends through the proxy.
                "Origin": "https://podcasts.example.com",
                # What a `proxy_pass http://app:8000;` without `proxy_set_header
                # Host $host;` makes the container see.
                "Host": "app:8000",
            },
            follow_redirects=False,
        )

    assert response.status_code == 403
    diagnosis = "\n".join(
        r.getMessage() for r in caplog.records if "same-origin check" in r.getMessage()
    )
    assert diagnosis, "no log line identifies the same-origin check as the cause"
    assert "podcasts.example.com" in diagnosis, "the received origin is not logged"
    assert "app:8000" in diagnosis, "the Host the container saw is not logged"
    assert "192.168.1.50:8000" in diagnosis, (
        "the configured Public Application URL - the expected value - is not logged"
    )
    assert "Public Application URL" in diagnosis, "the log does not name the fix"


def test_origin_less_post_is_still_rejected_and_still_explained(client):
    """A scripted POST sends no Origin at all. Still 403, still diagnosable."""
    response = client.post(ADMIN_POST, follow_redirects=False)

    assert response.status_code == 403
    assert "origin" in response.text.lower()


def test_diagnostics_do_not_leak_credential_bearing_paths(client, caplog):
    """A Referer can carry a feed token in its path. Only netlocs may be logged.

    Scoped to the `app.web.auth` diagnostics deliberately. `require_same_origin`
    in `app/web/router.py` logs the raw `origin` value, which is the full
    `Referer` URL when the browser sent no `Origin` header - so it still writes
    a token-bearing path to the log. That is a separate defect in a file this
    change does not own; it is not fixed here and this test does not paper over
    it, it just refuses to add a second copy of the same leak.
    """
    with caplog.at_level(logging.WARNING, logger="app.web.auth"):
        client.post(
            ADMIN_POST,
            headers={"Referer": "https://podcasts.example.com/feed/c2VjcmV0LXRva2Vu"},
            follow_redirects=False,
        )

    logged = "\n".join(
        r.getMessage() for r in caplog.records if r.name == "app.web.auth"
    )
    assert "podcasts.example.com" in logged
    assert "c2VjcmV0LXRva2Vu" not in logged, "a token-bearing path reached the log"


def test_a_correctly_configured_proxy_is_accepted(client):
    """The check is not weakened: with Host forwarded, the same POST goes through.

    Proves the 403s above are configuration, not an unconditional block, and
    that the documented fix ("forward the public Host header") actually works.
    """
    response = client.post(
        ADMIN_POST,
        headers={
            "Origin": "https://podcasts.example.com",
            "Host": "podcasts.example.com",
        },
        data={"check_interval": "60"},
        follow_redirects=False,
    )

    assert response.status_code != 403, response.text


def test_the_configured_public_url_alone_is_enough(client):
    """The second documented fix: set Public Application URL, Host untouched."""
    _set_settings(app_external_url="https://podcasts.example.com")

    response = client.post(
        ADMIN_POST,
        headers={"Origin": "https://podcasts.example.com", "Host": "app:8000"},
        data={"check_interval": "60"},
        follow_redirects=False,
    )

    assert response.status_code != 403, response.text

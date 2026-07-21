"""
Regression tests for app/web/security_headers.py.

This module has been touched 7 times in recent history per PROP-077 and has
zero prior coverage - these tests pin down the exact set of headers the
middleware is supposed to add to every response.

These tests exercise `/health` rather than `/` or an unmatched 404 path: the
web dashboard route (`GET /`) and the HTML 404 page both render Jinja2
templates via `templates.TemplateResponse(name, context, ...)` (old-style,
positional `name` first), which is incompatible with the Starlette version
resolved by this project's unpinned `fastapi`/`starlette` requirements and
raises `TypeError: cannot use 'tuple' as a dict key` before a response is
even produced - a pre-existing template-call/dependency-version bug in
app/web/error_handlers.py and app/web/router.py, unrelated to the security
headers middleware under test here. `/health` returns plain JSON and is
unaffected, so it exercises the same middleware stack without tripping over
that separate bug. The 404 case is still covered, forcing the JSON
(non-template) branch of the 404 handler via an `Accept: application/json`
header.
"""


def test_health_response_has_all_security_headers(client):
    response = client.get("/health")
    assert response.status_code == 200

    headers = response.headers
    for name in (
        "Content-Security-Policy",
        "Strict-Transport-Security",
        "X-Frame-Options",
        "X-Content-Type-Options",
        "Referrer-Policy",
        "Permissions-Policy",
        "X-XSS-Protection",
    ):
        assert name in headers, f"missing header: {name}"


def test_content_security_policy_directives(client):
    response = client.get("/health")
    csp = response.headers["Content-Security-Policy"]

    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "base-uri 'self'" in csp
    assert "form-action 'self'" in csp
    assert "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com https://static.cloudflareinsights.com" in csp
    assert "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com" in csp
    assert "font-src 'self' data: https://fonts.gstatic.com" in csp
    assert "connect-src 'self' https://cloudflareinsights.com" in csp


def test_strict_transport_security_is_unconditional(client):
    # Applied even over plain HTTP in the test client, per the comment in
    # security_headers.py: it supports reverse-proxy/load-balancer deployments
    # where the app sees HTTP but the client connects via HTTPS.
    response = client.get("/health")
    assert response.headers["Strict-Transport-Security"] == (
        "max-age=31536000; includeSubDomains; preload"
    )


def test_clickjacking_and_mime_sniffing_headers(client):
    response = client.get("/health")
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_referrer_policy(client):
    response = client.get("/health")
    assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_permissions_policy_denies_sensitive_features(client):
    response = client.get("/health")
    permissions_policy = response.headers["Permissions-Policy"]

    for feature in (
        "geolocation=()",
        "microphone=()",
        "camera=()",
        "payment=()",
        "usb=()",
        "magnetometer=()",
        "gyroscope=()",
        "accelerometer=()",
    ):
        assert feature in permissions_policy


def test_legacy_xss_protection_header(client):
    response = client.get("/health")
    assert response.headers["X-XSS-Protection"] == "1; mode=block"


def test_headers_present_on_404_json_response(client):
    # BaseHTTPMiddleware.dispatch runs after call_next regardless of the
    # downstream status code, so even error responses should carry the
    # security headers. Force the handler's JSON branch (see module
    # docstring) to avoid the unrelated template-rendering bug on the HTML
    # branch.
    response = client.get(
        "/this-route-does-not-exist", headers={"Accept": "application/json"}
    )
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" in response.headers

from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import RedirectResponse, JSONResponse, PlainTextResponse
from starlette.middleware.sessions import SessionMiddleware
from typing import Optional
import logging
from datetime import datetime

from app.infra.database import get_db_connection
from app.web.auth_utils import get_client_ip, is_ip_allowed, verify_password
from app.core.models import User

logger = logging.getLogger(__name__)

# Session key
SESSION_USER_KEY = "user_id"

# Methods the same-origin check in `require_admin_action` actually guards.
UNSAFE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


def _origin_diagnostics(request: Request) -> str:
    """Describe both sides of the same-origin comparison, for the log only.

    Every admin route is guarded by `require_admin_action`, which rejects a
    state-changing request whose `Origin` (or `Referer`) host is neither the
    `Host` header nor the operator-configured Public Application URL. Behind a
    reverse proxy that rewrites `Host` to the container name, or with the
    Public Application URL left pointing at the auto-detected LAN address,
    every admin form POST then fails with a generic 403 and nothing anywhere
    says why. This produces the one log line that names the received value and
    the expected one.

    Only netlocs are logged, never a full URL: a `Referer` can carry a feed
    token in its path, and tokens must not reach the log.
    """
    from urllib.parse import urlsplit

    def _netloc(value):
        if not value:
            return None
        value = value.strip()
        return urlsplit(value).netloc.lower() or value.lower()

    try:
        from app.core.utils import get_global_settings
        configured = get_global_settings().get("app_external_url")
    except Exception:  # pragma: no cover - settings unavailable
        configured = None

    return (
        f"origin={_netloc(request.headers.get('origin'))!r} "
        f"referer={_netloc(request.headers.get('referer'))!r} "
        f"host={_netloc(request.headers.get('host'))!r} "
        f"x-forwarded-host={_netloc(request.headers.get('x-forwarded-host'))!r} "
        f"public-application-url={_netloc(configured)!r}"
    )

def _forbidden(request: Request, detail: str):
    """Build a 403 response for use *inside* the middleware.

    `auth_middleware` is installed with `@app.middleware("http")`, which wraps
    it in Starlette's `BaseHTTPMiddleware`. Exception handlers - including
    FastAPI's `HTTPException` handler - are mounted inside the router, further
    down the stack, so an exception raised here has already escaped them by
    the time it propagates. The only thing left above is
    `ServerErrorMiddleware`, which turns it into a 500.

    That is not academic: a logged-in non-admin requesting /admin/* received a
    500 "Internal Server Error" instead of a 403, so a working authorization
    decision was reported as a broken server. The middleware must *return* its
    refusals, never raise them.

    The response shape follows the request: an API/XHR caller that asked for
    JSON gets JSON with the same `detail` key FastAPI's own handler produces,
    so nothing downstream has to special-case middleware-origin refusals.
    """
    accept = (request.headers.get("accept") or "").lower()
    wants_json = (
        "application/json" in accept
        or request.headers.get("x-requested-with", "").lower() == "xmlhttprequest"
        or request.url.path.startswith("/api/")
    )
    if wants_json:
        return JSONResponse(
            {"detail": detail}, status_code=status.HTTP_403_FORBIDDEN
        )
    return PlainTextResponse(detail, status_code=status.HTTP_403_FORBIDDEN)


def get_current_user(request: Request) -> Optional[User]:
    """Get the currently logged-in user from session."""
    user_id = request.session.get(SESSION_USER_KEY)
    
    # Check if auth is disabled globally - treat everyone as admin
    try:
        with get_db_connection() as conn:
            settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()
            if settings and not settings['auth_enabled']:
                return User(
                    id=0, 
                    username="admin", 
                    password_hash="", 
                    is_admin=True, 
                    created_at=datetime.now(), 
                    last_login=datetime.now()
                )
    except Exception as e:
        logger.error(f"Error checking auth settings: {e}")

    if not user_id:
        return None
    
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if row:
            return User.model_validate(dict(row))
    return None

def require_auth(request: Request) -> User:
    """Dependency that requires authentication."""
    # First check if auth is enabled globally
    with get_db_connection() as conn:
        settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()
        
    # If settings exist and auth is disabled, return a dummy admin user
    if settings and not settings['auth_enabled']:
        return User(
            id=0, 
            username="admin", 
            password_hash="", 
            is_admin=True, 
            created_at=datetime.now(), 
            last_login=datetime.now()
        )

    user = get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

def require_admin(request: Request) -> User:
    """Dependency that requires admin privileges."""
    user = require_auth(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return user

async def auth_middleware(request: Request, call_next):
    """
    Middleware to handle authentication and IP allowlisting.
    """
    path = request.url.path
    
    # Skip auth/IP check for specific paths
    # static: always public
    # feeds/audio: public to world (IP check skipped), but might be protected by Feed Auth elsewhere
    #
    # The prefixes below are matched with an explicit trailing slash and the
    # bare path is listed separately, so the exemption covers exactly the
    # subtree it names. This is load-bearing, not stylistic: the
    # administrative route /feed-token/rotate sits one hyphen away from
    # "/feed/" and would be silently made public by a looser prefix such as
    # "/feed". Do not drop the slashes.
    PUBLIC_PATHS = ("/login", "/request-access", "/submit-access-request")
    PUBLIC_PREFIXES = ("/static/", "/feeds/", "/feed/", "/audio/")
    if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
        return await call_next(request)
    
    # Check if auth is enabled
    with get_db_connection() as conn:
        settings = conn.execute("SELECT auth_enabled, ip_allowlist FROM app_settings WHERE id = 1").fetchone()
    
    if not settings:
        return await call_next(request)

    # 1. GLOBAL IP CHECK (High Priority)
    # If an allowlist is set, it applies to EVERYTHING (Admin, Feeds, Audio, Dashboard)
    if settings['ip_allowlist']:
        client_ip = get_client_ip(request)
        if not is_ip_allowed(client_ip, settings['ip_allowlist']):
            logger.warning(f"AUTH - IP blocked: {client_ip} - Path: {path}")
            # Returned, not raised - see `_forbidden`. Same defect as the
            # admin check below: raising here reported an IP allowlist denial
            # as a 500.
            return _forbidden(request, "Access denied from your IP address")

    # 2. USER AUTHENTICATION CHECK
    # Only if auth is enabled
    if settings['auth_enabled']:
        # Dashboard and Admin routes require user auth
        user = get_current_user(request)
        if not user:
            # Log the attempt
            client_ip = get_client_ip(request)
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO login_attempts (username, ip_address, success, user_agent) VALUES (?, ?, ?, ?)",
                    (None, client_ip, 0, request.headers.get("user-agent", ""))
                )
                conn.commit()
            
            logger.info(f"AUTH - Unauthorized access attempt: {client_ip} - Path: {path}")
            return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
        
        # Check if password change is required
        with get_db_connection() as conn:
            settings_row = conn.execute("SELECT require_password_change FROM app_settings WHERE id = 1").fetchone()
            if settings_row and settings_row['require_password_change'] and path != "/change-password":
                return RedirectResponse(url="/change-password", status_code=status.HTTP_302_FOUND)
                
        # 3. ADMIN PRIVILEGE CHECK
        # Protect /admin routes from non-admin users
        if path.startswith("/admin") and not user.is_admin:
            # Returned, not raised. See `_forbidden`: an HTTPException raised
            # in a BaseHTTPMiddleware never reaches FastAPI's exception
            # handler, so this correct 403 decision was surfacing as a 500.
            logger.info(
                "AUTH - Non-admin user %r denied admin path %s",
                getattr(user, "username", None), path,
            )
            return _forbidden(request, "Admin privileges required")

    response = await call_next(request)

    # A 403 on a state-changing request is, in practice, almost always the
    # same-origin check in `require_admin_action`. The response body is
    # deliberately generic, so without this the operator sees "Access Denied"
    # on every admin form and has nothing to debug from.
    if response.status_code == status.HTTP_403_FORBIDDEN and request.method in UNSAFE_METHODS:
        logger.warning(
            "AUTH - 403 on %s %s. Likely the admin same-origin check: %s. "
            "Fix by setting Admin > System > 'Public Application URL' to the exact "
            "URL the browser uses (scheme, host and port), or by making the reverse "
            "proxy forward the public Host header to this container.",
            request.method, path, _origin_diagnostics(request),
        )

    return response

def log_login_attempt(username: str, ip_address: str, success: bool, user_agent: str):
    """Log a login attempt to the database."""
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO login_attempts (username, ip_address, success, user_agent) VALUES (?, ?, ?, ?)",
            (username, ip_address, 1 if success else 0, user_agent)
        )
        conn.commit()
    
    status_str = "SUCCESS" if success else "FAILED"
    logger.info(f"AUTH - Login {status_str}: {username} from {ip_address}")

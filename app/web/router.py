from fastapi import APIRouter, Request, Form, Depends, BackgroundTasks, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from app.infra.repository import SubscriptionRepository, EpisodeRepository
from app.core.feed import FeedManager
from app.core.models import SubscriptionCreate
from app.web.auth import get_current_user, require_auth, require_admin, log_login_attempt, SESSION_USER_KEY
from app.web.auth_utils import hash_password, verify_password, generate_secure_password, get_client_ip
from app.web.rate_limiter import login_rate_limiter, check_rate_limit
from app.infra.database import get_db_connection
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

# Helper to get CSP nonce from request
def get_csp_nonce(request: Request) -> str:
    """Extract CSP nonce from request state (set by SecurityHeadersMiddleware)"""
    return getattr(request.state, 'csp_nonce', '')


# Add simple markdown filter
def simple_markdown(text):
    """Convert basic markdown to HTML: **bold**, bullets (*, -, •, 1.)"""
    if not text:
        return ""
    import re
    import html
    # Escape first: this output is rendered with |safe, and the text is
    # LLM-generated from third-party transcripts. Only the tags this
    # function emits itself may reach the page as markup.
    text = html.escape(text)
    # Handle both Unix and Windows line endings
    lines = text.replace('\r\n', '\n').split('\n')
    result = []
    in_list = False
    list_items = []
    
    def apply_bold(t):
        import re
        # Convert **text** to <strong>text</strong> (Bold)
        t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
        # Convert *text* to <strong>text</strong> (User requested single star bold)
        t = re.sub(r'\*(.+?)\*', r'<strong>\1</strong>', t)
        return t

    def flush_list():
        nonlocal in_list, list_items, result
        if in_list and list_items:
            # Inline styles as baseline, tailwind classes for styling
            list_html = '<ul class="list-disc ml-8 space-y-2 mb-4 text-white/90" style="list-style-type: disc; margin-left: 2rem; margin-bottom: 1rem;">\n' + \
                        '\n'.join(list_items) + '\n</ul>'
            result.append(list_html)
            list_items = []
            in_list = False

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if in_list:
                flush_list()
            continue
            
        # Match bullets: *, -, • or "1." followed by any whitespace (\s+)
        bullet_match = re.match(r'^([\*\-\•]|\d+\.)\s+(.+)$', stripped_line)
        
        if bullet_match:
            if not in_list:
                in_list = True
            content = bullet_match.group(2).strip()
            # Apply bolding only to the content of the list item
            content = apply_bold(content)
            list_items.append(f'<li class="mb-1">{content}</li>')
        else:
            if in_list:
                flush_list()
            # Apply bolding to the paragraph text
            processed_line = apply_bold(stripped_line)
            result.append(f'<p class="mb-2 text-white/90 leading-relaxed">{processed_line}</p>')
            
    if in_list:
        flush_list()

    return '\n'.join(result)

def clean_description(text):
    """Clean episode description: remove URLs, sponsors, promotional text, and HTML tags."""
    if not text:
        return ""
    import re
    import html
    
    # 0. Strip HTML tags first
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # 0.5. Decode HTML entities
    text = html.unescape(text)
    
    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove common promo codes/sponsor patterns
    # (Simple heuristic: if line starts with 'Sponsor' or 'Promo', cut off rest or remove line)
    # For now, let's just remove specific keywords/lines
    lines = text.split('\n')
    cleaned_lines = []
    
    cutoff_keywords = ["Sponsors:", "Support the show:", "Brought to you by:", "Advertise with us:", "See omnystudio.com/listener"]
    
    for line in lines:
        stripped = line.strip()
        # Check cutoff
        if any(keyword in stripped for keyword in cutoff_keywords):
           break # Stop processing description here (assuming footer junk follows)
           
        if stripped:
            cleaned_lines.append(stripped)
            
    result = " ".join(cleaned_lines)
    
    # 3. Clean up extra whitespace
    result = re.sub(r'\s+', ' ', result).strip()
    
    return result

templates.env.filters['simple_markdown'] = simple_markdown
templates.env.filters['clean_description'] = clean_description

sub_repo = SubscriptionRepository()
ep_repo = EpisodeRepository()

# Helper to get settings
def get_global_settings():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        if row:
            return dict(row)
    return {}

from app.core.utils import get_app_base_url


# --- One-shot session flash messages ---------------------------------------
#
# Used to hand a freshly generated credential to the very next rendered page
# without putting it in a URL. A redirect query string ends up in browser
# history, the server access log, and any outbound Referer header; the session
# cookie ends up in none of those. The value is popped (not read) so it lives
# for exactly one render.

FLASH_SESSION_KEY = "_flash"


def set_flash(request: Request, **payload):
    """Stash a one-shot message for the next rendered page."""
    request.session[FLASH_SESSION_KEY] = payload


def pop_flash(request: Request) -> dict:
    """Return and clear the pending flash payload. Empty dict when none."""
    flash = request.session.pop(FLASH_SESSION_KEY, None)
    return flash if isinstance(flash, dict) else {}


def _acceptable_origins(request: Request) -> set:
    """Hosts that a same-origin browser request may legitimately declare.

    Behind a reverse proxy the Host header the app sees is often the internal
    one while the browser's Origin carries the public name, so the configured
    external URL and the forwarded host both count.
    """
    from urllib.parse import urlsplit

    hosts = set()

    def _add(value):
        if not value:
            return
        value = value.strip()
        if "//" in value:
            value = urlsplit(value).netloc
        if value:
            hosts.add(value.lower())

    # Host, plus the operator-configured public name. Deliberately NOT
    # X-Forwarded-Host: it is set by the caller on any request the proxy does
    # not overwrite, so trusting it would let a request nominate its own
    # acceptable origin. app_external_url covers the reverse-proxy case
    # properly, because an operator configured it.
    _add(request.headers.get("host"))
    try:
        _add(get_global_settings().get("app_external_url"))
    except Exception:  # pragma: no cover - settings unavailable
        pass
    return hosts


def require_same_origin(request: Request) -> None:
    """Reject state-changing requests that are not same-origin browser posts.

    This is the *only* authorisation material available in standalone mode
    (auth_enabled = 0), where the app has no users, no sessions and therefore
    nothing to authenticate against - `require_admin` degrades to a no-op
    dummy admin there by design. It is a CSRF boundary, not an authentication
    boundary: it stops drive-by cross-site posts and blind scripted loops
    (curl sends no Origin), and it deliberately does not pretend to stop an
    attacker who can already reach the admin UI. In standalone mode that
    remaining boundary is the IP allowlist enforced in `auth_middleware`.

    Safe methods are untouched, so ordinary navigation still works.
    """
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return

    from urllib.parse import urlsplit

    origin = request.headers.get("origin")
    if not origin:
        referer = request.headers.get("referer")
        origin = referer or ""
    netloc = urlsplit(origin).netloc.lower() if origin else ""

    if not netloc or netloc not in _acceptable_origins(request):
        # Log the NETLOC, never the raw header. When the browser sends no
        # Origin we fall back to Referer, which is a full URL — and a feed
        # request's Referer carries the feed token in its path or query, so
        # logging `origin` verbatim writes a live bearer credential to disk.
        # That is the exact leak the access-log redaction filter exists to
        # prevent; this call site would have reintroduced it.
        logger.warning(
            "AUTH - Rejected cross-origin/originless %s %s (origin_netloc=%r)",
            request.method, request.url.path, netloc or None,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cross-origin or origin-less request rejected",
        )


def require_admin_action(request: Request) -> None:
    """Authorisation for every administrative route.

    Two independent checks, because neither alone covers both deployments:

    * `require_admin` is the real gate whenever user auth is enabled. Without
      it these routes relied entirely on `auth_middleware`, which skips its
      `/admin` privilege check when `auth_enabled` is 0 - so the switch that
      turns feed auth off sat on the public side of the door.
    * `require_same_origin` covers the standalone deployment, where there is
      no user to be an admin at all.
    """
    require_admin(request)
    require_same_origin(request)


def require_user_action(request: Request) -> None:
    """Authorisation for state-changing routes any signed-in user may perform.

    These are ordinary user actions - adding a subscription, downloading or
    reprocessing an episode, editing a subscription's settings - not
    administrative ones, so `require_admin` is the wrong gate: it would lock a
    non-admin user out of the app's normal function in order to close a hole
    that is not about privilege at all.

    What each half actually buys, per deployment:

    * `auth_enabled = 1` (multi-user): `require_auth` is a real authentication
      boundary. An anonymous request is rejected with 401 by the dependency
      itself rather than relying on `auth_middleware` alone, so the route
      stays guarded even if the middleware's path handling changes. It does
      *not* distinguish between users - any signed-in account may act on any
      subscription, because the data model has no per-user ownership.
    * `auth_enabled = 0` (standalone, the common deployment): `require_auth`
      is a deliberate no-op - it synthesises a dummy admin because there are
      no users to authenticate. Making these routes require a real login there
      would lock the owner out of adding a podcast, which is a worse outcome
      than the hole being closed. So in standalone mode `require_same_origin`
      is the entire boundary, exactly as it is for the admin routes: it stops
      drive-by cross-site posts and blind scripted loops (curl sends no
      Origin), and it does not pretend to stop anyone who can already reach
      the UI. That remaining boundary is the IP allowlist in
      `auth_middleware`.

    In both modes this is a CSRF boundary first; only in multi-user mode is it
    also an authentication one.
    """
    require_auth(request)
    require_same_origin(request)


# --- One-time "your feed URLs changed" upgrade notice ---------------------
#
# Feed URLs used to be base64(username:account_password). They are now
# base64(username:feed_token), and there is deliberately no password fallback
# on the feed routes. That means every URL an existing subscriber already
# pasted into a podcast app starts returning 401 the moment they upgrade.
# Podcast clients do not surface a 401 on a background refresh - they simply
# stop updating - so without this notice the user's podcasts stop on a random
# day with no signal anywhere.
#
# The marker is a file under DATA_DIR rather than an app_settings column, on
# purpose: the schema is owned elsewhere, and a file needs no migration to
# land. A *fresh* install is auto-acknowledged the first time the dashboard
# renders with zero subscriptions, so only an upgraded install - one that
# already had subscriptions when this code first ran - ever sees the banner.

FEED_URL_NOTICE_MARKER = "feed-url-migration-acknowledged"


def _feed_url_notice_marker_path() -> str:
    from app.core.config import settings as app_settings

    return os.path.join(app_settings.DATA_DIR, FEED_URL_NOTICE_MARKER)


def _feed_url_notice_acknowledged() -> bool:
    try:
        return os.path.exists(_feed_url_notice_marker_path())
    except Exception:  # pragma: no cover - unreadable DATA_DIR
        logger.exception("Could not read the feed-URL notice marker")
        return True  # fail closed: never nag when we cannot record a dismissal


def acknowledge_feed_url_notice() -> None:
    """Persist the dismissal so the banner does not come back on reload."""
    path = _feed_url_notice_marker_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(datetime.now().isoformat())
    except Exception:  # pragma: no cover - read-only DATA_DIR
        logger.exception("Could not persist the feed-URL notice dismissal")


def feed_url_notice_pending(global_settings: dict, subscription_count: int) -> bool:
    """True when this install still owes the user the feed-URL-changed notice."""
    if not is_feed_auth_enabled(global_settings):
        # Without feed auth the URLs never carried a credential, so nothing
        # about them changed.
        return False
    if _feed_url_notice_acknowledged():
        return False
    if subscription_count == 0:
        # Fresh install: there are no already-shared URLs to invalidate.
        # Acknowledge silently so the banner never appears later.
        acknowledge_feed_url_notice()
        return False
    return True


@router.post("/feed-url-notice/dismiss", dependencies=[Depends(require_same_origin)])
async def dismiss_feed_url_notice(request: Request):
    """Dismiss the one-time feed-URL-changed banner, permanently."""
    acknowledge_feed_url_notice()
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)


def is_feed_auth_enabled(global_settings: dict) -> bool:
    """True when feed URLs must carry a credential."""
    value = global_settings.get('enable_feed_auth')
    if value is None:
        return False
    return str(value).lower() in ('1', 'true', 'yes', 'on')


def build_feed_auth_token(global_settings: dict, user_obj=None) -> str:
    """Return the base64 ?auth= credential for feed URLs.

    The credential is a per-user (or, standalone, a per-install) random feed
    token - never the account password. The token is a bearer credential for
    read-only feed access only, is revocable on its own, and its leakage into
    podcast-client logs and proxies does not expose the login.

    Raises RuntimeError when no token can be resolved. That is deliberate: the
    old code silently fell back to the session-held account password, so a
    missing identity produced a working URL built from a real credential.
    Failing loudly is the point of this change.
    """
    from app.infra.database import ensure_feed_token, ensure_global_feed_token

    if global_settings.get('auth_enabled'):
        # Integrated auth: the feed credential belongs to the logged-in user.
        user_id = getattr(user_obj, "id", None)
        username = getattr(user_obj, "username", None)
        if not user_id or not username:
            raise RuntimeError(
                "Cannot build a feed URL: user authentication is enabled but "
                "no authenticated user was supplied."
            )
        credential = ensure_feed_token(user_id)
        auth_user = username
    else:
        # Standalone auth: one shared install-wide feed token.
        #
        # No `or 'feed'` fallback. The unified-feed validator authorises only
        # when the presented username equals app_settings.feed_auth_username,
        # so a guessed default would build a URL the app itself rejects with
        # 401 - a silently broken feed rather than an honest error. Same
        # fail-loud rule as the auth_enabled branch above.
        credential = ensure_global_feed_token()
        auth_user = global_settings.get('feed_auth_username')
        if not auth_user:
            raise RuntimeError(
                "Cannot build a feed URL: feed auth is enabled in standalone "
                "mode but no feed_auth_username is configured."
            )

    if not credential:
        raise RuntimeError("Cannot build a feed URL: no feed token available.")

    import base64
    return base64.b64encode(f"{auth_user}:{credential}".encode()).decode()


def append_feed_auth(url: str, global_settings: dict, user_obj=None) -> str:
    """Append the ?auth= feed token to url when feed auth is enabled."""
    if not is_feed_auth_enabled(global_settings):
        return url
    token = build_feed_auth_token(global_settings, user_obj)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}auth={token}"


def generate_rss_links(request: Request, sub, global_settings: dict, user_obj=None):
    """Consolidated logic for generating RSS links with optional auth injection."""
    base_url = get_app_base_url(global_settings, request)

    rss_url = append_feed_auth(
        f"{base_url}/feeds/{sub.slug}.xml", global_settings, user_obj
    )

    return {
        "rss": rss_url,
        "direct": rss_url,
        "apple": rss_url,  # Method 1: Direct HTTPS URL for manual "Follow a Show by URL"
        "pocket_casts": f"pktc://subscribe/{rss_url}",
        "overcast": f"overcast://x-callback-url/add?url={rss_url}",
        "castbox": f"castbox://subscribe?url={rss_url}",
        "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
    }

# Helper to get pending access requests count for sidebar badge
def get_pending_requests_count():
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        result = conn.execute("SELECT COUNT(*) FROM access_requests WHERE status = 'pending'").fetchone()
        return result[0] if result else 0

# --- Authentication Routes ---
@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Display login page or first-time setup."""
    with get_db_connection() as conn:
        # `initial_password` is deliberately NOT selected. /login is exempt
        # from auth_middleware, so anything rendered here is readable by
        # anyone who can reach the app. The generated admin password is
        # surfaced in the container logs at creation time instead; it must
        # never travel back out through an unauthenticated page.
        settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()
        user_count = conn.execute("SELECT COUNT(*) as count FROM users").fetchone()['count']
    
    # Check if this is first launch
    first_launch = user_count == 0
    
    return templates.TemplateResponse(request, "login.html", {
        "csp_nonce": get_csp_nonce(request),
        "first_launch": first_launch,
        "auth_enabled": settings['auth_enabled'] if settings else False
    })

@router.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login submission with rate limiting protection."""
    client_ip = get_client_ip(request)
    user_agent = request.headers.get("user-agent", "")
    
    # Check rate limit before processing login
    try:
        check_rate_limit(client_ip)
    except HTTPException as e:
        # Return user-friendly error page instead of raw exception
        return templates.TemplateResponse(request, "login.html", {
        "csp_nonce": get_csp_nonce(request),
            "error": e.detail,
            "first_launch": False,
            "auth_enabled": True,
            "rate_limited": True
        }, status_code=e.status_code)
    
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    
    if not user_row or not verify_password(password, user_row['password_hash']):
        # Record failed attempt and check if now locked
        is_locked = login_rate_limiter.record_failed_attempt(client_ip)
        log_login_attempt(username, client_ip, False, user_agent)
        
        error_msg = "Invalid username or password"
        if is_locked:
            error_msg = f"Too many failed login attempts. Your IP has been locked for {login_rate_limiter.lockout_seconds // 60} minutes."
        
        return templates.TemplateResponse(request, "login.html", {
        "csp_nonce": get_csp_nonce(request),
            "error": error_msg,
            "first_launch": False,
            "auth_enabled": True,
            "rate_limited": is_locked
        })
    
    # Successful login - clear rate limiting for this IP
    login_rate_limiter.record_successful_login(client_ip)
    log_login_attempt(username, client_ip, True, user_agent)
    
    # Update last login
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (datetime.now(), user_row['id']))
        conn.commit()
    
    # Set session
    request.session[SESSION_USER_KEY] = user_row['id']
    # The account password is deliberately NOT retained in the session. Feed
    # URLs are built from the user's feed token (see build_feed_auth_token).

    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)

@router.get("/logout")
async def logout(request: Request):
    """Handle logout."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

@router.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request, user: dict = Depends(require_auth)):
    """Display password change page."""
    with get_db_connection() as conn:
        settings = conn.execute("SELECT require_password_change FROM app_settings WHERE id = 1").fetchone()
    
    return templates.TemplateResponse(request, "change_password.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "required": settings['require_password_change'] if settings else False
    })

@router.post("/change-password", dependencies=[Depends(require_same_origin)])
async def change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...),
    user: dict = Depends(require_auth)
):
    """Handle password change submission."""
    if new_password != confirm_password:
        return templates.TemplateResponse(request, "change_password.html", {
        "csp_nonce": get_csp_nonce(request),
            "user": user,
            "error": "Passwords do not match"
        })
    
    # Verify current password
    with get_db_connection() as conn:
        user_row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user.id,)).fetchone()
    
    if not verify_password(current_password, user_row['password_hash']):
        return templates.TemplateResponse(request, "change_password.html", {
        "csp_nonce": get_csp_nonce(request),
            "user": user,
            "error": "Current password is incorrect"
        })
    
    # Update password
    new_hash = hash_password(new_password)
    with get_db_connection() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user.id))
        conn.execute("UPDATE app_settings SET require_password_change = 0 WHERE id = 1")
        conn.commit()
    
    # No password is kept in the session; the feed token is unaffected by a
    # password change and keeps working.

    return RedirectResponse(url="/admin/system?password_changed=1", status_code=status.HTTP_302_FOUND)

@router.get("/request-access", response_class=HTMLResponse)
async def request_access_page(request: Request):
    """Display access request form."""
    return templates.TemplateResponse(request, "request_access.html", {})

@router.post("/submit-access-request")
async def submit_access_request(
    request: Request,
    username: str = Form(...),
    email: str = Form(None),
    reason: str = Form(None)
):
    """Handle access request submission."""
    client_ip = get_client_ip(request)
    
    with get_db_connection() as conn:
        conn.execute(
            "INSERT INTO access_requests (username, email, reason, ip_address) VALUES (?, ?, ?, ?)",
            (username, email, reason, client_ip)
        )
        conn.commit()
    
    return templates.TemplateResponse(request, "request_access.html", {
        "csp_nonce": get_csp_nonce(request),
        "success": "Your access request has been submitted. You will be notified when it is reviewed."
    })

@router.get("/admin", response_class=RedirectResponse, dependencies=[Depends(require_admin_action)])
async def admin_root():
    return RedirectResponse(url="/admin/system")

@router.get("/settings", response_class=RedirectResponse)
async def view_settings_redirect():
    return RedirectResponse(url="/admin/system")

# --- Admin: System ---
@router.get("/admin/system", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_system(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse(request, "admin/system.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "settings": get_global_settings(),
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "system"
    })

@router.post("/admin/system/update", dependencies=[Depends(require_admin_action)])
async def update_system_settings(
    request: Request,
    concurrent_downloads: int = Form(2),
    retention_days: int = Form(30),
    check_interval_minutes: int = Form(60),
    app_external_url: str = Form(None),
    auth_enabled: bool = Form(False),
    ip_allowlist: str = Form(None),
    enable_feed_auth: bool = Form(False),
    feed_auth_username: str = Form(None),
    feed_auth_password: str = Form(None),
    redirect_to: str = Form(None)
):
    from app.infra.database import get_db_connection, ensure_global_feed_token

    # The submitted feed password is deliberately ignored and never stored.
    # Feed access is authenticated by a random feed token, not by any
    # password. The feed_auth_password column is left untouched here (dropping
    # it is a separate change); nothing reads it any more.
    del feed_auth_password

    # Standalone feed auth needs a username plus an install-wide feed token.
    # The token is generated on demand, so the only thing that can be missing
    # is the username.
    if enable_feed_auth and not auth_enabled:
        if not feed_auth_username:
            enable_feed_auth = False
        else:
            ensure_global_feed_token()

    with get_db_connection() as conn:
        # Get current settings
        current_settings = conn.execute("SELECT auth_enabled FROM app_settings WHERE id = 1").fetchone()

        # Check if auth is being enabled for the first time
        if auth_enabled and (not current_settings or not current_settings['auth_enabled']):
            # Check if ANY admin user exists (regardless of username)
            # This prevents re-creating 'admin' if the user renamed their account
            admin_exists = conn.execute("SELECT COUNT(*) as count FROM users WHERE is_admin = 1").fetchone()['count']
            
            if not admin_exists:
                # Create admin user with random password
                initial_password = generate_secure_password()
                password_hash = hash_password(initial_password)
                
                conn.execute(
                    "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                    ("admin", password_hash, 1)
                )
                
                # The plaintext password is NOT persisted. It used to be
                # written to app_settings.initial_password and rendered on
                # the unauthenticated /login page; the operator now reads it
                # once from the server log and nowhere else.
                conn.execute(
                    "UPDATE app_settings SET require_password_change = 1 WHERE id = 1"
                )
                logger.warning(
                    "AUTH - Created admin user 'admin' with one-time password: %s "
                    "(change it at first login; it is not stored anywhere)",
                    initial_password,
                )
        
        # Check if app_external_url is changing
        old_url = conn.execute("SELECT app_external_url FROM app_settings WHERE id = 1").fetchone()
        url_changed = old_url and old_url['app_external_url'] != app_external_url
        
        # Update settings
        conn.execute("""
            UPDATE app_settings SET concurrent_downloads = ?,
                retention_days = ?,
                check_interval_minutes = ?,
                app_external_url = ?,
                auth_enabled = ?,
                ip_allowlist = ?,
                enable_feed_auth = ?,
                feed_auth_username = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (concurrent_downloads, retention_days, check_interval_minutes, app_external_url,
              1 if auth_enabled else 0, ip_allowlist,
              1 if enable_feed_auth else 0,
              feed_auth_username if feed_auth_username else None))
        conn.commit()
    
    # Regenerate all feeds if the URL changed
    if url_changed:
        try:
            from app.core.rss_gen import RSSGenerator
            logger.info(f"Public URL changed to '{app_external_url}', regenerating all RSS feeds...")
            rss_gen = RSSGenerator()
            
            # Get all subscriptions
            from app.infra.repository import SubscriptionRepository
            sub_repo = SubscriptionRepository()
            subs = sub_repo.get_all()
            
            for sub in subs:
                rss_gen.generate_feed(sub.id)
            rss_gen.generate_unified_feed()
            
            logger.info(f"Successfully regenerated {len(subs) + 1} feeds with new URL.")
        except Exception as e:
            logger.error(f"Failed to regenerate feeds after URL change: {e}")
    
    url = redirect_to if redirect_to else "/admin/system?success=System+settings+updated"
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

# --- Admin: AI ---
@router.get("/admin/ai", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_ai(request: Request):
    from app.core.config import settings
    
    # helper to check which env vars are set
    env_keys = {
        "GEMINI_API_KEY": bool(settings.GEMINI_API_KEY),
        "OPENAI_API_KEY": bool(settings.OPENAI_API_KEY),
        "ANTHROPIC_API_KEY": bool(settings.ANTHROPIC_API_KEY),
        "OPENROUTER_API_KEY": bool(settings.OPENROUTER_API_KEY)
    }

    user = get_current_user(request)

    return templates.TemplateResponse(request, "admin/ai.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "settings": get_global_settings(),
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "ai",
        "env_keys": env_keys
    })

@router.post("/admin/ai/update", dependencies=[Depends(require_admin_action)])
async def update_ai_settings(
    request: Request,
    whisper_model: str = Form("base"),
    ai_model_cascade: str = Form(...),
    piper_model: str = Form("en_GB-cori-high.onnx"),
    active_ai_provider: str = Form("gemini"),
    openai_api_key: str = Form(None),
    anthropic_api_key: str = Form(None),
    openrouter_api_key: str = Form(None),
    gemini_api_keys: str = Form(None),
    openai_model: str = Form("gpt-4o"),
    anthropic_model: str = Form("claude-3-5-sonnet"),
    openrouter_model: str = Form("google/gemini-2.0-flash-001")
):
    from app.infra.database import get_db_connection
    import json
    try:
        json.loads(ai_model_cascade)
    except:
        ai_model_cascade = '["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]'
    
    # Validate gemini_api_keys is valid JSON array
    if gemini_api_keys:
        try:
            parsed_keys = json.loads(gemini_api_keys)
            if not isinstance(parsed_keys, list):
                gemini_api_keys = "[]"
        except:
            gemini_api_keys = "[]"
    else:
        gemini_api_keys = "[]"

    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings 
            SET whisper_model = ?,
                ai_model_cascade = ?,
                piper_model = ?,
                active_ai_provider = ?,
                openai_api_key = ?,
                anthropic_api_key = ?,
                openrouter_api_key = ?,
                gemini_api_keys = ?,
                openai_model = ?,
                anthropic_model = ?,
                openrouter_model = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        """, (
            whisper_model, ai_model_cascade, piper_model, active_ai_provider,
            openai_api_key, anthropic_api_key, openrouter_api_key, gemini_api_keys,
            openai_model, anthropic_model, openrouter_model
        ))
        conn.commit()
    return RedirectResponse(url="/admin/ai", status_code=303)

@router.post("/admin/ai/test", dependencies=[Depends(require_admin_action)])
async def test_ai_connection(
    provider: str = Form(...),
    api_key: str = Form(None),
    model: str = Form(None)
):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        
        # Create provider slightly differently depending on type to pass correct args
        # But our factory method handles it if we pass inputs
        # We need to map form inputs to factory args
        # The factory takes (provider_type, api_key, model, openrouter_key)
        # We passed api_key as generic.
        
        prov_instance = detector.create_provider(provider, api_key=api_key, model=model)
        result = prov_instance.test_connection()
        return {"status": "success", "message": result}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@router.get("/admin/ai/refresh/{provider}", dependencies=[Depends(require_admin_action)])
async def refresh_models(provider: str):
    try:
        from app.core.ai_services import AdDetector
        detector = AdDetector()
        # Create provider using saved settings (implies user must save key first usually, 
        # but we could allow passing key in query param if we wanted to be fancy. 
        # For now, rely on saved settings for Auth to keep it simple).
        prov_instance = detector.create_provider(provider) 
        models = prov_instance.list_models()
        return {"models": models}
    except Exception as e:
        return {"error": str(e)}

# --- Admin: Prompts ---
@router.get("/admin/prompts", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_prompts(request: Request):
    # Default prompts from ai_services.py
    default_prompts = {
        "ad_base": """Identify segments in the transcript that match the Targets.
Targets: {targets}
{custom_instr}
Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
Example: [{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}]""",
        "sponsor": "Sponsor messages, ad reads, promotional segments",
        "promo": "Cross-promotions, plugs for other shows or content",
        "summary": "Summarize the key points of this podcast episode in 3-5 bullet points."
    }
    
    user = get_current_user(request)

    return templates.TemplateResponse(request, "admin/prompts.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "settings": get_global_settings(),
        "default_prompts": default_prompts,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "prompts"
    })

@router.post("/admin/prompts", dependencies=[Depends(require_admin_action)])
async def save_prompts(request: Request):
    user = get_current_user(request)
    if not user or not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    form = await request.form()
    
    # Required variables for validation
    required_vars = {
        'ad_prompt_base': ['{targets}', '{custom_instr}'],
        'summary_prompt_template': ['{transcript_context}']
    }
    
    # Validate required variables
    for field, vars_needed in required_vars.items():
        value = form.get(field, '')
        for var in vars_needed:
            if var not in value:
                raise HTTPException(status_code=400, detail=f"{field} must include {var}")
    
    # Save to database
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings SET
                ad_prompt_base = ?,
                ad_target_sponsor = ?,
                ad_target_promo = ?,
                summary_prompt_template = ?
            WHERE id = 1
        """, (
            form.get('ad_prompt_base'),
            form.get('ad_target_sponsor'),
            form.get('ad_target_promo'),
            form.get('summary_prompt_template')
        ))
        conn.commit()
    
    return {"status": "success"}

@router.post("/admin/prompts/reset", dependencies=[Depends(require_admin_action)])
async def reset_prompts(request: Request):
    user = get_current_user(request)
    if not user or not getattr(user, 'is_admin', False):
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Default prompts
    defaults = {
        'summary': """You are a smart assistant. Write a short 2-3 sentence summary of this podcast episode.
The summary must:
1. NOT mention the podcast name, episode title, or date.
2. Start immediately with "This episode includes".
3. Briefly summarize key topics.
Transcript Context: {transcript_context}""",
        'ad_base': """Identify segments in the transcript that match the Targets.
Targets: {targets}
{custom_instr}
Return a JSON array of objects with "start", "end", "label" (Ad/Promo/Intro/Outro), and "reason" (brief explanation).
Example: [{"start": 0.0, "end": 10.0, "label": "Ad", "reason": "Sponsor read for XYZ"}]""",
        'sponsor': 'Sponsor messages, ad reads, promotional segments',
        'promo': 'Cross-promotions, plugs for other shows or content'
    }
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings SET
                summary_prompt_template = ?,
                ad_prompt_base = ?,
                ad_target_sponsor = ?,
                ad_target_promo = ?
            WHERE id = 1
        """, (defaults['summary'], defaults['ad_base'], defaults['sponsor'], defaults['promo']))
        conn.commit()
    
    return {"status": "success"}

# --- Admin: Queue ---
@router.get("/admin/queue", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_queue(request: Request):
    user = get_current_user(request)
    queue = ep_repo.get_queue()
    recently_processed = ep_repo.get_recently_processed(days=3)
    return templates.TemplateResponse(request, "admin/queue.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "queue": queue,
        "recently_processed": recently_processed,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "queue"
    })

@router.post("/admin/queue/cancel/{episode_id}", dependencies=[Depends(require_admin_action)])
async def cancel_episode(episode_id: int):
    # Soft delete an episode (marks as ignored, cleans up files)
    from app.core.processor import Processor
    proc = Processor()
    await proc.delete_episode(episode_id)
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/admin/queue/retry/{episode_id}", dependencies=[Depends(require_admin_action)])
async def retry_episode(episode_id: int):
    # Check if already processing?
    status = ep_repo.get_status(episode_id)
    if status == 'processing':
         return RedirectResponse(url="/admin/queue", status_code=303)
         
    # Force to pending (Background processor will pick it up)
    from app.core.processor import Processor
    proc = Processor()
    await proc.version_episode(episode_id)
    ep_repo.update_status(episode_id, "pending")
    return RedirectResponse(url="/admin/queue", status_code=303)

@router.post("/api/episodes/{episode_id}/reprocess", dependencies=[Depends(require_user_action)])
async def api_reprocess_episode(episode_id: int, skip_transcription: bool = False):
    import json
    logger.info(f"Reprocess request for {episode_id} with skip_transcription={skip_transcription}")
    
    # API version of retry - force status to pending
    current_status = ep_repo.get_status(episode_id)
    if current_status == 'processing':
         return {"status": "ignored", "reason": "already_processing"}
    
    # Set processing flags (like subscriptions.py does)
    flags = {'skip_transcription': skip_transcription}
    flags_json = json.dumps(flags)
    
    # Reset status with flags so processor respects skip_transcription
    from app.core.processor import Processor
    proc = Processor()
    await proc.version_episode(episode_id)
    ep_repo.reset_status(episode_id, processing_flags=flags_json)
    ep_repo.update_status(episode_id, "pending")
    return {"status": "ok"}

@router.post("/api/episodes/{episode_id}/ignore", dependencies=[Depends(require_user_action)])
async def api_ignore_episode(episode_id: int):
    # API version of cancel/delete - soft delete
    from app.core.processor import Processor
    proc = Processor()
    await proc.delete_episode(episode_id)
    return {"status": "ok"}

@router.post("/episodes/{episode_id}/download", dependencies=[Depends(require_user_action)])
async def manual_download_episode(episode_id: int, request: Request):
    # Update DB to pending
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("UPDATE episodes SET is_manual_download=1, status='pending' WHERE id=?", (episode_id,))
        conn.commit()
    
    # Background processor will see 'pending' and pick it up (polls every 10s)
    
    return RedirectResponse(url=request.headers.get("referer") or "/", status_code=303)


# --- Admin: Logs ---
@router.get("/admin/logs", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_logs(request: Request, lines: int = 1000, level: str = "ALL"):
    from app.core.config import settings
    log_path = os.path.join(settings.DATA_DIR, "app.log")
    logs = ""
    
    if os.path.exists(log_path):
        try:
            # Read relevant lines
            # For simplicity, read last N bytes then filter lines
            # Reading 1MB roughly
            with open(log_path, "r", encoding="utf-8") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 1024 * 1024)) # 1MB
                raw_logs = f.read()
                
            log_lines = raw_logs.splitlines()
            
            # Simple Filter
            filtered = []
            for line in log_lines:
                if level != "ALL" and level not in line:
                    continue
                filtered.append(line)
                
            # Take last N
            logs = "\n".join(filtered[-lines:])
            
        except Exception as e:
            logs = f"Error reading logs: {e}"
    else:
        logs = "Log file not found."

    
    user = get_current_user(request)

    return templates.TemplateResponse(request, "admin/logs.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "logs": logs,
        "pending_requests_count": get_pending_requests_count(),
        "active_tab": "logs",
        "current_lines": lines,
        "current_level": level
    })

# --- Admin: Access ---

@router.get("/subscribe/apple", response_class=HTMLResponse)
async def apple_subscribe_page(request: Request, url: str):
    """Render the Apple Podcasts subscription instruction page."""
    return templates.TemplateResponse(request, "apple_subscribe.html", {
        "csp_nonce": get_csp_nonce(request),
        "feed_url": url
    })

@router.get("/admin/access", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_access(request: Request):
    from app.infra.database import get_db_connection
    from datetime import datetime, timedelta
    
    # Load settings and user
    settings = get_global_settings()
    user = get_current_user(request)

    # Consume any one-shot flash (e.g. a freshly minted temporary password).
    # Popped, so a reload of this page will not show it again.
    flash = pop_flash(request)

    # Load pending access requests
    with get_db_connection() as conn:
        pending_requests = conn.execute(
            "SELECT * FROM access_requests WHERE status = 'pending' ORDER BY requested_at DESC"
        ).fetchall()
        
        # Load login history for last 30 days
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
        login_history = conn.execute(
            """SELECT * FROM login_attempts 
               WHERE timestamp > ? 
               ORDER BY timestamp DESC 
               LIMIT 100""",
            (thirty_days_ago,)
        ).fetchall()
        
        # Load active users
        active_users = conn.execute(
            "SELECT * FROM users ORDER BY created_at DESC"
        ).fetchall()
    
    return templates.TemplateResponse(request, "admin/access_requests.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "active_tab": "access",
        "settings": settings,
        "app_base_url": get_app_base_url(settings, request),
        "pending_requests": [dict(row) for row in pending_requests],
        "active_users": [dict(row) for row in active_users],
        "login_history": [dict(row) for row in login_history],
        "pending_requests_count": get_pending_requests_count(),
        "flash": flash,
    })


@router.post("/feed-token/rotate", dependencies=[Depends(require_user_action)])
async def rotate_own_feed_token(request: Request):
    """Rotate the current user's feed token, revoking every existing feed URL.

    Revocability is the point of a feed token: the old ?auth= value stops
    working immediately and the account password is untouched.

    `require_user_action`, not `require_admin_action`. Per-user revocability is
    the whole reason the token exists, and the dashboard renders the "Rotate
    Feed Token" button for every signed-in user whenever feed auth is on -
    under an admin gate a non-admin saw the button, got a 403 on click, and
    could never revoke their own leaked feed URL by any route at all. There is
    no privilege to protect here: the route reads the caller's own identity
    from the session and can only ever rewrite that user's row.

    The standalone branch below rotates the install-wide token, and it stays
    reachable under the same guard - not as a relaxation but because there is
    nothing to relax. That branch is selected by `auth_enabled = 0`, a
    deployment with no users and no sessions, where `require_admin` already
    degraded to a synthetic dummy admin and rejected nobody. "Admin-only" was
    never expressible there; `require_same_origin` was the entire boundary
    under the old dependency and remains the entire boundary under the new
    one, with the IP allowlist behind it. Conversely a non-admin in multi-user
    mode cannot reach the standalone branch to rotate the shared token,
    because the branch is chosen by the same `auth_enabled` flag that gave
    them a session in the first place.
    """
    from app.infra.database import rotate_feed_token, ensure_global_feed_token

    global_settings = get_global_settings()

    if global_settings.get('auth_enabled'):
        user = get_current_user(request)
        user_id = getattr(user, "id", None)
        if not user_id:
            return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        rotate_feed_token(user_id)
        logger.info(f"AUTH - Feed token rotated for user id {user_id}")
    else:
        # Standalone mode: rotate the install-wide token by clearing and
        # regenerating it.
        from app.infra.database import get_db_connection
        with get_db_connection() as conn:
            conn.execute("UPDATE app_settings SET feed_auth_token = NULL WHERE id = 1")
            conn.commit()
        ensure_global_feed_token()
        logger.info("AUTH - Global feed token rotated")

    set_flash(request, feed_token_rotated=True)
    return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)

@router.post("/admin/users/{user_id}/password", dependencies=[Depends(require_admin_action)])
async def admin_change_user_password(
    request: Request, 
    user_id: int, 
    password: str = Form(...),
    admin_user: dict = Depends(require_admin)
):
    """Admin route to force change a user's password."""
    # Prevent changing own password via this route (use /change-password instead)
    if user_id == admin_user.id:
        return RedirectResponse(
            url="/admin/access?error=Use+My+Profile+to+change+your+own+password", 
            status_code=status.HTTP_303_SEE_OTHER
        )

    # Hash new password
    new_hash = hash_password(password)
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        # Check if user exists
        user = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if not user:
            return RedirectResponse(
                url="/admin/access?error=User+not+found", 
                status_code=status.HTTP_303_SEE_OTHER
            )
            
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?", 
            (new_hash, user_id)
        )
        conn.commit()
    
    return RedirectResponse(
        url=f"/admin/access?success=Password+updated+for+{user['username']}", 
        status_code=status.HTTP_303_SEE_OTHER
    )

@router.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin_action)])
async def delete_user(user_id: int, request: Request, user: dict = Depends(require_admin)):
    # Check admin
    if user.id == user_id:
        return RedirectResponse(
            url="/admin/access?error=Cannot delete your own account", 
            status_code=status.HTTP_303_SEE_OTHER
        )

    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    
    return RedirectResponse(
        url="/admin/access?success=User deleted successfully", 
        status_code=status.HTTP_303_SEE_OTHER
    )

# --- Admin: Approve Access Request ---
@router.post("/admin/access-requests/{request_id}/approve", dependencies=[Depends(require_admin_action)])
async def approve_access_request(request: Request, request_id: int):
    from app.infra.database import get_db_connection
    from app.web.auth_utils import hash_password, generate_secure_password
    
    with get_db_connection() as conn:
        # Get the access request
        access_req = conn.execute(
            "SELECT * FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access?error=Request+not+found", status_code=303)
        
        # Check if username already exists
        existing_user = conn.execute(
            "SELECT id FROM users WHERE username = ?", (access_req['username'],)
        ).fetchone()
        
        if existing_user:
            # Update request status to denied with reason
            conn.execute(
                "UPDATE access_requests SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
                (request_id,)
            )
            conn.commit()
            return RedirectResponse(url="/admin/access?error=Username+already+exists", status_code=303)
        
        # Generate random password for the new user
        temp_password = generate_secure_password()
        password_hash = hash_password(temp_password)
        
        # Create the new user
        conn.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)",
            (access_req['username'], password_hash)
        )
        
        # Update access request status
        conn.execute(
            "UPDATE access_requests SET status = 'approved', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request approved: {access_req['username']} - Temp password generated")

    # Hand the temporary password to the next render via a one-shot session
    # flash. It must never go in the URL: query strings are recorded in
    # browser history, server access logs, and outbound Referer headers.
    set_flash(
        request,
        approved_username=access_req['username'],
        approved_temp_password=temp_password,
    )
    return RedirectResponse(url="/admin/access", status_code=303)

# --- Admin: Deny Access Request ---
@router.post("/admin/access-requests/{request_id}/deny", dependencies=[Depends(require_admin_action)])
async def deny_access_request(request: Request, request_id: int):
    from app.infra.database import get_db_connection
    
    with get_db_connection() as conn:
        # Get the access request to log username
        access_req = conn.execute(
            "SELECT username FROM access_requests WHERE id = ?", (request_id,)
        ).fetchone()
        
        if not access_req:
            return RedirectResponse(url="/admin/access?error=Request+not+found", status_code=303)
        
        # Update access request status to denied
        conn.execute(
            "UPDATE access_requests SET status = 'denied', reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (request_id,)
        )
        conn.commit()
        
        logger.info(f"AUTH - Access request denied: {access_req['username']}")
        
    return RedirectResponse(url="/admin/access?denied=1", status_code=303)

# --- Admin: Update User Username ---
@router.post("/admin/users/{user_id}/username", dependencies=[Depends(require_admin_action)])
async def update_user_username(request: Request, user_id: int, username: str = Form(...)):
    require_admin(request)
    with get_db_connection() as conn:
        # Check if username already exists
        existing = conn.execute("SELECT id FROM users WHERE username = ? AND id != ?", (username, user_id)).fetchone()
        if existing:
            return RedirectResponse(url="/admin/access?error=Username already exists", status_code=303)
            
        conn.execute("UPDATE users SET username = ? WHERE id = ?", (username, user_id))
        conn.commit()
    return RedirectResponse(url="/admin/access", status_code=303)

# --- Admin: Update Request Username ---
@router.post("/admin/access-requests/{request_id}/username", dependencies=[Depends(require_admin_action)])
async def update_request_username(request: Request, request_id: int, username: str = Form(...)):
    require_admin(request)
    with get_db_connection() as conn:
        conn.execute("UPDATE access_requests SET username = ? WHERE id = ?", (username, request_id))
        conn.commit()
    return RedirectResponse(url="/admin/access", status_code=303)

# Helper to render index with consistent data
def _render_index(request: Request, error: str = None):
    from app.infra.repository import SubscriptionRepository
    sub_repo = SubscriptionRepository()
    subs = sub_repo.get_all()
    
    # Calculate stats
    total_podcasts = len(subs)
    total_episodes = 0
    total_duration = 0 # seconds
    total_size = 0 # bytes
    
    from app.infra.database import get_db_connection
    with get_db_connection() as conn:
        rows = conn.execute("SELECT duration, file_size FROM episodes WHERE status = 'completed'").fetchall()
        total_episodes = len(rows)
        for row in rows:
            if row['duration']: total_duration += row['duration']
            if row['file_size']: total_size += row['file_size']
            
    stats = {
        "podcasts": total_podcasts,
        "episodes": total_episodes,
        "hours": round(total_duration / 3600, 1),
        "size_gb": round(total_size / (1024 * 1024 * 1024), 2)
    }

    user = get_current_user(request)
    
    subs_with_links = []
    global_settings = get_global_settings()

    # A stale-but-real config must degrade, not 500.
    #
    # `build_feed_auth_token` raises when feed auth is enabled and the identity
    # half of the credential is missing - in practice `feed_auth_username` NULL
    # on a standalone install. No current save path can produce that, but the
    # code this replaced tolerated it with an `or 'feed'` fallback, so an
    # upgraded database can genuinely hold it. Left unhandled it took out the
    # dashboard, which is the app's front page: the operator's first signal
    # that a setting was stale was the whole site erroring.
    #
    # The old fallback is deliberately not restored. It built a URL the
    # unified-feed validator answers with 401 - a feed that looks fine and
    # silently never updates. Emitting the links without a credential and
    # naming the setting in a banner is the honest failure.
    feed_auth_error = None
    if is_feed_auth_enabled(global_settings):
        try:
            build_feed_auth_token(global_settings, user)
        except RuntimeError as exc:
            feed_auth_error = str(exc)
            logger.warning(
                "Feed auth is enabled but no feed URL can be built, so the "
                "dashboard is rendering unauthenticated links: %s", exc
            )

    # Link-building settings only. `global_settings` itself is still passed to
    # the template untouched, so `settings.enable_feed_auth` keeps reflecting
    # what is actually configured rather than what happens to work.
    link_settings = global_settings
    if feed_auth_error:
        link_settings = dict(global_settings)
        link_settings['enable_feed_auth'] = 0

    for sub in subs:
        # Get completed episodes for this subscription
        with get_db_connection() as conn:
            episodes = conn.execute(
                """SELECT title, pub_date as published_date, status FROM episodes 
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 10""",
                (sub.id,)
            ).fetchall()
            
            # Get latest episode with AI summary
            latest_ep = conn.execute(
                """SELECT id, title, description, ai_summary, pub_date FROM episodes 
                   WHERE subscription_id = ? AND status = 'completed'
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()
            
            # Get the latest episode date (any status) for filtering/sorting
            latest_any_ep = conn.execute(
                """SELECT pub_date FROM episodes 
                   WHERE subscription_id = ?
                   ORDER BY pub_date DESC LIMIT 1""",
                (sub.id,)
            ).fetchone()
            
            # Count processing/pending episodes for this subscription
            processing_row = conn.execute(
                """SELECT COUNT(*) as count FROM episodes 
                   WHERE subscription_id = ? AND status IN ('processing', 'pending')""",
                (sub.id,)
            ).fetchone()
            processing_count = processing_row['count'] if processing_row else 0
        
        latest_summary = None
        latest_description = None
        latest_episode_date = None
        if latest_ep:
            latest_summary = latest_ep['ai_summary']
            latest_description = latest_ep['description']
        if latest_any_ep and latest_any_ep['pub_date']:
            # Convert to ISO format string for safe JS parsing
            d = latest_any_ep['pub_date']
            if hasattr(d, 'isoformat'):
                latest_episode_date = d.isoformat()
            else:
                latest_episode_date = str(d)
        
        subs_with_links.append({
            "sub": sub,
            "links": generate_rss_links(request, sub, link_settings, user),
            "episodes": [dict(ep) for ep in episodes],
            "episode_count": len(episodes),
            "processing_count": processing_count,
            "total_listens": ep_repo.get_subscription_listen_count(sub.id),
            "latest_ai_summary": latest_summary,
            "latest_description": latest_description,
            "latest_episode_date": latest_episode_date
        })

    # Get queue data for dashboard display
    queue = ep_repo.get_queue()

    user = get_current_user(request)
    
    # Determine if AI is configured (DB Overrides/Augments Env)
    from app.core.config import settings
    config_warning = not any([
        settings.GEMINI_API_KEY,
        settings.OPENAI_API_KEY,
        settings.ANTHROPIC_API_KEY,
        settings.OPENROUTER_API_KEY,
        global_settings.get('gemini_api_key'),
        global_settings.get('openai_api_key'),
        global_settings.get('anthropic_api_key'),
        global_settings.get('openrouter_api_key')
    ])

    # Generate Unified Links if subscriptions exist
    unified_links = None
    if subs:
        # Determine Base URL using consolidated logic
        base_url = get_app_base_url(global_settings, request)
        
        rss_url = append_feed_auth(
            f"{base_url}/feed/unified.xml", link_settings, user
        )

        unified_links = {
            "rss": rss_url,
            "direct": rss_url,
            "apple": rss_url,  # Method 1: Direct HTTPS URL
            "pocket_casts": f"pktc://subscribe/{rss_url}",
            "overcast": f"overcast://x-callback-url/add?url={rss_url}",
            "castbox": f"castbox://subscribe?url={rss_url}",
            "podcast_addict": f"podcastaddict://subscribe/{rss_url}"
        }

    return templates.TemplateResponse(request, "index.html", {
        "csp_nonce": get_csp_nonce(request), 
        "user": user,
        "subscriptions": subs_with_links, 
        "stats": stats,
        "error": error,
        "config_warning": config_warning,
        "queue": queue,
        "unified_links": unified_links,
        "settings": global_settings,
        "flash": pop_flash(request),
        "feed_url_notice": feed_url_notice_pending(global_settings, len(subs)),
        "feed_auth_error": feed_auth_error,
    })

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render_index(request)

from app.core.processor import Processor

# --- Admin: Global Subscription Settings ---
@router.get("/admin/global-subscription-settings", response_class=HTMLResponse, dependencies=[Depends(require_admin_action)])
async def admin_global_subscription_settings(request: Request):
    user = get_current_user(request)
    
    with get_db_connection() as conn:
        settings_row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        
    return templates.TemplateResponse(request, "admin/global_subscription_settings.html", {
        "csp_nonce": get_csp_nonce(request),
        "user": user,
        "settings": settings_row,
        "active_tab": "global_subs"
    })

@router.post("/admin/global-subscription-settings/update", dependencies=[Depends(require_admin_action)])
async def update_global_subscription_settings(
    request: Request,
    default_remove_ads: bool = Form(False),
    default_remove_promos: bool = Form(False),
    default_remove_intros: bool = Form(False),
    default_remove_outros: bool = Form(False),
    default_ai_rewrite_description: bool = Form(False),
    default_ai_audio_summary: bool = Form(False),
    default_append_title_intro: bool = Form(False),
    default_retention_limit: int = Form(1),
    default_retention_days: int = Form(30),
    default_manual_retention_days: int = Form(14),
    default_custom_instructions: str = Form(None)
):
    with get_db_connection() as conn:
        conn.execute("""
            UPDATE app_settings 
            SET default_remove_ads = ?, 
                default_remove_promos = ?, 
                default_remove_intros = ?, 
                default_remove_outros = ?, 
                default_ai_rewrite_description = ?,
                default_ai_audio_summary = ?, 
                default_append_title_intro = ?,
                default_retention_limit = ?,
                default_retention_days = ?,
                default_manual_retention_days = ?,
                default_custom_instructions = ?
            WHERE id = 1
        """, (
            default_remove_ads, default_remove_promos, default_remove_intros, default_remove_outros,
            default_ai_rewrite_description, default_ai_audio_summary, default_append_title_intro,
            default_retention_limit, default_retention_days, default_manual_retention_days,
            default_custom_instructions
        ))
        conn.commit()
        
    return RedirectResponse(url="/admin/global-subscription-settings?success=Settings updated", status_code=303)


@router.post("/add", response_class=HTMLResponse, dependencies=[Depends(require_user_action)])
async def add_subscription(request: Request, background_tasks: BackgroundTasks, feed_url: str = Form(...), initial_count: int = Form(1)):
    try:
        # Check if exists (quick DB check)
        existing = sub_repo.get_by_url(feed_url)
        if existing:
            return _render_index(request, error="Subscription already exists")
        
        # Create subscription with placeholder data
        # Fetch global defaults first
        with get_db_connection() as conn:
            app_settings = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
            
        # Use user-provided initial_count (from UI dropdown) as retention limit
        # The UI defaults this dropdown to the global default setting already.
        retention_limit = initial_count
        
        sub_create = SubscriptionCreate(feed_url=feed_url)
        new_sub = sub_repo.create(sub_create, "Loading...", f"loading-{int(__import__('time').time())}", None, "Fetching feed information...", retention_limit=retention_limit)
        
        # Apply other global defaults immediately
        sub_repo.update_settings(
            new_sub.id,
            remove_ads=bool(app_settings['default_remove_ads']),
            remove_promos=bool(app_settings['default_remove_promos']),
            remove_intros=bool(app_settings['default_remove_intros']),
            remove_outros=bool(app_settings['default_remove_outros']),
            custom_instructions=app_settings['default_custom_instructions'],
            append_summary=bool(app_settings['default_ai_audio_summary']), # Mapped correctly? Yes
            append_title_intro=bool(app_settings['default_append_title_intro']),
            ai_rewrite_description=bool(app_settings['default_ai_rewrite_description']),
            ai_audio_summary=bool(app_settings['default_ai_audio_summary']),
            retention_days=app_settings['default_retention_days'] or 30,
            manual_retention_days=app_settings['default_manual_retention_days'] or 14,
            retention_limit=retention_limit
        )

        
        # All heavy lifting happens in background
        async def setup_subscription(sub_id: int, url: str, limit: int):
            from app.core.processor import Processor
            from app.core.feed import FeedManager
            from app.infra.database import get_db_connection
            
            try:
                # Parse feed (network call)
                title, slug, image_url, description = FeedManager.parse_feed(url)
                
                # Update subscription with real data
                # Keep the settings we just set! Only update metadata.
                with get_db_connection() as conn:
                    conn.execute("""
                        UPDATE subscriptions 
                        SET title = ?, slug = ?, image_url = ?, description = ?
                        WHERE id = ?
                    """, (title, slug, image_url, description, sub_id))
                    conn.commit()
                
                # Now check feeds and process queue
                proc = Processor()
                await proc.check_feeds(subscription_id=sub_id, limit=limit)
                await proc.process_queue()
                
            except Exception as e:
                logger.error(f"Error setting up subscription {sub_id}: {e}")
        
        background_tasks.add_task(setup_subscription, new_sub.id, feed_url, retention_limit)
        
        return RedirectResponse(url="/", status_code=303)
    except Exception as e:
        return _render_index(request, error=str(e))

@router.get("/subscriptions/{id}", response_class=HTMLResponse)
async def view_subscription(request: Request, id: int):
    sub = sub_repo.get_by_id(id)
    if not sub:
        return RedirectResponse(url="/")
    
    # Initial page size for lazy loading
    INITIAL_PAGE_SIZE = 20
    
    # Get first batch of episodes using pagination
    episodes = ep_repo.get_by_subscription_paginated(id, limit=INITIAL_PAGE_SIZE, offset=0)
    total_episodes = ep_repo.count_by_subscription(id)
    has_more = total_episodes > INITIAL_PAGE_SIZE
    
    def format_duration(seconds: int) -> str:
        if not seconds:
            return "-"
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    # Generate Links
    global_settings = get_global_settings()
    
    # Get current user for nav bar and links
    from app.web.auth import get_current_user
    user = get_current_user(request)

    links = generate_rss_links(request, sub, global_settings, user)
    
    # Get total listen count for this subscription
    total_listens = ep_repo.get_subscription_listen_count(sub.id)

    return templates.TemplateResponse(request, "episodes.html", {
        "csp_nonce": get_csp_nonce(request), 
        "user": user,
        "subscription": sub, 
        "episodes": episodes,
        "links": links,
        "basename": lambda p: p.split('/')[-1] if p else '',
        "format_duration": format_duration,
        "total_listens": total_listens,
        "total_episodes": total_episodes,
        "has_more": has_more,
        "page_size": INITIAL_PAGE_SIZE
    })

@router.get("/api/subscriptions/{id}/episodes")
async def get_subscription_episodes_api(id: int, limit: int = 20, offset: int = 0, search: str = None):
    """Return episodes for a subscription as JSON for lazy loading. Supports search by title."""
    sub = sub_repo.get_by_id(id)
    if not sub:
        raise HTTPException(status_code=404, detail="Subscription not found")
    
    # Pass search to repository methods
    episodes = ep_repo.get_by_subscription_paginated(id, limit=limit, offset=offset, search=search)
    total = ep_repo.count_by_subscription(id, search=search)
    
    # Convert sqlite rows to dicts
    episodes_data = []
    for ep in episodes:
        ep_dict = dict(ep)
        # Ensure pub_date is a string for JSON serialization
        if ep_dict.get('pub_date') and hasattr(ep_dict['pub_date'], 'isoformat'):
            ep_dict['pub_date'] = ep_dict['pub_date'].isoformat()
        episodes_data.append(ep_dict)
    
    return {
        "episodes": episodes_data,
        "total": total,
        "offset": offset,
        "limit": limit,
        "search": search,
        "has_more": offset + len(episodes) < total,
        "subscription_slug": sub.slug
    }

@router.post("/subscriptions/{id}/settings", dependencies=[Depends(require_user_action)])
async def update_settings(
    id: int,
    background_tasks: BackgroundTasks,
    remove_ads: bool = Form(False),
    remove_promos: bool = Form(False),
    remove_intros: bool = Form(False),
    remove_outros: bool = Form(False),
    custom_instructions: str = Form(None),
    append_summary: bool = Form(False),
    append_title_intro: bool = Form(False),
    ai_rewrite_description: bool = Form(False),
    ai_audio_summary: bool = Form(False),
    retention_days: int = Form(30),
    manual_retention_days: int = Form(14),
    retention_limit: int = Form(1)
):
    sub_repo.update_settings(
        id, 
        remove_ads, 
        remove_promos, 
        remove_intros, 
        remove_outros, 
        custom_instructions, 
        append_summary, 
        append_title_intro,
        ai_rewrite_description,
        ai_audio_summary,
        retention_days,
        manual_retention_days,
        retention_limit
    )
    
    # Trigger processing if any ads/promos settings were changed
    from app.core.processor import Processor
    proc = Processor()
    
    async def post_update_tasks(sub_id):
        await proc.cleanup_old_episodes()
        await proc.check_feeds(sub_id)
        await proc.process_queue()

    background_tasks.add_task(post_update_tasks, id)
    return RedirectResponse(url=f"/subscriptions/{id}", status_code=303)

from fastapi.responses import FileResponse

@router.get("/episodes/{id}/transcript")
async def view_transcript(id: int, request: Request):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    import json
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.id, e.title, e.pub_date, e.duration, e.guid, s.slug as subscription_slug, e.transcript_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Episode not found")
            
        transcript_path = row['transcript_path']
        
        # Check standard paths if not recorded in DB or file missing
        if not transcript_path or not os.path.exists(transcript_path):
             episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
             potential_path = os.path.join(
                settings.get_episode_dir(row['subscription_slug'], episode_slug),
                "transcript.json"
            )
             if os.path.exists(potential_path):
                 transcript_path = potential_path
        
        if not transcript_path or not os.path.exists(transcript_path):
             raise HTTPException(status_code=404, detail="Transcript file not found")
             
        try:
            with open(transcript_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading transcript: {str(e)}")
    
    def format_duration(seconds: int) -> str:
        if not seconds:
            return "-"
        seconds = int(seconds)  # Convert to int to handle floats
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    return templates.TemplateResponse(request, "transcript.html", {
        "csp_nonce": get_csp_nonce(request),
        "episode": row,
        "transcript_data": data,
        "format_duration": format_duration
    })

@router.get("/artifacts/transcript/{id}")
async def get_transcript_json(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.guid, s.slug, e.transcript_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if row:
            # Try new hierarchical structure first
            episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
            new_path = os.path.join(
                settings.get_episode_dir(row['slug'], episode_slug),
                "transcript.json"
            )
            if os.path.exists(new_path):
                return FileResponse(new_path)
            
            # Fallback to old path for backward compatibility
            if row['transcript_path'] and os.path.exists(row['transcript_path']):
                return FileResponse(row['transcript_path'])
                
    raise HTTPException(status_code=404, detail="Transcript not found")

@router.get("/artifacts/report/{id}")
async def get_report(id: int):
    from app.infra.database import get_db_connection
    from app.core.config import settings
    
    with get_db_connection() as conn:
        row = conn.execute(
            """SELECT e.guid, s.slug, e.report_path, e.ad_report_path 
               FROM episodes e 
               JOIN subscriptions s ON e.subscription_id = s.id 
               WHERE e.id = ?""", 
            (id,)
        ).fetchone()
        
        if row:
            episode_slug = f"{row['guid']}".replace("/", "_").replace(" ", "_")
            episode_dir = settings.get_episode_dir(row['slug'], episode_slug)
            
            # Try new hierarchical structure first (prefer HTML)
            html_path = os.path.join(episode_dir, "report.html")
            if os.path.exists(html_path):
                return FileResponse(html_path)
            
            json_path = os.path.join(episode_dir, "report.json")
            if os.path.exists(json_path):
                return FileResponse(json_path)
            
            # Fallback to old paths for backward compatibility
            if row['report_path'] and os.path.exists(row['report_path']):
                return FileResponse(row['report_path'])
            if row['ad_report_path'] and os.path.exists(row['ad_report_path']):
                return FileResponse(row['ad_report_path'])
            
    raise HTTPException(status_code=404, detail="Report not found")

def _presented_feed_credential(request: Request):
    """The exact ``?auth=`` value to carry from a feed request into its
    enclosure URLs, or None.

    Feed and audio requests are authorised in exactly one place -
    ``app.web.middleware.feed_auth_middleware`` - which has already run and
    approved this request by the time a route body executes. These routes
    therefore do no validation of their own; they only need to forward a
    credential the client can reuse on the audio URLs the feed points at.

    Forwarding the caller's own credential verbatim is what keeps the two
    ends in agreement. The previous code decoded it, split it on ':' and
    re-encoded a base64("user:token") envelope, which meant a client that
    presented a bare token (a shape the middleware accepts) got audio URLs
    with no credential at all, and a 401 on every download.
    """
    presented = request.query_params.get('auth')
    if presented:
        return presented

    auth_header = request.headers.get('Authorization') or ''
    if auth_header.startswith('Basic '):
        # Hand back the same Basic blob; the middleware accepts it on the
        # audio routes too.
        encoded = auth_header[len('Basic '):].strip()
        if encoded:
            return encoded
    return None


def _inject_enclosure_credential(xml_content: str, credential: str) -> str:
    """Append ?auth=<credential> to every enclosure URL in the feed."""
    import re

    def inject(match):
        url = match.group(2)
        separator = "&" if "?" in url else "?"
        return f'{match.group(1)}{url}{separator}auth={credential}'

    return re.sub(r'(enclosure\s+url=")(https?://[^"]+)', inject, xml_content)


@router.get("/feeds/{slug}.xml")
async def get_individual_feed(slug: str, request: Request):
    """Serve individual podcast RSS feed with optional token injection for audio URLs."""
    from app.infra.repository import SubscriptionRepository
    from app.core.rss_gen import RSSGenerator
    from app.core.config import settings as app_settings
    from fastapi.responses import FileResponse, Response

    sub_repo = SubscriptionRepository()
    sub = sub_repo.get_by_slug(slug)
    if not sub:
        raise HTTPException(status_code=404, detail="Feed not found")
        
    file_path = os.path.join(app_settings.FEEDS_DIR, f"{slug}.xml")
    if not os.path.exists(file_path):
        gen = RSSGenerator()
        gen.generate_feed(sub.id)
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Feed generation failed")
    
    settings = get_global_settings()
    cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    # Authorisation already happened in feed_auth_middleware. All that is left
    # is forwarding the caller's credential onto the audio URLs.
    credential = _presented_feed_credential(request)
    if is_feed_auth_enabled(settings) and credential:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = _inject_enclosure_credential(f.read(), credential)
            return Response(content=xml_content, media_type="application/xml", headers=cache_headers)
        except Exception as e:
            logger.error(f"Error injecting auth into feed {slug}: {e}")

    return FileResponse(file_path, media_type="application/xml", headers=cache_headers)

@router.get("/feed/unified")
@router.get("/feed/unified.xml")
async def get_unified_feed(request: Request):
    """Serve the unified RSS feed.

    This route does NOT authorise. `/feed/` is covered by
    `app.web.middleware.feed_auth_middleware`, exactly like `/feeds/` and
    `/audio/`, so reaching this body means the credential already passed.

    The hand-rolled `Authorization`/`?auth=` check that used to live here was
    a second, independently maintained gate, and the two had drifted: the
    middleware treats the credential as an opaque token and ignores any
    username, while this route additionally demanded
    `username == feed_auth_username`. A bare token therefore passed the
    middleware and was then rejected here, and a URL built for one gate was
    invalid at the other. One gate, in one place.
    """
    from fastapi.responses import FileResponse, Response
    from app.core.config import settings as app_settings

    settings = get_global_settings()

    file_path = os.path.join(app_settings.FEEDS_DIR, "unified.xml")
    if not os.path.exists(file_path):
        # Generate on demand if missing
        from app.core.rss_gen import RSSGenerator
        gen = RSSGenerator()
        gen.generate_unified_feed()

    cache_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }

    # Forward the caller's own verified credential onto the audio URLs, so a
    # download is authorised by the same value that authorised the feed.
    credential = _presented_feed_credential(request)
    if is_feed_auth_enabled(settings) and credential:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_content = _inject_enclosure_credential(f.read(), credential)
            return Response(content=xml_content, media_type="application/xml", headers=cache_headers)
        except Exception as e:
            logger.error(f"Error injecting credentials into unified feed: {e}")
            # Fallback to the static file if injection fails.

    return FileResponse(file_path, media_type="application/xml", headers=cache_headers)

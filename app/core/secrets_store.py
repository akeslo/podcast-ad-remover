"""
Secrets that must never be a constant and never sit in the database in clear.

Two things live here:

* ``resolve_session_secret()`` - the SessionMiddleware signing key. Order:
  ``SESSION_SECRET_KEY`` env var, then ``SESSION_SECRET_KEY_FILE``, then a
  random key generated once and persisted (mode 0600) under the data dir so
  sessions survive restarts. There is no default value anywhere in source.

* ``encrypt_secret`` / ``decrypt_secret`` - Fernet encryption for the provider
  API keys stored in ``app_settings``. Key order: ``SECRETS_ENCRYPTION_KEY``
  env var (a Fernet key), else a generated key file under the data dir.

  Honest scope: with the generated key file, the key sits on the same volume
  as the database. That protects against a leaked DB file, backup or SQL dump,
  not against someone with the whole volume. Set ``SECRETS_ENCRYPTION_KEY``
  from a real secret store for more than that.

Encrypted values carry an ``enc:v1:`` prefix. ``decrypt_secret`` passes a
non-prefixed value through unchanged so a not-yet-migrated row still works;
``init_db`` migrates every plaintext value on upgrade.
"""
import json
import logging
import os
import secrets
from typing import Iterable, Optional

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

ENC_PREFIX = "enc:v1:"

# app_settings columns holding a single secret string.
SECRET_COLUMNS = (
    "gemini_api_key",
    "openai_api_key",
    "anthropic_api_key",
    "openrouter_api_key",
    "podcast_index_api_key",
    "podcast_index_api_secret",
)
# app_settings columns holding a JSON array of secret strings.
SECRET_LIST_COLUMNS = ("gemini_api_keys",)

# Rendered in admin forms in place of a saved secret. Submitting it back
# means "keep what is stored"; the real value never reaches the browser.
SAVED_SENTINEL = "__saved__"


def _secrets_dir() -> str:
    from app.core.config import settings

    path = os.path.join(settings.DATA_DIR, "secrets")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return path


def _read_or_create(path: str, generate) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if value:
            return value
    except FileNotFoundError:
        pass
    value = generate()
    # O_EXCL so two workers racing on first boot cannot both write; the loser
    # re-reads the winner's key.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(value)
    return value


def resolve_session_secret(env_value: Optional[str] = None) -> str:
    if env_value is None:
        env_value = os.environ.get("SESSION_SECRET_KEY")
    if env_value and env_value.strip():
        return env_value.strip()
    file_path = os.environ.get("SESSION_SECRET_KEY_FILE")
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            value = f.read().strip()
        if not value:
            raise RuntimeError(f"SESSION_SECRET_KEY_FILE {file_path} is empty")
        return value
    path = os.path.join(_secrets_dir(), "session_secret_key")
    existed = os.path.exists(path)
    value = _read_or_create(path, lambda: secrets.token_hex(32))
    if not existed:
        logger.warning(
            "SESSION_SECRET_KEY not set; generated a random key and stored it at %s",
            path,
        )
    return value


_fernet: Optional[Fernet] = None


def _get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        key = os.environ.get("SECRETS_ENCRYPTION_KEY", "").strip()
        if not key:
            path = os.path.join(_secrets_dir(), "secrets_encryption_key")
            key = _read_or_create(path, lambda: Fernet.generate_key().decode())
        _fernet = Fernet(key.encode())
    return _fernet


def reset_cache() -> None:
    """Tests only: forget the cached Fernet instance."""
    global _fernet
    _fernet = None


def is_encrypted(value) -> bool:
    return isinstance(value, str) and value.startswith(ENC_PREFIX)


def encrypt_secret(value: Optional[str]) -> Optional[str]:
    if value is None or value == "" or is_encrypted(value):
        return value
    return ENC_PREFIX + _get_fernet().encrypt(value.encode()).decode()


def decrypt_secret(value: Optional[str]) -> Optional[str]:
    if not is_encrypted(value):
        return value
    try:
        return _get_fernet().decrypt(value[len(ENC_PREFIX):].encode()).decode()
    except InvalidToken:
        logger.error(
            "Could not decrypt a stored API key (encryption key changed?); "
            "treating it as unset. Re-enter it in Admin."
        )
        return None


def encrypt_secret_list_json(value: Optional[str]) -> Optional[str]:
    """Encrypt each element of a JSON array of keys; returns JSON."""
    if not value:
        return value
    try:
        items = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return encrypt_secret(value)
    if not isinstance(items, list):
        return encrypt_secret(value)
    return json.dumps([encrypt_secret(k) if isinstance(k, str) else k for k in items])


def decrypt_secret_list_json(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    if is_encrypted(value):
        return decrypt_secret(value)
    try:
        items = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value
    if not isinstance(items, list):
        return value
    out = [decrypt_secret(k) if isinstance(k, str) else k for k in items]
    return json.dumps([k for k in out if k is not None])


def decrypt_settings(row) -> dict:
    """dict(row) with every secret column decrypted."""
    if row is None:
        return {}
    data = dict(row)
    for col in SECRET_COLUMNS:
        if col in data:
            data[col] = decrypt_secret(data[col])
    for col in SECRET_LIST_COLUMNS:
        if col in data:
            data[col] = decrypt_secret_list_json(data[col])
    return data


def mask_settings_for_display(data: dict) -> dict:
    """Replace saved secrets with SAVED_SENTINEL for template rendering."""
    out = dict(data)
    for col in SECRET_COLUMNS:
        if out.get(col):
            out[col] = SAVED_SENTINEL
    for col in SECRET_LIST_COLUMNS:
        raw = out.get(col)
        if raw:
            try:
                items = json.loads(raw)
                if isinstance(items, list):
                    out[col] = json.dumps(
                        [f"{SAVED_SENTINEL}:{i}" for i, _ in enumerate([k for k in items if k])]
                    )
                    continue
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
            out[col] = json.dumps([f"{SAVED_SENTINEL}:0"])
    return out


def resolve_submitted(submitted: Optional[str], stored_plain: Optional[str]) -> Optional[str]:
    """Form value -> plaintext to store. Sentinel keeps the stored value."""
    if submitted is None:
        return None
    submitted = submitted.strip()
    if submitted == SAVED_SENTINEL:
        return stored_plain
    return submitted or None


def resolve_submitted_list(submitted: Iterable, stored_plain_json: Optional[str]) -> list:
    """Map ``__saved__:<i>`` entries back to the i-th stored key."""
    try:
        stored = json.loads(stored_plain_json) if stored_plain_json else []
        if not isinstance(stored, list):
            stored = [stored_plain_json]
    except (json.JSONDecodeError, TypeError, ValueError):
        stored = [stored_plain_json]
    stored = [k for k in stored if k]
    out = []
    for item in submitted:
        if not isinstance(item, str) or not item.strip():
            continue
        item = item.strip()
        if item.startswith(SAVED_SENTINEL + ":"):
            try:
                idx = int(item.split(":", 1)[1])
            except ValueError:
                continue
            if 0 <= idx < len(stored):
                out.append(stored[idx])
            continue
        out.append(item)
    return out

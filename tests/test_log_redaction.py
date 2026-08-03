"""Access-log credential redaction.

Feed URLs carry a live bearer token in the `auth` query parameter and uvicorn's
access logger writes the full request line into app.log. Without redaction
every feed poll persists a working credential to disk.
"""
import logging

import pytest

from app.main import (
    REDACTED,
    CredentialRedactingFilter,
    redact_credentials,
)

TOKEN = "s3cret-feed-token-abcdefghijklmnop"


@pytest.fixture
def filt():
    return CredentialRedactingFilter()


def make_record(msg, args=None):
    return logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


def emitted(filt, record):
    assert filt.filter(record) is True
    return record.getMessage()


# --- the uvicorn access-log shape --------------------------------------------

def test_access_log_request_line_is_redacted(filt):
    record = make_record(
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1:5000", "GET", f"/feeds/show.xml?auth={TOKEN}", "1.1", 200),
    )
    message = emitted(filt, record)
    assert TOKEN not in message
    assert "auth=" + REDACTED in message
    assert "/feeds/show.xml" in message
    assert "200" in message


def test_audio_fetch_line_is_redacted(filt):
    record = make_record(
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.5:1234", "GET", f"/audio/ep.mp3?x=1&auth={TOKEN}&y=2", "1.1", 206),
    )
    message = emitted(filt, record)
    assert TOKEN not in message
    assert "y=2" in message


def test_unified_feed_line_is_redacted(filt):
    record = make_record(f"serving /feed/unified.xml?auth={TOKEN}")
    assert TOKEN not in emitted(filt, record)


# --- Authorization headers ----------------------------------------------------

def test_authorization_header_is_redacted(filt):
    record = make_record("Authorization: Basic ZmVlZHM6czNjcmV0")
    message = emitted(filt, record)
    assert "ZmVlZHM6czNjcmV0" not in message
    assert message == "Authorization: Basic " + REDACTED


def test_authorization_in_dict_repr_is_redacted(filt):
    record = make_record("headers={'authorization': 'Bearer %s'}" % TOKEN)
    assert TOKEN not in emitted(filt, record)


def test_dict_style_args_are_redacted(filt):
    record = make_record("%(path)s", ({"path": f"/feeds/a.xml?auth={TOKEN}"},))
    assert TOKEN not in emitted(filt, record)


# --- robustness ---------------------------------------------------------------

def test_unrelated_lines_are_untouched(filt):
    record = make_record("Database initialized at /data/app.db")
    assert emitted(filt, record) == "Database initialized at /data/app.db"


def test_non_string_message_does_not_crash(filt):
    record = make_record({"not": "a string"})
    assert filt.filter(record) is True


def test_malformed_record_is_not_swallowed(filt):
    """A record whose payload cannot be processed still gets emitted."""

    record = make_record("%s", ("ok",))

    class BadArgs(tuple):
        def __iter__(self):
            raise RuntimeError("cannot iterate")

    record.args = BadArgs(("x",))
    assert filt.filter(record) is True
    assert record.getMessage()


def test_redaction_is_idempotent(filt):
    once = redact_credentials(f"/feeds/a.xml?auth={TOKEN}")
    assert redact_credentials(once) == once


def test_redact_credentials_handles_empty_and_non_str():
    assert redact_credentials("") == ""
    assert redact_credentials(None) is None


# --- wired into the real handlers ---------------------------------------------

def test_app_log_handlers_carry_the_filter():
    import app.main as main

    for handler in (main.file_handler, main.stream_handler):
        assert any(
            isinstance(f, CredentialRedactingFilter) for f in handler.filters
        ), handler


def test_uvicorn_access_logger_output_is_redacted(tmp_path):
    """End to end: a record pushed through uvicorn.access lands redacted."""
    import app.main as main

    logger = logging.getLogger("uvicorn.access")
    log_path = tmp_path / "access.log"
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(CredentialRedactingFilter())

    previous = logger.handlers
    logger.handlers = [handler]
    try:
        logger.info(
            '%s - "%s %s HTTP/%s" %d',
            "127.0.0.1:5000",
            "GET",
            f"/feeds/show.xml?auth={TOKEN}",
            "1.1",
            200,
        )
    finally:
        handler.close()
        logger.handlers = previous

    contents = log_path.read_text()
    assert TOKEN not in contents
    assert "auth=" + REDACTED in contents
    assert main.REDACTED == REDACTED

"""Range-header parsing for the audio/video first-byte listen counter.

A malformed Range start used to reach int() unguarded and raise ValueError
out of the route, turning a bad client header into a 500.
"""

import pytest
from starlette.datastructures import Headers

from app.api.audio_routes import _is_first_byte_request as audio_first_byte
from app.api.video_routes import _is_first_byte_request as video_first_byte


class _FakeRequest:
    def __init__(self, range_value=None):
        raw = {} if range_value is None else {"Range": range_value}
        self.headers = Headers(raw)


ROUTES = [audio_first_byte, video_first_byte]


@pytest.mark.parametrize("fn", ROUTES)
@pytest.mark.parametrize(
    "range_value,expected",
    [
        (None, True),           # no Range header at all
        ("", True),             # empty Range header
        ("bytes=-500", True),   # suffix range, empty start
        ("bytes=0-1023", True),
        ("bytes=1023-", True),  # below the 1024 first-byte threshold
        ("bytes=1024-", False),
        ("bytes=50000-", False),
        ("items=0-10", False),  # non-bytes unit
        ("bytes=0", False),     # no dash, unparseable spec
        ("bytes=--100", True),  # start splits to "", same as a suffix range
    ],
)
def test_well_formed_ranges(fn, range_value, expected):
    assert fn(_FakeRequest(range_value)) is expected


@pytest.mark.parametrize("fn", ROUTES)
@pytest.mark.parametrize(
    "range_value",
    [
        "bytes=abc-100",
        "bytes= -100",
        "bytes=1.5-100",
        "bytes=0x10-100",
    ],
)
def test_malformed_start_does_not_raise(fn, range_value):
    """Non-numeric start returns False instead of raising ValueError."""
    assert fn(_FakeRequest(range_value)) is False

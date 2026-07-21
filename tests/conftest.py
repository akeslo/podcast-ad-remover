"""
Shared pytest fixtures for podcast-ad-remover.

Everything here is designed to be fully hermetic: no real AI-service calls, no
writes to a real database, no real network downloads. `DATA_DIR` is redirected
to a throwaway temp directory *before* `app.core.config.settings` (and
therefore anything that imports it, including `app.main`) is ever imported, so
the app never touches the real /data volume.
"""
import os
import shutil
import tempfile

import pytest

# Must happen before any `app.*` import - Settings() is instantiated at
# import time and reads DATA_DIR from the environment.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="podcast-ad-remover-test-")
os.environ["DATA_DIR"] = _TEST_DATA_DIR
os.environ.setdefault("SESSION_SECRET_KEY", "test-secret-key")
os.environ.setdefault("ENVIRONMENT", "development")

from app.infra.database import init_db  # noqa: E402
from app.main import app as fastapi_app  # noqa: E402

# Create the schema (empty tables, default settings row) in the temp DB.
# This is local sqlite bootstrapping only - no real data, no network calls.
init_db()


def pytest_sessionfinish(session, exitstatus):
    shutil.rmtree(_TEST_DATA_DIR, ignore_errors=True)


@pytest.fixture
def client():
    """
    A FastAPI TestClient bound to the real app.

    Deliberately NOT used as a context manager (`with TestClient(app) as c`):
    entering the context manager runs the app's lifespan, which spawns a
    background multiprocessing worker (`start_processor_process`) that polls
    feeds and calls real AI services. A bare TestClient never triggers
    lifespan/startup events, so requests still flow through the real
    middleware stack (including SecurityHeadersMiddleware) without any of
    that side effect.
    """
    from fastapi.testclient import TestClient

    return TestClient(fastapi_app)

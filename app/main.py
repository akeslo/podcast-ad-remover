from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import asyncio
import logging

from app.core.config import settings
from app.infra.database import init_db
from app.core.processor import Processor

# Configure logging
from logging.handlers import RotatingFileHandler
import os

log_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log_file = os.path.join(settings.DATA_DIR, "app.log")

file_handler = RotatingFileHandler(
    log_file,
    maxBytes=settings.LOG_MAX_BYTES,
    backupCount=settings.LOG_BACKUP_COUNT
)
file_handler.setFormatter(log_formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

# Root logger configuration
root_logger = logging.getLogger()
root_logger.setLevel(settings.LOG_LEVEL)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

# Capture uvicorn logs
for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    l = logging.getLogger(logger_name)
    l.handlers = [file_handler, stream_handler]
    l.propagate = False

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Podcast Ad Remover...")
    init_db()
    logger.info(f"Database initialized at {settings.DB_PATH}")
    import os
    if os.path.exists(settings.DB_PATH):
        size = os.path.getsize(settings.DB_PATH)
        logger.info(f"Database size: {size} bytes")
    else:
        logger.warning("Database file not found!")
    
    # Start background scheduler
    processor = Processor()
    asyncio.create_task(processor.run_loop())
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

from app.api import subscriptions
from app.web import router as web_router
from app.web.middleware import feed_auth_middleware
from app.web.auth import auth_middleware
from starlette.middleware.sessions import SessionMiddleware
import secrets

app = FastAPI(
    title="Podcast Ad Remover",
    lifespan=lifespan
)

# Add middleware (order matters - added in reverse of execution order)
# Execution order: SessionMiddleware -> auth_middleware -> feed_auth_middleware
app.middleware("http")(feed_auth_middleware)
app.middleware("http")(auth_middleware)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_urlsafe(32))

app.include_router(subscriptions.router, prefix="/api")
app.include_router(web_router.router)

# Mount static files
app.mount("/feeds", StaticFiles(directory=settings.FEEDS_DIR), name="feeds")
app.mount("/audio", StaticFiles(directory=settings.PODCASTS_DIR), name="audio")
# Mount general static files (css, js, images)
app.mount("/static", StaticFiles(directory="app/web/static"), name="static")

@app.get("/")
async def root():
    return {"message": "Podcast Ad Remover is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

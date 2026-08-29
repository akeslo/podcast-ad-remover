import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Core
    ENVIRONMENT: str = Field("production", description="Environment: development or production")
    GEMINI_API_KEY: str | None = Field(None, description="Google Gemini API Key (comma-separated for multiple keys)")
    OPENAI_API_KEY: str | None = Field(None, description="OpenAI API Key")
    ANTHROPIC_API_KEY: str | None = Field(None, description="Anthropic API Key")
    OPENROUTER_API_KEY: str | None = Field(None, description="OpenRouter API Key")
    # Podcast Index (api.podcastindex.org) credentials, used for podcast
    # search and the trending/browse discovery surfaces. These seed an
    # install; the operator-editable values in app_settings take precedence
    # once set, exactly as the AI provider keys do.
    PODCAST_INDEX_API_KEY: str | None = Field(None, description="Podcast Index API key")
    PODCAST_INDEX_API_SECRET: str | None = Field(None, description="Podcast Index API secret")
    LOG_LEVEL: str = "INFO"
    # No working default: a hardcoded default here is a public secret (it's in
    # this repo's source), so anyone can forge a session cookie against it.
    # Must be set via the SESSION_SECRET_KEY env var (e.g. `openssl rand -hex 32`).
    SESSION_SECRET_KEY: str = Field(..., description="Secret key for session encryption. Required - no default.")
    # The IP allowlist (app_settings.ip_allowlist) is the entire security boundary
    # in standalone/no-auth mode (see CLAUDE.local.md), and get_client_ip() honors
    # CF-Connecting-IP/X-Forwarded-For/X-Real-IP to support that allowlist behind a
    # real reverse proxy. Those headers are attacker-controlled on any request that
    # does NOT pass through a trusted proxy, so trusting them unconditionally lets
    # anyone spoof an allowlisted IP or defeat login rate-limiting. Default is off;
    # only set true once a reverse proxy in front of this app is guaranteed to
    # overwrite (not merge) these headers on every request.
    TRUST_PROXY_HEADERS: bool = Field(False, description="Trust CF-Connecting-IP/X-Forwarded-For/X-Real-IP from the incoming request. Only enable behind a reverse proxy that strips/overwrites these headers.")
    
    # Paths
    DATA_DIR: str = "/data"

    
    # Web
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    BASE_URL: str = "http://localhost:8000"
    
    # Processing
    CHECK_INTERVAL_MINUTES: int = 60
    WHISPER_MODEL: str = "base"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT: int = 5

    # Orphan / scratch reconciliation sweep (app/core/orphan_cleanup.py).
    #
    # This is IN ADDITION to Processor.cleanup_old_episodes, never a
    # replacement for it: retention deletes by `episodes` row, so anything the
    # database does not reference is invisible to it forever (stale
    # `loading-*` placeholder subscription directories, episode directories
    # whose row was hard-deleted, abandoned `.part` downloads).
    ORPHAN_CLEANUP_ENABLED: bool = Field(True, description="Run the filesystem-vs-database orphan reconciliation sweep during scheduled maintenance.")
    # Nothing is removed until it has been untouched for this long. A path is
    # only ever an orphan candidate if the database does not reference it, so
    # this is a second, independent safety net: it protects a directory that
    # was created on disk moments before the sweep read the database (the
    # `loading-<ts>` placeholder is created on disk and renamed on success, so
    # its name alone must never be the gate).
    ORPHAN_MIN_AGE_HOURS: int = Field(24, description="Minimum age (hours, by mtime) before an unreferenced path may be deleted.")
    # A `.part` file is an in-flight download. yt-dlp writes multi-GB video to
    # `.part` and moves it into place on success, so this threshold must be far
    # longer than any plausible single download. 24h is roughly 100x the
    # observed worst case.
    PART_FILE_MAX_AGE_HOURS: int = Field(24, description="Age (hours) after which a .part file is treated as an abandoned download.")

    # Retention for episodes that failed and will never be retried
    # (Processor.cleanup_old_episodes, selector 3).
    #
    # WHY THIS EXISTS: the count-based auto retention computes ROW_NUMBER() over
    # a set already filtered to `status='completed'`, so a non-completed episode
    # is never counted against `retention_limit` and never selected for
    # deletion. Its directory therefore survives forever, while ALSO being
    # invisible to the orphan sweep (which protects every episodes row whatever
    # its status). Neither mechanism could reclaim it. This is the third
    # selector that closes that gap.
    #
    # The grace window is what makes it safe. `EpisodeRepository.requeue_stuck`
    # turns every interrupted 'processing' episode into 'failed' on each
    # restart, so reaping a failed episode the instant it appears would silently
    # discard work the operator may still want to retry. A failed episode is
    # only reclaimed once it has sat terminal for this long AND has no retry
    # scheduled (`next_retry_at IS NULL`).
    FAILED_RETENTION_DAYS: int = Field(7, description="Days a failed episode with no scheduled retry keeps its files before retention reclaims them.")
    
    @property
    def DB_PATH(self) -> str:
        return os.path.join(self.DATA_DIR, "db", "podcasts.db")
    
    @property
    def PODCASTS_DIR(self) -> str:
        """Base directory for all podcast data organized by podcast/episode"""
        return os.path.join(self.DATA_DIR, "podcasts")
        
    @property
    def DOWNLOADS_DIR(self) -> str:
        """Deprecated: Use get_episode_dir() instead"""
        return os.path.join(self.DATA_DIR, "downloads")
        
    @property
    def TRANSCRIPTS_DIR(self) -> str:
        """Deprecated: Use get_episode_dir() instead"""
        return os.path.join(self.DATA_DIR, "transcripts")
        
    @property
    def FEEDS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "feeds")
        
    @property
    def AUDIO_DIR(self) -> str:
        """Deprecated: Use get_episode_dir() instead"""
        return os.path.join(self.DATA_DIR, "audio")

    @property
    def MODELS_DIR(self) -> str:
        return os.path.join(self.DATA_DIR, "models")
    
    def get_episode_dir(self, podcast_slug: str, episode_slug: str) -> str:
        """Get the directory path for a specific episode"""
        return os.path.join(self.PODCASTS_DIR, podcast_slug, episode_slug)

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
for path in [
    os.path.dirname(settings.DB_PATH),
    settings.PODCASTS_DIR,
    settings.FEEDS_DIR,
    settings.MODELS_DIR
]:
    os.makedirs(path, exist_ok=True)

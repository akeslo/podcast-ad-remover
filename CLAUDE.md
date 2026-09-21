# CLAUDE.md — podcast-ad-remover (AGPAR)

FastAPI app that downloads podcast episodes (RSS or YouTube), transcribes them locally with
faster-whisper, asks an LLM to mark ad segments, cuts them with FFmpeg and republishes a clean
RSS feed per subscription. Dockerized, runs on the N100 box; Gemini is the primary ad detector.

## Commands

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/uvicorn app.main:app --reload --port 8000     # dev server
.venv/bin/python -m pytest -q                          # test suite (369 passed, 2 skipped as of 2026-09-10)
python -m app.core.retention [--include-orphans]       # dry-run retention/orphan report; there is no --apply
./scripts/generate-lockfile.sh                         # regenerate requirements.lock (run on linux, see Gotchas)
docker build .                                         # uses requirements.lock when present
```

Use `.venv/bin/python -m pytest`, not the `.venv/bin/pytest` shebang, which still points at a
pre-reorg path.

## Environment

Required: `SESSION_SECRET_KEY` (no default, app refuses to start without it), `GEMINI_API_KEY`.
Optional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` (alternate ad-detection
providers), `PODCAST_INDEX_API_KEY` / `PODCAST_INDEX_API_SECRET` (discovery; DB-backed override in
`app_settings` wins), `BASE_URL`, `TRUST_PROXY_HEADERS` (default false), `CHECK_INTERVAL_MINUTES`,
`LOG_LEVEL`. See `env.example`.

## Architecture

- `app/main.py` mounts two routers: `app/web/router.py` (HTML UI, `/`) and `app/api/*` at `/api`.
  `app/api/subscriptions.py` holds 8 state-changing routes including `DELETE`; any auth audit must
  walk the mounted app, not one router.
- `app/core/processor.py` is the background pipeline: download → `audio.py`/`video.py` →
  `ai_services.py` (whisper transcription + provider ad detection) → cut → `rss_gen.py`.
- `app/core/ai_services.py`: all four providers (Gemini, OpenAI, Anthropic/Claude, OpenRouter)
  have multi-key rotation and transient-error retry (2026-09-20), sharing the same
  `TRANSIENT_ERROR_PATTERNS`/`RATE_LIMIT_ERROR_PATTERNS` classification Gemini originated.
- `app/core/retention.py` is the single selector for what retention may delete, consumed by both
  the deleter and the dry-run report. `orphan_cleanup.py` reaps directories no `episodes` row
  references and fails closed when the `subscriptions` table is empty.
- `app/core/youtube_feed.py` (`yt-dlp`, `extract_flat`) and `sponsorblock.py` add YouTube
  channels as subscriptions (`subscriptions.source_type = 'youtube'`).
- `app/core/discovery.py` tries Podcast Index first, falls back to iTunes; keep both paths.
- `app/infra/database.py`: SQLite, migrations run in `init_db()`, the only code guaranteed to
  execute on upgrade, so every backfill belongs there.
- `app/web/auth.py` + `middleware.py`: two auth modes. `auth_enabled = 1` makes `require_auth` a
  real boundary; `auth_enabled = 0` (standalone, the common install) deliberately does not.
  Feed URLs carry a per-user random token, never the account password.

## Gotchas

- Media routes are gated by two independent middlewares that must list the same prefixes;
  exempting one without the other is either a login-redirect loop or an unauthenticated leak.
- Never log a raw `Referer`: it is the `Origin` fallback and a feed request's Referer carries
  the live feed token. Log `urlsplit(...).netloc` only.
- `FileResponse(..., filename=...)` defaults to `Content-Disposition: attachment`, which makes
  podcast clients download instead of stream.
- Both `model.transcribe(...)` calls pass `condition_on_previous_text=False`; the default drops
  the 30 s window after a sign-off line on `tiny`/`base` models.
- `processed.mp3` is longer than the kept audio by design (title TTS and summary TTS are
  prepended after the cut).
- `piper-tts` is linux-only in `requirements.txt`; `requirements.lock` must be regenerated on
  linux. A floating `>=` pin does not float across CI rebuilds because of the Docker layer cache.
- `get_client_ip()` honours proxy headers only when `TRUST_PROXY_HEADERS=true`.
- The deploy workflow triggers on `main`; this repo has never had a `master` branch.

## Conventions

- Reclaimable retention statuses are an allowlist (`completed`, `failed` with `next_retry_at IS
  NULL` past 7 days), never a denylist. Reaping a failed episode soft-deletes it to `ignored` and
  keeps the guid, so it is never re-downloaded.
- Secrets outlive the code that wrote them: removing a write path does not clear the stored
  value; add a migration that does.
- Local, untracked session notes live in `CLAUDE.local.md`; keep durable project facts here.

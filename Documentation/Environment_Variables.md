# Environment Variables

The application is configured via environment variables.

## AI Provider Keys (Optional)
The application requires at least one API key to function (Gemini, OpenAI, Anthropic, or OpenRouter). You can set these via Environment Variables (recommended for Docker) or via the Admin UI.

**Note:** Settings in the **Admin UI** take priority over Environment Variables.

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API Key |
| `OPENAI_API_KEY` | OpenAI API Key |
| `ANTHROPIC_API_KEY` | Anthropic API Key |
| `OPENROUTER_API_KEY` | OpenRouter API Key |

## Optional / Defaults

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Directory for internal data (DB, temp) | `/data` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CHECK_INTERVAL_MINUTES` | How often to check for new episodes | `60` |
| `WHISPER_MODEL` | Whisper model size | `base` |
| `HOST` | Host to bind to | `0.0.0.0` |
| `PORT` | Port to bind to | `8000` |
| `BASE_URL` | Last-resort fallback for RSS feed URLs | `http://localhost:8000` |

## Public Application URL (stored setting, not an environment variable)

The public base URL is **not** an environment variable. It lives in the database
as `app_settings.app_external_url` and is edited at
**Admin > System > "Public Application URL"**. It is auto-detected on first boot
as `http://<LAN-IP>:<PORT>`, which is correct for direct LAN access and wrong
behind a reverse proxy.

It does two things:

1. Builds the RSS feed and audio URLs handed to podcast clients.
2. Is one of the two accepted origins for the same-origin check on every admin
   POST (the other being the `Host` header the app receives).

**If it is wrong and your proxy does not forward the public `Host` header, every
admin form POST returns 403** - including the one that would fix it. See
[Deployment](Deployment.md#behind-a-reverse-proxy-read-this-before-you-proxy-it).

# Deployment

## Docker Compose (Recommended)

```yaml
## Docker Compose (Recommended)

```yaml
services:
  app:
    build: .
    image: podcast-ad-remover
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data      # App data (DB, downloads, transcripts, feeds, audio)
    environment:
      # Optional: Pre-configure API Key (or set in Admin UI)
      - GEMINI_API_KEY=your_key_here
      # Optional: Config
      - BASE_URL=http://your-server-ip:8000
```

## Behind a Reverse Proxy (read this before you proxy it)

Every state-changing admin route is same-origin checked. The browser's `Origin`
header (falling back to `Referer`) must match either the `Host` header the app
receives, or the **Public Application URL** stored in the app
(Admin > System > "Public Application URL"). `X-Forwarded-Host` is deliberately
**not** trusted - it is caller-controlled on any request the proxy does not
overwrite.

Two ways to satisfy that check; do at least one:

1. **Forward the real Host.** In nginx, `proxy_set_header Host $host;`. A bare
   `proxy_pass http://app:8000;` sends `Host: app:8000` instead, and the check
   then compares your browser's public hostname against a container name.
2. **Set the Public Application URL** to the exact URL you type in the browser,
   scheme, host and port included - `https://podcasts.example.com`, not
   `http://192.168.1.50:8000`. On first boot the app auto-detects the latter,
   which is right for direct LAN access and wrong behind a proxy.

Also keep the `Origin` and `Referer` headers intact. A proxy configured to strip
them turns every admin save into a 403.

**If you get this wrong, every admin form POST returns 403** with
`Cross-origin or origin-less request rejected` - including the form that sets
the Public Application URL. To recover, reach the container directly
(`http://<docker-host>:8000`), which makes `Origin` and `Host` agree, and correct
the setting there. The server log names both sides of the comparison on every
such rejection:

```bash
grep "same-origin check" /data/app.log
```

Do not "fix" this by loosening the check. It is the only thing standing between
an install running in standalone mode (`auth_enabled = 0`) and a drive-by
cross-site POST that can disable feed authentication.

## Manual Docker Run

```bash
docker build -t podcast-ad-remover .
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/data \
  -v $(pwd)/public:/public \
  -e GEMINI_API_KEY=your_key_here \
  podcast-ad-remover
```

## Data Volumes
1. **`/data` (Internal)**:
    - `db/`: Database file.
    - `downloads/`: Temporary raw downloads.
    - `transcripts/`: Intermediate JSON transcripts.

2. **`/public` (External)**:
    - `feeds/`: RSS XML files.
    - `audio/`: Cleaned MP3 files.
    - `index.html`: Landing page.


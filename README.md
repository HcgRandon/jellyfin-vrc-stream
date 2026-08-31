# Jellyfin HLS Proxy for VRChat

[![Docker Build](https://github.com/C9Glax/jellyfin-vrc-stream/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/C9Glax/jellyfin-vrc-stream/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** Almost all of this repo was vibe-coded to get a quick working POC. Terminology and comments might be weird and nonsensical, but it is functional. I plan to clean this up properly later.

A simple FastAPI proxy service that:

## Benefits

- ✅ **Single transcode** - One Jellyfin session serves unlimited viewers
- ✅ **Hidden credentials** - API key stays server-side
- ✅ **Automatic stream selection** - Prefers Japanese audio + English subs
- ✅ **Cached delivery** - Fast segment serving from local cache
- ✅ **Simple** - No custom transcoding, just proxying Jellyfin
- ✅ **Time-limited, per-item links** - No more permanent, guessable URLs to your whole library
- ✅ **One click from Jellyfin** - The [Jellyfin plugin](jellyfin-plugin-vrc-share/) creates and copies a share link straight from the item page

## How It Works

1. Client requests: `GET /vod.m3u8?m={media_id}`
2. Proxy fetches HLS stream from Jellyfin
3. Proxy caches playlist and segments to disk
4. All clients read from the same cached files
5. Jellyfin only transcodes once, regardless of viewer count

## API Endpoints

[Link](API-Endpoints.md)

## Container Configuration

[Link](Container-Configuration.md)

## Installation

### Docker

**Using pre-built image from GitHub Container Registry:**
```bash
docker pull ghcr.io/c9glax/jellyfin-vrc-stream:latest
docker run -p 8000:8000 \
  -e JELLYFIN_URL=http://jellyfin:8096 \
  -e JELLYFIN_API_KEY=your_key \
  ghcr.io/c9glax/jellyfin-vrc-stream:latest
```

**Or build locally:**
```bash
docker build -t jellyfin-vrc-stream:latest .
docker run -p 8000:8000 \
  -e JELLYFIN_URL=http://jellyfin:8096 \
  -e JELLYFIN_API_KEY=your_key \
  jellyfin-vrc-stream:latest
```

### Kubernetes

1. Update secret in `deployment.yaml`:
```yaml
stringData:
  JELLYFIN_API_KEY: "your_actual_api_key_here"
```

2. Deploy:
```bash
kubectl --kubeconfig=/path/to/kubeconfig apply -f deployment.yaml
```

3. Get NodePort:
```bash
kubectl get svc jellyfin-vrc-stream-service
```

### Jellyfin Plugin (VR Share Link button)

[`jellyfin-plugin-vrc-share/`](jellyfin-plugin-vrc-share/) is a companion Jellyfin plugin that adds a **VR Share Link** button to every movie/episode detail page. Click it as an administrator and it creates a share link on this proxy and copies it to your clipboard - no manual ID lookup or `curl`/dashboard step needed.

#### Install

1. In Jellyfin, go to **Dashboard → Plugins → Repositories → Add Repository** and add:
   - **Repository URL:** `https://raw.githubusercontent.com/C9Glax/jellyfin-vrc-stream/main/jellyfin-plugin-vrc-share/manifest.json`
2. Restart Jellyfin to refresh repositories
3. Go to **Catalog**, find **VRC Share** under General, install it, and restart Jellyfin (again).
4. In **Plugins → VRC Share**, set **Proxy Base URL** and **Proxy Admin API Key** (matching `JELLYFIN_API_KEY` above).
5. Open any movie or episode as an administrator - a **VR Share Link** button appears next to Play.

## Architecture

```
┌─────────┐     ┌─────────┐     ┌──────────┐
│ VRChat  │────▶│  Proxy  │────▶│ Jellyfin │
│ Player1 │     │         │     │          │
└─────────┘     │ Caches  │     │ Trans-   │
┌─────────┐     │ Stream  │     │ codes    │
│ VRChat  │────▶│         │     │ Once     │
│ Player2 │     │         │     │          │
└─────────┘     └─────────┘     └──────────┘
┌─────────┐           │
│ VRChat  │───────────┘
│ Player3 │    All read same cache
└─────────┘
```

## Notes

- First viewer triggers Jellyfin transcoding
- Additional viewers join immediately from cache
- Subtitles are burned into video by Jellyfin
- **Quality defaults are optimized for high-quality single-stream fan-out** (40 Mbps video, 320 Kbps audio)
- Lower quality settings if bandwidth/storage is limited by adjusting env vars

### Cleanup & Resource Management

The proxy automatically manages cache to prevent OOM:

1. **Idle Stream Cleanup**
   - Streams not accessed for `STREAM_IDLE_TIMEOUT` seconds are removed
   - Default: 5 minutes (300s)
   - Cached files deleted, resources freed

2. **Size-Based Cleanup**
   - When cache exceeds `MAX_CACHE_SIZE_MB`, oldest streams are removed
   - Cleans down to 80% of limit to avoid thrashing
   - Default limit: 1.8 GB

3. **Background Task**
   - Runs every `CLEANUP_INTERVAL` seconds (default: 60s)
   - Performs both idle and size-based cleanup
   - Can be disabled by setting `CLEANUP_INTERVAL=0`

4. **Manual Management**
   - `POST /cleanup` - Trigger cleanup immediately
   - `DELETE /streams/{key}` - Stop specific stream
   - `GET /streams` - Monitor idle times and cache size

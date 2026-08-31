# Jellyfin HLS Proxy for VRChat

[![Docker Build](https://github.com/HcgRandon/jellyfin-vrc-stream/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/HcgRandon/jellyfin-vrc-stream/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** Almost all of this repo was vibe-coded to get a quick working POC. Terminology and comments might be weird and nonsensical, but it is functional. I plan to clean this up properly later.

A simple FastAPI proxy service that:
- **Streams once from Jellyfin** - Single transcoding session regardless of viewer count
- **Fans out to multiple clients** - All viewers watch the same stream
- **Hides API key** - Clients connect without exposing Jellyfin credentials
- **Time-limited share links** - Playback requires a per-item token that expires, instead of a permanent unauthenticated URL
- **One-click links from Jellyfin** - The companion [Jellyfin plugin](jellyfin-plugin-vrc-share/) adds a button to each item's page that creates and copies a share link for you

## How It Works

1. Client requests: `GET /vod.m3u8?m={media_id}`
2. Proxy fetches HLS stream from Jellyfin
3. Proxy caches playlist and segments to disk
4. All clients read from the same cached files
5. Jellyfin only transcodes once, regardless of viewer count

## Benefits

- ✅ **Single transcode** - One Jellyfin session serves unlimited viewers
- ✅ **Hidden credentials** - API key stays server-side
- ✅ **Automatic stream selection** - Prefers Japanese audio + English subs
- ✅ **Cached delivery** - Fast segment serving from local cache
- ✅ **Simple** - No custom transcoding, just proxying Jellyfin
- ✅ **Time-limited, per-item links** - No more permanent, guessable URLs to your whole library
- ✅ **One click from Jellyfin** - The [Jellyfin plugin](jellyfin-plugin-vrc-share/) creates and copies a share link straight from the item page

## API Endpoints

### VOD Mode (Seekable, Full Video)
```
GET /vod.m3u8?m={media_id}&token={share_token}
```

Uses Jellyfin's `main.m3u8` endpoint for full video playback with seeking support. Requires a valid share `token` (see [Share Links](#share-links)) or `admin_key`.

**Query Parameters:**
- `m` (required): Jellyfin media item ID
- `token` / `admin_key` (one required): share token from `POST /share`, or the admin API key
- `audio` (optional): Audio stream index (auto-selects jpn > eng if not specified)
- `subtitle` (optional): Subtitle stream index (auto-selects eng if not specified)

**Example:**
```bash
# Play using a share link (auto-selects best streams)
curl "http://proxy:8000/vod.m3u8?m=abc123&token=<share_token>"

# Manual selection using the admin key
curl "http://proxy:8000/vod.m3u8?m=abc123&audio=2&subtitle=5&admin_key=$ADMIN_API_KEY"
```

### Live Streaming Mode
```
GET /live.m3u8?m={media_id}&token={share_token}
```

Uses Jellyfin's `live.m3u8` endpoint for real-time streaming (no seeking). Requires a valid share `token` or `admin_key`, same as VOD mode.

**Query Parameters:**
- `m` (required): Jellyfin media item ID
- `token` / `admin_key` (one required): share token from `POST /share`, or the admin API key
- `audio` (optional): Audio stream index (auto-selects jpn > eng if not specified)
- `subtitle` (optional): Subtitle stream index (auto-selects eng if not specified)

**Example:**
```bash
# Live stream with auto-selected streams
curl "http://proxy:8000/live.m3u8?m=abc123&token=<share_token>"
```

### Share Links

Share links are single-item, time-limited tokens meant to be pasted into a VRChat video player without exposing the rest of the library. All share-management endpoints require `ADMIN_API_KEY`.

```
POST /share
```
Body (JSON):
```json
{
  "media_id": "abc123",
  "mode": "vod",
  "audio": null,
  "subtitle": null,
  "profile": null,
  "ttl_seconds": 3600
}
```
Only `media_id` is required; `ttl_seconds` defaults to `DEFAULT_SHARE_TTL_SECONDS`. Returns:
```json
{
  "token": "…",
  "url": "https://stream.example.com/vod.m3u8?m=abc123&token=…",
  "media_id": "abc123",
  "mode": "vod",
  "created_at": 1730000000.0,
  "expires_at": 1730003600.0
}
```

```
GET /shares
```
Lists active (non-expired) share links.

```
DELETE /share/{token}
```
Revokes a share link immediately.

**Example:**
```bash
curl -X POST -H "X-Admin-Key: $ADMIN_API_KEY" -H 'Content-Type: application/json' \
  -d '{"media_id":"abc123","ttl_seconds":3600}' \
  http://proxy:8000/share
```

### Get Segments
```
GET /vod/{media_id}/{segment_path:path}  # VOD mode segments
GET /live/{media_id}/{segment_file}      # Live mode segments
```

Automatically served after playlist request.

### List Active Streams
```
GET /streams
```

Returns list of currently active/cached streams with timing information.

### Management Endpoints

**Manual Cleanup:**
```
POST /cleanup
```

Manually trigger cleanup (idle streams + size-based cleanup).

**Delete Specific Stream:**
```
DELETE /streams/{stream_key}
```

Stop and cleanup a specific stream by its stream_key (from `/streams` endpoint).

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `JELLYFIN_URL` | Jellyfin server URL | `http://jellyfin:8096` |
| `JELLYFIN_API_KEY` | Jellyfin API key | (required) |
| `CACHE_DIR` | HLS cache directory | `/tmp/hls-cache` |
| `ADMIN_API_KEY` | **Required.** Key for admin/browsing endpoints and creating share links | (none - disables those endpoints) |
| `PUBLIC_BASE_URL` | External base URL used to build share link URLs | (falls back to request base URL) |
| `DEFAULT_SHARE_TTL_SECONDS` | Default share link lifetime in seconds | `86400` (24h) |
| `STREAM_IDLE_TIMEOUT` | Cleanup streams idle for N seconds (0=disable) | `300` (5 min) |
| `CLEANUP_INTERVAL` | Run cleanup every N seconds (0=disable) | `60` |
| `MAX_CACHE_SIZE_MB` | Max cache size in MB (0=disable) | `1800` (1.8 GB) |
| **Quality Settings** | | |
| `VIDEO_BITRATE` | Video bitrate in bits/sec | `40000000` (40 Mbps) |
| `AUDIO_BITRATE` | Audio bitrate in bits/sec | `320000` (320 Kbps) |
| `MAX_STREAMING_BITRATE` | Total bitrate cap in bits/sec | `50000000` (50 Mbps) |
| `MAX_WIDTH` | Maximum video width | `1920` |
| `MAX_HEIGHT` | Maximum video height | `1080` |
| `MAX_FRAMERATE` | Maximum framerate | `60` |
| `H264_PROFILE` | H.264 profile (baseline/main/high) | `high` |
| `H264_LEVEL` | H.264 level (41=1080p30, 42=1080p60) | `41` |
| `MAX_REF_FRAMES` | Reference frames for motion quality | `4` |

## Deployment

### Docker

**Using pre-built image from GitHub Container Registry:**
```bash
docker pull ghcr.io/hcgrandon/jellyfin-vrc-stream:latest
docker run -p 8000:8000 \
  -e JELLYFIN_URL=http://jellyfin:8096 \
  -e JELLYFIN_API_KEY=your_key \
  ghcr.io/hcgrandon/jellyfin-vrc-stream:latest
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

## Jellyfin Plugin (VR Share Link button)

[`jellyfin-plugin-vrc-share/`](jellyfin-plugin-vrc-share/) is a companion Jellyfin plugin that adds a **VR Share Link** button to every movie/episode detail page. Click it as an administrator and it creates a share link on this proxy and copies it to your clipboard - no manual ID lookup or `curl`/dashboard step needed.

### Requirements
- Jellyfin 10.11.x
- This proxy running with `ADMIN_API_KEY` set

### Build
```bash
cd jellyfin-plugin-vrc-share
dotnet build -c Release
```
The compiled plugin is at `Jellyfin.Plugin.VrcShare/bin/Release/net9.0/Jellyfin.Plugin.VrcShare.dll`.

### Install
1. Copy `Jellyfin.Plugin.VrcShare.dll` into a new folder under Jellyfin's plugin directory, e.g. `<jellyfin-config>/plugins/VRC Share_1.0.0.0/`.
2. Restart Jellyfin.
3. In the admin dashboard, go to **Plugins → VRC Share** and set:
   - **Proxy Base URL** - e.g. `https://stream.example.com`
   - **Proxy Admin API Key** - must match `ADMIN_API_KEY` on the proxy
   - **Default link lifetime** - defaults to 86400 seconds (24h)
4. Open any movie or episode as an administrator - a **VR Share Link** button appears next to Play. Click it, then paste the copied URL into your VRChat video player.

See [`jellyfin-plugin-vrc-share/README.md`](jellyfin-plugin-vrc-share/README.md) for details on how the button is injected into the web UI.

## VRChat Usage

Generate a share link (via `POST /share`, the `/manage` dashboard, or the Jellyfin plugin) and paste the returned URL directly into a VRChat video player:

```
https://proxy.example.com/vod.m3u8?m=<media_id>&token=<share_token>
```

**All viewers using the same URL watch the same synchronized stream, until the link expires.**

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

## Cleanup & Resource Management

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

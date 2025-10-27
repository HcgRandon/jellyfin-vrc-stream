# Jellyfin HLS Proxy for VRChat

[![Docker Build](https://github.com/HcgRandon/jellyfin-vrc-stream/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/HcgRandon/jellyfin-vrc-stream/actions/workflows/docker-publish.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A simple FastAPI proxy service that:
- **Streams once from Jellyfin** - Single transcoding session regardless of viewer count
- **Fans out to multiple clients** - All viewers watch the same stream
- **Hides API key** - Clients connect without exposing Jellyfin credentials

## How It Works

1. Client requests: `GET /media.m3u8?m={media_id}`
2. Proxy fetches HLS stream from Jellyfin
3. Proxy caches playlist and segments in memory
4. All clients read from the same cached files
5. Jellyfin only transcodes once, regardless of viewer count

## Benefits

- ✅ **Single transcode** - One Jellyfin session serves unlimited viewers
- ✅ **Hidden credentials** - API key stays server-side
- ✅ **Automatic stream selection** - Prefers Japanese audio + English subs
- ✅ **Memory-backed cache** - Fast delivery, SSD-friendly
- ✅ **Simple** - No custom transcoding, just proxying Jellyfin

## API Endpoints

### VOD Mode (Seekable, Full Video)
```
GET /media.m3u8?m={media_id}
```

Uses Jellyfin's `main.m3u8` endpoint for full video playback with seeking support.

**Query Parameters:**
- `m` (required): Jellyfin media item ID
- `audio` (optional): Audio stream index (auto-selects jpn > eng if not specified)
- `subtitle` (optional): Subtitle stream index (auto-selects eng if not specified)

**Example:**
```bash
# Auto-select best streams (Japanese audio + English subtitles)
curl http://proxy:8000/media.m3u8?m=abc123

# Manual selection
curl http://proxy:8000/media.m3u8?m=abc123&audio=2&subtitle=5
```

### Live Streaming Mode
```
GET /live.m3u8?m={media_id}
```

Uses Jellyfin's `live.m3u8` endpoint for real-time streaming (no seeking).

**Query Parameters:**
- `m` (required): Jellyfin media item ID
- `audio` (optional): Audio stream index (auto-selects jpn > eng if not specified)
- `subtitle` (optional): Subtitle stream index (auto-selects eng if not specified)

**Example:**
```bash
# Live stream with auto-selected streams
curl http://proxy:8000/live.m3u8?m=abc123
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
| `STREAM_IDLE_TIMEOUT` | Cleanup streams idle for N seconds (0=disable) | `300` (5 min) |
| `CLEANUP_INTERVAL` | Run cleanup every N seconds (0=disable) | `60` |
| `MAX_CACHE_SIZE_MB` | Max cache size in MB (0=disable) | `1800` (1.8 GB) |

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

## VRChat Usage

Use the proxy URL in VRChat video players:

**VOD Mode (recommended for full videos with seeking):**
```
http://proxy:8000/media.m3u8?m=<media_id>
```

**Live Mode (for real-time streaming without seeking):**
```
http://proxy:8000/live.m3u8?m=<media_id>
```

**All viewers using the same URL watch the same synchronized stream!**

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

## Cleanup & Resource Management

The proxy automatically manages cache to prevent OOM:

1. **Idle Stream Cleanup**
   - Streams not accessed for `STREAM_IDLE_TIMEOUT` seconds are removed
   - Default: 5 minutes (300s)
   - Cached files deleted, memory freed

2. **Size-Based Cleanup**
   - When cache exceeds `MAX_CACHE_SIZE_MB`, oldest streams are removed
   - Cleans down to 80% of limit to avoid thrashing
   - Default limit: 1.8 GB (leaves 200MB buffer from 2GB tmpfs)

3. **Background Task**
   - Runs every `CLEANUP_INTERVAL` seconds (default: 60s)
   - Performs both idle and size-based cleanup
   - Can be disabled by setting `CLEANUP_INTERVAL=0`

4. **Manual Management**
   - `POST /cleanup` - Trigger cleanup immediately
   - `DELETE /streams/{key}` - Stop specific stream
   - `GET /streams` - Monitor idle times and cache size

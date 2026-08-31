# Container Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `JELLYFIN_URL` | Jellyfin server URL | `http://jellyfin:8096` |
| `JELLYFIN_API_KEY` | **Required.** Jellyfin API key - also doubles as the proxy's own admin credential (send as `X-Admin-Key` header or `admin_key` query param) for admin/browsing endpoints and creating share links | (none - disables those endpoints) |
| `CACHE_DIR` | HLS cache directory | `/tmp/hls-cache` |
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
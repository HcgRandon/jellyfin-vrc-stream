# API Endpoints

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
curl "http://proxy:8000/vod.m3u8?m=abc123&audio=2&subtitle=5&admin_key=$JELLYFIN_API_KEY"
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

Share links are single-item, time-limited tokens meant to be pasted into a VRChat video player without exposing the rest of the library. All share-management endpoints require the `JELLYFIN_API_KEY` admin credential.

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
curl -X POST -H "X-Admin-Key: $JELLYFIN_API_KEY" -H 'Content-Type: application/json' \
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

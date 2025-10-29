#!/usr/bin/env python3
"""
Simple Jellyfin HLS Proxy for VRChat
- Streams once from Jellyfin
- Fans out to multiple clients
- Hides API key
"""

import re
import time
import shutil
import asyncio
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings
import urllib.request
import urllib.parse
import urllib.error
import json


class Settings(BaseSettings):
    jellyfin_url: str = "http://jellyfin:8096"
    jellyfin_api_key: str = ""
    cache_dir: str = "/tmp/hls-cache"
    stream_idle_timeout: int = 300  # 5 minutes
    locked_stream_idle_timeout: int = 86400  # 24 hours for locked streams
    cleanup_interval: int = 60  # Check every 60 seconds
    max_cache_size_mb: int = 0  # Disabled by default (0 = no size-based cleanup)

    # Quality settings - defaults optimized for high quality single-stream
    video_bitrate: int = 40000000  # 40 Mbps for very high quality
    audio_bitrate: int = 320000    # 320 Kbps for high quality audio
    max_streaming_bitrate: int = 50000000  # 50 Mbps total cap
    max_width: int = 1920  # Max resolution width
    max_height: int = 1080  # Max resolution height
    max_framerate: int = 60  # Max framerate

    # Encoding quality settings for better motion handling
    h264_profile: str = "high"   # baseline/main/high - high = better quality
    h264_level: str = "41"       # H.264 level (41 = 4.1, supports 1080p30, 42 = 4.2 supports 1080p60)
    max_ref_frames: int = 4      # More reference frames = better quality motion, but higher CPU (1-16)

    class Config:
        env_file = ".env"


settings = Settings()
app = FastAPI(title="Jellyfin HLS Proxy")

# Ensure cache directory exists
Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

# Mount static files if directory exists
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Track active streams
active_streams = {}

# Track pre-warm operations
prewarm_operations = {}

# Viewer activity timeout (if no segment request in this time, viewer is considered inactive)
VIEWER_TIMEOUT = 60  # seconds

# Lock persistence file
LOCK_FILE = Path(settings.cache_dir) / ".stream_locks.json"

# Startup recovery lock - prevents requests until recovery is complete
startup_complete = asyncio.Event()


def load_stream_locks():
    """Load locked stream keys from persistent storage"""
    if not LOCK_FILE.exists():
        return set()
    try:
        with open(LOCK_FILE, 'r') as f:
            data = json.load(f)
            return set(data.get('locked_streams', []))
    except Exception as e:
        print(f"Error loading stream locks: {e}")
        return set()


def save_stream_locks(locked_streams: set):
    """Save locked stream keys to persistent storage"""
    try:
        with open(LOCK_FILE, 'w') as f:
            json.dump({'locked_streams': list(locked_streams)}, f)
    except Exception as e:
        print(f"Error saving stream locks: {e}")


def get_locked_streams():
    """Get set of currently locked stream keys"""
    locked = set()
    for key, info in active_streams.items():
        if info.get('locked', False):
            locked.add(key)
    return locked


async def cleanup_task():
    """Background task to cleanup idle streams"""
    while True:
        try:
            await asyncio.sleep(settings.cleanup_interval)
            removed_idle = cleanup_idle_streams()
            removed_size = cleanup_by_size()
            if removed_idle > 0 or removed_size > 0:
                print(f"Cleanup: removed {removed_idle} idle stream(s), {removed_size} stream(s) due to size")
        except Exception as e:
            print(f"Error in cleanup task: {e}")


@app.on_event("startup")
async def startup_event():
    """Start background cleanup task and recover cached streams"""
    # Load locked streams from persistent storage
    locked_streams = load_stream_locks()
    if locked_streams:
        print(f"Loaded {len(locked_streams)} locked stream(s) from storage")

    # Try to recover streams from existing cache directories
    recovered = recover_cached_streams()
    if recovered > 0:
        print(f"Recovered {recovered} cached stream(s) from previous session")

    # Apply lock status to recovered streams
    for stream_key in locked_streams:
        if stream_key in active_streams:
            active_streams[stream_key]['locked'] = True
            print(f"Restored lock status for stream: {stream_key}")

    # Signal that startup/recovery is complete - clients can now make requests
    startup_complete.set()
    print("Startup complete - ready to accept requests")

    if settings.cleanup_interval > 0:
        asyncio.create_task(cleanup_task())
        print(f"Started cleanup task (interval: {settings.cleanup_interval}s, idle timeout: {settings.stream_idle_timeout}s, locked timeout: {settings.locked_stream_idle_timeout}s, max cache: {settings.max_cache_size_mb}MB)")
    else:
        print("Cleanup task disabled (cleanup_interval=0)")


def get_item_info(item_id: str):
    """Fetch item info from Jellyfin"""
    url = f"{settings.jellyfin_url}/Items/{item_id}?api_key={settings.jellyfin_api_key}"

    try:
        with urllib.request.urlopen(url, timeout=10.0) as response:
            return json.loads(response.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {e}")


def get_cache_size_mb():
    """Get total cache size in MB"""
    total_size = 0
    cache_path = Path(settings.cache_dir)
    if cache_path.exists():
        for item in cache_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    return total_size / (1024 * 1024)


def get_stream_cache_size_mb(stream_key: str):
    """Get cache size for a specific stream in MB"""
    total_size = 0
    cache_path = Path(settings.cache_dir) / stream_key
    if cache_path.exists():
        for item in cache_path.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    return total_size / (1024 * 1024)


def cleanup_idle_streams():
    """Remove idle streams and their cached files (respects locked streams)"""
    # Skip if disabled (0 or negative value)
    if settings.stream_idle_timeout <= 0:
        return 0

    current_time = time.time()
    streams_to_remove = []

    for stream_key, info in active_streams.items():
        idle_time = current_time - info.get('last_accessed', current_time)
        is_locked = info.get('locked', False)

        # Use different timeout for locked vs unlocked streams
        timeout = settings.locked_stream_idle_timeout if is_locked else settings.stream_idle_timeout

        if idle_time > timeout:
            streams_to_remove.append(stream_key)

    for stream_key in streams_to_remove:
        # Delete cached files
        cache_path = Path(settings.cache_dir) / stream_key
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"Cleaned up cache for idle stream: {stream_key}")
            except Exception as e:
                print(f"Error cleaning up cache for {stream_key}: {e}")

        # Remove from active streams
        del active_streams[stream_key]
        print(f"Removed idle stream: {stream_key}")

    return len(streams_to_remove)


def cleanup_by_size():
    """Remove oldest streams if cache is too large (skips locked streams)"""
    # Skip if disabled (0 or negative value)
    if settings.max_cache_size_mb <= 0:
        return 0

    cache_size_mb = get_cache_size_mb()

    if cache_size_mb < settings.max_cache_size_mb:
        return 0

    print(f"Cache size ({cache_size_mb:.1f} MB) exceeds limit ({settings.max_cache_size_mb} MB), cleaning up...")

    # Sort streams by last access time (oldest first), excluding locked streams
    sorted_streams = sorted(
        [(k, v) for k, v in active_streams.items() if not v.get('locked', False)],
        key=lambda x: x[1].get('last_accessed', 0)
    )

    removed = 0
    for stream_key, info in sorted_streams:
        if get_cache_size_mb() < settings.max_cache_size_mb * 0.8:  # Clean to 80% of limit
            break

        # Delete cached files
        cache_path = Path(settings.cache_dir) / stream_key
        if cache_path.exists():
            try:
                shutil.rmtree(cache_path)
                print(f"Cleaned up cache for stream (size limit): {stream_key}")
            except Exception as e:
                print(f"Error cleaning up cache for {stream_key}: {e}")

        # Remove from active streams
        del active_streams[stream_key]
        removed += 1

    return removed


def recover_cached_streams():
    """Try to recover stream entries from existing cache directories at startup"""
    cache_dir = Path(settings.cache_dir)
    if not cache_dir.exists():
        return 0

    recovered = 0

    # Iterate through cache directories
    for item in cache_dir.iterdir():
        if item.is_dir():
            stream_key = item.name

            # Skip if already in active_streams
            if stream_key in active_streams:
                continue

            # Parse stream_key format: {media_id}_{audio}_{subtitle}_{mode}
            parts = stream_key.rsplit('_', 3)
            if len(parts) != 4:
                print(f"Skipping invalid stream_key format: {stream_key}")
                continue

            media_id, audio_str, subtitle_str, mode = parts

            # Parse audio/subtitle (None if 'None')
            audio = int(audio_str) if audio_str != 'None' else None
            subtitle = int(subtitle_str) if subtitle_str != 'None' else None

            # Validate mode
            if mode not in ['vod', 'live']:
                print(f"Skipping invalid mode: {stream_key}")
                continue

            # Create minimal stream entry - will be fully initialized on first access
            active_streams[stream_key] = {
                'media_id': media_id,
                'audio': audio,
                'subtitle': subtitle,
                'mode': mode,
                'created_at': time.time(),
                'last_accessed': time.time(),
                'recovered': True  # Flag to indicate this was recovered from cache
            }

            print(f"Recovered cached stream: {stream_key}")
            recovered += 1

    return recovered


def cleanup_orphaned_cache():
    """Remove cache directories that don't correspond to any active stream"""
    cache_dir = Path(settings.cache_dir)
    if not cache_dir.exists():
        return 0

    removed = 0
    active_keys = set(active_streams.keys())

    # Iterate through cache directories
    for item in cache_dir.iterdir():
        if item.is_dir():
            stream_key = item.name
            # If this stream_key is not in active_streams, it's orphaned
            if stream_key not in active_keys:
                try:
                    shutil.rmtree(item)
                    print(f"Cleaned up orphaned cache directory: {stream_key}")
                    removed += 1
                except Exception as e:
                    print(f"Error cleaning up orphaned cache {stream_key}: {e}")

    return removed


def find_best_streams(item_info):
    """Find best audio (jpn > eng) and subtitle (eng) streams"""
    media_streams = item_info.get('MediaSources', [{}])[0].get('MediaStreams', [])

    # Find audio
    audio_streams = [s for s in media_streams if s.get('Type') == 'Audio']
    audio_index = None
    for stream in audio_streams:
        if stream.get('Language') == 'jpn':
            audio_index = stream.get('Index')
            break
    if audio_index is None:
        for stream in audio_streams:
            if stream.get('Language') == 'eng':
                audio_index = stream.get('Index')
                break
    if audio_index is None and audio_streams:
        audio_index = audio_streams[0].get('Index')

    # Find subtitles
    subtitle_streams = [s for s in media_streams if s.get('Type') == 'Subtitle']
    subtitle_index = None
    for stream in subtitle_streams:
        if stream.get('Language') == 'eng' and 'full' in stream.get('DisplayTitle', '').lower():
            subtitle_index = stream.get('Index')
            break
    if subtitle_index is None:
        for stream in subtitle_streams:
            if stream.get('Language') == 'eng':
                subtitle_index = stream.get('Index')
                break

    return audio_index, subtitle_index


async def fetch_and_cache(url: str, cache_path: Path, timeout: float = 60.0) -> Path:
    """Fetch content from URL and cache it with streaming"""
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    def _blocking_fetch():
        # Stream the content in chunks instead of loading all at once
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with open(cache_path, 'wb') as f:
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        return cache_path

    try:
        # Run blocking I/O in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_blocking_fetch)
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=f"Jellyfin error: {e.reason}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=504, detail=f"Connection timeout: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content: {e}")


def get_client_ip(request: Request) -> str:
    """Get real client IP from proxy headers or direct connection"""
    # Check X-Forwarded-For header first (set by Traefik)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        # X-Forwarded-For can be a comma-separated list, get the first (original client)
        return forwarded_for.split(',')[0].strip()

    # Fall back to direct connection IP
    return request.client.host if request.client else 'unknown'


def rewrite_playlist_vod(content: bytes, media_id: str) -> str:
    """Rewrite VOD playlist URLs - preserves query params but strips api_key for security"""
    text = content.decode('utf-8')
    lines = text.split('\n')
    new_lines = []

    def strip_api_key(url: str) -> str:
        """Remove api_key from URL to avoid exposing it to clients"""
        if '?' not in url:
            return url
        path, query = url.split('?', 1)
        params = [p for p in query.split('&') if not p.startswith('api_key=')]
        if params:
            return f"{path}?{'&'.join(params)}"
        return path

    for line in lines:
        if line and not line.startswith('#'):
            # Rewrite .m3u8 playlist references, stripping hls*/session/ prefix like we do for .ts
            # Format: hls1/main/playlist.m3u8?query -> playlist.m3u8?query
            m3u8_match = re.search(r'(?:hls\d*/[^/]+/)?(.+\.m3u8.*)', line)
            if m3u8_match and '.m3u8' in line:
                playlist_path = m3u8_match.group(1)  # Strip hls*/session/ prefix
                playlist_path = strip_api_key(playlist_path)  # Strip api_key
                new_lines.append(f"/vod/{media_id}/{playlist_path}")
                continue

            # VOD segments have query params we need to preserve (but not api_key)
            # Format: hls1/main/0.ts?mediaSourceId=...&runtimeTicks=...
            ts_match = re.search(r'hls\d*/[^/]+/(.+\.ts.*)', line)
            if ts_match:
                segment_path = ts_match.group(1)  # Includes query params, strips hls*/session/ prefix
                segment_path = strip_api_key(segment_path)  # Strip api_key
                new_lines.append(f"/vod/{media_id}/{segment_path}")
                continue
        new_lines.append(line)

    return '\n'.join(new_lines)


def rewrite_playlist_live(content: bytes, stream_key: str) -> str:
    """Rewrite live playlist URLs - use stream_key to distinguish different audio/subtitle combos"""
    text = content.decode('utf-8')
    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if line and not line.startswith('#'):
            # Live segments are simpler
            match = re.search(r'hls\d*/[^/]+/([^/?]+\.ts)', line)
            if match:
                segment_file = match.group(1)
                new_lines.append(f"/live/{stream_key}/{segment_file}")
                continue
        new_lines.append(line)

    return '\n'.join(new_lines)


@app.get("/")
async def root():
    return {
        "service": "Jellyfin HLS Proxy",
        "active_streams": len(active_streams),
        "cache_size_mb": round(get_cache_size_mb(), 2),
        "cache_limit_mb": settings.max_cache_size_mb
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/manage", response_class=HTMLResponse)
async def management_portal():
    """Serve management portal"""
    static_file = Path(__file__).parent / "static" / "index.html"
    if static_file.exists():
        return HTMLResponse(content=static_file.read_text())
    raise HTTPException(status_code=404, detail="Management portal not found")


@app.get("/media/{media_id}/streams")
async def get_media_streams(media_id: str):
    """Get available audio and subtitle streams for a media item"""
    try:
        item_info = get_item_info(media_id)
        media_streams = item_info.get('MediaSources', [{}])[0].get('MediaStreams', [])

        # Get audio streams
        audio_streams = []
        for stream in media_streams:
            if stream.get('Type') == 'Audio':
                audio_streams.append({
                    'index': stream.get('Index'),
                    'language': stream.get('Language', 'und'),
                    'title': stream.get('DisplayTitle', ''),
                    'codec': stream.get('Codec', ''),
                })

        # Get subtitle streams
        subtitle_streams = []
        for stream in media_streams:
            if stream.get('Type') == 'Subtitle':
                subtitle_streams.append({
                    'index': stream.get('Index'),
                    'language': stream.get('Language', 'und'),
                    'title': stream.get('DisplayTitle', ''),
                    'codec': stream.get('Codec', ''),
                })

        # Get default selections
        default_audio, default_subtitle = find_best_streams(item_info)

        return {
            'audio_streams': audio_streams,
            'subtitle_streams': subtitle_streams,
            'default_audio': default_audio,
            'default_subtitle': default_subtitle,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch streams: {e}")


@app.get("/series/{series_id}/episodes")
async def get_series_episodes(series_id: str):
    """Get all episodes for a series"""
    url = f"{settings.jellyfin_url}/Shows/{series_id}/Episodes?Fields=Overview,PrimaryImageAspectRatio&api_key={settings.jellyfin_api_key}"

    try:
        def fetch():
            with urllib.request.urlopen(url, timeout=10.0) as response:
                return json.loads(response.read())

        data = await asyncio.to_thread(fetch)

        episodes = []
        for item in data.get('Items', []):
            item_id = item.get('Id')
            image_url = None
            if item.get('ImageTags', {}).get('Primary'):
                image_url = f"{settings.jellyfin_url}/Items/{item_id}/Images/Primary?maxHeight=300&quality=90"

            episodes.append({
                'id': item_id,
                'name': item.get('Name'),
                'type': 'Episode',
                'series': item.get('SeriesName'),
                'season': item.get('ParentIndexNumber'),
                'episode': item.get('IndexNumber'),
                'overview': item.get('Overview', '')[:200] if item.get('Overview') else '',
                'image': image_url,
            })

        return {"episodes": episodes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch episodes: {e}")


@app.get("/recent")
async def get_recent_media(limit: int = Query(20, description="Number of items to return")):
    """Get recently added media"""
    url = f"{settings.jellyfin_url}/Items?SortBy=DateCreated&SortOrder=Descending&Recursive=true&IncludeItemTypes=Series,Movie&Fields=Overview,PrimaryImageAspectRatio&Limit={limit}&api_key={settings.jellyfin_api_key}"

    try:
        def fetch():
            with urllib.request.urlopen(url, timeout=10.0) as response:
                return json.loads(response.read())

        data = await asyncio.to_thread(fetch)

        items = []
        for item in data.get('Items', []):
            item_id = item.get('Id')
            item_type = item.get('Type')

            image_url = None
            if item.get('ImageTags', {}).get('Primary'):
                image_url = f"{settings.jellyfin_url}/Items/{item_id}/Images/Primary?maxHeight=300&quality=90"

            result = {
                'id': item_id,
                'name': item.get('Name'),
                'type': item_type,
                'year': item.get('ProductionYear'),
                'overview': item.get('Overview', '')[:200] if item.get('Overview') else '',
                'image': image_url,
            }

            if item_type == 'Series':
                result['is_series'] = True

            items.append(result)

        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent media: {e}")


@app.get("/search")
async def search_media(q: str = Query("", description="Search query")):
    """Search Jellyfin media library"""
    if not q:
        return {"items": []}

    # Search Jellyfin Items API - only Series and Movies (episodes via expand)
    url = f"{settings.jellyfin_url}/Items?searchTerm={urllib.parse.quote(q)}&Recursive=true&IncludeItemTypes=Series,Movie&Fields=Overview,PrimaryImageAspectRatio&api_key={settings.jellyfin_api_key}"

    try:
        def fetch():
            with urllib.request.urlopen(url, timeout=10.0) as response:
                return json.loads(response.read())

        data = await asyncio.to_thread(fetch)

        # Simplify response for frontend
        items = []
        for item in data.get('Items', []):
            item_id = item.get('Id')
            item_type = item.get('Type')

            # Build image URL if item has primary image
            image_url = None
            if item.get('ImageTags', {}).get('Primary'):
                image_url = f"{settings.jellyfin_url}/Items/{item_id}/Images/Primary?maxHeight=300&quality=90"

            result = {
                'id': item_id,
                'name': item.get('Name'),
                'type': item_type,
                'year': item.get('ProductionYear'),
                'overview': item.get('Overview', '')[:200] if item.get('Overview') else '',
                'image': image_url,
            }

            # Add episode-specific info
            if item_type == 'Episode':
                result['series'] = item.get('SeriesName')
                result['season'] = item.get('ParentIndexNumber')
                result['episode'] = item.get('IndexNumber')
            elif item_type == 'Series':
                # For series, we want to fetch episodes
                result['is_series'] = True

            items.append(result)

        return {"items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


async def _get_stream_playlist(m: str, audio: Optional[int], subtitle: Optional[int], mode: str):
    """Common playlist logic for both VOD and live modes"""
    # Wait for startup/recovery to complete
    await startup_complete.wait()

    stream_key = f"{m}_{audio}_{subtitle}_{mode}"

    # Check if stream needs initialization (new or recovered from cache)
    needs_init = (stream_key not in active_streams or
                  active_streams[stream_key].get('recovered', False) or
                  'jellyfin_url' not in active_streams[stream_key])

    if needs_init:
        item_info = get_item_info(m)

        if audio is None or subtitle is None:
            auto_audio, auto_sub = find_best_streams(item_info)
            if audio is None:
                audio = auto_audio
            if subtitle is None:
                subtitle = auto_sub

        # Use a unique DeviceId that includes subtitle info to break Jellyfin's cache
        # This ensures we get a fresh transcode with subtitles
        device_suffix = f"-a{audio}-s{subtitle}" if subtitle is not None else f"-a{audio}"

        params = {
            'mediaSourceId': m,
            'api_key': settings.jellyfin_api_key,
            'DeviceId': f'jellyfin-proxy{device_suffix}',
            'VideoCodec': 'h264',
            'AudioCodec': 'aac',
            'VideoBitrate': str(settings.video_bitrate),
            'AudioBitrate': str(settings.audio_bitrate),
            'MaxStreamingBitrate': str(settings.max_streaming_bitrate),
            'MaxWidth': str(settings.max_width),
            'MaxHeight': str(settings.max_height),
            'MaxFramerate': str(settings.max_framerate),
            'MaxAudioChannels': '2',  # Force stereo for compatibility
            'Profile': settings.h264_profile,
            'Level': settings.h264_level,
            'MaxRefFrames': str(settings.max_ref_frames),
            'SegmentContainer': 'ts',
            'MinSegments': '2',
        }

        if audio is not None:
            params['AudioStreamIndex'] = str(audio)

        if subtitle is not None:
            params['SubtitleMethod'] = 'Encode'
            params['SubtitleStreamIndex'] = str(subtitle)

        param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
        endpoint = "main.m3u8" if mode == "vod" else "live.m3u8"
        jellyfin_url = f"{settings.jellyfin_url}/Videos/{m}/{endpoint}?{param_str}"

        active_streams[stream_key] = {
            'jellyfin_url': jellyfin_url,
            'media_id': m,
            'audio': audio,
            'subtitle': subtitle,
            'mode': mode,
            'last_accessed': time.time(),
            'created_at': time.time(),
        }
        # Note: This overwrites the stream dict, automatically clearing 'recovered' flag

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

    # Check if playlist is already cached (from prewarm or previous request)
    if 'playlist_content' in active_streams[stream_key]:
        content = active_streams[stream_key]['playlist_content']
    else:
        # Fetch from Jellyfin only if not cached
        jellyfin_url = active_streams[stream_key]['jellyfin_url']

        try:
            # Run blocking urllib call in thread pool to avoid blocking event loop
            def fetch_playlist():
                with urllib.request.urlopen(jellyfin_url, timeout=60.0) as response:
                    return response.read()

            content = await asyncio.to_thread(fetch_playlist)
            # Cache the playlist content to avoid creating multiple Jellyfin sessions
            active_streams[stream_key]['playlist_content'] = content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch playlist: {e}")

    # Store session info (but don't overwrite if already set by prewarm!)
    if content and 'session_id' not in active_streams[stream_key]:
        match = re.search(rb'(hls\d*)/([^/]+)/', content)
        if match:
            hls_dir = match.group(1).decode('utf-8')
            session_id = match.group(2).decode('utf-8')
            active_streams[stream_key]['hls_dir'] = hls_dir
            active_streams[stream_key]['session_id'] = session_id

    # Rewrite based on mode
    if mode == "vod":
        rewritten = rewrite_playlist_vod(content, m)
    else:
        rewritten = rewrite_playlist_live(content, stream_key)

    return PlainTextResponse(
        rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/vod.m3u8")
async def get_vod_playlist(
    m: str = Query(..., description="Media ID"),
    audio: Optional[int] = Query(None, description="Audio stream index"),
    subtitle: Optional[int] = Query(None, description="Subtitle stream index"),
):
    """Get VOD HLS playlist (seekable, full video)"""
    return await _get_stream_playlist(m, audio, subtitle, "vod")


@app.get("/live.m3u8")
async def get_live_playlist(
    m: str = Query(..., description="Media ID"),
    audio: Optional[int] = Query(None, description="Audio stream index"),
    subtitle: Optional[int] = Query(None, description="Subtitle stream index"),
):
    """Get live HLS playlist (real-time streaming)"""
    return await _get_stream_playlist(m, audio, subtitle, "live")


@app.get("/vod/{media_id}/{segment_path:path}")
async def get_vod_segment(media_id: str, segment_path: str, request: Request):
    """Get VOD segment with query params preserved"""
    # Wait for startup/recovery to complete
    await startup_complete.wait()

    # Parse audio/subtitle from query params to find the correct stream
    query_params = dict(request.query_params)
    req_audio = query_params.get('AudioStreamIndex')
    req_subtitle = query_params.get('SubtitleStreamIndex')

    # Convert to int or None to match stream info
    req_audio = int(req_audio) if req_audio else None
    req_subtitle = int(req_subtitle) if req_subtitle else None

    stream_key = None
    for key, info in active_streams.items():
        if (info['media_id'] == media_id and
            info.get('mode') == 'vod' and
            info.get('audio') == req_audio and
            info.get('subtitle') == req_subtitle):
            stream_key = key
            break

    if not stream_key:
        raise HTTPException(status_code=404, detail=f"Stream not found: media_id={media_id}, audio={req_audio}, subtitle={req_subtitle}")

    if 'session_id' not in active_streams[stream_key]:
        raise HTTPException(status_code=404, detail="Stream found but not initialized (no session_id)")

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

    session_id = active_streams[stream_key]['session_id']
    hls_dir = active_streams[stream_key].get('hls_dir', 'hls1')

    # Extract segment filename for cache key (without query params)
    segment_file = segment_path.split('?')[0]

    # Check if this is a nested playlist or a segment
    is_playlist = segment_file.endswith('.m3u8')

    # Track viewer for .ts segments (not playlists)
    if not is_playlist and segment_file.endswith('.ts'):
        client_ip = get_client_ip(request)
        if 'viewers' not in active_streams[stream_key]:
            active_streams[stream_key]['viewers'] = {}
        active_streams[stream_key]['viewers'][client_ip] = time.time()

    # Build Jellyfin URL preserving query params and adding api_key back
    query_string = request.url.query
    if query_string:
        # Add api_key to the query string if not already present
        if 'api_key=' not in query_string:
            query_string = f"api_key={settings.jellyfin_api_key}&{query_string}"
        segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment_path}?{query_string}"
    else:
        segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment_path}?api_key={settings.jellyfin_api_key}"

    # For playlists, fetch and rewrite (don't cache - always fetch fresh)
    if is_playlist:
        try:
            def fetch_playlist():
                with urllib.request.urlopen(segment_url, timeout=60.0) as response:
                    return response.read()

            content = await asyncio.to_thread(fetch_playlist)
            rewritten = rewrite_playlist_vod(content, media_id)

            return PlainTextResponse(
                rewritten,
                media_type="application/vnd.apple.mpegurl",
                headers={"Access-Control-Allow-Origin": "*"}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch nested playlist: {e}")

    # For segments, cache and serve
    cache_path = Path(settings.cache_dir) / stream_key / segment_file

    if cache_path.exists():
        return FileResponse(
            cache_path,
            media_type="video/mp2t",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    # Use longer timeout for first segment (transcoding startup)
    segment_num = segment_file.split('.')[0]
    timeout = 120.0 if segment_num in ['0', '1', '2'] else 60.0

    await fetch_and_cache(segment_url, cache_path, timeout=timeout)

    return FileResponse(
        cache_path,
        media_type="video/mp2t",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/live/{stream_key}/{segment_file}")
async def get_live_segment(stream_key: str, segment_file: str, request: Request):
    """Get live segment - use stream_key to distinguish different audio/subtitle combos"""
    # Wait for startup/recovery to complete
    await startup_complete.wait()

    if stream_key not in active_streams or 'session_id' not in active_streams[stream_key]:
        raise HTTPException(status_code=404, detail="Stream not found or not initialized")

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

    # Track viewer for .ts segments
    if segment_file.endswith('.ts'):
        client_ip = get_client_ip(request)
        if 'viewers' not in active_streams[stream_key]:
            active_streams[stream_key]['viewers'] = {}
        active_streams[stream_key]['viewers'][client_ip] = time.time()

    media_id = active_streams[stream_key]['media_id']
    session_id = active_streams[stream_key]['session_id']
    hls_dir = active_streams[stream_key].get('hls_dir', 'hls1')
    cache_path = Path(settings.cache_dir) / stream_key / segment_file

    if cache_path.exists():
        return FileResponse(
            cache_path,
            media_type="video/mp2t",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment_file}?api_key={settings.jellyfin_api_key}"

    # Use longer timeout for first segment (transcoding startup)
    segment_num = segment_file.split('.')[0]
    timeout = 120.0 if segment_num in ['0', '1', '2'] else 60.0

    await fetch_and_cache(segment_url, cache_path, timeout=timeout)

    return FileResponse(
        cache_path,
        media_type="video/mp2t",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/streams")
async def list_streams():
    """List active streams with media details"""
    current_time = time.time()
    streams = []

    for key, info in active_streams.items():
        # Count cached segments
        cache_path = Path(settings.cache_dir) / key
        segments_cached = 0
        if cache_path.exists():
            segments_cached = len(list(cache_path.glob("*.ts")))

        # Calculate total segments from playlist if available
        total_segments = None
        cache_percentage = None

        if 'playlist_content' in info:
            try:
                playlist_text = info['playlist_content'].decode('utf-8')
                # Count non-comment, non-empty lines that are segments
                total_segments = sum(1 for line in playlist_text.split('\n')
                                    if line and not line.startswith('#') and '.ts' in line)

                if total_segments > 0:
                    cache_percentage = round((segments_cached / total_segments) * 100, 1)
            except:
                pass

        # Count active viewers (those who accessed segments in last VIEWER_TIMEOUT seconds)
        active_viewers = 0
        if 'viewers' in info:
            # Clean up stale viewers while counting
            stale_viewers = []
            for viewer_ip, last_seen in info['viewers'].items():
                if current_time - last_seen <= VIEWER_TIMEOUT:
                    active_viewers += 1
                else:
                    stale_viewers.append(viewer_ip)
            # Remove stale viewers
            for viewer_ip in stale_viewers:
                del info['viewers'][viewer_ip]

        stream_info = {
            "stream_key": key,
            "media_id": info['media_id'],
            "audio": info.get('audio'),
            "subtitle": info.get('subtitle'),
            "mode": info.get('mode'),
            "idle_seconds": int(current_time - info.get('last_accessed', current_time)),
            "age_seconds": int(current_time - info.get('created_at', current_time)),
            "cache_size_mb": round(get_stream_cache_size_mb(key), 2),
            "segments_cached": segments_cached,
            "total_segments": total_segments,
            "cache_percentage": cache_percentage,
            "active_viewers": active_viewers,
            "locked": info.get('locked', False),
        }

        # Try to fetch media details
        try:
            item_info = get_item_info(info['media_id'])
            stream_info['media_name'] = item_info.get('Name', 'Unknown')
            stream_info['media_type'] = item_info.get('Type', 'Unknown')

            # Add image if available
            if item_info.get('ImageTags', {}).get('Primary'):
                stream_info['media_image'] = f"{settings.jellyfin_url}/Items/{info['media_id']}/Images/Primary?maxHeight=100&quality=80"

            # Add series info for episodes
            if item_info.get('Type') == 'Episode':
                stream_info['series_name'] = item_info.get('SeriesName')
                stream_info['season'] = item_info.get('ParentIndexNumber')
                stream_info['episode'] = item_info.get('IndexNumber')
        except:
            # If fetching fails, just use defaults
            stream_info['media_name'] = 'Unknown'
            stream_info['media_type'] = 'Unknown'

        streams.append(stream_info)

    return {
        "streams": streams,
        "cache_size_mb": round(get_cache_size_mb(), 2)
    }


@app.delete("/streams/{stream_key}")
async def delete_stream(stream_key: str):
    """Manually stop and cleanup a stream"""
    if stream_key not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Delete cached files
    cache_path = Path(settings.cache_dir) / stream_key
    if cache_path.exists():
        try:
            shutil.rmtree(cache_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete cache: {e}")

    # Remove from active streams
    del active_streams[stream_key]

    # Update lock persistence if stream was locked
    locked_streams = get_locked_streams()
    save_stream_locks(locked_streams)

    return {"status": "deleted", "stream_key": stream_key}


@app.post("/streams/{stream_key}/lock")
async def lock_stream(stream_key: str):
    """Lock a stream to prevent cleanup (24 hour idle timeout instead of 5 min)"""
    if stream_key not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Set locked flag
    active_streams[stream_key]['locked'] = True

    # Persist to disk
    locked_streams = get_locked_streams()
    save_stream_locks(locked_streams)

    return {"status": "locked", "stream_key": stream_key}


@app.post("/streams/{stream_key}/unlock")
async def unlock_stream(stream_key: str):
    """Unlock a stream to allow normal cleanup (5 min idle timeout)"""
    if stream_key not in active_streams:
        raise HTTPException(status_code=404, detail="Stream not found")

    # Remove locked flag
    active_streams[stream_key]['locked'] = False

    # Persist to disk
    locked_streams = get_locked_streams()
    save_stream_locks(locked_streams)

    return {"status": "unlocked", "stream_key": stream_key}


@app.post("/prewarm/start")
async def start_prewarm(
    media_id: str,
    audio: Optional[int] = None,
    subtitle: Optional[int] = None,
    mode: str = "vod"
):
    """Start pre-warming a stream by fetching segments in advance (VOD only)"""
    import uuid

    # Only allow pre-warming for VOD mode
    if mode != "vod":
        raise HTTPException(status_code=400, detail="Pre-warming is only supported for VOD mode")

    # Generate unique prewarm ID
    prewarm_id = str(uuid.uuid4())

    # Initialize prewarm operation
    prewarm_operations[prewarm_id] = {
        "id": prewarm_id,
        "media_id": media_id,
        "audio": audio,
        "subtitle": subtitle,
        "mode": mode,
        "status": "initializing",
        "start_time": time.time(),
        "segments_cached": 0,
        "total_segments": 0,
        "bytes_cached": 0,
        "error": None,
        "task": None
    }

    # Start async background task
    task = asyncio.create_task(prewarm_worker(prewarm_id))
    prewarm_operations[prewarm_id]["task"] = task

    return {"prewarm_id": prewarm_id, "status": "started"}


async def prewarm_worker(prewarm_id: str):
    """Background worker that fetches all segments for a VOD stream"""
    try:
        op = prewarm_operations[prewarm_id]
        media_id = op["media_id"]
        audio = op["audio"]
        subtitle = op["subtitle"]

        # Initialize the stream first
        op["status"] = "initializing"

        # Start VOD stream
        stream_key = f"{media_id}_{audio}_{subtitle}_vod"

        # Check if already active
        if stream_key in active_streams:
            jellyfin_url = active_streams[stream_key]['jellyfin_url']
        else:
            # Initialize new stream
            m = media_id
            # Use a unique DeviceId that includes subtitle info to break Jellyfin's cache
            # This ensures we get a fresh transcode with subtitles
            device_suffix = f"-a{audio}-s{subtitle}" if subtitle is not None else f"-a{audio}"

            params = {
                'mediaSourceId': m,
                'api_key': settings.jellyfin_api_key,
                'DeviceId': f'jellyfin-proxy{device_suffix}',
                'VideoCodec': 'h264',
                'AudioCodec': 'aac',
                'VideoBitrate': str(settings.video_bitrate),
                'AudioBitrate': str(settings.audio_bitrate),
                'MaxStreamingBitrate': str(settings.max_streaming_bitrate),
                'MaxWidth': str(settings.max_width),
                'MaxHeight': str(settings.max_height),
                'MaxFramerate': str(settings.max_framerate),
                'MaxAudioChannels': '2',
                'Profile': settings.h264_profile,
                'Level': settings.h264_level,
                'MaxRefFrames': str(settings.max_ref_frames),
                'SegmentContainer': 'ts',
                'MinSegments': '2',
            }

            if audio is not None:
                params['AudioStreamIndex'] = str(audio)
            if subtitle is not None:
                params['SubtitleMethod'] = 'Encode'
                params['SubtitleStreamIndex'] = str(subtitle)

            jellyfin_url = f"{settings.jellyfin_url}/Videos/{m}/main.m3u8?{urllib.parse.urlencode(params)}"

            # Store stream info
            active_streams[stream_key] = {
                'media_id': media_id,
                'audio': audio,
                'subtitle': subtitle,
                'mode': 'vod',
                'jellyfin_url': jellyfin_url,
                'created_at': time.time(),
                'last_accessed': time.time()
            }

        # Fetch the playlist to get segment list
        op["status"] = "fetching_playlist"

        def fetch_playlist():
            with urllib.request.urlopen(jellyfin_url, timeout=120.0) as response:
                return response.read()

        content = await asyncio.to_thread(fetch_playlist)

        # Cache the playlist content to avoid creating multiple Jellyfin sessions
        active_streams[stream_key]['playlist_content'] = content

        # Extract and store session info (critical for playback!)
        # Don't overwrite if already set to avoid creating conflicting sessions
        if content and 'session_id' not in active_streams[stream_key]:
            match = re.search(rb'(hls\d*)/([^/]+)/', content)
            if match:
                hls_dir = match.group(1).decode('utf-8')
                session_id = match.group(2).decode('utf-8')
                active_streams[stream_key]['hls_dir'] = hls_dir
                active_streams[stream_key]['session_id'] = session_id
            else:
                raise Exception("Failed to extract session info from playlist")

        # Parse playlist to extract segments
        text = content.decode('utf-8')
        lines = text.split('\n')
        segments = []

        for line in lines:
            if line and not line.startswith('#'):
                # VOD segments come with full path and query params: hls1/main/0.ts?params...
                # Store the segment filename with query params for cache key
                match = re.search(r'hls\d*/[^/]+/(.+\.ts.*)', line)
                if match:
                    # Store the full relative path from hls*/session/ onwards
                    segments.append(match.group(1))

        op["total_segments"] = len(segments)
        op["status"] = "warming"

        # Fetch all segments
        for i, segment in enumerate(segments):
            # Check if cancelled
            if prewarm_id not in prewarm_operations or prewarm_operations[prewarm_id]["status"] == "cancelled":
                op["status"] = "cancelled"
                return

            # Extract just the filename for cache path (without query params)
            segment_file = segment.split('?')[0]
            cache_path = Path(settings.cache_dir) / stream_key / segment_file

            if not cache_path.exists():
                # Build segment URL
                session_id = active_streams[stream_key].get('session_id')
                hls_dir = active_streams[stream_key].get('hls_dir', 'hls1')

                # For VOD, segment already has full path with all query params from Jellyfin
                # Just prepend the Jellyfin base URL and Videos path
                segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment}"

                # Fetch and cache
                timeout = 120.0 if i < 3 else 60.0
                await fetch_and_cache(segment_url, cache_path, timeout=timeout)

            # Update progress
            if cache_path.exists():
                op["segments_cached"] = i + 1
                op["bytes_cached"] = sum(f.stat().st_size for f in (Path(settings.cache_dir) / stream_key).glob("*.ts"))

        # Mark as ready
        op["status"] = "ready"

    except Exception as e:
        if prewarm_id in prewarm_operations:
            prewarm_operations[prewarm_id]["status"] = "error"
            prewarm_operations[prewarm_id]["error"] = str(e)
        print(f"Pre-warm error: {e}")


@app.get("/prewarm/{prewarm_id}/status")
async def get_prewarm_status(prewarm_id: str):
    """Get status of a pre-warm operation"""
    if prewarm_id not in prewarm_operations:
        raise HTTPException(status_code=404, detail="Pre-warm operation not found")

    op = prewarm_operations[prewarm_id]

    # Calculate elapsed time
    elapsed = time.time() - op["start_time"]

    # Get media info
    media_info = {}
    try:
        item_info = get_item_info(op["media_id"])
        media_info["name"] = item_info.get("Name", "Unknown")
        media_info["type"] = item_info.get("Type", "Unknown")

        if item_info.get('Type') == 'Episode':
            media_info['series_name'] = item_info.get('SeriesName')
            media_info['season'] = item_info.get('ParentIndexNumber')
            media_info['episode'] = item_info.get('IndexNumber')
    except:
        media_info["name"] = "Unknown"
        media_info["type"] = "Unknown"

    # Calculate progress percentage
    progress = 0
    if op["total_segments"] > 0:
        progress = (op["segments_cached"] / op["total_segments"]) * 100

    return {
        "prewarm_id": prewarm_id,
        "media_id": op["media_id"],
        "media_info": media_info,
        "audio": op["audio"],
        "subtitle": op["subtitle"],
        "mode": op["mode"],
        "status": op["status"],
        "elapsed_seconds": int(elapsed),
        "segments_cached": op["segments_cached"],
        "total_segments": op["total_segments"],
        "progress_percent": round(progress, 1),
        "bytes_cached": op["bytes_cached"],
        "error": op["error"]
    }


@app.post("/prewarm/{prewarm_id}/cancel")
async def cancel_prewarm(prewarm_id: str):
    """Cancel a pre-warm operation"""
    if prewarm_id not in prewarm_operations:
        raise HTTPException(status_code=404, detail="Pre-warm operation not found")

    op = prewarm_operations[prewarm_id]

    if op["status"] in ["ready", "error", "cancelled"]:
        return {"status": "already_finished", "current_status": op["status"]}

    # Mark as cancelled
    op["status"] = "cancelled"

    # Cancel the task
    if op["task"] and not op["task"].done():
        op["task"].cancel()

    return {"status": "cancelled"}


@app.get("/prewarm/list")
async def list_prewarms():
    """List all pre-warm operations"""
    operations = []

    for prewarm_id, op in prewarm_operations.items():
        elapsed = time.time() - op["start_time"]
        progress = 0
        if op["total_segments"] > 0:
            progress = (op["segments_cached"] / op["total_segments"]) * 100

        operations.append({
            "prewarm_id": prewarm_id,
            "media_id": op["media_id"],
            "audio": op["audio"],
            "subtitle": op["subtitle"],
            "mode": op["mode"],
            "status": op["status"],
            "elapsed_seconds": int(elapsed),
            "progress_percent": round(progress, 1),
            "segments_cached": op["segments_cached"],
            "total_segments": op["total_segments"]
        })

    return {"operations": operations}


@app.delete("/prewarm/{prewarm_id}")
async def delete_prewarm(prewarm_id: str):
    """Delete a pre-warm operation from the list"""
    if prewarm_id not in prewarm_operations:
        raise HTTPException(status_code=404, detail="Pre-warm operation not found")

    # Cancel if still running
    op = prewarm_operations[prewarm_id]
    if op["task"] and not op["task"].done():
        op["task"].cancel()

    del prewarm_operations[prewarm_id]

    return {"status": "deleted"}


@app.post("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup"""
    removed_idle = cleanup_idle_streams()
    removed_size = cleanup_by_size()
    return {
        "removed_idle": removed_idle,
        "removed_size": removed_size,
        "cache_size_mb": round(get_cache_size_mb(), 2)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

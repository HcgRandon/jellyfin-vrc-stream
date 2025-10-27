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
    cleanup_interval: int = 60  # Check every 60 seconds
    max_cache_size_mb: int = 1800  # 1.8 GB (leave 200MB buffer from 2GB limit)

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
    """Start background cleanup task"""
    if settings.cleanup_interval > 0:
        asyncio.create_task(cleanup_task())
        print(f"Started cleanup task (interval: {settings.cleanup_interval}s, idle timeout: {settings.stream_idle_timeout}s, max cache: {settings.max_cache_size_mb}MB)")
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


def cleanup_idle_streams():
    """Remove idle streams and their cached files"""
    # Skip if disabled (0 or negative value)
    if settings.stream_idle_timeout <= 0:
        return 0

    current_time = time.time()
    streams_to_remove = []

    for stream_key, info in active_streams.items():
        idle_time = current_time - info.get('last_accessed', current_time)
        if idle_time > settings.stream_idle_timeout:
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
    """Remove oldest streams if cache is too large"""
    # Skip if disabled (0 or negative value)
    if settings.max_cache_size_mb <= 0:
        return 0

    cache_size_mb = get_cache_size_mb()

    if cache_size_mb < settings.max_cache_size_mb:
        return 0

    print(f"Cache size ({cache_size_mb:.1f} MB) exceeds limit ({settings.max_cache_size_mb} MB), cleaning up...")

    # Sort streams by last access time (oldest first)
    sorted_streams = sorted(
        active_streams.items(),
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


def fetch_and_cache(url: str, cache_path: Path, timeout: float = 60.0) -> Path:
    """Fetch content from URL and cache it with streaming"""
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
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
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=f"Jellyfin error: {e.reason}")
    except urllib.error.URLError as e:
        raise HTTPException(status_code=504, detail=f"Connection timeout: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch content: {e}")


def rewrite_playlist_vod(content: bytes, media_id: str) -> str:
    """Rewrite VOD playlist URLs - preserves query params"""
    text = content.decode('utf-8')
    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if line and not line.startswith('#'):
            # VOD segments have query params we need to preserve
            # Format: hls1/main/0.ts?mediaSourceId=...&runtimeTicks=...
            match = re.search(r'hls\d*/[^/]+/(.+\.ts.*)', line)
            if match:
                segment_path = match.group(1)  # Includes query params
                new_lines.append(f"/vod/{media_id}/{segment_path}")
                continue
        new_lines.append(line)

    return '\n'.join(new_lines)


def rewrite_playlist_live(content: bytes, media_id: str) -> str:
    """Rewrite live playlist URLs - simple segments"""
    text = content.decode('utf-8')
    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if line and not line.startswith('#'):
            # Live segments are simpler
            match = re.search(r'hls\d*/[^/]+/([^/?]+\.ts)', line)
            if match:
                segment_file = match.group(1)
                new_lines.append(f"/live/{media_id}/{segment_file}")
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
        with urllib.request.urlopen(url, timeout=10.0) as response:
            data = json.loads(response.read())

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
        with urllib.request.urlopen(url, timeout=10.0) as response:
            data = json.loads(response.read())

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
        with urllib.request.urlopen(url, timeout=10.0) as response:
            data = json.loads(response.read())
            print(f"DEBUG: Jellyfin search returned {data.get('TotalRecordCount', 0)} items")

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

            print(f"DEBUG: Returning {len(items)} items to frontend")
            return {"items": items}
    except Exception as e:
        print(f"DEBUG: Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


async def _get_stream_playlist(m: str, audio: Optional[int], subtitle: Optional[int], mode: str):
    """Common playlist logic for both VOD and live modes"""
    stream_key = f"{m}_{audio}_{subtitle}_{mode}"

    if stream_key not in active_streams:
        item_info = get_item_info(m)

        if audio is None or subtitle is None:
            auto_audio, auto_sub = find_best_streams(item_info)
            if audio is None:
                audio = auto_audio
            if subtitle is None:
                subtitle = auto_sub

        params = {
            'mediaSourceId': m,
            'api_key': settings.jellyfin_api_key,
            'DeviceId': 'jellyfin-proxy',
            'VideoCodec': 'h264',
            'AudioCodec': 'aac',
            'VideoBitrate': str(settings.video_bitrate),
            'AudioBitrate': str(settings.audio_bitrate),
            'MaxStreamingBitrate': str(settings.max_streaming_bitrate),
            'MaxWidth': str(settings.max_width),
            'MaxHeight': str(settings.max_height),
            'MaxFramerate': str(settings.max_framerate),
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

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

    jellyfin_url = active_streams[stream_key]['jellyfin_url']

    try:
        with urllib.request.urlopen(jellyfin_url, timeout=30.0) as response:
            content = response.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch playlist: {e}")

    # Store session info
    if content:
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
        rewritten = rewrite_playlist_live(content, m)

    return PlainTextResponse(
        rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/media.m3u8")
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

    stream_key = None
    for key, info in active_streams.items():
        if info['media_id'] == media_id and info.get('mode') == 'vod':
            stream_key = key
            break

    if not stream_key or 'session_id' not in active_streams[stream_key]:
        raise HTTPException(status_code=404, detail="Stream not found or not initialized")

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

    session_id = active_streams[stream_key]['session_id']
    hls_dir = active_streams[stream_key].get('hls_dir', 'hls1')

    # Extract segment filename for cache key (without query params)
    segment_file = segment_path.split('?')[0]
    cache_path = Path(settings.cache_dir) / stream_key / segment_file

    if cache_path.exists():
        return FileResponse(
            cache_path,
            media_type="video/mp2t",
            headers={"Access-Control-Allow-Origin": "*"}
        )

    # Build Jellyfin URL preserving query params
    query_string = request.url.query
    if query_string:
        segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment_path}?{query_string}"
    else:
        segment_url = f"{settings.jellyfin_url}/Videos/{media_id}/{hls_dir}/{session_id}/{segment_path}"

    # Use longer timeout for first segment (transcoding startup)
    segment_num = segment_file.split('.')[0]
    timeout = 120.0 if segment_num in ['0', '1', '2'] else 60.0

    fetch_and_cache(segment_url, cache_path, timeout=timeout)

    return FileResponse(
        cache_path,
        media_type="video/mp2t",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/live/{media_id}/{segment_file}")
async def get_live_segment(media_id: str, segment_file: str):
    """Get live segment - simple filename"""

    stream_key = None
    for key, info in active_streams.items():
        if info['media_id'] == media_id and info.get('mode') == 'live':
            stream_key = key
            break

    if not stream_key or 'session_id' not in active_streams[stream_key]:
        raise HTTPException(status_code=404, detail="Stream not found or not initialized")

    # Update last accessed time
    active_streams[stream_key]['last_accessed'] = time.time()

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

    fetch_and_cache(segment_url, cache_path, timeout=timeout)

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
        stream_info = {
            "stream_key": key,
            "media_id": info['media_id'],
            "audio": info.get('audio'),
            "subtitle": info.get('subtitle'),
            "mode": info.get('mode'),
            "idle_seconds": int(current_time - info.get('last_accessed', current_time)),
            "age_seconds": int(current_time - info.get('created_at', current_time)),
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

    return {"status": "deleted", "stream_key": stream_key}


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

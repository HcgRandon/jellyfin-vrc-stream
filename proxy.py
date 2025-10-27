#!/usr/bin/env python3
"""
Simple Jellyfin HLS Proxy for VRChat
- Streams once from Jellyfin
- Fans out to multiple clients
- Hides API key
"""

import re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic_settings import BaseSettings
import urllib.request
import json


class Settings(BaseSettings):
    jellyfin_url: str = "http://jellyfin:8096"
    jellyfin_api_key: str = ""
    cache_dir: str = "/tmp/hls-cache"

    class Config:
        env_file = ".env"


settings = Settings()
app = FastAPI(title="Jellyfin HLS Proxy")

# Ensure cache directory exists
Path(settings.cache_dir).mkdir(parents=True, exist_ok=True)

# Track active streams
active_streams = {}


def get_item_info(item_id: str):
    """Fetch item info from Jellyfin"""
    url = f"{settings.jellyfin_url}/Items/{item_id}?api_key={settings.jellyfin_api_key}"

    try:
        with urllib.request.urlopen(url, timeout=10.0) as response:
            return json.loads(response.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch item: {e}")


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


def fetch_and_cache(url: str, cache_path: Path) -> Path:
    """Fetch content from URL and cache it"""
    if cache_path.exists():
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with urllib.request.urlopen(url, timeout=30.0) as response:
            content = response.read()
            cache_path.write_bytes(content)
            return cache_path
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
        "active_streams": len(active_streams)
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
            'VideoBitrate': '2500000',
            'AudioBitrate': '128000',
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
        }

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

    fetch_and_cache(segment_url, cache_path)

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

    fetch_and_cache(segment_url, cache_path)

    return FileResponse(
        cache_path,
        media_type="video/mp2t",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/streams")
async def list_streams():
    """List active streams"""
    return {
        "streams": [
            {
                "media_id": info['media_id'],
                "audio": info.get('audio'),
                "subtitle": info.get('subtitle'),
            }
            for key, info in active_streams.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

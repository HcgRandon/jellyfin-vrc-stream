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
from fastapi import FastAPI, HTTPException, Query
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


def rewrite_playlist(content: bytes, media_id: str) -> str:
    """Rewrite playlist URLs to point to our proxy"""
    text = content.decode('utf-8')
    lines = text.split('\n')
    new_lines = []

    for line in lines:
        if line and not line.startswith('#'):
            match = re.search(r'hls\d*/[^/]+/([^/]+\.ts)', line)
            if match:
                segment_file = match.group(1)
                new_lines.append(f"/s/{media_id}/{segment_file}")
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


@app.get("/media.m3u8")
async def get_playlist(
    m: str = Query(..., description="Media ID"),
    audio: Optional[int] = Query(None, description="Audio stream index"),
    subtitle: Optional[int] = Query(None, description="Subtitle stream index"),
):
    """Get HLS playlist - proxy from Jellyfin and cache"""

    # Create stream key
    stream_key = f"{m}_{audio}_{subtitle}"

    # If not cached, fetch item info and determine streams
    if stream_key not in active_streams:
        item_info = get_item_info(m)

        # Auto-select streams if not specified
        if audio is None or subtitle is None:
            auto_audio, auto_sub = find_best_streams(item_info)
            if audio is None:
                audio = auto_audio
            if subtitle is None:
                subtitle = auto_sub

        # Build Jellyfin URL
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
        jellyfin_url = f"{settings.jellyfin_url}/Videos/{m}/main.m3u8?{param_str}"

        active_streams[stream_key] = {
            'jellyfin_url': jellyfin_url,
            'media_id': m,
            'audio': audio,
            'subtitle': subtitle,
        }

    # Fetch playlist from Jellyfin
    jellyfin_url = active_streams[stream_key]['jellyfin_url']
    cache_path = Path(settings.cache_dir) / stream_key / "playlist.m3u8"

    # Always fetch fresh playlist
    try:
        with urllib.request.urlopen(jellyfin_url, timeout=30.0) as response:
            content = response.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch playlist: {e}")

    # Rewrite playlist to point to our proxy
    rewritten = rewrite_playlist(content, m)

    # Store session info for segment fetching
    if content:
        match = re.search(rb'(hls\d*)/([^/]+)/', content)
        if match:
            hls_dir = match.group(1).decode('utf-8')
            session_id = match.group(2).decode('utf-8')
            active_streams[stream_key]['hls_dir'] = hls_dir
            active_streams[stream_key]['session_id'] = session_id

    return PlainTextResponse(
        rewritten,
        media_type="application/vnd.apple.mpegurl",
        headers={"Access-Control-Allow-Origin": "*"}
    )


@app.get("/s/{media_id}/{segment_file}")
async def get_segment(media_id: str, segment_file: str):
    """Get HLS segment - proxy from Jellyfin and cache"""

    stream_key = None
    for key, info in active_streams.items():
        if info['media_id'] == media_id:
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

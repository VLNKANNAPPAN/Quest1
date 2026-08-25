"""Audio acquisition, preprocessing, and acoustic fingerprinting."""

from .fingerprint import compute_audio_fingerprint
from .downloader import (
    get_video_id,
    get_light_metadata,
    download_audio_only,
    download_video,
    MediaManager,
)

__all__ = [
    "compute_audio_fingerprint",
    "get_video_id",
    "get_light_metadata",
    "download_audio_only",
    "download_video",
    "MediaManager",
]

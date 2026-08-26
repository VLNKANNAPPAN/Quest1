"""Tests for audio downloading, probing, and caching."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from video_dialogue.audio.downloader import (
    check_has_audio,
    download_audio_only_with_metadata,
    get_video_id,
)
from video_dialogue.core.models import VideoMetadata


def test_check_has_audio_valid_wav(synthetic_audio_path: Path):
    """Ground truth ffprobe check correctly detects audio in a valid WAV file."""
    assert check_has_audio(synthetic_audio_path) is True


def test_check_has_audio_missing_or_empty(tmp_path: Path):
    """Missing or empty files return False without throwing exceptions."""
    missing = tmp_path / "missing.wav"
    assert check_has_audio(missing) is False

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    assert check_has_audio(empty) is False


def test_download_audio_cached_skips_network(tmp_path: Path, synthetic_audio_path: Path):
    """When the audio is already cached on disk, no remote network calls are made."""
    cache_dir = tmp_path / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = "https://example.com/stream-with-muxed-hls"
    vid_id = get_video_id(url)
    cached_wav = cache_dir / f"{vid_id}.wav"
    cached_wav.write_bytes(synthetic_audio_path.read_bytes())

    with patch("video_dialogue.audio.downloader.yt_dlp.YoutubeDL") as mock_ydl:
        audio_path, meta = download_audio_only_with_metadata(
            url_or_path=url,
            output_dir=cache_dir,
        )

        mock_ydl.assert_not_called()
        assert audio_path == cached_wav
        assert meta.has_audio is True
        assert meta.duration > 0


def test_remote_download_overrides_unreliable_acodec(tmp_path: Path, synthetic_audio_path: Path):
    """Pre-download acodec='none' is overridden by ground-truth ffprobe on the extracted audio."""
    cache_dir = tmp_path / "audio"
    cache_dir.mkdir(parents=True, exist_ok=True)

    url = "https://ok.ru/video/mocked_hls"
    vid_id = get_video_id(url)

    def mock_download(*args, **kwargs):
        raw_file = cache_dir / f"{vid_id}_raw.mp4"
        raw_file.write_bytes(synthetic_audio_path.read_bytes())
        return {
            "duration": 3261.0,
            "vcodec": "h264",
            "acodec": "none",  # Muxed HLS bug scenario
            "fps": 25.0,
        }

    with patch("video_dialogue.audio.downloader.yt_dlp.YoutubeDL") as mock_ydl_class:
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = mock_download
        mock_ydl_class.return_value.__enter__.return_value = mock_ydl_instance

        audio_path, meta = download_audio_only_with_metadata(
            url_or_path=url,
            output_dir=cache_dir,
        )

        assert audio_path.exists()
        assert meta.has_audio is True
        assert meta.duration > 0

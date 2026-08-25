"""Pytest fixtures for video dialogue retrieval tests."""

import subprocess
from pathlib import Path
from typing import List, Dict, Any

import pytest

import sys
# Ensure src and root are in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from video_dialogue.config.settings import PipelineConfig
from video_dialogue.database.db import DatabaseManager
from video_dialogue.search.normalizer import normalize_text


@pytest.fixture
def temp_config(tmp_path: Path) -> PipelineConfig:
    """Provide a PipelineConfig initialized with temporary isolated directories."""
    config = PipelineConfig(cache_dir=tmp_path / "cache")
    config.ensure_directories()
    return config


@pytest.fixture
def temp_db(tmp_path: Path) -> DatabaseManager:
    """Provide a DatabaseManager backed by a temporary file."""
    db_path = tmp_path / "test_pipeline.db"
    return DatabaseManager(db_path)


@pytest.fixture
def sample_transcript_words() -> List[str]:
    """Sample normalized words for search engine testing."""
    text = (
        "my at my mind at my mind the weather today is fine the cat sat on the mat "
        "my mind rebels at stagnation the dog ran fast my mind is at home"
    )
    return normalize_text(text).split()


@pytest.fixture
def synthetic_transcript() -> List[Dict[str, Any]]:
    """Synthetic timestamped word transcript."""
    words = [
        "the", "weather", "today", "is", "fine", "the", "cat", "sat", "on", "the", "mat",
        "my", "mind", "rebels", "at", "stagnation", "the", "dog", "ran", "fast"
    ]
    return [
        {"word": w, "start": float(i), "end": float(i) + 0.85}
        for i, w in enumerate(words)
    ]


@pytest.fixture
def synthetic_audio_path(tmp_path: Path) -> Path:
    """Generate a 3-second 440Hz test sine tone WAV file using FFmpeg."""
    audio_path = tmp_path / "test_tone_440.wav"
    cmd = [
        "ffmpeg", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=3",
        "-ar", "16000", "-ac", "1",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


@pytest.fixture
def synthetic_video_path(tmp_path: Path) -> Path:
    """Generate a 25-second test MP4 video with color bars and audio tone matching synthetic transcript duration."""
    video_path = tmp_path / "test_video.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=25:size=320x240:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return video_path

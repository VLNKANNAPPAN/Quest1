"""Tests for SQLite database storage and fingerprint deduplication."""

import pytest
from video_dialogue.database.db import DatabaseManager
from video_dialogue.core.models import VideoMetadata


def test_insert_and_get_video(temp_db: DatabaseManager):
    meta = VideoMetadata(duration=120.5, fps=29.97, width=1920, height=1080, video_codec="h264", audio_codec="aac")
    temp_db.insert_video(
        video_id="vid_001",
        url="https://example.com/video1.mp4",
        fingerprint="fp_abc123",
        metadata=meta,
    )

    # Lookup by ID
    rec = temp_db.get_video_by_id("vid_001")
    assert rec is not None
    assert rec.video_id == "vid_001"
    assert rec.url == "https://example.com/video1.mp4"
    assert rec.audio_fingerprint == "fp_abc123"
    assert rec.duration == 120.5
    assert rec.fps == 29.97
    assert rec.width == 1920
    assert rec.height == 1080
    assert rec.has_audio is True

    # Lookup by fingerprint
    fp_rec = temp_db.get_video_by_fingerprint("fp_abc123")
    assert fp_rec is not None
    assert fp_rec.video_id == "vid_001"


def test_fingerprint_deduplication(temp_db: DatabaseManager):
    meta = VideoMetadata(duration=50.0, fps=25.0)
    temp_db.insert_video("orig_id", "https://siteA.com/clip.mp4", "unique_fingerprint_999", meta)

    # Look up by fingerprint returns original record
    found = temp_db.get_video_by_fingerprint("unique_fingerprint_999")
    assert found is not None
    assert found.video_id == "orig_id"
    assert found.url == "https://siteA.com/clip.mp4"


def test_transcript_caching(temp_db: DatabaseManager):
    meta = VideoMetadata(duration=30.0, fps=25.0)
    temp_db.insert_video("vid_100", "local://test.mp4", "fp_100", meta)

    sample_transcript = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.6, "end": 1.0},
    ]

    # No transcript initially
    assert temp_db.get_transcript("vid_100", "small") is None

    # Insert transcript for model "small"
    temp_db.insert_transcript("vid_100", "small", sample_transcript)
    cached_small = temp_db.get_transcript("vid_100", "small")
    assert cached_small is not None
    assert len(cached_small) == 2
    assert cached_small[0]["word"] == "hello"

    # Model size separation: "medium" is still empty
    assert temp_db.get_transcript("vid_100", "medium") is None


def test_list_and_clear_videos(temp_db: DatabaseManager):
    meta = VideoMetadata(duration=10.0, fps=24.0)
    temp_db.insert_video("v1", "url1", "fp1", meta)
    temp_db.insert_video("v2", "url2", "fp2", meta)

    videos = temp_db.list_videos()
    assert len(videos) == 2

    temp_db.clear_all()
    assert len(temp_db.list_videos()) == 0

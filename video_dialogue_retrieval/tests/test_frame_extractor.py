"""Tests for timestamp-to-frame calculation and FFmpeg frame extraction."""

from pathlib import Path
from PIL import Image

from video_dialogue.video.frame_extractor import timestamp_to_frame, extract_frame


def test_timestamp_to_frame():
    assert timestamp_to_frame(0.0, 25.0) == 0
    assert timestamp_to_frame(1.0, 25.0) == 25
    assert timestamp_to_frame(1.04, 25.0) == 26
    assert timestamp_to_frame(10.0, 29.97) == 300


def test_extract_frame_from_video(synthetic_video_path: Path, tmp_path: Path):
    out_frame = tmp_path / "extracted_frame_1s.jpg"
    extracted_path = extract_frame(synthetic_video_path, timestamp=1.0, output_path=out_frame)

    assert extracted_path.exists()
    assert extracted_path.stat().st_size > 0

    # Verify Pillow can open the extracted JPEG
    with Image.open(extracted_path) as img:
        assert img.format == "JPEG"
        assert img.width == 320
        assert img.height == 240

"""Frame calculation and FFmpeg-based high-quality frame extraction."""

import logging
import subprocess
from pathlib import Path
from typing import Union

from ..media_tools import ffmpeg_executable

logger = logging.getLogger(__name__)


def timestamp_to_frame(timestamp: float, fps: float) -> int:
    """Calculate discrete frame number for a timestamp given video frame rate (FPS).

    Args:
        timestamp: Time in seconds.
        fps: Frames per second.

    Returns:
        Frame integer index (rounded).
    """
    if fps <= 0:
        fps = 25.0
    return int(round(timestamp * fps))


def extract_frame(
    video_path: Union[str, Path],
    timestamp: float,
    output_path: Union[str, Path],
) -> Path:
    """Extract a single high-quality JPEG frame at the specified timestamp using FFmpeg.

    Uses input seeking before -i for fast positioning and high quality -q:v 2.

    Args:
        video_path: Path to the input video container.
        timestamp: Seek position in seconds.
        output_path: Destination path for the saved JPEG image.

    Returns:
        Path to the saved frame image file.
    """
    vpath = Path(video_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    cmd = [
        ffmpeg_executable(),
        "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", str(vpath),
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Frame extraction failed to produce valid image at {out_path}")

    return out_path

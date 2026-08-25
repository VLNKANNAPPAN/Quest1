"""Locate the FFmpeg executable used by media-processing stages."""

from __future__ import annotations

import shutil


def ffmpeg_executable() -> str:
    """Return a system FFmpeg path or the bundled imageio-ffmpeg binary.

    ``imageio-ffmpeg`` ships a platform-specific FFmpeg executable, so a fresh
    Python install can run remote downloads, audio conversion and frame
    extraction without asking the user to install FFmpeg separately.
    """
    return shutil.which("ffmpeg") or _bundled_ffmpeg()


def _bundled_ffmpeg() -> str:
    try:
        import imageio_ffmpeg
    except ImportError as exc:  # pragma: no cover - package dependency guards this
        raise RuntimeError("FFmpeg is unavailable; install imageio-ffmpeg or add ffmpeg to PATH.") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()

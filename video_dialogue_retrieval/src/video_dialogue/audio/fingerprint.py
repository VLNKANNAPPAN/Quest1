"""Acoustic fingerprinting using Chromaprint with AcoustID and FFmpeg fallbacks."""

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import Union

from ..media_tools import ffmpeg_executable

logger = logging.getLogger(__name__)

try:
    import acoustid
    ACOUSTID_AVAILABLE = True
except ImportError:
    ACOUSTID_AVAILABLE = False


def _fingerprint_via_ffmpeg(audio_path: Path) -> str:
    """Generate Chromaprint fingerprint using FFmpeg's built-in chromaprint muxer."""
    cmd = [
        ffmpeg_executable(),
        "-v", "error",
        "-i", str(audio_path),
        "-f", "chromaprint",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    fingerprint = proc.stdout.strip()
    if not fingerprint:
        raise RuntimeError(f"FFmpeg chromaprint returned empty fingerprint for {audio_path}")
    return fingerprint


def _fingerprint_via_sha256(audio_path: Path) -> str:
    """Fallback hash fingerprint when chromaprint is entirely unavailable."""
    hasher = hashlib.sha256()
    with open(audio_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"sha256_{hasher.hexdigest()[:32]}"


def compute_audio_fingerprint(audio_path: Union[str, Path]) -> str:
    """Compute an acoustic fingerprint for an audio file.

    Tries pyacoustid first (calling fpcalc). If fpcalc is missing or fails,
    seamlessly falls back to FFmpeg's built-in chromaprint muxer.

    Args:
        audio_path: Path to the WAV/audio file.

    Returns:
        String representation of the audio fingerprint.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # 1. Try acoustid / fpcalc if available
    if ACOUSTID_AVAILABLE:
        try:
            _duration, fp = acoustid.fingerprint_file(str(path))
            if isinstance(fp, bytes):
                return fp.decode("utf-8", errors="ignore")
            if fp:
                return str(fp)
        except Exception as exc:
            logger.debug("acoustid.fingerprint_file failed (%s); trying ffmpeg chromaprint fallback", exc)

    # 2. Fallback to FFmpeg chromaprint muxer
    try:
        return _fingerprint_via_ffmpeg(path)
    except Exception as exc:
        logger.warning("FFmpeg chromaprint extraction failed (%s); falling back to SHA256", exc)

    # 3. Last-resort fallback to audio content hash
    return _fingerprint_via_sha256(path)

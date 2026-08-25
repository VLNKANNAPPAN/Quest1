"""Small, dependency-free terminal progress reporting helpers."""

from __future__ import annotations

import sys
import time
from typing import TextIO


class TerminalProgress:
    """Print concise pipeline status updates suitable for an ordinary terminal."""

    def __init__(self, enabled: bool = True, stream: TextIO | None = None) -> None:
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self._last_download_percent = -5.0
        self._last_asr_seconds = -15.0
        self._started_at = time.monotonic()
        # Audio is the first and most common acquisition stage; callers can
        # switch this to "video" before lazy full-video retrieval.
        self._download_label = "audio"

    def stage(self, message: str) -> None:
        if self.enabled:
            print(f"[PROGRESS] {message}", file=self.stream, flush=True)

    def workflow_stage(self, number: int, total: int, message: str) -> None:
        """Report a named pipeline stage and its overall workflow position."""
        percent = max(0, min(100, round((number - 1) * 100 / total)))
        self.stage(f"Overall {percent:3d}% | Step {number}/{total}: {message}")

    def set_download_label(self, label: str) -> None:
        """Set the label used by the next yt-dlp progress-hook invocation."""
        self._download_label = label
        self._last_download_percent = -5.0

    def elapsed_seconds(self) -> float:
        """Return the elapsed time since this reporter was created."""
        return time.monotonic() - self._started_at

    def download(self, data: dict) -> None:
        """yt-dlp progress-hook callback; throttle updates to five-percent steps."""
        status = data.get("status")
        if status == "finished":
            self.stage(f"Download complete ({self._download_label}); preparing next step.")
            return
        if status != "downloading":
            return

        downloaded = data.get("downloaded_bytes") or 0
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        if not total:
            return
        percent = downloaded * 100 / total
        if percent < 100 and percent - self._last_download_percent < 5:
            return
        self._last_download_percent = percent
        speed = data.get("speed")
        eta = data.get("eta")
        details = [f"{percent:5.1f}%"]
        if speed:
            details.append(f"{speed / 1024 / 1024:.1f} MiB/s")
        if eta is not None:
            details.append(f"ETA {eta}s")
        self.stage(f"Downloading {self._download_label}: " + " | ".join(details))

    def asr_segment(self, end_seconds: float, segment_count: int, duration: float | None) -> None:
        """Report inference advancement while faster-whisper yields segments."""
        if end_seconds - self._last_asr_seconds < 15 and duration is not None and end_seconds < duration:
            return
        self._last_asr_seconds = end_seconds
        if duration and duration > 0:
            self.stage(
                f"ASR: {min(end_seconds / duration * 100, 100):5.1f}% "
                f"({end_seconds:.1f}s / {duration:.1f}s; {segment_count} segments)"
            )
        else:
            self.stage(f"ASR: processed {end_seconds:.1f}s ({segment_count} segments)")

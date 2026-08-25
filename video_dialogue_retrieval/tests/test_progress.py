"""Tests for visible but throttled terminal progress reporting."""

from io import StringIO

from video_dialogue.progress import TerminalProgress


def test_download_progress_is_throttled_and_completes():
    stream = StringIO()
    progress = TerminalProgress(stream=stream)

    progress.download({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 100})
    progress.download({"status": "downloading", "downloaded_bytes": 3, "total_bytes": 100})
    progress.download({"status": "downloading", "downloaded_bytes": 10, "total_bytes": 100})
    progress.download({"status": "finished"})

    output = stream.getvalue()
    assert output.count("Downloading audio") == 2
    assert "Download complete" in output


def test_asr_progress_includes_percentage():
    stream = StringIO()
    progress = TerminalProgress(stream=stream)

    progress.asr_segment(16.0, 2, 80.0)

    assert "ASR:  20.0%" in stream.getvalue()

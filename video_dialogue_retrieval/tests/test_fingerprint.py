"""Tests for acoustic fingerprinting determinism and discrimination."""

import subprocess
from pathlib import Path
import pytest

from video_dialogue.audio.fingerprint import compute_audio_fingerprint


def test_fingerprint_deterministic(synthetic_audio_path: Path):
    fp1 = compute_audio_fingerprint(synthetic_audio_path)
    fp2 = compute_audio_fingerprint(synthetic_audio_path)

    assert fp1, "Fingerprint must not be empty"
    assert fp1 == fp2, "Fingerprint must be deterministic for identical audio file"


def test_fingerprint_content_sensitivity(tmp_path: Path):
    audio1 = tmp_path / "tone_300.wav"
    audio2 = tmp_path / "tone_880.wav"

    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=3", "-ar", "16000", "-ac", "1", str(audio1)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=880:duration=3", "-ar", "16000", "-ac", "1", str(audio2)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    fp1 = compute_audio_fingerprint(audio1)
    fp2 = compute_audio_fingerprint(audio2)

    assert fp1 != fp2, "Different audio frequencies must produce different fingerprints"


def test_fingerprint_missing_file():
    with pytest.raises(FileNotFoundError):
        compute_audio_fingerprint("non_existent_file_xyz123.wav")

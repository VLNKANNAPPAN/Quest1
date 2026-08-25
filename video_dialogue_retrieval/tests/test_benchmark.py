"""Tests for search and model benchmark suites."""

from unittest.mock import MagicMock
import pandas as pd

from video_dialogue.benchmark.benchmark import (
    run_variant_benchmark,
    compare_model_sizes,
    DEFAULT_SEARCH_VARIANTS,
)


def test_variant_benchmark_synthetic(synthetic_transcript):
    known_dialogues = [("My mind rebels at stagnation", 11.0, 2.0)]
    df = run_variant_benchmark(
        synthetic_transcript,
        known_dialogues,
        variants=DEFAULT_SEARCH_VARIANTS,
        runs=2,
    )

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "top1_accuracy" in df.columns
    assert "mean_time_ms" in df.columns
    assert (df["top1_accuracy"] == 1.0).all()


def test_compare_model_sizes_mocked(temp_db, tmp_path, synthetic_transcript):
    mock_transcriber = MagicMock()
    mock_transcriber.transcribe.return_value = synthetic_transcript

    fake_audio = tmp_path / "fake.wav"
    fake_audio.touch()

    df = compare_model_sizes(
        db_manager=temp_db,
        video_id="test_vid",
        audio_path=fake_audio,
        model_sizes=("tiny", "base"),
        known_dialogues=[("My mind rebels at stagnation", 11.0, 2.0)],
        force=True,
        transcriber=mock_transcriber,
    )

    assert len(df) == 2
    assert "model_size" in df.columns
    assert "transcribe_time_s" in df.columns
    assert df["word_count"].iloc[0] == len(synthetic_transcript)

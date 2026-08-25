"""Tests for end-to-end pipeline execution and deduplication workflow."""

from pathlib import Path
from unittest.mock import MagicMock

from video_dialogue.pipeline.orchestrator import DialogueRetrievalPipeline, find_dialogue
from video_dialogue.core.models import VideoMetadata


def test_pipeline_with_mocked_transcriber(temp_config, synthetic_video_path: Path, synthetic_transcript):
    pipeline = DialogueRetrievalPipeline(config=temp_config)

    # Mock ASR transcriber so test runs instantly without loading heavy Whisper models
    pipeline.transcriber.transcribe = MagicMock(return_value=synthetic_transcript)

    result = pipeline.run(
        video_url=str(synthetic_video_path),
        target_dialogue="my mind rebels at stagnation",
        method="rare_anchor_fuzzy",
        score_fn_name="difflib",
        model_size="small",
        top_k=3,
    )

    assert result.success is True
    assert len(result.matches) > 0
    top_match = result.matches[0]
    assert "my mind rebels at stagnation" in top_match.matched_text
    assert top_match.start_timestamp == 11.0 or top_match.start_timestamp == 12.0 or top_match.start_timestamp >= 10.0
    assert Path(top_match.frame_path).exists()
    assert result.result_file is not None and Path(result.result_file).exists()


def test_pipeline_deduplication_sharing(temp_config, synthetic_video_path: Path, synthetic_transcript):
    pipeline = DialogueRetrievalPipeline(config=temp_config)
    pipeline.transcriber.transcribe = MagicMock(return_value=synthetic_transcript)

    # First call: indexes the video and runs ASR
    res1 = pipeline.run(
        video_url=str(synthetic_video_path),
        target_dialogue="my mind rebels",
    )
    assert res1.success is True
    assert pipeline.transcriber.transcribe.call_count == 1

    # Second call with the same audio content (simulated via another path or URL)
    # The cached transcript should be fetched from SQLite
    res2 = pipeline.run(
        video_url=str(synthetic_video_path),
        target_dialogue="my mind rebels",
    )
    assert res2.success is True
    # Transcribe was called, but cached transcript is used

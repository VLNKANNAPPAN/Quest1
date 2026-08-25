"""End-to-end pipeline orchestrator for audio-first, fingerprint-deduplicated dialogue retrieval."""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from ..config.settings import PipelineConfig, get_default_config
from ..core.models import VideoRecord, DialogueMatch, PipelineResult
from ..database.db import DatabaseManager
from ..audio.downloader import MediaManager, get_video_id
from ..audio.fingerprint import compute_audio_fingerprint
from ..asr.transcriber import WhisperTranscriber
from ..search.normalizer import normalize_text, build_word_index
from ..search.index import InvertedIndex
from ..search.engine import search_dialogue
from ..video.frame_extractor import timestamp_to_frame, extract_frame
from ..progress import TerminalProgress

logger = logging.getLogger(__name__)


class DialogueRetrievalPipeline:
    """Orchestrates end-to-end dialogue retrieval and frame localization."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        db_manager: Optional[DatabaseManager] = None,
        media_manager: Optional[MediaManager] = None,
        transcriber: Optional[WhisperTranscriber] = None,
    ):
        self.config = config or get_default_config()
        self.config.ensure_directories()

        self.db = db_manager or DatabaseManager(self.config.db_path)
        self.media = media_manager or MediaManager(self.config)
        self.transcriber = transcriber or WhisperTranscriber(db_manager=self.db)

    def get_or_create_video_record(self, url_or_path: str, progress: Optional[TerminalProgress] = None) -> VideoRecord:
        """Audio-first acquisition: downloads audio and computes fingerprint before full video.

        If the fingerprint has already been indexed in SQLite under a previous URL or filename,
        the earlier video_id is reused so cached transcripts are retrieved.
        """
        video_id = get_video_id(url_or_path)
        if progress:
            progress.workflow_stage(1, 5, "Acquiring the audio track.")
            progress.set_download_label("audio")
        audio_path = self.media.get_audio(url_or_path, progress_hook=progress.download if progress else None)
        if progress:
            progress.workflow_stage(2, 5, "Computing audio fingerprint.")
        fingerprint = compute_audio_fingerprint(audio_path)

        existing = self.db.get_video_by_fingerprint(fingerprint)
        if existing is not None:
            if existing.video_id != video_id:
                logger.info(
                    "Fingerprint match! Content was previously registered as '%s' (video_id=%s). "
                    "Reusing cached video_id for transcript sharing.",
                    existing.url,
                    existing.video_id,
                )
                light_meta = self.media.get_metadata(url_or_path)
                existing.fps = light_meta.fps or existing.fps
                existing.duration = light_meta.duration or existing.duration
                existing.current_url = url_or_path
                return existing
            else:
                existing.current_url = url_or_path
                return existing

        # First time seeing this audio fingerprint
        light_meta = self.media.get_metadata(url_or_path)
        self.db.insert_video(video_id, url_or_path, fingerprint, light_meta)
        record = self.db.get_video_by_fingerprint(fingerprint)
        if record is None:
            record = VideoRecord(
                video_id=video_id,
                url=url_or_path,
                audio_fingerprint=fingerprint,
                duration=light_meta.duration,
                fps=light_meta.fps,
                width=light_meta.width,
                height=light_meta.height,
                video_codec=light_meta.video_codec,
                audio_codec=light_meta.audio_codec,
                has_audio=light_meta.has_audio,
                current_url=url_or_path,
            )
        else:
            record.current_url = url_or_path
        return record

    def run(
        self,
        video_url: str,
        target_dialogue: str,
        method: Optional[str] = None,
        score_fn_name: Optional[str] = None,
        model_size: Optional[str] = None,
        top_k: Optional[int] = None,
        use_inverted_index: bool = True,
        force_asr: bool = False,
        progress: Optional[TerminalProgress] = None,
    ) -> PipelineResult:
        """Run the complete retrieval workflow for a video and dialogue query."""
        method = method or self.config.default_search_method
        score_fn_name = score_fn_name or self.config.default_score_fn
        model_size = model_size or self.config.default_model_size
        top_k = top_k or self.config.default_top_k

        # 1. Audio-first acquisition & fingerprint dedup
        record = self.get_or_create_video_record(video_url, progress=progress)
        video_id = record.video_id
        fps = record.fps or 25.0

        if not record.has_audio:
            return PipelineResult(
                success=False,
                video={"url": video_url, "duration_seconds": record.duration, "fps": fps},
                query={"dialogue": target_dialogue},
                message="Video has no audio track.",
            )

        # 2. Speech-to-Text with DB transcript cache
        audio_path = self.config.audio_dir / f"{video_id}.wav"
        if not audio_path.exists():
            audio_path = self.media.get_audio(video_url, progress_hook=progress.download if progress else None)

        if progress:
            progress.workflow_stage(3, 5, f"Starting ASR with the {model_size!r} Whisper model.")
        transcript = self.transcriber.transcribe(
            video_id=video_id,
            audio_path=audio_path,
            model_size=model_size,
            force=force_asr,
            beam_size=self.config.whisper_beam_size,
            vad_filter=self.config.whisper_vad_filter,
            min_silence_duration_ms=self.config.whisper_min_silence_duration_ms,
            progress_callback=progress.asr_segment if progress else None,
            audio_duration=record.duration,
        )

        if not transcript:
            return PipelineResult(
                success=False,
                video={"url": video_url, "duration_seconds": record.duration, "fps": fps},
                query={"dialogue": target_dialogue},
                message="No speech detected in audio track.",
            )

        # 3. Token normalization & search
        if progress:
            progress.workflow_stage(4, 5, f"Searching {len(transcript)} transcribed words.")
        transcript_words = build_word_index(transcript)
        target_words = normalize_text(target_dialogue).split()
        if not target_words:
            raise ValueError("Target dialogue cannot be empty.")

        inverted_index = (
            InvertedIndex.from_words(transcript_words) if use_inverted_index else None
        )

        raw_results = search_dialogue(
            transcript_words=transcript_words,
            target_words=target_words,
            method=method,
            score_fn_name=score_fn_name,
            inverted_index=inverted_index,
            length_tolerance=self.config.fuzzy_length_tolerance,
            extra_context=self.config.fuzzy_extra_context,
        )

        if not raw_results:
            return PipelineResult(
                success=False,
                video={"url": video_url, "duration_seconds": record.duration, "fps": fps},
                query={
                    "dialogue": target_dialogue,
                    "normalized": " ".join(target_words),
                    "method": method,
                    "score_fn": score_fn_name,
                    "model_size": model_size,
                },
                message="Dialogue was not found.",
            )

        # 4. Lazy video download (only after confirmed match) & frame extraction
        if progress:
            progress.workflow_stage(5, 5, "Match found; acquiring video for frame extraction.")
            progress.set_download_label("video")
        video_path = self.media.get_video(video_url, progress_hook=progress.download if progress else None)

        matches: List[DialogueMatch] = []
        for rank, result in enumerate(raw_results[:top_k], start=1):
            start_idx, end_idx = result.start_index, result.end_index
            start_time = float(transcript[start_idx]["start"])
            end_time = float(transcript[end_idx - 1]["end"])
            start_frame = timestamp_to_frame(start_time, fps)
            end_frame = timestamp_to_frame(end_time, fps)

            frame_path = self.config.frame_dir / f"{video_id}_frame_{start_frame}.jpg"
            if not frame_path.exists() and video_path.exists():
                try:
                    extract_frame(video_path, start_time, frame_path)
                except Exception as exc:
                    logger.warning("Frame extraction error: %s", exc)

            matched_text = " ".join(transcript[i]["word"] for i in range(start_idx, end_idx))
            matches.append(
                DialogueMatch(
                    rank=rank,
                    matched_text=matched_text,
                    start_timestamp=start_time,
                    end_timestamp=end_time,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    score=float(result.score),
                    frame_path=str(frame_path),
                    anchor=result.anchor,
                )
            )

        final_result = PipelineResult(
            success=True,
            video={"url": video_url, "duration_seconds": record.duration, "fps": fps},
            query={
                "dialogue": target_dialogue,
                "normalized": " ".join(target_words),
                "method": method,
                "score_fn": score_fn_name,
                "model_size": model_size,
            },
            matches=matches,
        )

        result_path = self.config.result_dir / f"{video_id}_{model_size}_result.json"
        result_path.write_text(json.dumps(final_result.to_dict(), indent=2))
        final_result.result_file = str(result_path)

        if progress:
            elapsed = progress.elapsed_seconds()
            progress.stage(f"Overall 100% | Finished in {elapsed:.1f}s; saved report to {result_path}.")

        return final_result


def find_dialogue(
    video_url: str,
    target_dialogue: str,
    method: str = "rare_anchor_fuzzy",
    score_fn_name: str = "difflib",
    model_size: str = "small",
    top_k: int = 5,
    use_inverted_index: bool = True,
    config: Optional[PipelineConfig] = None,
    progress: Optional[TerminalProgress] = None,
) -> Dict[str, Any]:
    """Convenience function providing a drop-in API matching the v3 notebook."""
    pipeline = DialogueRetrievalPipeline(config=config)
    result = pipeline.run(
        video_url=video_url,
        target_dialogue=target_dialogue,
        method=method,
        score_fn_name=score_fn_name,
        model_size=model_size,
        top_k=top_k,
        use_inverted_index=use_inverted_index,
        progress=progress,
    )
    return result.to_dict()

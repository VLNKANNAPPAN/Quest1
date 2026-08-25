"""Whisper ASR transcriber with word-level alignment, VAD filtering, and DB caching."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable

from faster_whisper import WhisperModel

from ..core.models import WordTimestamp
from ..database.db import DatabaseManager

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Manages Whisper models and performs word-level speech transcription with DB caching."""

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ):
        self.db_manager = db_manager
        self.device = device or self._default_device()
        if compute_type:
            self.compute_type = compute_type
        else:
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.models_dir = Path(models_dir or "cache/models").resolve()
        self._model_cache: Dict[str, WhisperModel] = {}

    @staticmethod
    def _default_device() -> str:
        """Use CUDA when PyTorch is present and sees a GPU; otherwise use CPU."""
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _resolve_model_dir(self, model_size_or_path: str) -> str:
        """Ensure model is downloaded to local directory without Windows symlink issues."""
        p = Path(model_size_or_path)
        if p.exists() and p.is_dir():
            return str(p)

        # Standard Systran model name mapping
        target_dir = self.models_dir / model_size_or_path
        if (target_dir / "model.bin").exists() or (target_dir / "model.safetensors").exists():
            return str(target_dir)

        repo_id = f"Systran/faster-whisper-{model_size_or_path}"
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading Whisper model '%s' to %s...", repo_id, target_dir)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id, local_dir=str(target_dir))
            return str(target_dir)
        except Exception as exc:
            logger.warning("Local snapshot download failed (%s); trying direct model_size", exc)
            return model_size_or_path

    def get_model(self, model_size: str = "small") -> WhisperModel:
        """Retrieve a cached or newly instantiated WhisperModel."""
        key = f"{model_size}:{self.device}:{self.compute_type}"
        if key not in self._model_cache:
            resolved_path = self._resolve_model_dir(model_size)
            logger.info(
                "Loading Whisper model '%s' (from %s) on %s (%s)...",
                model_size,
                resolved_path,
                self.device,
                self.compute_type,
            )
            self._model_cache[key] = WhisperModel(
                resolved_path, device=self.device, compute_type=self.compute_type
            )
        return self._model_cache[key]

    def transcribe(
        self,
        video_id: str,
        audio_path: Union[str, Path],
        model_size: str = "small",
        force: bool = False,
        beam_size: int = 5,
        vad_filter: bool = True,
        min_silence_duration_ms: int = 300,
        progress_callback: Optional[Callable[[float, int, Optional[float]], None]] = None,
        audio_duration: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Transcribe an audio file into a list of word timestamp dictionaries.

        If a cached transcript exists in SQLite and force is False, the cached
        transcript is returned immediately without running ASR inference.
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        # Check DB cache
        if not force and self.db_manager is not None:
            cached = self.db_manager.get_transcript(video_id, model_size)
            if cached is not None:
                logger.info("Found cached transcript for video %s (%s)", video_id, model_size)
                return cached

        model = self.get_model(model_size)
        logger.info("Transcribing %s with model '%s'...", path.name, model_size)
        segments, _info = model.transcribe(
            str(path),
            beam_size=beam_size,
            word_timestamps=True,
            vad_filter=vad_filter,
            vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
        )

        transcript: List[Dict[str, Any]] = []
        segment_count = 0
        for segment in segments:
            segment_count += 1
            if progress_callback:
                progress_callback(float(segment.end), segment_count, audio_duration)
            if not segment.words:
                continue
            for word in segment.words:
                transcript.append({
                    "word": word.word.strip(),
                    "start": float(word.start),
                    "end": float(word.end),
                })

        transcript.sort(key=lambda w: w["start"])

        # Save to DB cache
        if self.db_manager is not None:
            self.db_manager.insert_transcript(video_id, model_size, transcript)

        return transcript

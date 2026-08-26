"""Pipeline configuration and default settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PipelineConfig:
    """Central configuration for the Video Dialogue Retrieval pipeline."""

    # Base directories
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    video_dir: Optional[Path] = None
    audio_dir: Optional[Path] = None
    frame_dir: Optional[Path] = None
    result_dir: Optional[Path] = None
    db_path: Optional[Path] = None

    # Audio & ASR parameters
    sample_rate: int = 16000
    default_model_size: str = "small"
    whisper_beam_size: int = 2
    whisper_vad_filter: bool = True
    whisper_min_silence_duration_ms: int = 300

    # Download & cache safety
    audio_max_bitrate_kbps: int = 96
    video_max_height: int = 480
    concurrent_fragment_downloads: int = 5
    dedup_duration_tolerance_seconds: float = 2.0

    # Search & matching parameters
    fuzzy_length_tolerance: int = 2
    fuzzy_extra_context: int = 2
    default_search_method: str = "rare_anchor_fuzzy"
    default_score_fn: str = "difflib"
    default_top_k: int = 5

    def __post_init__(self):
        """Derive sub-directory paths if not explicitly specified and create them."""
        self.cache_dir = Path(self.cache_dir).resolve()
        if self.video_dir is None:
            self.video_dir = self.cache_dir / "videos"
        else:
            self.video_dir = Path(self.video_dir).resolve()

        if self.audio_dir is None:
            self.audio_dir = self.cache_dir / "audio"
        else:
            self.audio_dir = Path(self.audio_dir).resolve()

        if self.frame_dir is None:
            self.frame_dir = self.cache_dir / "frames"
        else:
            self.frame_dir = Path(self.frame_dir).resolve()

        if self.result_dir is None:
            self.result_dir = self.cache_dir / "results"
        else:
            self.result_dir = Path(self.result_dir).resolve()

        if self.db_path is None:
            self.db_path = self.cache_dir / "pipeline.db"
        else:
            self.db_path = Path(self.db_path).resolve()

    def ensure_directories(self) -> None:
        """Create all required cache directories on disk."""
        for directory in [self.cache_dir, self.video_dir, self.audio_dir, self.frame_dir, self.result_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        if self.db_path and self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)


def get_default_config(cache_dir: Optional[Path] = None) -> PipelineConfig:
    """Return a PipelineConfig instance initialized with default or custom cache directory."""
    if cache_dir:
        config = PipelineConfig(cache_dir=Path(cache_dir))
    else:
        config = PipelineConfig()
    config.ensure_directories()
    return config

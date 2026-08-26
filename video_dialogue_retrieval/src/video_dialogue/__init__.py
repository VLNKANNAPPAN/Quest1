"""Video Dialogue Retrieval package.

Audio-first, fingerprint-deduplicated video dialogue retrieval and frame localization.
"""

from .config.settings import PipelineConfig, get_default_config
from .core.models import (
    VideoMetadata,
    VideoRecord,
    WordTimestamp,
    DialogueMatch,
    PipelineResult,
)
from .database.db import DatabaseManager
from .audio.fingerprint import compute_audio_fingerprint
from .audio.downloader import MediaManager, get_video_id, get_light_metadata, check_has_audio
from .asr.transcriber import WhisperTranscriber
from .search.normalizer import normalize_text, build_word_index
from .search.scorers import get_score_fn, difflib_score, rapidfuzz_score, embedding_score
from .search.index import InvertedIndex, choose_anchor, build_rarity_fn
from .search.engine import (
    exact_phrase_search,
    fuzzy_sliding_window_search,
    rare_anchor_fuzzy_search,
    search_dialogue,
)
from .video.frame_extractor import timestamp_to_frame, extract_frame
from .pipeline.orchestrator import DialogueRetrievalPipeline, find_dialogue
from .benchmark.benchmark import (
    run_variant_benchmark,
    compare_model_sizes,
    DEFAULT_SEARCH_VARIANTS,
)

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "PipelineConfig",
    "get_default_config",
    "VideoMetadata",
    "VideoRecord",
    "WordTimestamp",
    "DialogueMatch",
    "PipelineResult",
    "DatabaseManager",
    "compute_audio_fingerprint",
    "MediaManager",
    "get_video_id",
    "get_light_metadata",
    "check_has_audio",
    "WhisperTranscriber",
    "normalize_text",
    "build_word_index",
    "get_score_fn",
    "difflib_score",
    "rapidfuzz_score",
    "embedding_score",
    "InvertedIndex",
    "choose_anchor",
    "build_rarity_fn",
    "exact_phrase_search",
    "fuzzy_sliding_window_search",
    "rare_anchor_fuzzy_search",
    "search_dialogue",
    "timestamp_to_frame",
    "extract_frame",
    "DialogueRetrievalPipeline",
    "find_dialogue",
    "run_variant_benchmark",
    "compare_model_sizes",
    "DEFAULT_SEARCH_VARIANTS",
]

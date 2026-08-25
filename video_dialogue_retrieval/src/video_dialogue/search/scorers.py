"""Pluggable text similarity scorers for dialogue matching."""

import logging
from difflib import SequenceMatcher
from typing import List, Callable, Dict

from rapidfuzz import fuzz as rf_fuzz

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _EMBED_MODEL = None

    def _get_embed_model():
        global _EMBED_MODEL
        if _EMBED_MODEL is None:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _EMBED_MODEL

    def embedding_score(a_words: List[str], b_words: List[str]) -> float:
        """Compute cosine similarity between sentence embeddings of two word sequences."""
        model = _get_embed_model()
        emb = model.encode([" ".join(a_words), " ".join(b_words)], convert_to_tensor=True)
        return float(st_util.cos_sim(emb[0], emb[1]).item())

    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False

    def embedding_score(a_words: List[str], b_words: List[str]) -> float:
        raise RuntimeError(
            "sentence-transformers is not installed. Install it with: "
            "pip install sentence-transformers"
        )


def difflib_score(a_words: List[str], b_words: List[str]) -> float:
    """Calculate SequenceMatcher similarity ratio (0.0 to 1.0) on joined word strings."""
    return SequenceMatcher(None, " ".join(a_words), " ".join(b_words)).ratio()


def rapidfuzz_score(a_words: List[str], b_words: List[str]) -> float:
    """Calculate RapidFuzz Levenshtein similarity ratio (normalized to 0.0 - 1.0)."""
    return rf_fuzz.ratio(" ".join(a_words), " ".join(b_words)) / 100.0


SCORE_FNS: Dict[str, Callable[[List[str], List[str]], float]] = {
    "difflib": difflib_score,
    "rapidfuzz": rapidfuzz_score,
    "embedding": embedding_score,
}


def get_score_fn(name: str) -> Callable[[List[str], List[str]], float]:
    """Retrieve scoring function by name ('difflib', 'rapidfuzz', 'embedding')."""
    normalized_name = name.lower().strip()
    if normalized_name not in SCORE_FNS:
        raise ValueError(
            f"Unknown score_fn '{name}'. Available scorers: {list(SCORE_FNS.keys())}"
        )
    if normalized_name == "embedding" and not EMBEDDING_AVAILABLE:
        raise RuntimeError("sentence-transformers is not installed.")
    return SCORE_FNS[normalized_name]

"""Search, indexing, and similarity scoring module."""

from .normalizer import normalize_text, build_word_index
from .scorers import (
    get_score_fn,
    difflib_score,
    rapidfuzz_score,
    embedding_score,
    SCORE_FNS,
)
from .index import InvertedIndex, choose_anchor, build_rarity_fn
from .engine import (
    exact_phrase_search,
    fuzzy_sliding_window_search,
    rare_anchor_fuzzy_search,
    search_dialogue,
)

__all__ = [
    "normalize_text",
    "build_word_index",
    "get_score_fn",
    "difflib_score",
    "rapidfuzz_score",
    "embedding_score",
    "SCORE_FNS",
    "InvertedIndex",
    "choose_anchor",
    "build_rarity_fn",
    "exact_phrase_search",
    "fuzzy_sliding_window_search",
    "rare_anchor_fuzzy_search",
    "search_dialogue",
]

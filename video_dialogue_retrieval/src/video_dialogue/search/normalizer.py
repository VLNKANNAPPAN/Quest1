"""Text normalization and token indexing utilities."""

import re
from typing import List, Dict, Any, Union
from ..core.models import WordTimestamp


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and collapsing whitespace.

    Examples:
        >>> normalize_text("Hello, World!  ")
        'hello world'
        >>> normalize_text("My mind rebels at stagnation.")
        'my mind rebels at stagnation'
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_word_index(transcript: List[Union[WordTimestamp, Dict[str, Any]]]) -> List[str]:
    """Extract and normalize a sequential list of words from a transcript."""
    words: List[str] = []
    for item in transcript:
        if isinstance(item, WordTimestamp):
            w = item.word
        elif isinstance(item, dict):
            w = item.get("word", "")
        else:
            w = str(item)
        normalized = normalize_text(w)
        if normalized:
            words.append(normalized)
        else:
            words.append(w.strip().lower())
    return words

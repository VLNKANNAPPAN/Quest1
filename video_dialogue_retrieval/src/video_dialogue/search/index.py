"""Inverted index and dynamic term rarity algorithms for rare-anchor search."""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Optional, Callable, Set


class InvertedIndex:
    """Inverted index mapping words to their 0-indexed positions in a transcript."""

    def __init__(self, index_dict: Optional[Dict[str, List[int]]] = None):
        self._index: Dict[str, List[int]] = index_dict or defaultdict(list)

    @classmethod
    def from_words(cls, transcript_words: List[str]) -> "InvertedIndex":
        """Construct an inverted index from an ordered list of normalized words."""
        index = defaultdict(list)
        for i, word in enumerate(transcript_words):
            index[word].append(i)
        return cls(dict(index))

    def get(self, word: str) -> List[int]:
        """Return positions of a word in the transcript."""
        return self._index.get(word, [])

    def __contains__(self, word: str) -> bool:
        return word in self._index

    def __getitem__(self, word: str) -> List[int]:
        return self._index[word]

    def to_dict(self) -> Dict[str, List[int]]:
        return dict(self._index)


def build_rarity_fn(transcript_words: List[str]) -> Callable[[str], float]:
    """Compute per-call word rarity (IDF-inspired) based on transcript token frequencies.

    Calculates: log(N / (1 + freq(word)))
    Ensures rare words have high rarity scores, while stopwords/common words have low scores.
    """
    freq = Counter(transcript_words)
    n = max(1, len(transcript_words))

    def rarity(word: str) -> float:
        return math.log(n / (1.0 + freq.get(word, 0)))

    return rarity


def choose_anchor(
    target_words: List[str],
    transcript_words: List[str],
    rarity_fn: Optional[Callable[[str], float]] = None,
) -> Optional[str]:
    """Select the most informative/rare anchor word from the target query that appears in transcript."""
    vocab: Set[str] = set(transcript_words)
    candidates = [w for w in target_words if w in vocab]
    if not candidates:
        return None

    if rarity_fn is None:
        rarity_fn = build_rarity_fn(transcript_words)

    return max(candidates, key=rarity_fn)

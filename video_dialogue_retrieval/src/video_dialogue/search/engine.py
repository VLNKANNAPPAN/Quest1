"""Search retrieval engines: Exact phrase matching, sliding window fuzzy, and rare-anchor fuzzy."""

import logging
from typing import List, Dict, Any, Optional, Callable, Union

from ..core.models import SearchResultItem
from .index import InvertedIndex, build_rarity_fn, choose_anchor
from .scorers import get_score_fn, difflib_score

logger = logging.getLogger(__name__)


def exact_phrase_search(
    transcript_words: List[str], target_words: List[str]
) -> List[SearchResultItem]:
    """Find exact contiguous occurrences of target_words in transcript_words."""
    matches: List[SearchResultItem] = []
    n = len(target_words)
    if n == 0 or len(transcript_words) < n:
        return matches

    for i in range(len(transcript_words) - n + 1):
        if transcript_words[i : i + n] == target_words:
            matches.append(
                SearchResultItem(
                    start_index=i,
                    end_index=i + n,
                    score=1.0,
                    method="exact",
                )
            )
    return matches


def fuzzy_sliding_window_search(
    transcript_words: List[str],
    target_words: List[str],
    score_fn: Optional[Callable[[List[str], List[str]], float]] = None,
    length_tolerance: int = 2,
) -> List[SearchResultItem]:
    """Exhaustive fuzzy sliding window across all window lengths [N-tol, N+tol]."""
    if score_fn is None:
        score_fn = difflib_score

    n = len(target_words)
    if n == 0 or not transcript_words:
        return []

    lo = max(1, n - length_tolerance)
    hi = n + length_tolerance

    results: List[SearchResultItem] = []
    for wlen in range(lo, hi + 1):
        if wlen > len(transcript_words):
            continue
        for i in range(len(transcript_words) - wlen + 1):
            window = transcript_words[i : i + wlen]
            score = float(score_fn(target_words, window))
            results.append(
                SearchResultItem(
                    start_index=i,
                    end_index=i + wlen,
                    score=score,
                    method="fuzzy_sliding_window",
                )
            )

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def rare_anchor_fuzzy_search(
    transcript_words: List[str],
    target_words: List[str],
    score_fn: Optional[Callable[[List[str], List[str]], float]] = None,
    extra_context: int = 2,
    length_tolerance: int = 2,
    inverted_index: Optional[Union[InvertedIndex, Dict[str, List[int]]]] = None,
) -> List[SearchResultItem]:
    """Optimized fuzzy search anchored on the rarest word in target query.

    Selects the rarest token in target_words (using per-call IDF rarity),
    locates all occurrences of that anchor, and evaluates candidate windows only in the
    neighborhood around each anchor occurrence.
    """
    if score_fn is None:
        score_fn = difflib_score

    if not target_words or not transcript_words:
        return []

    rarity_fn = build_rarity_fn(transcript_words)
    anchor = choose_anchor(target_words, transcript_words, rarity_fn)
    if anchor is None:
        return []

    anchor_offset = target_words.index(anchor)
    n = len(target_words)

    if inverted_index is not None:
        if isinstance(inverted_index, InvertedIndex):
            anchor_positions = inverted_index.get(anchor)
        else:
            anchor_positions = inverted_index.get(anchor, [])
    else:
        anchor_positions = [i for i, w in enumerate(transcript_words) if w == anchor]

    results: List[SearchResultItem] = []
    for anchor_index in anchor_positions:
        expected_start = anchor_index - anchor_offset
        start = max(0, expected_start - extra_context)
        end = min(len(transcript_words), expected_start + n + extra_context)
        region = transcript_words[start:end]

        lo = max(1, n - length_tolerance)
        hi = n + length_tolerance
        best: Optional[SearchResultItem] = None

        for wlen in range(lo, hi + 1):
            if wlen > len(region):
                continue
            for i in range(len(region) - wlen + 1):
                window = region[i : i + wlen]
                score = float(score_fn(target_words, window))
                if best is None or score > best.score:
                    best = SearchResultItem(
                        start_index=start + i,
                        end_index=start + i + wlen,
                        score=score,
                        method="rare_anchor_fuzzy",
                        anchor=anchor,
                    )
        if best is not None:
            results.append(best)

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def search_dialogue(
    transcript_words: List[str],
    target_words: List[str],
    method: str = "rare_anchor_fuzzy",
    score_fn_name: str = "difflib",
    inverted_index: Optional[Union[InvertedIndex, Dict[str, List[int]]]] = None,
    length_tolerance: int = 2,
    extra_context: int = 2,
) -> List[SearchResultItem]:
    """Unified entrypoint for dialogue retrieval algorithms."""
    normalized_method = method.lower().strip()
    score_fn = get_score_fn(score_fn_name) if normalized_method != "exact" else None

    if normalized_method == "exact":
        return exact_phrase_search(transcript_words, target_words)
    elif normalized_method in ("fuzzy", "fuzzy_sliding_window"):
        return fuzzy_sliding_window_search(
            transcript_words,
            target_words,
            score_fn=score_fn,
            length_tolerance=length_tolerance,
        )
    elif normalized_method in ("rare_anchor_fuzzy", "rare_anchor"):
        return rare_anchor_fuzzy_search(
            transcript_words,
            target_words,
            score_fn=score_fn,
            extra_context=extra_context,
            length_tolerance=length_tolerance,
            inverted_index=inverted_index,
        )
    else:
        raise ValueError(
            f"Unknown search method '{method}'. Valid methods: "
            "['exact', 'fuzzy', 'rare_anchor_fuzzy']"
        )

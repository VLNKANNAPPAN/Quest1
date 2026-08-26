"""Search retrieval engines: Exact phrase matching, sliding window fuzzy, and rare-anchor fuzzy."""

import logging
from typing import List, Dict, Any, Optional, Callable, Union

from ..core.models import SearchResultItem
from .index import InvertedIndex, build_rarity_fn, choose_anchor, choose_anchors
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
    """Optimized fuzzy search anchored on rare words with multi-anchor retry and sliding window fallback.

    Iterates candidate anchor tokens ordered from rarest to common. If the top anchor yields
    no strong match (or is absent from the transcript due to ASR misspelling), it automatically
    retries with the next rarest anchor words before falling back to full sliding window search.
    """
    if score_fn is None:
        score_fn = difflib_score

    if not target_words or not transcript_words:
        return []

    rarity_fn = build_rarity_fn(transcript_words)
    candidate_anchors = choose_anchors(target_words, transcript_words, rarity_fn)

    if not candidate_anchors:
        logger.info("No query anchor survived ASR; falling back to exhaustive fuzzy search.")
        results = fuzzy_sliding_window_search(
            transcript_words, target_words, score_fn=score_fn,
            length_tolerance=length_tolerance,
        )
        for result in results:
            result.method = "rare_anchor_fuzzy_fallback"
        return results

    all_anchor_results: List[SearchResultItem] = []
    n = len(target_words)

    for anchor in candidate_anchors:
        anchor_offset = target_words.index(anchor)
        if inverted_index is not None:
            if isinstance(inverted_index, InvertedIndex):
                anchor_positions = inverted_index.get(anchor)
            else:
                anchor_positions = inverted_index.get(anchor, [])
        else:
            anchor_positions = [i for i, w in enumerate(transcript_words) if w == anchor]

        anchor_results: List[SearchResultItem] = []
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
                anchor_results.append(best)

        anchor_results.sort(key=lambda r: r.score, reverse=True)
        if anchor_results and anchor_results[0].score >= 0.70:
            return anchor_results

        all_anchor_results.extend(anchor_results)

    if all_anchor_results:
        all_anchor_results.sort(key=lambda r: r.score, reverse=True)
        return all_anchor_results

    # Exhaustive fallback
    logger.info("All candidate anchors produced low scores; falling back to exhaustive fuzzy search.")
    results = fuzzy_sliding_window_search(
        transcript_words, target_words, score_fn=score_fn,
        length_tolerance=length_tolerance,
    )
    for result in results:
        result.method = "rare_anchor_fuzzy_fallback"
    return results


def auto_search(
    transcript_words: List[str],
    target_words: List[str],
    inverted_index: Optional[Union[InvertedIndex, Dict[str, List[int]]]] = None,
    length_tolerance: int = 2,
    extra_context: int = 2,
    high_confidence_threshold: float = 0.85,
) -> List[SearchResultItem]:
    """Automatic cascading search strategy that selects the best method without user intervention.

    The cascade logic:
    1. Try **exact phrase match** first (O(N), instant). If found, return immediately with score 1.0.
    2. Try **rare-anchor fuzzy search with RapidFuzz** (fastest fuzzy method, bounded by anchor
       neighborhoods). If the best match scores >= high_confidence_threshold, return it.
    3. Fall back to **full fuzzy sliding window with RapidFuzz** (exhaustive but catches all edge
       cases where anchor selection fails or ASR heavily garbles the rare words).

    This gives the user the best result without requiring any algorithm knowledge.
    """
    if not target_words or not transcript_words:
        return []

    # Stage 1: Exact match (instant, highest confidence)
    exact_results = exact_phrase_search(transcript_words, target_words)
    if exact_results:
        logger.info("Auto-search: Exact match found (score=1.0). Returning immediately.")
        for r in exact_results:
            r.method = "auto:exact"
        return exact_results

    # Stage 2: Rare-anchor fuzzy with RapidFuzz (fast, bounded search)
    rapidfuzz_score_fn = get_score_fn("rapidfuzz")
    anchor_results = rare_anchor_fuzzy_search(
        transcript_words,
        target_words,
        score_fn=rapidfuzz_score_fn,
        extra_context=extra_context,
        length_tolerance=length_tolerance,
        inverted_index=inverted_index,
    )
    if anchor_results and anchor_results[0].score >= high_confidence_threshold:
        logger.info(
            "Auto-search: Rare-anchor fuzzy match (score=%.3f >= %.2f threshold). Returning.",
            anchor_results[0].score, high_confidence_threshold,
        )
        for r in anchor_results:
            r.method = f"auto:{r.method}"
        return anchor_results

    # Stage 3: Exhaustive fuzzy sliding window (catches everything the anchor missed)
    logger.info(
        "Auto-search: Anchor best=%.3f < %.2f threshold; escalating to exhaustive sliding window.",
        anchor_results[0].score if anchor_results else 0.0, high_confidence_threshold,
    )
    sliding_results = fuzzy_sliding_window_search(
        transcript_words,
        target_words,
        score_fn=rapidfuzz_score_fn,
        length_tolerance=length_tolerance,
    )
    for r in sliding_results:
        r.method = f"auto:{r.method}"

    # Merge: keep the best from both stages, deduplicated by (start_index, end_index)
    seen = set()
    merged: List[SearchResultItem] = []
    for r in sorted(anchor_results + sliding_results, key=lambda x: x.score, reverse=True):
        key = (r.start_index, r.end_index)
        if key not in seen:
            seen.add(key)
            merged.append(r)

    return merged


def search_dialogue(
    transcript_words: List[str],
    target_words: List[str],
    method: str = "auto",
    score_fn_name: str = "rapidfuzz",
    inverted_index: Optional[Union[InvertedIndex, Dict[str, List[int]]]] = None,
    length_tolerance: int = 2,
    extra_context: int = 2,
) -> List[SearchResultItem]:
    """Unified entrypoint for dialogue retrieval algorithms.

    When method='auto' (default), the system automatically cascades through
    exact -> rare-anchor fuzzy -> sliding window, picking the best strategy.
    """
    normalized_method = method.lower().strip()

    if normalized_method == "auto":
        return auto_search(
            transcript_words,
            target_words,
            inverted_index=inverted_index,
            length_tolerance=length_tolerance,
            extra_context=extra_context,
        )

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
            "['auto', 'exact', 'fuzzy', 'rare_anchor_fuzzy']"
        )

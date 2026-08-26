"""Tests for dialogue search retrieval algorithms."""

import pytest
from video_dialogue.search.normalizer import normalize_text
from video_dialogue.search.index import InvertedIndex
from video_dialogue.search.scorers import difflib_score, rapidfuzz_score
from video_dialogue.search.engine import (
    exact_phrase_search,
    fuzzy_sliding_window_search,
    rare_anchor_fuzzy_search,
    search_dialogue,
)


def test_exact_phrase_search(sample_transcript_words):
    target = ["my", "mind", "rebels", "at", "stagnation"]
    matches = exact_phrase_search(sample_transcript_words, target)
    assert len(matches) == 1
    assert matches[0].score == 1.0
    assert matches[0].method == "exact"
    matched_words = sample_transcript_words[matches[0].start_index : matches[0].end_index]
    assert matched_words == target


def test_exact_phrase_search_no_match(sample_transcript_words):
    target = ["sherlock", "holmes", "london"]
    matches = exact_phrase_search(sample_transcript_words, target)
    assert len(matches) == 0


def test_fuzzy_sliding_window_search(sample_transcript_words):
    target = ["my", "mind", "rebels", "at", "stagnation"]
    results = fuzzy_sliding_window_search(sample_transcript_words, target, score_fn=difflib_score)
    assert len(results) > 0
    top = results[0]
    assert top.score == 1.0
    assert sample_transcript_words[top.start_index : top.end_index] == target


def test_rare_anchor_fuzzy_index_equivalence(sample_transcript_words):
    target = ["my", "mind", "rebels", "at", "stagnation"]
    inv_index = InvertedIndex.from_words(sample_transcript_words)

    linear_results = rare_anchor_fuzzy_search(sample_transcript_words, target, score_fn=difflib_score, inverted_index=None)
    indexed_results = rare_anchor_fuzzy_search(sample_transcript_words, target, score_fn=difflib_score, inverted_index=inv_index)

    assert len(linear_results) == len(indexed_results)
    assert linear_results[0].start_index == indexed_results[0].start_index
    assert linear_results[0].end_index == indexed_results[0].end_index
    assert abs(linear_results[0].score - indexed_results[0].score) < 1e-6


def test_rare_anchor_falls_back_when_asr_loses_every_query_word():
    transcript = ["i", "really", "adore", "geography", "today"]
    target = ["freaking", "love", "maps"]

    results = rare_anchor_fuzzy_search(transcript, target, score_fn=difflib_score)

    assert results
    assert results[0].method == "rare_anchor_fuzzy_fallback"


def test_search_dialogue_dispatch(sample_transcript_words):
    target = ["my", "mind", "rebels", "at", "stagnation"]

    exact_res = search_dialogue(sample_transcript_words, target, method="exact")
    assert len(exact_res) == 1

    fuzzy_res = search_dialogue(sample_transcript_words, target, method="fuzzy", score_fn_name="rapidfuzz")
    assert len(fuzzy_res) > 0

    anchor_res = search_dialogue(sample_transcript_words, target, method="rare_anchor_fuzzy", score_fn_name="difflib")
    assert len(anchor_res) > 0
    assert anchor_res[0].anchor in ("rebels", "stagnation")

    with pytest.raises(ValueError):
        search_dialogue(sample_transcript_words, target, method="unsupported_method")

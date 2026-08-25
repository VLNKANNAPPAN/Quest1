"""Tests for similarity scoring algorithms."""

import pytest
from video_dialogue.search.scorers import (
    difflib_score,
    rapidfuzz_score,
    get_score_fn,
    SCORE_FNS,
)


def test_difflib_score():
    words_a = ["my", "mind", "rebels"]
    words_b = ["my", "mind", "rebels"]
    assert difflib_score(words_a, words_b) == 1.0

    words_c = ["my", "mind", "sleeps"]
    score = difflib_score(words_a, words_c)
    assert 0.0 < score < 1.0


def test_rapidfuzz_score():
    words_a = ["my", "mind", "rebels"]
    words_b = ["my", "mind", "rebels"]
    assert rapidfuzz_score(words_a, words_b) == 1.0

    words_c = ["my", "mind", "rebbels"]
    score = rapidfuzz_score(words_a, words_c)
    assert score > 0.90


def test_scorer_agreement_on_misspelling():
    target = ["my", "mind", "rebels", "at", "stagnation"]
    misspelled = ["my", "mind", "rebbels", "at", "stagnashun"]
    unrelated = ["the", "quick", "brown", "fox", "jumps"]

    score_diff_match = difflib_score(target, misspelled)
    score_diff_unrelated = difflib_score(target, unrelated)
    assert score_diff_match > score_diff_unrelated

    score_rf_match = rapidfuzz_score(target, misspelled)
    score_rf_unrelated = rapidfuzz_score(target, unrelated)
    assert score_rf_match > score_rf_unrelated


def test_get_score_fn():
    assert get_score_fn("difflib") is difflib_score
    assert get_score_fn("rapidfuzz") is rapidfuzz_score

    with pytest.raises(ValueError):
        get_score_fn("non_existent_scorer")

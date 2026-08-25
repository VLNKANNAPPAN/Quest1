"""Tests for text normalization and token extraction."""

from video_dialogue.search.normalizer import normalize_text, build_word_index
from video_dialogue.core.models import WordTimestamp


def test_normalize_text_basic():
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("  Spaces   and \n newlines \t ") == "spaces and newlines"
    assert normalize_text("My mind rebels at stagnation...") == "my mind rebels at stagnation"
    assert normalize_text("") == ""


def test_build_word_index():
    transcript_dicts = [
        {"word": "Hello,", "start": 0.0, "end": 0.5},
        {"word": "world!", "start": 0.6, "end": 1.0},
    ]
    assert build_word_index(transcript_dicts) == ["hello", "world"]

    transcript_models = [
        WordTimestamp(word="DeepMind", start=0.0, end=0.4),
        WordTimestamp(word="Agents!", start=0.5, end=0.9),
    ]
    assert build_word_index(transcript_models) == ["deepmind", "agents"]

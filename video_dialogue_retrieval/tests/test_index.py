"""Tests for InvertedIndex and dynamic term rarity anchor selection."""

from video_dialogue.search.normalizer import normalize_text
from video_dialogue.search.index import InvertedIndex, build_rarity_fn, choose_anchor


def test_inverted_index():
    words = ["the", "cat", "sat", "on", "the", "mat"]
    index = InvertedIndex.from_words(words)

    assert index.get("the") == [0, 4]
    assert index.get("cat") == [1]
    assert "sat" in index
    assert index.get("dog") == []


def test_rarity_anchor_dynamic_selection():
    # Video A: 'mind' appears many times, 'rebels' appears once
    video_a = normalize_text(
        "my at my mind at my mind the weather today is fine the cat sat on the mat "
        "my mind rebels at stagnation the dog ran fast my mind is at home"
    ).split()

    # Video B: 'rebels' appears many times, 'stagnation' appears once
    video_b = normalize_text(
        "rebels rebels rebels rebels rebels rebels rebels rebels "
        "my my my mind mind mind at at at "
        "my mind rebels at stagnation"
    ).split()

    target = normalize_text("my mind rebels at stagnation").split()

    anchor_a = choose_anchor(target, video_a)
    anchor_b = choose_anchor(target, video_b)

    assert anchor_a == "rebels", "In video_a, 'rebels' or 'stagnation' is much rarer than 'mind'"
    assert anchor_b == "stagnation", "In video_b, 'stagnation' is the rarest token"


def test_choose_anchor_no_match():
    transcript = ["hello", "world"]
    target = ["apples", "oranges"]
    assert choose_anchor(target, transcript) is None

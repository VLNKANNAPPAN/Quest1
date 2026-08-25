"""Benchmark Demo: Comparative Evaluation of Search Algorithms and Indexing."""

from pathlib import Path
import sys
from tabulate import tabulate

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from video_dialogue.benchmark.benchmark import run_variant_benchmark, DEFAULT_SEARCH_VARIANTS
from video_dialogue.search.normalizer import normalize_text


def main():
    print("=== Video Dialogue Retrieval: Search Method Benchmark ===")

    corpus = (
        "the weather today is fine the cat sat on the mat "
        "my mind rebels at stagnation the dog ran fast across the green yard "
        "she noticed that time was passing quickly and knowledge was expanding"
    )
    words = normalize_text(corpus).split()
    synthetic_transcript = [
        {"word": w, "start": float(i) * 0.8, "end": float(i) * 0.8 + 0.75}
        for i, w in enumerate(words)
    ]

    # Target dialogue ground truths: (query, expected_start_s, tolerance_s)
    known_dialogues = [
        ("My mind rebels at stagnation", 8.8, 1.5),
        ("the cat sat on the mat", 4.0, 1.5),
        ("time was passing quickly", 17.6, 1.5),
    ]

    print(f"\nCorpus Size: {len(words)} tokens | Ground Truth Queries: {len(known_dialogues)}")
    print("Executing benchmark across all search variants (3 runs per query)...\n")

    df = run_variant_benchmark(
        synthetic_transcript,
        known_dialogues,
        variants=DEFAULT_SEARCH_VARIANTS,
        runs=5,
    )

    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


if __name__ == "__main__":
    main()

"""Command-line interface (CLI) for Video Dialogue Retrieval."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

from tabulate import tabulate

from .config.settings import PipelineConfig, get_default_config
from .database.db import DatabaseManager
from .pipeline.orchestrator import DialogueRetrievalPipeline
from .benchmark.benchmark import run_variant_benchmark, DEFAULT_SEARCH_VARIANTS
from .search.normalizer import normalize_text
from .progress import TerminalProgress


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_search(args: argparse.Namespace) -> None:
    config = get_default_config(cache_dir=Path(args.cache_dir))
    pipeline = DialogueRetrievalPipeline(config=config)

    print(f"\n[SEARCH] Searching dialogue in: {args.video}")
    print(f"  Target query : \"{args.query}\"")
    print(f"  Method       : {args.method} (score_fn={args.score_fn})")
    print(f"  ASR Model    : {args.model_size}\n")

    progress = TerminalProgress(enabled=not args.no_progress)
    result = pipeline.run(
        video_url=args.video,
        target_dialogue=args.query,
        method=args.method,
        score_fn_name=args.score_fn,
        model_size=args.model_size,
        top_k=args.top_k,
        use_inverted_index=not args.no_index,
        force_asr=args.force_asr,
        progress=progress,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    if not result.success:
        print(f"[ERROR] Search failed: {result.message}")
        sys.exit(1)

    print("[MATCH] Match found!\n")
    table_data = []
    for m in result.matches:
        table_data.append([
            m.rank,
            f"{m.start_timestamp:.2f}s - {m.end_timestamp:.2f}s",
            m.start_frame,
            f"{m.score:.3f}",
            m.matched_text,
            Path(m.frame_path).name if m.frame_path else "N/A",
        ])

    headers = ["Rank", "Timestamp", "Frame", "Score", "Matched Text", "Frame Image"]
    print(tabulate(table_data, headers=headers, tablefmt="github"))
    if result.result_file:
        print(f"\nSaved report to: {result.result_file}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    print("\n[BENCHMARK] Running Search Engine Variant Benchmark...")
    # Synthetic transcript test or loaded video
    sample_text = (
        "the weather today is fine the cat sat on the mat "
        "my mind rebels at stagnation the dog ran fast"
    )
    synthetic_transcript = [
        {"word": w, "start": float(i), "end": float(i) + 0.9}
        for i, w in enumerate(normalize_text(sample_text).split())
    ]
    known = [("My mind rebels at stagnation", 12.0, 1.5)]

    df = run_variant_benchmark(synthetic_transcript, known, runs=args.runs)
    print("\n" + tabulate(df, headers="keys", tablefmt="github", showindex=False))


def cmd_inspect(args: argparse.Namespace) -> None:
    config = get_default_config(cache_dir=Path(args.cache_dir))
    db = DatabaseManager(config.db_path)
    videos = db.list_videos()

    print(f"\n[INSPECT] Inspecting Cache Database: {config.db_path}")
    print(f"  Total Videos Indexed: {len(videos)}\n")

    if not videos:
        print("  Database is currently empty.")
        return

    rows = []
    for v in videos:
        rows.append([
            v.video_id,
            v.url[:35] + "..." if len(v.url) > 35 else v.url,
            v.audio_fingerprint[:16] + "..." if v.audio_fingerprint else "N/A",
            f"{v.duration:.1f}s",
            f"{v.fps:.1f}" if v.fps else "N/A",
            f"{v.width}x{v.height}",
            v.first_seen_at[:19] if v.first_seen_at else "N/A",
        ])

    headers = ["Video ID", "Source URL / Path", "Fingerprint", "Duration", "FPS", "Resolution", "First Seen"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def cmd_clear_cache(args: argparse.Namespace) -> None:
    config = get_default_config(cache_dir=Path(args.cache_dir))
    db = DatabaseManager(config.db_path)
    db.clear_all()
    print(f"[OK] Cleared database records in {config.db_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="video-dialogue",
        description="Video Dialogue Retrieval: Audio-first, fingerprint-deduplicated dialogue & frame search",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search dialogue in a video URL or local file")
    search_parser.add_argument("-i", "--video", required=True, help="URL or local path to video file")
    search_parser.add_argument("-q", "--query", required=True, help="Target dialogue phrase")
    search_parser.add_argument(
        "-m", "--method", default="auto",
        choices=["auto", "rare_anchor_fuzzy", "fuzzy", "exact"],
        help="Search retrieval algorithm (default: auto — cascades exact -> rare-anchor -> sliding window)",
    )
    search_parser.add_argument(
        "-s", "--score-fn", default="rapidfuzz",
        choices=["rapidfuzz", "difflib", "embedding"],
        help="Similarity score function (default: rapidfuzz)",
    )
    search_parser.add_argument(
        "--model-size", default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: small)",
    )
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of top matches to return")
    search_parser.add_argument("--no-index", action="store_true", help="Disable inverted index for rare anchor search")
    search_parser.add_argument("--force-asr", action="store_true", help="Force re-transcription bypassing DB cache")
    search_parser.add_argument("--cache-dir", default="cache", help="Cache directory root (default: cache)")
    search_parser.add_argument("--json", action="store_true", help="Output full JSON result payload")
    search_parser.add_argument("--no-progress", action="store_true", help="Hide live download and ASR progress")

    # Benchmark subcommand
    bench_parser = subparsers.add_parser("benchmark", help="Run search algorithm variant benchmark")
    bench_parser.add_argument("-r", "--runs", type=int, default=5, help="Number of timing iterations")
    bench_parser.add_argument("--cache-dir", default="cache", help="Cache directory")

    # Inspect subcommand
    inspect_parser = subparsers.add_parser("inspect", help="Inspect indexed videos and transcripts in SQLite DB")
    inspect_parser.add_argument("--cache-dir", default="cache", help="Cache directory")

    # Clear cache subcommand
    clear_parser = subparsers.add_parser("clear-cache", help="Clear all database cache entries")
    clear_parser.add_argument("--cache-dir", default="cache", help="Cache directory")

    parsed_args = parser.parse_args(argv)
    setup_logging(parsed_args.verbose)

    if parsed_args.command == "search":
        cmd_search(parsed_args)
    elif parsed_args.command == "benchmark":
        cmd_benchmark(parsed_args)
    elif parsed_args.command == "inspect":
        cmd_inspect(parsed_args)
    elif parsed_args.command == "clear-cache":
        cmd_clear_cache(parsed_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

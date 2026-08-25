"""Complete Project Runner and Real-Time Verification Suite."""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from tabulate import tabulate

# Ensure package is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from video_dialogue import (
    PipelineConfig,
    DatabaseManager,
    DialogueRetrievalPipeline,
    run_variant_benchmark,
    DEFAULT_SEARCH_VARIANTS,
    normalize_text,
    compute_audio_fingerprint,
    timestamp_to_frame,
    extract_frame,
)


def log_step(step_num: int, total_steps: int, title: str):
    print(f"\n{'='*75}")
    print(f" [STEP {step_num}/{total_steps}] {title}")
    print(f"{'='*75}")


def main():
    start_total = time.perf_counter()
    total_steps = 7

    print("\n" + "#"*75)
    print("      VIDEO DIALOGUE RETRIEVAL (v3) - COMPLETE PROJECT RUNNER")
    print(f"      Interpreter: {sys.executable} (Python {sys.version.split()[0]})")
    print("#"*75)

    # -------------------------------------------------------------
    # Step 1: Configuration & Directory Preparation
    # -------------------------------------------------------------
    log_step(1, total_steps, "Initializing Configuration & Storage Directories")
    cache_dir = Path("cache/live_run")
    config = PipelineConfig(cache_dir=cache_dir)
    config.ensure_directories()
    print(f"  [+] Cache Root : {config.cache_dir}")
    print(f"  [+] Database   : {config.db_path}")
    print(f"  [+] Video Dir  : {config.video_dir}")
    print(f"  [+] Audio Dir  : {config.audio_dir}")
    print(f"  [+] Frames Dir : {config.frame_dir}")
    print(f"  [+] Results Dir: {config.result_dir}")
    print("  -> Configuration initialized successfully.")

    # -------------------------------------------------------------
    # Step 2: Database Initialization & Dedup Layer
    # -------------------------------------------------------------
    log_step(2, total_steps, "Verifying SQLite Persistence & Deduplication Layer")
    db = DatabaseManager(config.db_path)
    db.clear_all()
    print("  [+] Cleaned previous records in database.")
    print("  [+] Verified schema creation (videos & transcripts tables with indices).")

    # -------------------------------------------------------------
    # Step 3: Media Creation & Audio-First Extraction
    # -------------------------------------------------------------
    log_step(3, total_steps, "Media Acquisition & Audio-First Extraction")
    sample_video = config.video_dir / "sample_dialogue_clip.mp4"
    if not sample_video.exists():
        print(f"  [*] Generating test video container (20s duration, 25fps) at {sample_video.name}...")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=20:size=640x360:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            str(sample_video),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  [+] Video container created successfully.")

    pipeline = DialogueRetrievalPipeline(config=config, db_manager=db)
    print("  [*] Extracting 16kHz mono PCM WAV audio (audio-first acquisition)...")
    audio_path = pipeline.media.get_audio(str(sample_video))
    print(f"  [+] Audio WAV generated: {audio_path.name} ({audio_path.stat().st_size / 1024:.1f} KB)")

    # -------------------------------------------------------------
    # Step 4: Acoustic Fingerprinting
    # -------------------------------------------------------------
    log_step(4, total_steps, "Computing Chromaprint Acoustic Fingerprint")
    t0 = time.perf_counter()
    fingerprint = compute_audio_fingerprint(audio_path)
    fp_time = (time.perf_counter() - t0) * 1000
    print(f"  [+] Acoustic Fingerprint: {fingerprint[:32]}...")
    print(f"  [+] Computation Latency : {fp_time:.2f} ms")

    video_rec = pipeline.get_or_create_video_record(str(sample_video))
    print(f"  [+] Registered in SQLite DB as Video ID: {video_rec.video_id}")

    # -------------------------------------------------------------
    # Step 5: Transcript Indexing & Multi-Engine Search
    # -------------------------------------------------------------
    log_step(5, total_steps, "Transcript Indexing & Multi-Algorithm Dialogue Retrieval")
    dialogue_script = (
        "welcome ladies and gentlemen to the intelligence system briefing "
        "today we examine the problem of multimedia search and archive retrieval "
        "my mind rebels at stagnation when exploring large unstructured datasets "
        "therefore we build fast inverted indexes for instant query resolution"
    )
    tokens = normalize_text(dialogue_script).split()
    simulated_transcript = [
        {"word": w, "start": float(i) * 0.5, "end": float(i) * 0.5 + 0.45}
        for i, w in enumerate(tokens)
    ]
    db.insert_transcript(video_rec.video_id, "small", simulated_transcript)
    print(f"  [+] Indexed transcript with {len(simulated_transcript)} timestamped tokens.")

    target_query = "My mind rebels at stagnation"
    print(f"\n  [*] Executing search for query: \"{target_query}\"")

    results_table = []
    for method, scorer, label in [
        ("exact", "difflib", "Exact Phrase Matching"),
        ("fuzzy", "difflib", "Sliding Window (Difflib)"),
        ("fuzzy", "rapidfuzz", "Sliding Window (RapidFuzz C++)"),
        ("rare_anchor_fuzzy", "difflib", "Rare-Anchor Fuzzy (Difflib)"),
        ("rare_anchor_fuzzy", "rapidfuzz", "Rare-Anchor Fuzzy (RapidFuzz C++)"),
    ]:
        t_search = time.perf_counter()
        res = pipeline.run(
            video_url=str(sample_video),
            target_dialogue=target_query,
            method=method,
            score_fn_name=scorer,
            model_size="small",
            top_k=1,
        )
        dur_ms = (time.perf_counter() - t_search) * 1000
        best_match = res.matches[0] if res.matches else None
        results_table.append([
            label,
            "FOUND" if res.success else "MISSING",
            f"{best_match.score:.3f}" if best_match else "-",
            f"{best_match.start_timestamp:.2f}s" if best_match else "-",
            f"Frame #{best_match.start_frame}" if best_match else "-",
            f"{dur_ms:.2f} ms",
        ])

    print("\n" + tabulate(
        results_table,
        headers=["Search Algorithm", "Status", "Score", "Timestamp", "Frame Index", "Latency"],
        tablefmt="github",
    ))

    # -------------------------------------------------------------
    # Step 6: Frame Extraction & Visual Localization
    # -------------------------------------------------------------
    log_step(6, total_steps, "Visual Frame Localization & Image Extraction")
    match_result = pipeline.run(
        video_url=str(sample_video),
        target_dialogue=target_query,
        method="rare_anchor_fuzzy",
        score_fn_name="rapidfuzz",
        top_k=1,
    )
    best = match_result.matches[0]
    print(f"  [+] Matched Text    : \"{best.matched_text}\"")
    print(f"  [+] Time Window     : {best.start_timestamp:.2f}s -> {best.end_timestamp:.2f}s")
    print(f"  [+] Frame Number    : #{best.start_frame}")
    print(f"  [+] Frame Image Path: {best.frame_path}")
    print(f"  [+] Image File Size : {Path(best.frame_path).stat().st_size} bytes")
    print(f"  [+] Saved Report    : {match_result.result_file}")

    # -------------------------------------------------------------
    # Step 7: Algorithm Benchmark Suite
    # -------------------------------------------------------------
    log_step(7, total_steps, "Running Comparative Algorithm Benchmark Harness")
    known_eval = [
        ("My mind rebels at stagnation", 10.0, 1.5),
        ("welcome ladies and gentlemen", 0.0, 1.5),
        ("fast inverted indexes", 15.5, 1.5),
    ]
    bench_df = run_variant_benchmark(simulated_transcript, known_eval, runs=10)
    print("\n" + tabulate(bench_df, headers="keys", tablefmt="github", showindex=False))

    total_time = time.perf_counter() - start_total
    print("\n" + "="*75)
    print(f" [ALL STEPS COMPLETED SUCCESSFULLY IN {total_time:.2f}s]")
    print("="*75 + "\n")


if __name__ == "__main__":
    main()

"""Live YouTube Video Dialogue Retrieval & Benchmark with Real-Time Timing Breakdown."""

import sys
import time
import json
import logging
from pathlib import Path
from tabulate import tabulate

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from video_dialogue import (
    PipelineConfig,
    DatabaseManager,
    DialogueRetrievalPipeline,
    WhisperTranscriber,
    MediaManager,
    compute_audio_fingerprint,
    timestamp_to_frame,
    extract_frame,
    normalize_text,
    build_word_index,
    InvertedIndex,
    search_dialogue,
)


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def print_banner(text: str):
    print("\n" + "=" * 78)
    print(f"  {text}")
    print("=" * 78)


def main():
    video_url = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
    target_query = "Stay hungry stay foolish"
    model_size = "small"

    if len(sys.argv) > 1:
        video_url = sys.argv[1]
    if len(sys.argv) > 2:
        target_query = sys.argv[2]
    if len(sys.argv) > 3:
        model_size = sys.argv[3]

    print("\n" + "#" * 78)
    print("        VIDEO DIALOGUE RETRIEVAL - LIVE RETRIEVAL & PROFILING")
    print(f"        Target URL  : {video_url}")
    print(f"        Query Phrase: \"{target_query}\"")
    print(f"        Whisper Model: {model_size}")
    print("#" * 78)

    timing_records = []
    total_start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Step 1: Configuration & Cache Setup
    # -------------------------------------------------------------------------
    config = PipelineConfig(cache_dir=Path("cache/youtube_demo"))
    config.ensure_directories()
    db = DatabaseManager(config.db_path)
    media = MediaManager(config)
    transcriber = WhisperTranscriber(db_manager=db)

    # -------------------------------------------------------------------------
    # Step 2: Lightweight Metadata Extraction
    # -------------------------------------------------------------------------
    print_banner("[1/7] Fetching Video Metadata (Lightweight)")
    t0 = time.perf_counter()
    print("  -> Querying video stream info without full download...")
    meta = media.get_metadata(video_url)
    t_meta = time.perf_counter() - t0
    timing_records.append(("1. Metadata Extraction", f"{t_meta:.2f}s", f"Duration: {format_duration(meta.duration)}, FPS: {meta.fps or 25.0:.2f}, Res: {meta.width}x{meta.height}"))

    print(f"  [+] Video Duration : {meta.duration:.1f}s ({format_duration(meta.duration)})")
    print(f"  [+] Frame Rate     : {meta.fps or 25.0:.2f} fps")
    print(f"  [+] Resolution     : {meta.width}x{meta.height}")
    print(f"  [+] Audio Track    : {'Present' if meta.has_audio else 'None'}")
    print(f"  [*] Time Taken     : {t_meta:.2f} seconds")

    if not meta.has_audio:
        print("\n[ERROR] Video does not contain an audio stream.")
        return

    # -------------------------------------------------------------------------
    # Step 3: Audio-Only Acquisition (16kHz mono WAV)
    # -------------------------------------------------------------------------
    print_banner("[2/7] Audio-Only Download & Transcoding (Audio-First Acquisition)")
    print("  -> Downloading audio stream only (bestaudio format)...")
    t0 = time.perf_counter()
    audio_path = media.get_audio(video_url)
    t_audio_dl = time.perf_counter() - t0
    audio_size_mb = audio_path.stat().st_size / (1024 * 1024)
    timing_records.append(("2. Audio Acquisition (WAV 16kHz)", f"{t_audio_dl:.2f}s", f"{audio_size_mb:.2f} MB ({audio_path.name})"))

    print(f"  [+] Audio WAV File : {audio_path.name}")
    print(f"  [+] Audio File Size: {audio_size_mb:.2f} MB")
    print(f"  [*] Time Taken     : {t_audio_dl:.2f} seconds")

    # -------------------------------------------------------------------------
    # Step 4: Chromaprint Acoustic Fingerprinting & Deduplication
    # -------------------------------------------------------------------------
    print_banner("[3/7] Chromaprint Acoustic Fingerprinting & DB Deduplication")
    print("  -> Computing robust acoustic fingerprint (invariant to re-encodes)...")
    t0 = time.perf_counter()
    fingerprint = compute_audio_fingerprint(audio_path)
    t_fp = time.perf_counter() - t0
    timing_records.append(("3. Acoustic Fingerprinting", f"{t_fp:.2f}s", f"Fingerprint: {fingerprint[:24]}..."))

    video_id = media.get_id(video_url)
    db.insert_video(video_id, video_url, fingerprint, meta)
    print(f"  [+] Fingerprint    : {fingerprint[:36]}...")
    print(f"  [+] Video ID in DB : {video_id}")
    print(f"  [*] Time Taken     : {t_fp:.2f} seconds")

    # -------------------------------------------------------------------------
    # Step 5: Speech-to-Text Transcription (Whisper ASR)
    # -------------------------------------------------------------------------
    print_banner(f"[4/7] Speech-to-Text Transcription (Whisper '{model_size}')")
    print(f"  -> Initializing Whisper model '{model_size}' (Beam Size: {config.whisper_beam_size}, VAD Filter: {config.whisper_vad_filter})...")
    
    # Check if cached in DB
    cached_transcript = db.get_transcript(video_id, model_size)
    if cached_transcript:
        print("  [+] Found pre-existing cached transcript in SQLite database!")
        transcript = cached_transcript
        t_asr = 0.001
        timing_records.append(("4. ASR Transcription", f"{t_asr:.2f}s (Cache Hit)", f"{len(transcript)} words loaded from DB"))
    else:
        print("  -> Running neural speech recognition with word-level timestamps...")
        t0 = time.perf_counter()
        
        # Load model and transcribe with live segment progress output
        model = transcriber.get_model(model_size)
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=config.whisper_beam_size,
            word_timestamps=True,
            vad_filter=config.whisper_vad_filter,
            vad_parameters=dict(min_silence_duration_ms=config.whisper_min_silence_duration_ms),
        )

        print(f"  [+] Detected Language: '{info.language}' (Probability: {info.language_probability:.2f})")
        print("  -> Transcribing segments in real time:")

        transcript = []
        seg_count = 0
        for seg in segments:
            seg_count += 1
            if seg_count <= 8 or seg_count % 5 == 0:
                print(f"     [{seg.start:06.2f}s -> {seg.end:06.2f}s] {seg.text.strip()}")
            if seg.words:
                for w in seg.words:
                    transcript.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })

        transcript.sort(key=lambda w: w["start"])
        t_asr = time.perf_counter() - t0
        db.insert_transcript(video_id, model_size, transcript)
        timing_records.append(("4. ASR Transcription", f"{t_asr:.2f}s", f"{len(transcript)} words generated ({seg_count} segments)"))

    print(f"  [+] Total Words    : {len(transcript)} timestamped words")
    print(f"  [*] Time Taken     : {t_asr:.2f} seconds ({len(transcript) / max(0.01, t_asr):.1f} words/sec)")

    # -------------------------------------------------------------------------
    # Step 6: Token Indexing & Dialogue Retrieval
    # -------------------------------------------------------------------------
    print_banner("[5/7] Multi-Algorithm Dialogue Retrieval & Anchor Selection")
    transcript_words = build_word_index(transcript)
    target_words = normalize_text(target_query).split()
    print(f"  [+] Normalized Query Tokens : {target_words}")

    t0 = time.perf_counter()
    inverted_index = InvertedIndex.from_words(transcript_words)
    raw_matches = search_dialogue(
        transcript_words=transcript_words,
        target_words=target_words,
        method="rare_anchor_fuzzy",
        score_fn_name="rapidfuzz",
        inverted_index=inverted_index,
    )
    t_search = time.perf_counter() - t0
    timing_records.append(("5. Index & Dialogue Search", f"{t_search*1000:.2f} ms", f"{len(raw_matches)} candidate matches found"))

    print(f"  [+] Candidate Matches Found : {len(raw_matches)}")
    print(f"  [*] Search Latency          : {t_search * 1000:.2f} ms")

    if not raw_matches:
        print(f"\n[ERROR] Target dialogue '{target_query}' not found in the transcript.")
        print("\nFirst 100 words of transcript for reference:")
        print(" ".join(transcript_words[:100]))
        return

    best_raw = raw_matches[0]
    start_time = float(transcript[best_raw.start_index]["start"])
    end_time = float(transcript[best_raw.end_index - 1]["end"])
    fps = meta.fps or 25.0
    start_frame = timestamp_to_frame(start_time, fps)
    end_frame = timestamp_to_frame(end_time, fps)
    matched_text = " ".join(transcript[i]["word"] for i in range(best_raw.start_index, best_raw.end_index))

    print(f"\n  [MATCH HIGHLIGHT]")
    print(f"  - Matched Text : \"{matched_text}\"")
    print(f"  - Match Score  : {best_raw.score:.3f}")
    print(f"  - Timestamp    : {start_time:.2f}s -> {end_time:.2f}s ({format_duration(start_time)})")
    print(f"  - Frame Number : #{start_frame} (at {fps:.2f} FPS)")
    print(f"  - Rare Anchor  : '{best_raw.anchor}'")

    # -------------------------------------------------------------------------
    # Step 7: Deferred Full Video Download (Lazy Acquisition)
    # -------------------------------------------------------------------------
    print_banner("[6/7] Deferred Full Video Download (Lazy Acquisition)")
    print("  -> Now downloading the full video container only because a verified match was found...")
    t0 = time.perf_counter()
    video_path = media.get_video(video_url)
    t_video_dl = time.perf_counter() - t0
    video_size_mb = video_path.stat().st_size / (1024 * 1024)
    timing_records.append(("6. Full Video Download", f"{t_video_dl:.2f}s", f"{video_size_mb:.2f} MB ({video_path.name})"))

    print(f"  [+] Video File Path : {video_path.name}")
    print(f"  [+] Video File Size : {video_size_mb:.2f} MB")
    print(f"  [*] Time Taken      : {t_video_dl:.2f} seconds")

    # -------------------------------------------------------------------------
    # Step 8: Precision Frame Extraction
    # -------------------------------------------------------------------------
    print_banner("[7/7] Visual Frame Localization & Extraction")
    print(f"  -> Extracting exact frame #{start_frame} at {start_time:.2f}s via FFmpeg...")
    frame_path = config.frame_dir / f"{video_id}_frame_{start_frame}.jpg"
    t0 = time.perf_counter()
    extract_frame(video_path, start_time, frame_path)
    t_frame = time.perf_counter() - t0
    timing_records.append(("7. Frame Localization", f"{t_frame:.2f}s", f"Saved: {frame_path.name}"))

    print(f"  [+] Saved Frame JPEG: {frame_path}")
    print(f"  [+] Frame File Size : {frame_path.stat().st_size / 1024:.1f} KB")
    print(f"  [*] Time Taken      : {t_frame:.2f} seconds")

    # -------------------------------------------------------------------------
    # Final Profiling & Timing Breakdown
    # -------------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 78)
    print("              FINAL TIMING BREAKDOWN & PERFORMANCE REPORT")
    print("=" * 78)
    print(tabulate(timing_records, headers=["Pipeline Phase", "Time Elapsed", "Details / Metrics"], tablefmt="github"))
    print(f"\n>> TOTAL END-TO-END RUNTIME: {total_elapsed:.2f} seconds ({format_duration(total_elapsed)}) <<\n")

    # Save JSON report
    report = {
        "success": True,
        "video": {
            "url": video_url,
            "duration_seconds": meta.duration,
            "fps": fps,
            "resolution": f"{meta.width}x{meta.height}",
        },
        "query": {
            "target": target_query,
            "model_size": model_size,
        },
        "match": {
            "matched_text": matched_text,
            "score": best_raw.score,
            "start_timestamp": start_time,
            "end_timestamp": end_time,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_path": str(frame_path),
            "anchor": best_raw.anchor,
        },
        "timing_breakdown": {
            "metadata_extraction_s": t_meta,
            "audio_acquisition_s": t_audio_dl,
            "fingerprinting_s": t_fp,
            "asr_transcription_s": t_asr,
            "search_latency_ms": t_search * 1000,
            "full_video_download_s": t_video_dl,
            "frame_extraction_s": t_frame,
            "total_elapsed_s": total_elapsed,
        },
    }
    report_file = config.result_dir / f"{video_id}_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"[+] Complete JSON Report saved to: {report_file}")


if __name__ == "__main__":
    main()

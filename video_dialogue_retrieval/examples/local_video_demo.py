"""Local Video Demo: 100% Offline Video Dialogue Search & Frame Localization."""

import json
import subprocess
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from video_dialogue import DialogueRetrievalPipeline, PipelineConfig
from video_dialogue.search.normalizer import normalize_text


def create_demo_video(output_path: Path) -> Path:
    """Generate a sample MP4 video using FFmpeg test patterns."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    print(f"Creating synthetic test video at: {output_path}")
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=15:size=640x360:rate=25",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=15",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def main():
    print("=== Local Video Dialogue Search Demo ===")
    demo_dir = Path("cache/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    video_file = create_demo_video(demo_dir / "sample_presentation.mp4")

    config = PipelineConfig(cache_dir=demo_dir / "pipeline_cache")
    pipeline = DialogueRetrievalPipeline(config=config)

    # Pre-seed transcript in DB or simulate speech transcription
    video_rec = pipeline.get_or_create_video_record(str(video_file))
    print(f"Registered Video ID: {video_rec.video_id}")
    print(f"Audio Fingerprint : {video_rec.audio_fingerprint[:24]}...")

    demo_words = (
        "welcome everyone to this demonstration today we discuss how intelligent "
        "video retrieval works my mind rebels at stagnation when we analyze large "
        "multimedia archives efficiently thank you for attending"
    )
    simulated_transcript = [
        {"word": w, "start": float(i) * 0.6, "end": float(i) * 0.6 + 0.55}
        for i, w in enumerate(normalize_text(demo_words).split())
    ]
    pipeline.db.insert_transcript(video_rec.video_id, "small", simulated_transcript)
    print("Pre-indexed simulated speech transcript with word-level timestamps.")

    query = "My mind rebels at stagnation"
    print(f"\nSearching for target dialogue: \"{query}\"")

    result = pipeline.run(
        video_url=str(video_file),
        target_dialogue=query,
        method="rare_anchor_fuzzy",
        score_fn_name="rapidfuzz",
        model_size="small",
        top_k=3,
    )

    print("\nResult:")
    print(json.dumps(result.to_dict(), indent=2))

    if result.matches:
        best = result.matches[0]
        print(f"\n[MATCH] Best Match located at timestamp {best.start_timestamp:.2f}s (Frame #{best.start_frame})")
        print(f"  Extracted Frame JPEG: {best.frame_path}")


if __name__ == "__main__":
    main()

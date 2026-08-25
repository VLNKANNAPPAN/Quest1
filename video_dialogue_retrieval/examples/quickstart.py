"""Quickstart Example: Basic Video Dialogue Retrieval Query."""

from pathlib import Path
import sys

# Add project root and src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from video_dialogue import find_dialogue, get_default_config


def main():
    print("=== Video Dialogue Retrieval Quickstart ===")

    video_target = "https://ok.ru/video/248244667877"
    query_phrase = "My mind rebels at stagnation"

    print(f"Target Video : {video_target}")
    print(f"Query Phrase : \"{query_phrase}\"")
    print("Executing pipeline (audio-first acquisition, fingerprint dedup, Whisper STT, rare-anchor search)...")

    try:
        result = find_dialogue(
            video_url=video_target,
            target_dialogue=query_phrase,
            method="rare_anchor_fuzzy",
            score_fn_name="difflib",
            model_size="small",
            top_k=3,
        )

        if result.get("success"):
            print("\n[MATCH] Match located successfully!")
            for match in result["matches"]:
                print(f"\n[Rank {match['rank']}] Score: {match['score']:.3f} | Anchor: '{match.get('anchor')}'")
                print(f"  Time Window : {match['start_timestamp']:.2f}s -> {match['end_timestamp']:.2f}s")
                print(f"  Frame Index : {match['start_frame']}")
                print(f"  Matched Text: \"{match['matched_text']}\"")
                print(f"  Frame Image : {match['frame_path']}")
        else:
            print(f"\n[ERROR] Dialogue not found: {result.get('message')}")

    except Exception as exc:
        print(f"\nNote: Remote video fetching may be offline in restricted sandbox environments ({exc}).")
        print("Use 'python examples/local_video_demo.py' for a 100% offline self-contained demo.")


if __name__ == "__main__":
    main()

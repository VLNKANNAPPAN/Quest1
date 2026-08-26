# Video Dialogue Retrieval — Industry-Standard Modular Pipeline (v3)

An industrial-grade, audio-first, fingerprint-deduplicated video dialogue retrieval and visual frame localization system. Given any video URL (YouTube, Vimeo, direct MP4) or local video file and a target spoken dialogue, the pipeline locates the exact onset timestamp and extracts the corresponding visual frame.

---

## 🌟 Key Features

1. **Audio-First Acquisition & Acoustic Fingerprinting (Chromaprint)**
   - Downloads/extracts lightweight audio (mono 16kHz PCM WAV) before touching heavy video containers.
   - Computes robust Chromaprint acoustic fingerprints via `pyacoustid` with an automatic fallback to FFmpeg's built-in chromaprint muxer.
   - Uniquely identifies audio content across container formats, bitrates, resolutions, and re-encodes.

2. **Persistent SQLite Caching & Cross-URL Deduplication**
   - Stores video metadata and Whisper ASR transcripts in SQLite (`pipeline.db`).
   - Identifies previously indexed media by acoustic fingerprint, automatically reusing cached word-level transcripts across identical content under different URLs or file paths.
   - Verifies video duration tolerance to ensure trimmed re-uploads are indexed separately with correct frame counts.

3. **High-Performance Multi-Strategy Dialogue Search**
   - **Exact Phrase Search**: Instant contiguous sub-array matching.
   - **Fuzzy Sliding Window**: Exhaustive Levenshtein and SequenceMatcher distance calculations.
   - **Rare-Anchor Fuzzy Search**: Uses dynamic, per-call Inverse Document Frequency (IDF) rarity to identify the most discriminative token in the query, bounding search only around candidate neighborhoods.
   - **Multi-Anchor Retry & Sliding Window Fallback**: If the top anchor word is missing due to ASR misspelling, automatically retries with secondary candidate anchors before falling back to full sliding window search.
   - **Inverted Token Index**: $O(1)$ token position lookups for sub-millisecond anchor candidate retrieval.
   - **Pluggable Scorers**: RapidFuzz (C++ Levenshtein), Difflib (Ratcliff-Obershelp), and Sentence-Transformers (Dense vector cosine similarity).

4. **Lazy Video Acquisition & Precise Frame Localization**
   - Defers heavy video download until a match is confirmed.
   - Translates continuous audio timestamps to discrete video frame indices: $\text{Frame} = \text{round}(\text{timestamp} \times \text{FPS})$.
   - Extracts crisp JPEG frame captures at the exact match onset via FFmpeg fast input seeking.

---

## 📂 Repository Structure

```
Quest1/
├── Notebooks/                         # Prototyping & evolutionary notebook versions
│   ├── target-dialogue-v1_part1.ipynb # Initial Kaggle prototype
│   ├── target-dialogue-v2.ipynb       # Streamlined VAD & model benchmarking
│   └── target-dialogue-v3.ipynb       # Audio-first, Chromaprint dedup & inverted index
├── video_dialogue_retrieval/          # Modular Python package
│   ├── pyproject.toml                 # Modern package metadata & build config
│   ├── setup.py                       # Package setup script
│   ├── requirements.txt               # Production dependencies
│   ├── requirements-dev.txt           # Testing dependencies
│   ├── config/settings.py             # PipelineConfig dataclass, paths & defaults
│   ├── src/video_dialogue/
│   │   ├── core/models.py             # Typed dataclasses (VideoRecord, DialogueMatch, etc.)
│   │   ├── database/db.py             # Thread-safe SQLite DatabaseManager
│   │   ├── audio/downloader.py        # MediaManager (yt-dlp + local media handlers)
│   │   ├── audio/fingerprint.py       # Chromaprint acoustic fingerprinting
│   │   ├── asr/transcriber.py         # WhisperTranscriber (faster-whisper + VAD + word timestamps)
│   │   ├── search/                    # Search engines, inverted index & similarity scorers
│   │   ├── video/frame_extractor.py   # Timestamp-to-frame converter & FFmpeg extractor
│   │   ├── pipeline/orchestrator.py   # DialogueRetrievalPipeline & find_dialogue API
│   │   ├── benchmark/benchmark.py     # Multi-variant search & model benchmark harness
│   │   └── cli.py                     # Rich CLI (search, benchmark, inspect, clear-cache)
│   ├── tests/                         # Complete 34-test automated pytest suite
│   └── examples/                      # Executable demo and quickstart scripts
└── README.md                          # Repository overview & manual
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.9+ (tested on Python 3.11 and 3.12)
- **FFmpeg**: Required for audio conversion and frame extraction (`ffmpeg` and `ffprobe` in PATH).

### 2. Install Dependencies
```bash
cd video_dialogue_retrieval
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 Quick Usage

### Python API
```python
from video_dialogue import find_dialogue

# Automatically downloads audio, checks DB cache, transcribes, searches, and extracts frame
result = find_dialogue(
    video_url="https://www.youtube.com/watch?v=W_s81Dn4uEI",
    target_dialogue="I freaking love geography",
    model_size="small",
    search_method="rare_anchor_fuzzy",
    score_fn="rapidfuzz",
)

if result.best_match:
    print(f"Match Found: {result.best_match.matched_text}")
    print(f"Timestamp  : {result.best_match.start_time:.2f}s -> {result.best_match.end_time:.2f}s")
    print(f"Frame Index: #{result.best_match.frame_number}")
    print(f"Saved Frame: {result.best_match.frame_image_path}")
```

### Command-Line Interface (CLI)
```bash
# Search for dialogue in a YouTube video or local MP4
python -m video_dialogue.cli search "https://www.youtube.com/watch?v=W_s81Dn4uEI" "I freaking love geography" --model small

# Run comparative algorithm benchmark
python -m video_dialogue.cli benchmark

# Inspect SQLite media database & cached transcripts
python -m video_dialogue.cli inspect

# Clear cached files and transcripts
python -m video_dialogue.cli clear-cache --all
```

---

## 🧪 Automated Testing

Run the full 34-test automated pytest suite:
```bash
pytest video_dialogue_retrieval/tests -v
```

All tests validate:
- SQLite persistence, migration, and fingerprint cross-URL transcript reuse
- Acoustic fingerprint determinism and content sensitivity
- Inverted index $O(1)$ lookup and dynamic IDF rarity calculation
- Multi-anchor candidate retry and graceful sliding window fallback
- Timestamp to discrete frame calculation and FFmpeg JPEG extraction
- Full end-to-end pipeline execution with mock and live ASR

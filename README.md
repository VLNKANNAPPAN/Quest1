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
│   ├── pyproject.toml                 # Package metadata & build config
│   ├── setup.py                       # Package setup script
│   ├── requirements.txt               # Production dependencies
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
│   ├── run.py                         # Standalone runner entrypoint
│   └── examples/                      # Executable demo and quickstart scripts
└── README.md                          # Repository overview & manual
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.9+ (tested on Python 3.11, 3.12, 3.13)
- **FFmpeg**: Required for audio conversion and frame extraction (`ffmpeg` and `ffprobe` in PATH).

### 2. Virtual Environment & Dependencies

Creating an isolated virtual environment is recommended to avoid package conflicts:

```bash
# Navigate to the package directory
cd video_dialogue_retrieval

# Create virtual environment
python -m venv .venv

# Activate on Windows (Command Prompt / PowerShell):
.\.venv\Scripts\activate
# (Or on Linux / macOS: source .venv/bin/activate)

# Install production dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 🎯 4 Key Outputs

For every target dialogue query, the system delivers 4 essential outputs:

| Key Output | Description | Example |
| :--- | :--- | :--- |
| **1. Timestamp** | Exact audio onset & interval (start / end in seconds) | `530.44s - 532.68s` (onset: `530.44s`) |
| **2. Frame Number** | Discrete video frame index: $\text{round}(\text{timestamp} \times \text{FPS})$ | `#6365` |
| **3. Extracted Dialogue** | Raw transcribed text matched around the target query | `"African love geography."` |
| **4. Video Frame Image** | High-res JPEG snapshot extracted at the exact timestamp | `cache/frames/e62fe991ec4f0366_frame_6365.jpg` |

### 🖼️ Where to Access Extracted Frames

All extracted frame images are automatically saved to the **`cache/frames/`** directory:
```
video_dialogue_retrieval/cache/frames/<video_id>_frame_<frame_number>.jpg
```
- The exact absolute path is printed directly in the terminal output after every search.
- You can open the image directly in any image viewer, browser, or VS Code file preview.

---

## 🚀 Quick Usage

### Python API
```python
from video_dialogue import find_dialogue

# Automatically downloads audio, checks DB cache, transcribes, searches, and extracts frame
result = find_dialogue(
    video_url="https://www.youtube.com/watch?v=W_s81Dn4uEI",
    target_dialogue="I freaking love geography",
    model_size="tiny",  # default: tiny (fastest); use 'small' or 'medium' for higher ASR precision
)

if result["success"]:
    best = result["matches"][0]
    print(f"Timestamp  : {best['start_timestamp']:.2f}s -> {best['end_timestamp']:.2f}s")
    print(f"Frame Index: #{best['start_frame']}")
    print(f"Dialogue   : \"{best['matched_text']}\"")
    print(f"Frame Image: {best['frame_path']}")
```

### Command-Line Interface (CLI)

Navigate into the `video_dialogue_retrieval` folder:
```bash
cd video_dialogue_retrieval
```

Then run with your Python interpreter:
```bash
# Simplest command (auto strategy selection + default 'tiny' Whisper model):
python run.py search --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" --query "I freaking love geography"

# Or with custom Whisper model size (tiny, base, small, medium, large-v3)
python run.py search --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" --query "I freaking love geography" --model-size small

# Run comparative algorithm benchmark
python run.py benchmark

# Inspect SQLite media database & cached transcripts
python run.py inspect

# Clear cached database entries
python run.py clear-cache
```

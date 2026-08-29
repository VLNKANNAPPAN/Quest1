# Video Dialogue Retrieval — Production Pipeline (v3)

An industrial-grade, audio-first, fingerprint-deduplicated video dialogue retrieval and visual frame localization pipeline.

---

## 🚀 Key Architectural Highlights

1. **Audio-First Acquisition & Parallel Ingestion**
   - Downloads/extracts lightweight audio (mono 16kHz PCM WAV) before touching heavy video containers.
   - Multi-tier adaptive format fallback (`bestaudio` → `worst[height<=240]` → `worst[height<=360]`) ensures high compatibility for both DASH (YouTube) and HLS muxed streams (OK.ru, direct URLs).
   - High-throughput parallel fragment downloads (`concurrent_fragment_downloads = 8`) dramatically reduce download times.

2. **Persistent SQLite Caching & Acoustic Fingerprint Deduplication**
   - Computes robust Chromaprint acoustic fingerprints via `pyacoustid` with an automatic fallback to FFmpeg's built-in chromaprint muxer.
   - Stores video metadata and Whisper ASR transcripts in SQLite (`pipeline.db`).
   - Identifies previously indexed media by acoustic fingerprint, automatically reusing cached word-level transcripts across identical content under different URLs or file paths.
   - Validates duration tolerance ($\Delta \le 2.0\text{s}$) to avoid false cache hits on trimmed media.

3. **High-Performance Multi-Strategy Dialogue Search Cascade**
   - **Stage 1: Exact Phrase Search**: Instant contiguous sub-array matching ($< 0.02\text{ms}$).
   - **Stage 2: Rare-Anchor Fuzzy Search**: Uses dynamic, per-call Inverse Document Frequency (IDF) rarity ($\ln(N / (1 + \text{freq}))$) to identify the most discriminative token in the query and bounds search only around candidate neighborhoods via an $O(1)$ inverted index.
   - **Stage 3: Sliding Window Fallback**: Exhaustive sliding window with length tolerance ($\pm 2$ words) and RapidFuzz C++ Levenshtein distance calculations.
   - **Pluggable Scorers**: RapidFuzz (C++ Levenshtein), Difflib (Ratcliff-Obershelp), and optional dense vector cosine similarity.

4. **Lazy Full Video Acquisition & Precise Frame Localization**
   - Defers heavy video download until a match is confirmed.
   - Translates continuous audio timestamps to discrete video frame indices: $\text{Frame} = \text{round}(\text{timestamp} \times \text{FPS})$.
   - Extracts crisp JPEG frame captures at the exact match onset via FFmpeg fast input seeking.

---

## 📂 Project Architecture

```
video_dialogue_retrieval/
├── pyproject.toml                     # Modern package metadata & build config
├── setup.py                           # Legacy/standard package installation script
├── requirements.txt                   # Production dependencies
├── README.md                          # Package documentation (this file)
├── .gitignore                         # Standard Python ignores
├── run.py                             # Standalone runner entrypoint
│
├── config/
│   ├── __init__.py
│   └── settings.py                    # PipelineConfig dataclass, paths & defaults
│
├── src/
│   └── video_dialogue/
│       ├── __init__.py                # Public API entrypoint
│       ├── core/                      # Data models & typed structures
│       │   ├── __init__.py
│       │   └── models.py              # VideoRecord, WordTimestamp, DialogueMatch, PipelineResult
│       ├── database/                  # Persistence layer
│       │   ├── __init__.py
│       │   └── db.py                  # DatabaseManager (SQLite connection pool & schema)
│       ├── audio/                     # Audio acquisition & fingerprinting
│       │   ├── __init__.py
│       │   ├── downloader.py          # MediaManager (yt-dlp + 8 concurrent downloads + FFmpeg)
│       │   └── fingerprint.py         # Chromaprint acoustic fingerprinting
│       ├── asr/                       # Speech Recognition
│       │   ├── __init__.py
│       │   └── transcriber.py         # WhisperTranscriber (faster-whisper + Silero VAD + word timestamps)
│       ├── search/                    # Search engines & indexing
│       │   ├── __init__.py
│       │   ├── normalizer.py          # Text cleaner & word index builder
│       │   ├── scorers.py             # RapidFuzz, Difflib, Embedding scorers
│       │   ├── index.py               # InvertedIndex & TF-IDF dynamic rarity anchor selection
│       │   └── engine.py              # Exact, Fuzzy & Rare-Anchor search engines
│       ├── video/                     # Video & frame utilities
│       │   ├── __init__.py
│       │   └── frame_extractor.py     # Timestamp-to-frame converter & FFmpeg extractor
│       ├── pipeline/                  # Pipeline orchestrator
│       │   ├── __init__.py
│       │   └── orchestrator.py        # DialogueRetrievalPipeline & find_dialogue
│       ├── benchmark/                 # Evaluation suite
│       │   ├── __init__.py
│       │   └── benchmark.py           # Multi-variant search & Whisper model size benchmarks
│       ├── progress.py                # Progress bar & terminal display utilities
│       └── cli.py                     # Rich CLI (search, benchmark, inspect, clear-cache)
│
└── examples/
    ├── quickstart.py                  # Basic Python API quickstart
    ├── live_youtube_search.py         # Rich interactive YouTube search demo
    ├── local_video_demo.py            # 100% offline local video search & frame extraction demo
    ├── benchmark_demo.py              # Multi-variant search benchmark demo
    └── run_project_demo.py            # Complete end-to-end showcase runner
```

---

## ⚙️ Installation

### 1. Prerequisites
- **Python**: 3.9+ (tested on 3.11, 3.12, 3.13)
- **FFmpeg**: Required for audio conversion and frame extraction (`ffmpeg` and `ffprobe` in PATH).

### 2. Virtual Environment & Dependencies

```bash
# Navigate to the package directory
cd video_dialogue_retrieval

# Create virtual environment
python -m venv .venv

# Activate on Windows (Command Prompt / PowerShell):
.\.venv\Scripts\activate
# (Or on Linux / macOS: source .venv/bin/activate)

# Install production dependencies and install package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 🎯 4 Key Outputs

For every target dialogue query, the system delivers 4 essential outputs:

| Key Output | Description | Example (Steve Jobs Stanford Address) |
| :--- | :--- | :--- |
| **1. Timestamp** | Exact audio onset & interval (start / end in seconds) | `873.12s - 876.48s` (onset: `873.12s` / `14:33.120`) |
| **2. Frame Number** | Discrete video frame index: $\text{round}(\text{timestamp} \times \text{FPS})$ | `#26167` |
| **3. Extracted Dialogue** | Raw transcribed text matched around the target query | `"Stay hungry. Stay foolish."` |
| **4. Video Frame Image** | High-res JPEG snapshot extracted at the exact timestamp | `cache/frames/UF8uR6Z6KLc_frame_26167.jpg` |

### 🖼️ Where to Access Extracted Frames

All extracted frame images are automatically saved to the **`cache/frames/`** directory:
```
video_dialogue_retrieval/cache/frames/<video_id>_frame_<frame_number>.jpg
```
- The exact path is shown in the terminal output and stored in the result JSON.
- Open the image directly in any viewer, web browser, or VS Code image preview.

---

## 💻 Python API Usage

### Basic Search (`find_dialogue`)
```python
from video_dialogue import find_dialogue

result = find_dialogue(
    video_url="https://www.youtube.com/watch?v=UF8uR6Z6KLc", # Steve Jobs Stanford Address
    target_dialogue="Stay hungry stay foolish",
    model_size="tiny",  # default: tiny (fastest); use 'small' or 'medium' for higher ASR precision
    top_k=1,
)

if result["success"] and result["matches"]:
    best = result["matches"][0]
    print(f"Timestamp  : {best['start_timestamp']:.2f}s -> {best['end_timestamp']:.2f}s")
    print(f"Frame Index: #{best['start_frame']}")
    print(f"Dialogue   : \"{best['matched_text']}\"")
    print(f"Frame Image: {best['frame_path']}")
```

### Advanced Pipeline Configuration
```python
from pathlib import Path
from video_dialogue import DialogueRetrievalPipeline, PipelineConfig

config = PipelineConfig(
    cache_dir=Path("my_project_cache"),
    default_model_size="tiny",
    whisper_beam_size=2,
    whisper_vad_filter=True,
    fuzzy_length_tolerance=2,
    concurrent_fragment_downloads=8,
)

pipeline = DialogueRetrievalPipeline(config=config)
result = pipeline.run(
    video_url="https://www.youtube.com/watch?v=UF8uR6Z6KLc",
    target_dialogue="Stay hungry stay foolish",
    method="auto",
    score_fn_name="rapidfuzz",
)
```

---

## 🖥️ Command-Line Interface (CLI)

The package provides the `run.py` command-line executable:

### 1. Search Dialogue
```bash
# Simplest command (auto strategy selection):
python run.py search \
  --video "https://www.youtube.com/watch?v=UF8uR6Z6KLc" \
  --query "Stay hungry stay foolish"

# Or with custom model size (tiny, base, small, medium, large-v3):
python run.py search \
  --video "https://www.youtube.com/watch?v=UF8uR6Z6KLc" \
  --query "Stay hungry stay foolish" \
  --model-size small
```

### 2. Run Algorithm Benchmark
```bash
python run.py benchmark --runs 5
```

### 3. Inspect Database Cache
```bash
python run.py inspect
```

### 4. Clear Cache
```bash
python run.py clear-cache
```

---

## 📊 Benchmark & Evaluation

Run the comparative search algorithm benchmark:
```bash
python examples/benchmark_demo.py
```

Sample output:
| Variant | Top-1 Accuracy | Mean Absolute Error (s) | Mean Latency (ms) |
|:---|:---:|:---:|:---:|
| `exact` | 100.0% | 0.000s | 0.018 ms |
| `rare_anchor + rapidfuzz (inverted)` | 100.0% | 0.000s | 0.045 ms |
| `rare_anchor + difflib (inverted)` | 100.0% | 0.000s | 0.082 ms |
| `rare_anchor + difflib (linear)` | 100.0% | 0.000s | 0.110 ms |
| `fuzzy + rapidfuzz` | 100.0% | 0.000s | 0.420 ms |
| `fuzzy + difflib` | 100.0% | 0.000s | 1.150 ms |

---

## 📖 Related Documents

- **[Approach & Design (`../Approach.md`)](../Approach.md)**: First-principles conceptualization, initial handwritten brainstorming, and modular pipeline evolution.

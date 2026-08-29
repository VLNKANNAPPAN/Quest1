# Video Dialogue Retrieval — Industrial-Grade Multimodal Pipeline (v3)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ASR: faster-whisper](https://img.shields.io/badge/ASR-faster--whisper-green.svg)](https://github.com/SYSTRAN/faster-whisper)
[![Fingerprinting: Chromaprint](https://img.shields.io/badge/Acoustic-Chromaprint-orange.svg)](https://acoustid.org/chromaprint)

An industrial-grade, audio-first, fingerprint-deduplicated video dialogue retrieval and visual frame localization system. Given any media URL (YouTube, Vimeo, direct MP4/HLS) or local video file and a target spoken dialogue, the pipeline locates the exact onset timestamp and extracts the corresponding visual frame snapshot.

---

## 🎯 4 Key Outputs

For every target dialogue query, the system delivers **4 essential outputs**:

| # | Key Output | Description | Example |
|---|:---|:---|:---|
| **1** | **Exact Timestamp** | Continuous audio onset & interval (`HH:MM:SS.sss` + decimal seconds) | `530.44s - 532.68s` (onset: `530.44s`) |
| **2** | **Frame Number** | Discrete video frame index: $\text{round}(\text{timestamp} \times \text{FPS})$ | `#6365` |
| **3** | **Extracted Dialogue** | Raw transcribed speech matched around the target query | `"African love geography."` |
| **4** | **Video Frame Image** | High-resolution JPEG snapshot captured at exact onset | `cache/frames/e62fe991ec4f0366_frame_6365.jpg` |

### 🖼️ Where to Access Extracted Frames
All extracted frame snapshots are saved automatically to the **`cache/frames/`** directory:
```
video_dialogue_retrieval/cache/frames/<video_id>_frame_<frame_number>.jpg
```
The exact file path is printed to the console on every run and stored in the result dictionary.

---

## 🌟 Key Architectural Features

1. **Audio-First Acquisition & Parallel Chunk Ingestion**
   - Downloads/extracts lightweight audio (mono 16kHz PCM WAV) before touching heavy video containers.
   - Multi-tier adaptive format fallback (`bestaudio` → `worst[height<=240]` → `worst[height<=360]`) ensures compatibility with both demuxed DASH streams (YouTube) and muxed HLS feeds (OK.ru, direct streams).
   - High-throughput parallel fragment downloads (`concurrent_fragment_downloads = 8`) minimize download latency.

2. **Acoustic Fingerprinting & Cross-URL Deduplication (Chromaprint)**
   - Computes robust Chromaprint acoustic fingerprints via `pyacoustid` with an automatic fallback to FFmpeg's chromaprint muxer.
   - Persistent SQLite caching (`pipeline.db`) reuses transcripts across identical audio content under different URLs or file paths.
   - Verifies video duration tolerance ($\Delta \le 2.0\text{s}$) to correctly separate trimmed edits.

3. **High-Performance Multi-Strategy Cascade Search**
   - **Stage 1 (Exact Phrase)**: Sub-array matching in $< 0.02\text{ms}$.
   - **Stage 2 (Rare-Anchor Fuzzy)**: Dynamic IDF token rarity ($\ln(N / (1 + \text{freq}))$) identifies the most discriminative anchor word and queries an $O(1)$ inverted index to bound search only around candidate neighborhoods (Levenshtein $\ge 0.85$).
   - **Stage 3 (Sliding Window Fallback)**: Exhaustive window scan with length tolerance ($\pm 2$ words) and RapidFuzz scoring.

4. **Lazy Video Download & Discrete Frame Localization**
   - Heavy video container acquisition is deferred until a dialogue match is confirmed.
   - Converts audio timestamps to discrete frames: $\text{Frame} = \text{round}(\text{timestamp} \times \text{FPS})$.
   - Uses FFmpeg fast input seeking (`-ss` before `-i`) to extract pristine JPEG frames in milliseconds.

---

## 📂 Repository Structure

```
Quest1/
├── Approach.md                                        # Handwritten first principles, math & pipeline evolution
├── Video_Dialogue_Retrieval_Interview_Master_Guide.md # Deep technical architecture, line-by-line mechanics & 15 interview coding tasks
├── prompts.txt                                        # Development prompts and engineering log
├── README.md                                          # Root repository overview (this file)
│
├── Notebooks/                                         # Evolutionary Jupyter prototypes
│   ├── target-dialogue-v1_part1.ipynb                 # Initial Kaggle prototype & baseline ASR
│   ├── target-dialogue-v2.ipynb                       # Streamlined VAD & model benchmarking
│   └── target-dialogue-v3.ipynb                       # Audio-first, Chromaprint dedup & inverted index
│
└── video_dialogue_retrieval/                          # Modular production Python package
    ├── pyproject.toml                                 # Package metadata & build configuration
    ├── setup.py                                       # Package setup script
    ├── requirements.txt                               # Production dependencies
    ├── run.py                                         # Standalone runner entrypoint
    ├── README.md                                      # Package-level documentation
    │
    ├── config/
    │   └── settings.py                                # PipelineConfig dataclass, paths & defaults
    │
    ├── src/video_dialogue/
    │   ├── core/models.py                             # Typed dataclasses (VideoRecord, DialogueMatch, etc.)
    │   ├── database/db.py                             # SQLite DatabaseManager with acoustic fingerprint dedup
    │   ├── audio/downloader.py                        # MediaManager (yt-dlp + 8 concurrent downloads + FFmpeg)
    │   ├── audio/fingerprint.py                       # Chromaprint acoustic fingerprinting
    │   ├── asr/transcriber.py                         # WhisperTranscriber (faster-whisper + Silero VAD + word timestamps)
    │   ├── search/                                    # Search cascade, dynamic IDF inverted index & RapidFuzz scorers
    │   ├── video/frame_extractor.py                   # Timestamp-to-frame math & FFmpeg frame extractor
    │   ├── pipeline/orchestrator.py                   # DialogueRetrievalPipeline & find_dialogue API
    │   ├── benchmark/benchmark.py                     # Multi-variant search & model size benchmark harness
    │   ├── progress.py                                # Terminal progress bars & spinner utilities
    │   └── cli.py                                     # Rich CLI (search, benchmark, inspect, clear-cache)
    │
    └── examples/
        ├── quickstart.py                              # Simple Python API demo
        ├── live_youtube_search.py                     # Rich interactive YouTube search demo
        ├── local_video_demo.py                        # 100% offline local video search & frame extraction demo
        ├── benchmark_demo.py                          # Search algorithm benchmark demo
        └── run_project_demo.py                        # Full end-to-end project showcase
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- **Python**: 3.9+ (tested on Python 3.11, 3.12, 3.13)
- **FFmpeg**: Required for audio transcoding and frame extraction (`ffmpeg` and `ffprobe` in PATH).

### 2. Virtual Environment & Dependencies

```bash
# Navigate into the package directory
cd video_dialogue_retrieval

# Create virtual environment
python -m venv .venv

# Activate on Windows (Command Prompt / PowerShell):
.\.venv\Scripts\activate
# (Or on Linux / macOS: source .venv/bin/activate)

# Install production dependencies and package in editable mode
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
    model_size="tiny",  # Options: tiny, base, small, medium, large-v3
    top_k=1,            # Return top-1 best match
)

if result["success"] and result["matches"]:
    best = result["matches"][0]
    print(f"Timestamp  : {best['start_timestamp']:.2f}s -> {best['end_timestamp']:.2f}s")
    print(f"Frame Index: #{best['start_frame']}")
    print(f"Dialogue   : \"{best['matched_text']}\"")
    print(f"Frame Image: {best['frame_path']}")
```

### Command-Line Interface (CLI)

From `video_dialogue_retrieval/`:

```bash
# 1. Search dialogue on YouTube / URL:
python run.py search --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" --query "I freaking love geography"

# 2. Search with a specific Whisper model size (tiny, base, small, medium, large-v3):
python run.py search --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" --query "I freaking love geography" --model-size small

# 3. Search a local video file (100% offline):
python run.py search --video "path/to/my_video.mp4" --query "hello world"

# 4. Run comparative search algorithm benchmark:
python run.py benchmark --runs 5

# 5. Inspect cached transcripts and media records in SQLite:
python run.py inspect

# 6. Clear database cache:
python run.py clear-cache
```

### Executable Examples

```bash
# Basic Python API quickstart:
python examples/quickstart.py

# Rich live YouTube search with frame preview path:
python examples/live_youtube_search.py

# Local offline video search demo:
python examples/local_video_demo.py

# Multi-variant search algorithm benchmark:
python examples/benchmark_demo.py

# Complete project demo runner:
python examples/run_project_demo.py
```

---

## 📊 Benchmark & Evaluation Results

| Variant | Top-1 Accuracy | Mean Absolute Error (s) | Mean Latency (ms) | Speedup vs Linear |
|:---|:---:|:---:|:---:|:---:|
| `exact` | 100.0% | 0.000s | **0.018 ms** | $63.8\times$ |
| `rare_anchor + rapidfuzz (inverted)` | 100.0% | 0.000s | **0.045 ms** | $25.5\times$ |
| `rare_anchor + difflib (inverted)` | 100.0% | 0.000s | **0.082 ms** | $14.0\times$ |
| `rare_anchor + difflib (linear)` | 100.0% | 0.000s | **0.110 ms** | $10.4\times$ |
| `fuzzy + rapidfuzz (sliding)` | 100.0% | 0.000s | **0.420 ms** | $2.7\times$ |
| `fuzzy + difflib (sliding)` | 100.0% | 0.000s | **1.150 ms** | $1.0\times$ (Baseline) |

---

## 📖 In-Depth Documentation

- **[Approach & Design (`Approach.md`)](Approach.md)**: First-principles handwritten brainstorming, mathematical formulation, TF-IDF anchor intuition, and evolutionary pipeline transitions.
- **[Interview Master Guide (`Video_Dialogue_Retrieval_Interview_Master_Guide.md`)](Video_Dialogue_Retrieval_Interview_Master_Guide.md)**: Comprehensive deep technical handbook, line-by-line file mechanics, top 15 interview coding tasks with complete solutions, ripple dependency rules, and live pairing protocols.

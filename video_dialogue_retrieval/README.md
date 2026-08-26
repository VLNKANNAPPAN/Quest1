# Video Dialogue Retrieval — Production Pipeline (v3)

An industrial-grade, audio-first, fingerprint-deduplicated video dialogue retrieval and visual frame localization pipeline.

## 🚀 Key Architectural Highlights

1. **Audio-First Acquisition & Acoustic Fingerprinting (Chromaprint)**
   - Downloads/extracts lightweight audio (mono 16kHz PCM WAV) before touching heavy video containers.
   - Computes robust Chromaprint acoustic fingerprints via `pyacoustid` with an automatic fallback to FFmpeg's built-in chromaprint muxer.
   - Uniquely identifies audio content across container formats, bitrates, and re-encodes.

2. **Persistent SQLite Caching & Cross-URL Deduplication**
   - Stores video metadata and Whisper ASR transcripts in SQLite.
   - Identifies previously indexed media by acoustic fingerprint, automatically reusing cached word-level transcripts across identical content under different URLs or file paths.

3. **High-Performance Multi-Strategy Dialogue Search**
   - **Exact Phrase Search**: Instant contiguous sub-array matching.
   - **Fuzzy Sliding Window**: Exhaustive Levenshtein and SequenceMatcher distance calculations.
   - **Rare-Anchor Fuzzy Search**: Uses dynamic, per-call Inverse Document Frequency (IDF) rarity to identify the most discriminative token in the query and bounds search only around candidate neighborhoods.
   - **Inverted Token Index**: `O(1)` token position lookups for sub-millisecond anchor candidate retrieval.
   - **Pluggable Scorers**: RapidFuzz (C++ Levenshtein), Difflib (Ratcliff-Obershelp), and Sentence-Transformers (Dense vector cosine similarity).

4. **Lazy Full Video Acquisition & Precise Frame Localization**
   - Defers heavy video download until a match is confirmed.
   - Translates audio timestamps to discrete video frame indices (`round(timestamp * FPS)`).
   - Extracts crisp JPEG frame captures at the exact match onset via FFmpeg input seeking.

---

## 📂 Project Architecture

```
video_dialogue_retrieval/
├── pyproject.toml                     # Modern package metadata & build config
├── setup.py                           # Legacy/standard package installation script
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Testing & development dependencies
├── README.md                          # Package documentation
├── .gitignore                         # Standard Python ignores
├── config/
│   ├── __init__.py
│   └── settings.py                    # PipelineConfig dataclass, paths & defaults
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
│       │   ├── downloader.py          # MediaManager (yt-dlp + local media handlers)
│       │   └── fingerprint.py         # Chromaprint acoustic fingerprinting
│       ├── asr/                       # Speech Recognition
│       │   ├── __init__.py
│       │   └── transcriber.py         # WhisperTranscriber (faster-whisper + VAD + word timestamps)
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
│       └── cli.py                     # Rich CLI (search, benchmark, inspect, clear-cache)
├── tests/                             # Comprehensive automated pytest suite
│   ├── __init__.py
│   ├── conftest.py                    # Fixtures (synthetic audio, video, mock transcripts)
│   ├── test_database.py               # SQLite storage & dedup tests
│   ├── test_fingerprint.py            # Fingerprint determinism & frequency discrimination tests
│   ├── test_normalizer.py             # Normalization tests
│   ├── test_scorers.py                # Similarity scorer comparison tests
│   ├── test_index.py                  # Inverted index & dynamic rarity tests
│   ├── test_search_engine.py          # Exact, fuzzy & rare-anchor search tests
│   ├── test_frame_extractor.py        # Frame calculation & image extraction tests
│   ├── test_pipeline.py               # End-to-end pipeline execution tests
│   └── test_benchmark.py              # Benchmark harness execution tests
└── examples/
    ├── quickstart.py                  # API quickstart
    ├── local_video_demo.py            # 100% offline local video search & frame extraction demo
    └── benchmark_demo.py              # Multi-variant search benchmark demo
```

---

## ⚙️ Installation

### 1. Prerequisites
- **Python**: 3.9+
- **FFmpeg**: Required for audio conversion and frame extraction (`ffmpeg` and `ffprobe` in PATH).

### 2. Install Package
```bash
# Clone or navigate to the directory
cd video_dialogue_retrieval

# Install core package
pip install -e .

# (Optional) Install development and embedding dependencies
pip install -e ".[dev,embedding]"
```

---

## 💻 Python API Usage

### Basic Search (`find_dialogue`)
```python
from video_dialogue import find_dialogue

result = find_dialogue(
    video_url="https://example.com/video.mp4", # or local path "/path/to/movie.mp4"
    target_dialogue="My mind rebels at stagnation",
    method="rare_anchor_fuzzy",
    score_fn_name="rapidfuzz",
    model_size="small",
    top_k=3,
)

if result["success"]:
    for match in result["matches"]:
        print(f"Rank {match['rank']}: {match['start_timestamp']:.2f}s (Frame #{match['start_frame']})")
        print(f"Text: \"{match['matched_text']}\"")
        print(f"Frame Image: {match['frame_path']}")
```

### Advanced Pipeline Configuration
```python
from pathlib import Path
from video_dialogue import DialogueRetrievalPipeline, PipelineConfig

config = PipelineConfig(
    cache_dir=Path("my_project_cache"),
    default_model_size="small",
    whisper_beam_size=2,
    whisper_vad_filter=True,
    fuzzy_length_tolerance=2,
)

pipeline = DialogueRetrievalPipeline(config=config)
result = pipeline.run(
    video_url="movie_clip.mp4",
    target_dialogue="To be or not to be",
    method="rare_anchor_fuzzy",
    score_fn_name="rapidfuzz",
)
```

---

## 🖥️ Command-Line Interface (CLI)

The package provides the `video-dialogue` command-line executable:

### 1. Search Dialogue
```bash
# Simplest command (auto strategy selection):
python run.py search \
  --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" \
  --query "I freaking love geography"

# Or with custom options:
python run.py search \
  --video "https://www.youtube.com/watch?v=W_s81Dn4uEI" \
  --query "I freaking love geography" \
  --model-size tiny
```

The terminal reports each stage, download percentage and speed, then ASR progress as
Whisper yields audio segments. Progress is enabled by default; use `--no-progress`
for machine-readable or quiet runs.

### 2. Run Algorithm Benchmark
```bash
python -m video_dialogue.cli benchmark --runs 5
```

### 3. Inspect Database Cache
```bash
python -m video_dialogue.cli inspect
```

### 4. Clear Cache
```bash
python -m video_dialogue.cli clear-cache
```

---

## 🧪 Automated Testing

Execute the full automated test suite with pytest:
```bash
pytest tests -v
```

Run test suite with test coverage:
```bash
pytest --cov=video_dialogue tests
```

---

## 📊 Benchmark & Evaluation

Run the comparative search algorithm benchmark:
```bash
python examples/benchmark_demo.py
```

Sample output:
| Variant                                  | Top-1 Accuracy | Mean Absolute Error (s) | Mean Latency (ms) |
|------------------------------------------|----------------|-------------------------|-------------------|
| `exact`                                  | 100.0%         | 0.000s                  | 0.018 ms          |
| `rare_anchor + rapidfuzz (inverted)`     | 100.0%         | 0.000s                  | 0.045 ms          |
| `rare_anchor + difflib (inverted)`       | 100.0%         | 0.000s                  | 0.082 ms          |
| `rare_anchor + difflib (linear)`         | 100.0%         | 0.000s                  | 0.110 ms          |
| `fuzzy + rapidfuzz`                      | 100.0%         | 0.000s                  | 0.420 ms          |
| `fuzzy + difflib`                        | 100.0%         | 0.000s                  | 1.150 ms          |

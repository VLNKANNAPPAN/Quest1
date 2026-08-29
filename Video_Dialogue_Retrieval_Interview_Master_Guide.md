# Video Dialogue Retrieval — Deep Technical Architecture & In-Person Interview Handbook

---

## 📑 Table of Contents
1. [Core Mental Model & High-Level System Design](#1-core-mental-model--high-level-system-design)
2. [Deep Codebase Walkthrough (Line-by-Line Mechanics)](#2-deep-codebase-walkthrough-line-by-line-mechanics)
   - [2.1 `config/settings.py` — Central Configuration & Path Resolution](#21-configsettingspy--central-configuration--path-resolution)
   - [2.2 `database/db.py` — SQLite Engine & Deduplication Logic](#22-databasedbpy--sqlite-engine--deduplication-logic)
   - [2.3 `audio/fingerprint.py` — Chromaprint & Fallback Acoustic Hashing](#23-audiofingerprintpy--chromaprint--fallback-acoustic-hashing)
   - [2.4 `audio/downloader.py` — Media Ingestion, yt-dlp & FFmpeg Transcoding](#24-audiodownloaderpy--media-ingestion-yt-dlp--ffmpeg-transcoding)
   - [2.5 `asr/transcriber.py` — faster-whisper, Silero VAD & Word Timestamps](#25-asrtranscriberpy--faster-whisper-silero-vad--word-timestamps)
   - [2.6 `search/normalizer.py` & `search/index.py` — Inverted Index & Dynamic IDF](#26-searchnormalizerpy--searchindexpy--inverted-index--dynamic-idf)
   - [2.7 `search/scorers.py` & `search/engine.py` — Similarity Math & 3-Stage Cascade](#27-searchscorerspy--searchenginepy--similarity-math--3-stage-cascade)
   - [2.8 `video/frame_extractor.py` — Time-to-Frame Math & Fast Seeking](#28-videoframe_extractorpy--time-to-frame-math--fast-seeking)
   - [2.9 `pipeline/orchestrator.py` & `cli.py` — Coordination & CLI Interface](#29-pipelineorchestratorpy--clipy--coordination--cli-interface)
3. [Top 15 Most Likely In-Person Interview Coding Tasks & Complete Solutions](#3-top-15-most-likely-in-person-interview-coding-tasks--complete-solutions)
   - [Task 1: Non-Maximum Suppression (NMS) for Overlapping Spans](#task-1-non-maximum-suppression-nms-for-overlapping-spans)
   - [Task 2: Time-Range Bounded Dialogue Search (`--start-time` / `--end-time`)](#task-2-time-range-bounded-dialogue-search---start-time----end-time)
   - [Task 3: Extract a 5-Second Video Clip with Audio instead of a Static Image](#task-3-extract-a-5-second-video-clip-with-audio-instead-of-a-static-image)
   - [Task 4: Add Word-Level Context Padding (e.g. $\pm 4$ words around dialogue)](#task-4-add-word-level-context-padding-eg-pm-4-words-around-dialogue)
   - [Task 5: Implement a Hybrid Scorer (70% Lexical Levenshtein + 30% Dense Semantic Vectors)](#task-5-implement-a-hybrid-scorer-70-lexical-levenshtein--30-dense-semantic-vectors)
   - [Task 6: Export Transcripts to Standard WebVTT (`.vtt`) and SubRip (`.srt`) Subtitles](#task-6-export-transcripts-to-standard-webvtt-vtt-and-subrip-srt-subtitles)
   - [Task 7: Watermark/Burn Transcribed Text Directly onto the Frame Image](#task-7-watermarkburn-transcribed-text-directly-onto-the-frame-image)
   - [Task 8: Add Score Threshold Filtering (`--min-score`) with Custom Fallback](#task-8-add-score-threshold-filtering---min-score-with-custom-fallback)
   - [Task 9: Transcribe Foreign Speech and Translate to English on the Fly](#task-9-transcribe-foreign-speech-and-translate-to-english-on-the-fly)
   - [Task 10: Extract Start Frame AND End Frame for Dialogue Span](#task-10-extract-start-frame-and-end-frame-for-dialogue-span)
   - [Task 11: Implement Phrase Proximity Search (Tolerate up to $K$ intervening gap words)](#task-11-implement-phrase-proximity-search-tolerate-up-to-k-intervening-gap-words)
   - [Task 12: Add Cookie / Authentication Support for Private/Restricted Streams](#task-12-add-cookie--authentication-support-for-privaterestricted-streams)
   - [Task 13: Add Custom Vocabulary Bias via `initial_prompt` in Whisper](#task-13-add-custom-vocabulary-bias-via-initial_prompt-in-whisper)
   - [Task 14: Batch URL & Query Search via CSV/JSON input](#task-14-batch-url--query-search-via-csvjson-input)
   - [Task 15: Add an LRU / Age-Based Cache Purge Function to SQLite](#task-15-add-an-lru--age-based-cache-purge-function-to-sqlite)
4. [Master Task-to-File Navigation Matrix](#4-master-task-to-file-navigation-matrix)
5. [The 5 Critical Ripple Dependency Rules (Never Break the Build)](#5-the-5-critical-ripple-dependency-rules-never-break-the-build)
6. [Git Mastery & Live Pairing Protocols](#6-git-mastery--live-pairing-protocols)

---

# 1. Core Mental Model & High-Level System Design

The system solves the problem of multimodal video search by enforcing an **Audio-First, Fingerprint-Deduplicated, Lazy-Video Architecture**:

```
                       Input: Video URL / Path + Dialogue Query
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         YouTube / Vimeo                                 OK.ru / Direct MP4
     (DASH Demuxed Audio Stream)                      (HLS Muxed Video+Audio)
     Downloads 8MB audio only                         Downloads 240p stream (8 parallel chunks)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                             16kHz Mono PCM WAV Audio
                                         │
                                         ▼
                        Chromaprint Acoustic Fingerprint
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │      SQLite Database (pipeline.db)   │
                      │  Check: Fingerprint & Duration Delta │
                      └──────────────┬───────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │ Cache Hit                       │ Cache Miss
                    ▼                                 ▼
         Load Cached Transcript              faster-whisper ASR (int8)
         (Sub-millisecond: 0.001s)           + Silero VAD (min_silence=300ms)
                    │                        + Word-Level Timestamps (DTW)
                    │                                 │
                    │                                 ▼
                    │                        Save to SQLite DB
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                     Text Normalization (Regex $O(N)$)
                     + Inverted Index ($O(1)$ Token Lookups)
                     + Dynamic IDF Rarity: ln(N / (1 + freq))
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │    Auto-Search 3-Stage       │
                      │    Cascade Engine            │
                      └──────────────┬───────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
      Stage 1                     Stage 2                     Stage 3
   Exact Phrase             Rare-Anchor Fuzzy           Sliding Window
   Match (Score=1.0)        (RapidFuzz Score >= 0.85)   (Exhaustive [N-2, N+2])
         │                           │                           │
         └───────────────────────────┼───────────────────────────┘
                                     ▼
                       Dialogue Match Confirmed!
                                     │
                                     ▼
                      Lazy Video Download (yt-dlp @ 480p)
                                     │
                                     ▼
                      Discrete Frame Index Calculation:
                      Frame # = round(Timestamp_onset × FPS)
                                     │
                                     ▼
                      FFmpeg Fast Input Seeking:
                      ffmpeg -ss <timestamp> -i <video> -q:v 2 <frame.jpg>
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │        4 KEY OUTPUTS         │
                      │ 1. Timestamp (Onset & Span)  │
                      │ 2. Discrete Frame Number     │
                      │ 3. Transcribed Text Match    │
                      │ 4. High-Res JPEG Frame Image │
                      └──────────────────────────────┘
```

---

# 2. Deep Codebase Walkthrough (Line-by-Line Mechanics)

---

## 2.1 `config/settings.py` — Central Configuration & Path Resolution

### Key Code Mechanics:
```python
@dataclass
class PipelineConfig:
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    video_dir: Optional[Path] = None
    audio_dir: Optional[Path] = None
    frame_dir: Optional[Path] = None
    result_dir: Optional[Path] = None
    db_path: Optional[Path] = None
    ...
```

### Why `__post_init__` is used:
In Python `@dataclass`, default field values are instantiated before the instance is initialized. In `__post_init__`:
1. It resolves `self.cache_dir = Path(self.cache_dir).resolve()`.
2. It checks if subdirectories (`video_dir`, `audio_dir`, `frame_dir`, `result_dir`, `db_path`) are `None`.
3. If they are `None`, it automatically anchors them as child paths: `self.cache_dir / "videos"`, `self.cache_dir / "frames"`, etc.
4. If the user passes a specific directory (e.g. `frame_dir=Path("E:/frames")`), it respects the override while leaving the rest under `cache_dir`.
5. `ensure_directories()` creates all parent directories using `mkdir(parents=True, exist_ok=True)`.

---

## 2.2 `database/db.py` — SQLite Engine & Deduplication Logic

### Database Architecture:
- **Connection Pattern**: `with self.get_connection() as conn:` ensures connection opening, thread-safety, transaction autocommit on success, and automatic rollback on unhandled exceptions.
- **Row Access**: `conn.row_factory = sqlite3.Row` allows dictionary-style column indexing (`row["duration"]`, `dict(row)`).
- **Foreign Key Cascading**: `conn.execute("PRAGMA foreign_keys = ON")` ensures that deleting a video row from `videos` automatically purges all related transcripts from `transcripts`.

### Schema Deep Dive:
```sql
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    audio_fingerprint TEXT,
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    video_codec TEXT,
    audio_codec TEXT,
    has_audio INTEGER,
    first_seen_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_videos_fingerprint ON videos(audio_fingerprint);

CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT NOT NULL,
    model_size TEXT NOT NULL,
    transcript_json TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (video_id, model_size),
    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
);
```

### Critical Methods:
- `get_video_by_fingerprint(fingerprint)`: Performs $O(1)$ B-Tree index lookup. Used for cross-URL deduplication.
- `insert_transcript(video_id, model_size, transcript)`: Serializes list of `WordTimestamp` objects into a JSON string and stores it with UTC ISO timestamp.

---

## 2.3 `audio/fingerprint.py` — Chromaprint & Fallback Acoustic Hashing

### The 3-Tier Fallback Hierarchy:
1. **Tier 1 (`pyacoustid`)**: Calls AcoustID C-library / `fpcalc` binary.
2. **Tier 2 (`ffmpeg -f chromaprint -`)**:
   ```python
   cmd = [ffmpeg_executable(), "-v", "error", "-i", str(audio_path), "-f", "chromaprint", "-"]
   ```
   Uses FFmpeg's internal libchromaprint muxer to pipe the fingerprint directly to stdout without needing `fpcalc` on the machine.
3. **Tier 3 (Memory-Safe SHA256 Chunk Hash)**:
   ```python
   hasher = hashlib.sha256()
   with open(audio_path, "rb") as f:
       while chunk := f.read(65536):  # 64 KB chunks
           hasher.update(chunk)
   return f"sha256_{hasher.hexdigest()[:32]}"
   ```
   Prevents out-of-memory errors by streaming in 64KB chunks rather than reading gigabytes into RAM.

---

## 2.4 `audio/downloader.py` — Media Ingestion, yt-dlp & FFmpeg Transcoding

### Key Ingestion Mechanics:
1. **Deterministic Video ID**:
   ```python
   hashlib.sha256(url_or_path.strip().encode("utf-8")).hexdigest()[:16]
   ```
2. **Audio-First Single-Pass Ingestion** (`download_audio_only_with_metadata`):
   - Format: `"format": "bestaudio[abr<=96]/bestaudio/worst[height<=240]/worst"`
   - Uses `concurrent_fragment_downloads: 8` for multi-threaded HLS chunk fetching.
   - Pipes raw stream into FFmpeg:
     ```bash
     ffmpeg -y -i <raw_stream> -vn -ac 1 -ar 16000 -c:a pcm_s16le <video_id>.wav
     ```
   - Immediately deletes the raw stream (`raw_path.unlink()`) to keep disk usage minimal.
3. **Audio Stream FPS Sanity Guard**:
   Audio streams in `yt-dlp` format metadata often report packet rates like `0.074 fps`. `_metadata_from_info` filters strictly for `vcodec != 'none'` and `fps >= 1.0` to preserve the real video framerate (e.g. 24.0, 25.0, 29.97).
4. **Ground-Truth Audio Verification** (`check_has_audio`):
   Runs `ffprobe -show_entries stream=codec_type` on the actual decoded WAV file on disk to guarantee an audio track exists before invoking neural ASR.

---

## 2.5 `asr/transcriber.py` — faster-whisper, Silero VAD & Word Timestamps

### Why `faster-whisper` (CTranslate2) is superior:
- Implemented in C++ with customized kernel quantization (`int8` on CPU, `float16` on CUDA).
- $4\times$ faster than standard PyTorch Whisper, consuming $50\%$ less memory.

### Transcription Execution Flow:
```python
segments, _info = model.transcribe(
    str(path),
    beam_size=beam_size,            # Default: 2
    word_timestamps=True,           # Dynamic Time Warping (DTW) word alignment
    vad_filter=vad_filter,          # Silero VAD
    vad_parameters=dict(min_silence_duration_ms=300),
)
```
- Iterates over segments $\to$ flattens `segment.words` $\to$ outputs list of `{"word": str, "start": float, "end": float}` sorted by `start` timestamp.
- Stores output in SQLite `transcripts` table.

---

## 2.6 `search/normalizer.py` & `search/index.py` — Inverted Index & Dynamic IDF

### Text Normalizer:
```python
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Strip all punctuation
    text = re.sub(r"\s+", " ", text)     # Collapse whitespace
    return text.strip()
```

### Inverted Index ($O(1)$ Lookup):
```python
class InvertedIndex:
    @classmethod
    def from_words(cls, transcript_words: List[str]) -> "InvertedIndex":
        index = defaultdict(list)
        for i, word in enumerate(transcript_words):
            index[word].append(i)
        return cls(dict(index))
```

### Dynamic IDF Rarity Formula:
```python
def build_rarity_fn(transcript_words: List[str]) -> Callable[[str], float]:
    freq = Counter(transcript_words)
    n = max(1, len(transcript_words))
    return lambda word: math.log(n / (1.0 + freq.get(word, 0)))
```
$$\text{rarity}(w) = \ln\left(\frac{N}{1.0 + \text{freq}(w)}\right)$$
- Computed fresh per search call (stateless $\to$ zero cross-video contamination).

---

## 2.7 `search/scorers.py` & `search/engine.py` — Similarity Math & 3-Stage Cascade

### Scorers:
- **`rapidfuzz_score`**: `rf_fuzz.ratio(" ".join(a), " ".join(b)) / 100.0` (C++ Levenshtein distance).
- **`difflib_score`**: `SequenceMatcher(None, str_a, str_b).ratio()` (Ratcliff-Obershelp).
- **`embedding_score`**: Cosine similarity between 384-dimensional `all-MiniLM-L6-v2` dense sentence embeddings.

### The 3-Stage Search Cascade (`auto_search`):
1. **Stage 1 (Exact Phrase Match)**:
   - Slices `transcript_words[i : i + N] == target_words`.
   - Complexity: $O(M)$, completes in $0.2\text{ms}$. If found, returns `score = 1.0` immediately.
2. **Stage 2 (Rare-Anchor Bounded Fuzzy)**:
   - Finds rarest token in query via `choose_anchors()`.
   - Look up anchor positions in `InvertedIndex` in $O(1)$.
   - Bounds search to window $[\text{anchor\_idx} - \text{offset} - 2, \text{anchor\_idx} - \text{offset} + N + 2]$.
   - If $\text{score} \ge 0.85$, returns immediately. If top anchor fails, retries next candidate anchor.
3. **Stage 3 (Full Sliding Window Fallback)**:
   - Scans all window lengths $[N-2, N+2]$ across the entire transcript.

---

## 2.8 `video/frame_extractor.py` — Time-to-Frame Math & Fast Seeking

### Time to Discrete Frame Index:
$$\text{Frame Number} = \text{round}(\text{timestamp}_{\text{onset}} \times \text{FPS}_{\text{video}})$$

### FFmpeg Fast Input Seeking:
```python
cmd = [
    ffmpeg_executable(),
    "-y",
    "-ss", f"{timestamp:.6f}",  # Placed BEFORE -i for container header fast seek
    "-i", str(vpath),
    "-frames:v", "1",
    "-q:v", "2",                # High quality JPEG
    str(out_path),
]
```
- Fast input seeking takes $\approx 15\text{ms}$ versus output seeking which decodes every preceding frame taking $10\text{s}$.

---

## 2.9 `pipeline/orchestrator.py` & `cli.py` — Coordination & CLI Interface

- Coordinates `MediaManager` $\to$ `WhisperTranscriber` $\to$ `InvertedIndex` $\to$ `search_dialogue` $\to$ `extract_frame`.
- Writes JSON payload to `cache/results/<video_id>_<model_size>_result.json`.
- Formats CLI output into GitHub-style tables using `tabulate`.

---

# 3. Top 15 Most Likely In-Person Interview Coding Tasks & Complete Solutions

---

### Task 1: Non-Maximum Suppression (NMS) for Overlapping Spans
> **Question**: *"Our search results often contain redundant overlapping windows (e.g. index 10-14 and index 10-15). Implement a filter that keeps only distinct, non-overlapping occurrences."*

**File**: [`src/video_dialogue/search/engine.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/engine.py)

```python
def filter_non_overlapping(results: List[SearchResultItem], top_k: int = 5) -> List[SearchResultItem]:
    """Filter out overlapping token spans, keeping the highest scoring matches."""
    # Ensure results are sorted by score descending
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
    selected: List[SearchResultItem] = []
    
    for r in sorted_results:
        # Check if candidate overlaps with any already selected match
        overlaps = any(
            not (r.end_index <= s.start_index or r.start_index >= s.end_index)
            for s in selected
        )
        if not overlaps:
            selected.append(r)
        if len(selected) >= top_k:
            break
            
    return selected
```

---

### Task 2: Time-Range Bounded Dialogue Search (`--start-time` / `--end-time`)
> **Question**: *"Allow the user to restrict dialogue search to a specific time interval in the video (e.g. between 300s and 600s)."*

**Files**: [`cli.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/cli.py) & [`pipeline/orchestrator.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/pipeline/orchestrator.py)

**Step 1 in `cli.py`**:
```python
search_parser.add_argument("--start-time", type=float, default=None, help="Start time boundary in seconds")
search_parser.add_argument("--end-time", type=float, default=None, help="End time boundary in seconds")
```

**Step 2 in `pipeline/orchestrator.py` inside `run()`**:
```python
# Filter transcript before word indexing if time bounds are provided:
start_bound = kwargs.get("start_time")
end_bound = kwargs.get("end_time")

if start_bound is not None or end_bound is not None:
    t_min = start_bound if start_bound is not None else 0.0
    t_max = end_bound if end_bound is not None else float("inf")
    transcript = [
        item for item in transcript
        if t_min <= item["start"] and item["end"] <= t_max
    ]
```

---

### Task 3: Extract a 5-Second Video Clip with Audio instead of a Static Image
> **Question**: *"Instead of extracting a single JPEG frame, extract a 5-second MP4 video clip starting 0.5s before the dialogue begins."*

**File**: [`src/video_dialogue/video/frame_extractor.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/video/frame_extractor.py)

```python
def extract_video_clip(
    video_path: Union[str, Path],
    start_timestamp: float,
    duration: float = 5.0,
    output_path: Union[str, Path] = "clip.mp4",
) -> Path:
    """Extract a short synchronized audio/video MP4 clip."""
    vpath = Path(video_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    seek_start = max(0.0, start_timestamp - 0.5)
    
    cmd = [
        ffmpeg_executable(),
        "-y",
        "-ss", f"{seek_start:.6f}",
        "-i", str(vpath),
        "-t", f"{duration:.2f}",
        "-c:v", "libx264",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path
```

---

### Task 4: Add Word-Level Context Padding (e.g. $\pm 4$ words around dialogue)
> **Question**: *"The extracted text is sometimes too concise. Add 4 words of leading and trailing context in the output."*

**File**: [`src/video_dialogue/pipeline/orchestrator.py:L226`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/pipeline/orchestrator.py#L226)

```python
# In DialogueRetrievalPipeline.run():
pad = 4
ctx_start = max(0, start_idx - pad)
ctx_end = min(len(transcript), end_idx + pad)

raw_words = [transcript[i]["word"] for i in range(ctx_start, ctx_end)]
matched_text = " ".join(raw_words)

if ctx_start > 0:
    matched_text = "... " + matched_text
if ctx_end < len(transcript):
    matched_text = matched_text + " ..."
```

---

### Task 5: Implement a Hybrid Scorer (70% Lexical Levenshtein + 30% Dense Semantic Vectors)
> **Question**: *"Create a composite scoring function combining RapidFuzz string similarity and Sentence-Transformer cosine similarity."*

**File**: [`src/video_dialogue/search/scorers.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/scorers.py)

```python
def hybrid_score(a_words: List[str], b_words: List[str], alpha: float = 0.7) -> float:
    """Compute weighted hybrid score: alpha * RapidFuzz + (1 - alpha) * Cosine Embedding."""
    rf_score = rapidfuzz_score(a_words, b_words)
    if EMBEDDING_AVAILABLE:
        emb_score = embedding_score(a_words, b_words)
        return float((alpha * rf_score) + ((1.0 - alpha) * max(0.0, emb_score)))
    return rf_score

# Register in dictionary:
SCORE_FNS["hybrid"] = hybrid_score
```

---

### Task 6: Export Transcripts to Standard WebVTT (`.vtt`) and SubRip (`.srt`) Subtitles
> **Question**: *"Write a standalone export utility to convert cached Whisper transcripts into `.srt` and `.vtt` subtitle files."*

**File**: [`src/video_dialogue/database/db.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/database/db.py)

```python
def export_to_srt_and_vtt(transcript: List[Dict[str, Any]], base_path: Union[str, Path]) -> None:
    def format_timestamp(s: float, sep: str = ",") -> str:
        ms = int(round((s - int(s)) * 1000))
        m, sec = divmod(int(s), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}{sep}{ms:03d}"

    # 1. SRT Export
    srt_lines = []
    for i, item in enumerate(transcript, start=1):
        start = format_timestamp(item["start"], ",")
        end = format_timestamp(item["end"], ",")
        srt_lines.append(f"{i}\n{start} --> {end}\n{item['word']}\n")
    Path(f"{base_path}.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    # 2. VTT Export
    vtt_lines = ["WEBVTT\n"]
    for i, item in enumerate(transcript, start=1):
        start = format_timestamp(item["start"], ".")
        end = format_timestamp(item["end"], ".")
        vtt_lines.append(f"{i}\n{start} --> {end}\n{item['word']}\n")
    Path(f"{base_path}.vtt").write_text("\n".join(vtt_lines), encoding="utf-8")
```

---

### Task 7: Watermark/Burn Transcribed Text Directly onto the Frame Image
> **Question**: *"Use FFmpeg's `drawtext` video filter to stamp the transcribed dialogue text and timestamp onto the extracted JPEG frame."*

**File**: [`src/video_dialogue/video/frame_extractor.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/video/frame_extractor.py)

```python
def extract_frame_with_overlay(
    video_path: Union[str, Path],
    timestamp: float,
    dialogue_text: str,
    output_path: Union[str, Path],
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Escape single quotes and colons for FFmpeg filter syntax
    safe_text = dialogue_text.replace("'", "").replace(":", "-")
    label = f"{safe_text} [{timestamp:.2f}s]"
    
    filter_str = (
        f"drawtext=text='{label}':fontcolor=white:fontsize=22:"
        f"box=1:boxcolor=black@0.6:boxborderw=5:x=(w-text_w)/2:y=h-45"
    )
    
    cmd = [
        ffmpeg_executable(),
        "-y",
        "-ss", f"{timestamp:.6f}",
        "-i", str(video_path),
        "-vf", filter_str,
        "-frames:v", "1",
        "-q:v", "2",
        str(out_path),
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return out_path
```

---

### Task 8: Add Score Threshold Filtering (`--min-score`) with Custom Fallback
> **Question**: *"Ensure only matches scoring above a user-specified threshold are returned. If none meet the threshold, report a clear error message."*

**Files**: [`cli.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/cli.py) & [`pipeline/orchestrator.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/pipeline/orchestrator.py)

**In `pipeline/orchestrator.py`**:
```python
def run(self, ..., min_score: float = 0.70):
    ...
    # Filter raw results
    qualified_results = [r for r in raw_results if r.score >= min_score]
    
    if not qualified_results:
        return PipelineResult(
            success=False,
            video={"url": video_url, "duration_seconds": record.duration, "fps": fps},
            query={"dialogue": target_dialogue},
            message=f"No matches met the minimum confidence threshold of {min_score:.2f} (Best was {raw_results[0].score:.3f}).",
        )
```

---

### Task 9: Transcribe Foreign Speech and Translate to English on the Fly
> **Question**: *"Allow searching English dialogue in foreign videos (e.g. Spanish, German, Japanese) by translating speech during ASR."*

**File**: [`src/video_dialogue/asr/transcriber.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/asr/transcriber.py#L112)

```python
# In WhisperTranscriber.transcribe():
segments, _info = model.transcribe(
    str(path),
    beam_size=beam_size,
    word_timestamps=True,
    vad_filter=vad_filter,
    task="translate",  # <--- Forces Whisper to transcribe foreign speech and translate to English!
    vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
)
```

---

### Task 10: Extract Start Frame AND End Frame for Dialogue Span
> **Question**: *"Extract two image frames for every match: one at the start of speech and one at the end of speech."*

**File**: [`src/video_dialogue/pipeline/orchestrator.py:L215-L225`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/pipeline/orchestrator.py#L215-L225)

```python
start_time = float(transcript[start_idx]["start"])
end_time = float(transcript[end_idx - 1]["end"])
start_frame = timestamp_to_frame(start_time, fps)
end_frame = timestamp_to_frame(end_time, fps)

start_frame_path = self.config.frame_dir / f"{video_id}_start_frame_{start_frame}.jpg"
end_frame_path = self.config.frame_dir / f"{video_id}_end_frame_{end_frame}.jpg"

extract_frame(video_path, start_time, start_frame_path)
extract_frame(video_path, end_time, end_frame_path)
```

---

### Task 11: Implement Phrase Proximity Search (Tolerate up to $K$ intervening gap words)
> **Question**: *"Implement a search engine that matches query keywords even if there are up to 3 unrelated words spoken between them."*

**File**: [`src/video_dialogue/search/engine.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/engine.py)

```python
def proximity_search(transcript_words: List[str], target_words: List[str], max_gap: int = 3) -> List[SearchResultItem]:
    """Find windows where all target words appear in order with at most max_gap words between them."""
    results = []
    n = len(target_words)
    if n == 0 or len(transcript_words) < n:
        return results

    for i in range(len(transcript_words)):
        if transcript_words[i] != target_words[0]:
            continue
        
        curr_t_idx = 1
        curr_tr_idx = i + 1
        gap_count = 0
        
        while curr_t_idx < n and curr_tr_idx < len(transcript_words):
            if transcript_words[curr_tr_idx] == target_words[curr_t_idx]:
                curr_t_idx += 1
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > max_gap:
                    break
            curr_tr_idx += 1
            
        if curr_t_idx == n:
            score = n / (curr_tr_idx - i) # Proximity density score
            results.append(SearchResultItem(start_index=i, end_index=curr_tr_idx, score=score, method="proximity"))

    return sorted(results, key=lambda r: r.score, reverse=True)
```

---

### Task 12: Add Cookie / Authentication Support for Private/Restricted Streams
> **Question**: *"Support downloading age-restricted YouTube videos by loading session cookies from the local browser."*

**File**: [`src/video_dialogue/audio/downloader.py:L218`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/audio/downloader.py#L218)

```python
# In download_audio_only_with_metadata() and download_video():
ydl_opts = {
    "outtmpl": tmp_template,
    "format": f"bestaudio[abr<={max_bitrate_kbps}]/bestaudio/worst[height<=240]/worst",
    "cookiesfrombrowser": ("chrome",),  # <--- Automatically extracts Chrome browser session cookies
    "noplaylist": True,
    "quiet": True,
}
```

---

### Task 13: Add Custom Vocabulary Bias via `initial_prompt` in Whisper
> **Question**: *"When transcribing medical or domain-specific videos, bias Whisper's beam search decoder towards custom industry terms."*

**File**: [`src/video_dialogue/asr/transcriber.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/asr/transcriber.py)

```python
def transcribe(
    self,
    video_id: str,
    audio_path: Union[str, Path],
    model_size: str = "tiny",
    initial_prompt: Optional[str] = None, # <--- Added parameter
    ...
):
    ...
    segments, _info = model.transcribe(
        str(path),
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=vad_filter,
        initial_prompt=initial_prompt,     # <--- e.g. "Stanford, Steve Jobs, Macintosh, Pixar"
    )
```

---

### Task 14: Batch URL & Query Search via CSV/JSON input
> **Question**: *"Write a CLI subcommand `batch` that takes a CSV file of `(url, query)` rows and produces a single consolidated summary report."*

**File**: [`src/video_dialogue/cli.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/cli.py)

```python
import csv

def cmd_batch(args: argparse.Namespace) -> None:
    """Process batch CSV containing columns: url, query"""
    config = get_default_config(cache_dir=Path(args.cache_dir))
    pipeline = DialogueRetrievalPipeline(config=config)
    
    summary_rows = []
    with open(args.csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row["url"]
            query = row["query"]
            print(f"[BATCH] Processing: {url} -> \"{query}\"")
            res = pipeline.run(video_url=url, target_dialogue=query)
            if res.success:
                best = res.matches[0]
                summary_rows.append({
                    "url": url, "query": query, "status": "FOUND",
                    "timestamp": f"{best.start_timestamp:.2f}s", "frame": best.start_frame, "score": best.score
                })
            else:
                summary_rows.append({"url": url, "query": query, "status": "NOT_FOUND", "timestamp": "N/A", "frame": "N/A", "score": 0.0})

    print("\n" + tabulate(summary_rows, headers="keys", tablefmt="github"))
```

---

### Task 15: Add an LRU / Age-Based Cache Purge Function to SQLite
> **Question**: *"Write a database maintenance method that purges video records and cached media older than $X$ days to prevent disk exhaustion."*

**File**: [`src/video_dialogue/database/db.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/database/db.py)

```python
from datetime import datetime, timezone, timedelta

def purge_records_older_than(self, days: int = 7) -> int:
    """Delete video records, transcripts, and return count of purged entries."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with self.get_connection() as conn:
        cursor = conn.execute("DELETE FROM videos WHERE first_seen_at < ?", (cutoff,))
        purged_count = cursor.rowcount
        conn.commit()
    return purged_count
```

---

# 4. Master Task-to-File Navigation Matrix

| Concern | Primary Source File | Dependent Files |
| :--- | :--- | :--- |
| **CLI flags / argument parsing** | [`src/video_dialogue/cli.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/cli.py) | `settings.py`, `orchestrator.py` |
| **Database schemas, SQLite queries** | [`src/video_dialogue/database/db.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/database/db.py) | `core/models.py` |
| **Acoustic fingerprinting & hashing** | [`src/video_dialogue/audio/fingerprint.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/audio/fingerprint.py) | `database/db.py` |
| **Media download, yt-dlp & FFmpeg audio**| [`src/video_dialogue/audio/downloader.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/audio/downloader.py) | `settings.py` |
| **Whisper ASR, Silero VAD, word timestamps**| [`src/video_dialogue/asr/transcriber.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/asr/transcriber.py) | `settings.py`, `database/db.py` |
| **Text normalization, regex, token lists** | [`src/video_dialogue/search/normalizer.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/normalizer.py) | `search/index.py` |
| **Inverted index, Dynamic IDF Rarity** | [`src/video_dialogue/search/index.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/index.py) | `search/engine.py` |
| **Similarity scoring functions** | [`src/video_dialogue/search/scorers.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/scorers.py) | `cli.py` (choices) |
| **Search retrieval algorithms & cascade** | [`src/video_dialogue/search/engine.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/search/engine.py) | `pipeline/orchestrator.py` |
| **Frame calculation, FFmpeg frame capture**| [`src/video_dialogue/video/frame_extractor.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/video/frame_extractor.py)| `pipeline/orchestrator.py` |
| **End-to-end orchestration & report JSON**| [`src/video_dialogue/pipeline/orchestrator.py`](file:///c:/Users/computer/Desktop/Quest1/video_dialogue_retrieval/src/video_dialogue/pipeline/orchestrator.py)| `cli.py` |

---

# 5. The 5 Critical Ripple Dependency Rules (Never Break the Build)

1. **Rule 1: CLI Flags**:
   - `cli.py` (`add_argument`) $\to$ `settings.py` (dataclass field) $\to$ `orchestrator.py` (`run()` parameter).
2. **Rule 2: New Similarity Scorer**:
   - `scorers.py` (define function) $\to$ `SCORE_FNS` registry $\to$ `cli.py` (`--score-fn` choices).
3. **Rule 3: Database Column Addition**:
   - `db.py` (`_init_db` SQL `CREATE TABLE`) $\to$ `db.py` (`insert_video` SQL `INSERT`) $\to$ `core/models.py` (`VideoRecord` dataclass).
4. **Rule 4: Output Field Addition**:
   - `core/models.py` (`DialogueMatch` dataclass) $\to$ `orchestrator.py` (`run()` constructor call) $\to$ `cli.py` (add column to `table_data` & `headers`).
5. **Rule 5: New Search Strategy**:
   - `engine.py` (define function) $\to$ `search_dialogue()` router $\to$ `cli.py` (`--method` choices).

---

# 6. Git Mastery & Live Pairing Protocols

```bash
# 1. Check working directory status
git status

# 2. Create and switch to an isolated feature branch
git checkout -b feature/interview-task

# 3. Stage the modified files
git add src/video_dialogue/search/engine.py

# 4. Commit with descriptive semantic message
git commit -m "feat(search): implement non-maximum suppression for overlapping matches"

# 5. Push branch to remote
git push origin feature/interview-task
```

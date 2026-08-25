"""Benchmark harness for evaluating search variants and Whisper model sizes."""

import time
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union

import pandas as pd

from ..core.models import BenchmarkVariant, WordTimestamp
from ..database.db import DatabaseManager
from ..asr.transcriber import WhisperTranscriber
from ..search.normalizer import normalize_text, build_word_index
from ..search.index import InvertedIndex
from ..search.scorers import EMBEDDING_AVAILABLE
from ..search.engine import search_dialogue


DEFAULT_SEARCH_VARIANTS: List[Dict[str, Any]] = [
    {"label": "exact", "method": "exact"},
    {"label": "fuzzy + difflib", "method": "fuzzy", "score_fn_name": "difflib"},
    {"label": "fuzzy + rapidfuzz", "method": "fuzzy", "score_fn_name": "rapidfuzz"},
    {
        "label": "rare_anchor + difflib (linear scan)",
        "method": "rare_anchor_fuzzy",
        "score_fn_name": "difflib",
        "use_index": False,
    },
    {
        "label": "rare_anchor + difflib (inverted index)",
        "method": "rare_anchor_fuzzy",
        "score_fn_name": "difflib",
        "use_index": True,
    },
    {
        "label": "rare_anchor + rapidfuzz (inverted index)",
        "method": "rare_anchor_fuzzy",
        "score_fn_name": "rapidfuzz",
        "use_index": True,
    },
]

if EMBEDDING_AVAILABLE:
    DEFAULT_SEARCH_VARIANTS.append({
        "label": "rare_anchor + embedding (inverted index)",
        "method": "rare_anchor_fuzzy",
        "score_fn_name": "embedding",
        "use_index": True,
    })


def run_variant_benchmark(
    transcript: List[Union[WordTimestamp, Dict[str, Any]]],
    known_dialogues: List[Tuple[str, float, float]],
    variants: Optional[List[Dict[str, Any]]] = None,
    runs: int = 3,
) -> pd.DataFrame:
    """Benchmark accuracy, mean absolute error (MAE), and latency across search variants.

    Args:
        transcript: List of word timestamp records.
        known_dialogues: List of (dialogue_text, expected_start_time, tolerance_seconds) ground truth tuples.
        variants: List of variant dictionaries or BenchmarkVariant objects.
        runs: Number of benchmark repetitions for latency timing.

    Returns:
        pd.DataFrame containing top1_accuracy, mean_abs_error_s, and mean_time_ms sorted by latency.
    """
    variants_list = variants if variants is not None else DEFAULT_SEARCH_VARIANTS
    transcript_words = build_word_index(transcript)
    inverted_index = InvertedIndex.from_words(transcript_words)
    rows: List[Dict[str, Any]] = []

    for var in variants_list:
        if isinstance(var, BenchmarkVariant):
            label = var.label
            method = var.method
            score_fn_name = var.score_fn_name
            use_idx = var.use_index
        else:
            label = var["label"]
            method = var["method"]
            score_fn_name = var.get("score_fn_name", "difflib")
            use_idx = var.get("use_index", False)

        idx = inverted_index if use_idx else None
        correct = 0
        errors: List[float] = []
        times: List[float] = []

        for dialogue, expected_start, tolerance in known_dialogues:
            target_words = normalize_text(dialogue).split()
            if not target_words:
                continue

            t0 = time.perf_counter()
            results = None
            for _ in range(runs):
                results = search_dialogue(
                    transcript_words,
                    target_words,
                    method=method,
                    score_fn_name=score_fn_name,
                    inverted_index=idx,
                )
            elapsed_ms = (time.perf_counter() - t0) / max(1, runs) * 1000.0
            times.append(elapsed_ms)

            if results:
                best_start_idx = results[0].start_index
                predicted_start = float(
                    transcript[best_start_idx]["start"]
                    if isinstance(transcript[best_start_idx], dict)
                    else transcript[best_start_idx].start
                )
                err = abs(predicted_start - expected_start)
                errors.append(err)
                if err <= tolerance:
                    correct += 1
            else:
                errors.append(float("nan"))

        total_queries = len(known_dialogues)
        top1_acc = (correct / total_queries) if total_queries > 0 else float("nan")
        mean_err = pd.Series(errors).dropna().mean() if errors else float("nan")
        mean_lat = pd.Series(times).mean() if times else 0.0

        rows.append({
            "variant": label,
            "top1_accuracy": top1_acc,
            "mean_abs_error_s": mean_err,
            "mean_time_ms": mean_lat,
        })

    df = pd.DataFrame(rows)
    if not df.empty and "mean_time_ms" in df.columns:
        df = df.sort_values("mean_time_ms").reset_index(drop=True)
    return df


def compare_model_sizes(
    db_manager: DatabaseManager,
    video_id: str,
    audio_path: Union[str, Path],
    model_sizes: Tuple[str, ...] = ("tiny", "base", "small", "medium"),
    known_dialogues: Optional[List[Tuple[str, float, float]]] = None,
    method: str = "rare_anchor_fuzzy",
    force: bool = True,
    transcriber: Optional[WhisperTranscriber] = None,
) -> pd.DataFrame:
    """Evaluate Whisper model sizes for transcription time, word count, and retrieval accuracy."""
    known_dialogues = known_dialogues or []
    trans = transcriber or WhisperTranscriber(db_manager=db_manager)
    rows: List[Dict[str, Any]] = []

    for model_size in model_sizes:
        t0 = time.perf_counter()
        transcript = trans.transcribe(
            video_id=video_id,
            audio_path=audio_path,
            model_size=model_size,
            force=force,
        )
        transcribe_s = time.perf_counter() - t0

        if known_dialogues:
            acc_df = run_variant_benchmark(
                transcript,
                known_dialogues,
                variants=[{
                    "label": method,
                    "method": method,
                    "score_fn_name": "difflib",
                    "use_index": True,
                }],
            )
            row = acc_df.iloc[0].to_dict()
        else:
            row = {
                "variant": method,
                "top1_accuracy": float("nan"),
                "mean_abs_error_s": float("nan"),
                "mean_time_ms": 0.0,
            }

        row.update({
            "model_size": model_size,
            "transcribe_time_s": transcribe_s,
            "word_count": len(transcript),
        })
        rows.append(row)

    cols = [
        "model_size",
        "word_count",
        "transcribe_time_s",
        "top1_accuracy",
        "mean_abs_error_s",
        "mean_time_ms",
    ]
    return pd.DataFrame(rows)[cols]

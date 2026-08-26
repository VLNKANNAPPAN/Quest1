"""SQLite database manager for video records, fingerprint deduplication, and ASR transcript caching."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from ..core.models import VideoMetadata, VideoRecord, WordTimestamp


class DatabaseManager:
    """Manages SQLite storage for deduplicating videos and caching ASR transcripts."""

    def __init__(self, db_path: Union[str, Path] = "cache/pipeline.db"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Create and return a new SQLite connection configured with row access and foreign keys."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Create tables and indices if they do not already exist."""
        with self.get_connection() as conn:
            conn.execute("""
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
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_videos_fingerprint ON videos(audio_fingerprint)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    video_id TEXT NOT NULL,
                    model_size TEXT NOT NULL,
                    transcript_json TEXT NOT NULL,
                    created_at TEXT,
                    PRIMARY KEY (video_id, model_size),
                    FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
                )
            """)
            conn.commit()

    def get_video_by_id(self, video_id: str) -> Optional[VideoRecord]:
        """Look up a video record by its unique internal video_id."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
            if row is None:
                return None
            return VideoRecord.from_dict(dict(row))

    def get_video_by_fingerprint(self, fingerprint: str) -> Optional[VideoRecord]:
        """Look up a video record by its audio fingerprint for cross-URL deduplication."""
        if not fingerprint:
            return None
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE audio_fingerprint = ?", (fingerprint,)
            ).fetchone()
            if row is None:
                return None
            return VideoRecord.from_dict(dict(row))

    def insert_video(
        self,
        video_id: str,
        url: str,
        fingerprint: Optional[str],
        metadata: Union[VideoMetadata, VideoRecord, Dict[str, Any]],
    ) -> None:
        """Insert or replace a video entry in the database."""
        if hasattr(metadata, "to_dict"):
            meta_dict = metadata.to_dict()
        else:
            meta_dict = metadata

        now_iso = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO videos
                   (video_id, url, audio_fingerprint, duration, fps, width, height,
                    video_codec, audio_codec, has_audio, first_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    video_id,
                    url,
                    fingerprint,
                    meta_dict.get("duration", 0.0),
                    meta_dict.get("fps"),
                    meta_dict.get("width", 0),
                    meta_dict.get("height", 0),
                    meta_dict.get("video_codec"),
                    meta_dict.get("audio_codec"),
                    1 if meta_dict.get("has_audio", True) else 0,
                    now_iso,
                ),
            )
            conn.commit()

    def get_transcript(
        self, video_id: str, model_size: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch a cached transcript for a specific video and whisper model size."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT transcript_json FROM transcripts WHERE video_id = ? AND model_size = ?",
                (video_id, model_size),
            ).fetchone()
            if row and row["transcript_json"]:
                return json.loads(row["transcript_json"])
            return None

    def insert_transcript(
        self,
        video_id: str,
        model_size: str,
        transcript: List[Union[WordTimestamp, Dict[str, Any]]],
    ) -> None:
        """Store or update the transcript JSON for a video and model size."""
        raw_list = [
            t.to_dict() if isinstance(t, WordTimestamp) else t for t in transcript
        ]
        now_iso = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO transcripts (video_id, model_size, transcript_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (video_id, model_size, json.dumps(raw_list), now_iso),
            )
            conn.commit()

    def list_videos(self) -> List[VideoRecord]:
        """Return all indexed video records."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM videos ORDER BY first_seen_at DESC").fetchall()
            return [VideoRecord.from_dict(dict(r)) for r in rows]

    def clear_all(self) -> None:
        """Delete all cached video records and transcripts."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM transcripts")
            conn.execute("DELETE FROM videos")
            conn.commit()

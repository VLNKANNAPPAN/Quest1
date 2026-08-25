"""Data models representing video entities, transcripts, search results, and benchmark records."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any


@dataclass
class VideoMetadata:
    """Lightweight metadata extracted from a video source."""
    duration: float = 0.0
    fps: Optional[float] = None
    width: int = 0
    height: int = 0
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    has_audio: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        return cls(
            duration=float(data.get("duration", 0.0)),
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            video_codec=data.get("video_codec"),
            audio_codec=data.get("audio_codec"),
            has_audio=bool(data.get("has_audio", True)),
        )


@dataclass
class VideoRecord:
    """Persistent video record with fingerprint deduplication key."""
    video_id: str
    url: str
    audio_fingerprint: Optional[str] = None
    duration: float = 0.0
    fps: Optional[float] = None
    width: int = 0
    height: int = 0
    video_codec: Optional[str] = None
    audio_codec: Optional[str] = None
    has_audio: bool = True
    first_seen_at: Optional[str] = None
    current_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoRecord":
        return cls(
            video_id=str(data["video_id"]),
            url=str(data.get("url", "")),
            audio_fingerprint=data.get("audio_fingerprint"),
            duration=float(data.get("duration", 0.0)),
            fps=float(data["fps"]) if data.get("fps") is not None else None,
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
            video_codec=data.get("video_codec"),
            audio_codec=data.get("audio_codec"),
            has_audio=bool(data.get("has_audio", True)),
            first_seen_at=data.get("first_seen_at"),
            current_url=data.get("current_url", data.get("url")),
        )


@dataclass
class WordTimestamp:
    """Word-level alignment timestamp from ASR."""
    word: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return {"word": self.word, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WordTimestamp":
        return cls(
            word=str(data["word"]),
            start=float(data["start"]),
            end=float(data["end"]),
        )


@dataclass
class SearchResultItem:
    """Individual match candidate produced by a search engine."""
    start_index: int
    end_index: int
    score: float
    method: str
    anchor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "score": self.score,
            "method": self.method,
        }
        if self.anchor is not None:
            d["anchor"] = self.anchor
        return d


@dataclass
class DialogueMatch:
    """Enriched dialogue retrieval match with timestamps, frame index, and visual asset path."""
    rank: int
    matched_text: str
    start_timestamp: float
    end_timestamp: float
    start_frame: int
    end_frame: int
    score: float
    frame_path: str
    anchor: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Complete end-to-end retrieval output report."""
    success: bool
    video: Optional[Dict[str, Any]] = None
    query: Optional[Dict[str, Any]] = None
    matches: List[DialogueMatch] = field(default_factory=list)
    message: Optional[str] = None
    result_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {"success": self.success}
        if self.message is not None:
            res["message"] = self.message
        if self.video is not None:
            res["video"] = self.video
        if self.query is not None:
            res["query"] = self.query
        if self.matches:
            res["matches"] = [m.to_dict() if isinstance(m, DialogueMatch) else m for m in self.matches]
        if self.result_file is not None:
            res["result_file"] = self.result_file
        return res


@dataclass
class BenchmarkVariant:
    """Specification of a search algorithm variant for benchmarking."""
    label: str
    method: str
    score_fn_name: str = "difflib"
    use_index: bool = False

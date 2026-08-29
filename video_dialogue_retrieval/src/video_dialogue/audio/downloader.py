"""Audio and video acquisition supporting both remote URLs and local media files."""

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Union, Dict, Any, Callable

import yt_dlp

from ..core.models import VideoMetadata
from ..config.settings import PipelineConfig, get_default_config
from ..media_tools import ffmpeg_executable

logger = logging.getLogger(__name__)

NETWORK_RETRY_ATTEMPTS = 3


def _with_network_retries(operation: Callable[[], Any], description: str) -> Any:
    """Run a yt-dlp operation with bounded exponential-backoff retries."""
    for attempt in range(1, NETWORK_RETRY_ATTEMPTS + 1):
        try:
            return operation()
        except yt_dlp.utils.DownloadError:
            if attempt == NETWORK_RETRY_ATTEMPTS:
                raise
            delay_seconds = 2 ** (attempt - 1)
            logger.warning(
                "%s failed (attempt %d/%d); retrying in %ds.",
                description, attempt, NETWORK_RETRY_ATTEMPTS, delay_seconds,
            )
            time.sleep(delay_seconds)


def _metadata_from_info(info: Dict[str, Any]) -> VideoMetadata:
    """Convert yt-dlp's already-fetched info dictionary to project metadata.

    FPS is extracted only from video stream formats (vcodec != 'none') to avoid
    picking up meaningless audio-stream frame rates (e.g. 0.07 fps).
    """
    fps = info.get("fps")
    # Validate: a real video FPS should be >= 1.0; anything below is an audio
    # stream artefact or placeholder value.
    if not fps or fps < 1.0:
        fps = None
        for fmt in info.get("formats", []):
            vcodec = fmt.get("vcodec", "none")
            fmt_fps = fmt.get("fps")
            # Only consider actual video streams with sensible FPS values
            if vcodec not in (None, "none") and fmt_fps and fmt_fps >= 1.0:
                fps = fmt_fps
                break
    return VideoMetadata(
        duration=float(info.get("duration") or 0.0),
        fps=float(fps) if fps else None,
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        video_codec=info.get("vcodec"),
        audio_codec=info.get("acodec"),
        has_audio=info.get("acodec") not in (None, "none"),
    )


def get_video_id(url_or_path: str) -> str:
    """Generate a deterministic 16-character identifier for a URL or local file path."""
    normalized = str(url_or_path).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def is_local_file(url_or_path: str) -> bool:
    """Check whether the input target is an existing local file or a remote URL."""
    p = Path(url_or_path)
    return p.exists() and p.is_file()


def get_local_metadata(file_path: Path) -> VideoMetadata:
    """Extract media format metadata from a local video/audio file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json",
        str(file_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(proc.stdout)
    except Exception as exc:
        logger.warning("ffprobe metadata extraction failed for %s: %s", file_path, exc)
        return VideoMetadata(duration=0.0, fps=25.0, width=1280, height=720, has_audio=True)

    duration = float(info.get("format", {}).get("duration", 0.0))
    fps: Optional[float] = None
    width = 0
    height = 0
    video_codec = None
    audio_codec = None
    has_audio = False

    for stream in info.get("streams", []):
        ctype = stream.get("codec_type")
        if ctype == "video" and not video_codec:
            video_codec = stream.get("codec_name")
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            r_rate = stream.get("r_frame_rate", "")
            if "/" in r_rate:
                num, den = r_rate.split("/")
                if float(den) > 0:
                    fps = float(num) / float(den)
            elif r_rate:
                fps = float(r_rate)
        elif ctype == "audio" and not audio_codec:
            audio_codec = stream.get("codec_name")
            has_audio = True

    return VideoMetadata(
        duration=duration,
        fps=fps or 25.0,
        width=width,
        height=height,
        video_codec=video_codec,
        audio_codec=audio_codec,
        has_audio=has_audio,
    )


def get_light_metadata(url_or_path: str) -> VideoMetadata:
    """Fetch duration, FPS, resolution and codec metadata without downloading full media."""
    if is_local_file(url_or_path):
        return get_local_metadata(Path(url_or_path))

    ydl_opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    def extract() -> Dict[str, Any]:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_path, download=False)

    return _metadata_from_info(_with_network_retries(extract, "Metadata request"))


def check_has_audio(audio_path: Union[str, Path]) -> bool:
    """Ground truth: probe the actual extracted audio file, don't trust
    yt-dlp's pre-download acodec field (unreliable for muxed HLS streams)."""
    p = Path(audio_path)
    if not p.exists() or p.stat().st_size == 0:
        return False
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(p),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return "audio" in result.stdout.lower()
    except Exception as exc:
        logger.warning("ffprobe audio check failed for %s: %s", audio_path, exc)
        return False


def download_audio_only(
    url_or_path: str,
    output_dir: Union[str, Path] = "cache/audio",
    sample_rate: int = 16000,
    max_bitrate_kbps: int = 96,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """Acquire audio track only (mono 16kHz PCM WAV) for fingerprinting and ASR."""
    audio_path, _metadata = download_audio_only_with_metadata(
        url_or_path, output_dir, sample_rate, max_bitrate_kbps, progress_hook,
    )
    return audio_path


def download_audio_only_with_metadata(
    url_or_path: str,
    output_dir: Union[str, Path] = "cache/audio",
    sample_rate: int = 16000,
    max_bitrate_kbps: int = 96,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[Path, VideoMetadata]:
    """Download/extract audio and return metadata with ground-truth audio probe."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = get_video_id(url_or_path)
    audio_path = out_dir / f"{video_id}.wav"

    if audio_path.exists():
        logger.info("Reusing cached audio: %s", audio_path.name)
        # Ground truth: probe the actual cached audio file directly on disk,
        # skipping unnecessary and fragile remote metadata network requests.
        meta = get_local_metadata(audio_path)
        meta.has_audio = check_has_audio(audio_path)
        return audio_path, meta

    if is_local_file(url_or_path):
        cmd = [
            ffmpeg_executable(),
            "-y",
            "-i", str(url_or_path),
            "-vn",
            "-ac", "1",
            "-ar", str(sample_rate),
            "-c:a", "pcm_s16le",
            str(audio_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        meta = get_local_metadata(Path(url_or_path))
        meta.has_audio = check_has_audio(audio_path)
        return audio_path, meta

    # Remote URL download
    tmp_template = str(out_dir / f"{video_id}_raw.%(ext)s")
    ydl_opts = {
        "outtmpl": tmp_template,
        "format": f"bestaudio[abr<={max_bitrate_kbps}]/bestaudio/worst[height<=240]/worst[height<=360]/worst",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook] if progress_hook else [],
        "ffmpeg_location": ffmpeg_executable(),
        "concurrent_fragment_downloads": 8,  # Parallel fragment downloads
    }

    def download() -> Dict[str, Any]:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url_or_path, download=True)

    info = _with_network_retries(download, "Audio download")

    raw_files = [p for p in out_dir.glob(f"{video_id}_raw.*") if not p.name.endswith((".mhtml", ".json"))]
    if not raw_files:
        raise RuntimeError(f"Audio download produced no audio stream for {url_or_path}")
    raw_path = raw_files[0]

    cmd = [
        ffmpeg_executable(),
        "-y",
        "-i", str(raw_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(audio_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw_path.unlink(missing_ok=True)
    meta = _metadata_from_info(info)
    meta.has_audio = check_has_audio(audio_path)
    if not meta.duration or meta.duration == 0.0:
        local_meta = get_local_metadata(audio_path)
        meta.duration = local_meta.duration
    return audio_path, meta


def download_video(
    url_or_path: str,
    output_dir: Union[str, Path] = "cache/videos",
    force: bool = False,
    max_height: int = 480,
    concurrent_fragments: int = 5,
    progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Path:
    """Download full video container or return local path — deferred until frame extraction."""
    if is_local_file(url_or_path):
        return Path(url_or_path).resolve()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_id = get_video_id(url_or_path)
    output_path = out_dir / f"{video_id}.mp4"

    if output_path.exists() and not force:
        return output_path

    ydl_opts = {
        "outtmpl": str(output_path),
        "format": f"bestvideo[height<={max_height}]+bestaudio/best[height<={max_height}]/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [progress_hook] if progress_hook else [],
        "ffmpeg_location": ffmpeg_executable(),
        "concurrent_fragment_downloads": concurrent_fragments,
    }
    def download() -> None:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url_or_path])

    _with_network_retries(download, "Video download")

    return output_path


class MediaManager:
    """High-level media manager coordinating audio and video asset workflows."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_default_config()

    def get_id(self, url_or_path: str) -> str:
        return get_video_id(url_or_path)

    def get_metadata(self, url_or_path: str) -> VideoMetadata:
        return get_light_metadata(url_or_path)

    def get_audio(self, url_or_path: str, progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None) -> Path:
        return download_audio_only(
            url_or_path,
            output_dir=self.config.audio_dir,
            sample_rate=self.config.sample_rate,
            max_bitrate_kbps=self.config.audio_max_bitrate_kbps,
            progress_hook=progress_hook,
        )

    def get_audio_with_metadata(
        self, url_or_path: str, progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> tuple[Path, VideoMetadata]:
        """Acquire audio and its metadata without a second request when downloading."""
        return download_audio_only_with_metadata(
            url_or_path,
            output_dir=self.config.audio_dir,
            sample_rate=self.config.sample_rate,
            max_bitrate_kbps=self.config.audio_max_bitrate_kbps,
            progress_hook=progress_hook,
        )

    def get_video(self, url_or_path: str, force: bool = False, progress_hook: Optional[Callable[[Dict[str, Any]], None]] = None) -> Path:
        return download_video(
            url_or_path,
            output_dir=self.config.video_dir,
            force=force,
            max_height=self.config.video_max_height,
            concurrent_fragments=self.config.concurrent_fragment_downloads,
            progress_hook=progress_hook,
        )

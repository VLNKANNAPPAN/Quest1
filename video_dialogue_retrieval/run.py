"""Run the video-dialogue CLI directly from a source checkout."""

import os
import sys
import warnings
from pathlib import Path

# Silence noisy external library warnings (TensorFlow oneDNN, optree, etc.)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_dialogue.cli import main


if __name__ == "__main__":
    main()


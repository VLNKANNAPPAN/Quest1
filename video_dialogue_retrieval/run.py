"""Run the video-dialogue CLI directly from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_dialogue.cli import main


if __name__ == "__main__":
    main()

from setuptools import setup, find_packages

setup(
    name="video-dialogue-retrieval",
    version="0.3.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "yt-dlp>=2023.12.30",
        "faster-whisper>=0.10.0",
        "rapidfuzz>=3.0.0",
        "pyacoustid>=1.2.0",
        "tabulate>=0.9.0",
        "imageio-ffmpeg>=0.6.0",
    ],
    extras_require={
        "embedding": ["sentence-transformers>=2.2.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "pandas>=2.0.0", "Pillow>=9.5.0", "torch>=2.0.0"],
    },
    entry_points={
        "console_scripts": [
            "video-dialogue=video_dialogue.cli:main",
        ],
    },
    python_requires=">=3.9",
)

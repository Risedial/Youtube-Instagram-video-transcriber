"""Processing package for YouTube Whisper Transcriber.

This package contains core processing modules for video downloading,
audio transcription, and related processing operations.
"""

__version__ = "1.0.0"
__author__ = "OPBL Project"

# Import main processing components for easier access
try:
    from .video_downloader import VideoDownloader
    from .whisper_transcriber import WhisperTranscriber
except ImportError:
    # Components not yet implemented - will be added in Phases 4-5
    pass

__all__ = [
    "VideoDownloader",
    "WhisperTranscriber",
]

"""Models package for YouTube Whisper Transcriber.

This package contains AI model management, Whisper model handling,
and model-related utilities.
"""

__version__ = "1.0.0"
__author__ = "OPBL Project"

# Import model components for easier access
try:
    from .model_manager import (
        ModelManager,
        download_model,
        get_available_models,
        get_model_info,
    )
except ImportError:
    # Components not yet implemented - will be added in Phase 5
    pass

__all__ = [
    "ModelManager",
    "download_model",
    "get_available_models",
    "get_model_info",
]

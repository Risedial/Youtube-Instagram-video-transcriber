"""Utilities package for YouTube Whisper Transcriber.

This package contains utility functions, helpers, and common operations
used throughout the application.
"""

__version__ = "1.0.0"
__author__ = "OPBL Project"

# Import utility components for easier access
try:
    from .file_manager import FileManager
    from .validators import (
        validate_youtube_url,
        validate_file_path,
        validate_model_name,
    )
    from .state_manager import StateManager
    from .error_handler import ErrorHandler
except ImportError:
    # Components not yet implemented - will be added in Phase 6
    pass

__all__ = [
    "FileManager",
    "validate_youtube_url",
    "validate_file_path", 
    "validate_model_name",
    "StateManager",
    "ErrorHandler",
]

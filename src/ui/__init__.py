"""UI package for YouTube Whisper Transcriber.

This package contains all user interface components including the main window,
reusable UI components, and GUI-related utilities.
"""

__version__ = "1.0.0"
__author__ = "OPBL Project"

# Import main UI components for easier access
try:
    from .main_window import MainWindow
    from .components import (
        ProgressBar,
        FileSelector,
        ModelSelector,
        URLInput,
    )
except ImportError:
    # Components not yet implemented - will be added in Phase 3
    pass

__all__ = [
    "MainWindow",
    "ProgressBar", 
    "FileSelector",
    "ModelSelector",
    "URLInput",
]

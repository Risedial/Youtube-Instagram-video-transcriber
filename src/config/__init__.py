"""Configuration package for YouTube Whisper Transcriber.

This package contains application configuration, settings management,
and environment variable handling.
"""

__version__ = "1.0.0"
__author__ = "OPBL Project"

# Import configuration components for easier access
try:
    from .settings import (
        Settings,
        load_settings,
        save_settings,
        get_default_settings,
    )
except ImportError:
    # Components not yet implemented - will be added in Phase 6
    pass

__all__ = [
    "Settings",
    "load_settings",
    "save_settings",
    "get_default_settings",
]

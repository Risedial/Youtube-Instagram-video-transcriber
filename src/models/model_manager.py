"""Whisper model management for YouTube Whisper Transcriber.

This module will handle Whisper model downloading, caching, and management.
To be implemented in Phase 5: Whisper Transcription Integration.
"""

from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import os


class ModelManager:
    """Whisper model management and caching.
    
    This class will implement:
    - Automatic model downloading and caching
    - Model validation and integrity checks
    - Storage optimization and cleanup
    - Model information and metadata
    
    To be fully implemented in Phase 5.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        """Initialize model manager.
        
        Args:
            cache_dir: Directory for model cache, uses default if None
        """
        self.cache_dir = cache_dir or self._get_default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model information
        self.model_info = {
            "tiny": {
                "size": "39MB",
                "memory": "1GB", 
                "speed": "fastest",
                "accuracy": "good",
                "languages": "multilingual"
            },
            "base": {
                "size": "74MB",
                "memory": "1GB",
                "speed": "fast", 
                "accuracy": "better",
                "languages": "multilingual"
            },
            "small": {
                "size": "244MB",
                "memory": "2GB",
                "speed": "medium",
                "accuracy": "great", 
                "languages": "multilingual"
            },
            "medium": {
                "size": "769MB",
                "memory": "5GB",
                "speed": "slow",
                "accuracy": "excellent",
                "languages": "multilingual"
            },
            "large": {
                "size": "1550MB", 
                "memory": "10GB",
                "speed": "slowest",
                "accuracy": "best",
                "languages": "multilingual"
            }
        }
        
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        if os.name == 'nt':  # Windows
            cache_base = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData' / 'Local'))
        else:  # Unix-like
            cache_base = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
            
        return cache_base / 'youtube_whisper' / 'models'
        
    def is_model_cached(self, model_name: str) -> bool:
        """Check if model is already cached.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if model is cached and valid
            
        To be implemented in Phase 5.
        """
        if model_name not in self.model_info:
            return False
            
        # Implementation placeholder - will check for model files
        model_path = self.cache_dir / f"{model_name}.pt"
        return model_path.exists()
        
    def download_model(self, model_name: str, progress_callback: Optional[callable] = None) -> bool:
        """Download and cache Whisper model.
        
        Args:
            model_name: Name of model to download
            progress_callback: Progress update function
            
        Returns:
            True if download successful
            
        To be implemented in Phase 5.
        """
        if model_name not in self.model_info:
            self.logger.error(f"Invalid model name: {model_name}")
            return False
            
        if self.is_model_cached(model_name):
            self.logger.info(f"Model {model_name} already cached")
            return True
            
        self.logger.info(f"Downloading model: {model_name}")
        
        if progress_callback:
            progress_callback(0.0, f"Starting download of {model_name} model...")
            
        # Implementation placeholder - will use whisper model downloading
        # This will include:
        # - Model download with progress tracking
        # - Integrity verification
        # - Cache storage
        # - Error handling
        
        if progress_callback:
            progress_callback(100.0, f"Model {model_name} downloaded successfully")
            
        return True
        
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to cached model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Path to model file, None if not cached
        """
        if not self.is_model_cached(model_name):
            return None
            
        return self.cache_dir / f"{model_name}.pt"
        
    def get_available_models(self) -> List[str]:
        """Get list of available model names.
        
        Returns:
            List of model names
        """
        return list(self.model_info.keys())
        
    def get_model_info(self, model_name: str) -> Optional[Dict[str, str]]:
        """Get information about a model.
        
        Args:
            model_name: Name of model
            
        Returns:
            Dictionary with model information
        """
        return self.model_info.get(model_name)
        
    def get_cached_models(self) -> List[str]:
        """Get list of currently cached models.
        
        Returns:
            List of cached model names
        """
        cached = []
        for model_name in self.model_info.keys():
            if self.is_model_cached(model_name):
                cached.append(model_name)
        return cached
        
    def get_cache_size(self) -> int:
        """Get total size of model cache in bytes.
        
        Returns:
            Cache size in bytes
        """
        total_size = 0
        if self.cache_dir.exists():
            for file_path in self.cache_dir.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        return total_size
        
    def clear_cache(self, model_name: Optional[str] = None) -> bool:
        """Clear model cache.
        
        Args:
            model_name: Specific model to clear, all if None
            
        Returns:
            True if successful
        """
        try:
            if model_name:
                model_path = self.cache_dir / f"{model_name}.pt"
                if model_path.exists():
                    model_path.unlink()
                    self.logger.info(f"Cleared cache for model: {model_name}")
            else:
                if self.cache_dir.exists():
                    import shutil
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info("Cleared all model cache")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False


# Utility functions
def download_model(model_name: str, cache_dir: Optional[Path] = None, progress_callback: Optional[callable] = None) -> bool:
    """Download Whisper model.
    
    Args:
        model_name: Name of model to download
        cache_dir: Cache directory, uses default if None  
        progress_callback: Progress update function
        
    Returns:
        True if download successful
    """
    manager = ModelManager(cache_dir)
    return manager.download_model(model_name, progress_callback)


def get_available_models() -> List[str]:
    """Get list of available Whisper models.
    
    Returns:
        List of model names
    """
    manager = ModelManager()
    return manager.get_available_models()


def get_model_info(model_name: str) -> Optional[Dict[str, str]]:
    """Get information about a Whisper model.
    
    Args:
        model_name: Name of model
        
    Returns:
        Dictionary with model information
    """
    manager = ModelManager()
    return manager.get_model_info(model_name)


def is_model_available(model_name: str, cache_dir: Optional[Path] = None) -> bool:
    """Check if model is available (cached).
    
    Args:
        model_name: Name of model to check
        cache_dir: Cache directory, uses default if None
        
    Returns:
        True if model is available
    """
    manager = ModelManager(cache_dir)
    return manager.is_model_cached(model_name)
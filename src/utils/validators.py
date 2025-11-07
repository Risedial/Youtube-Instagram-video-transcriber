"""Input validation utilities for YouTube Whisper Transcriber.

This module contains comprehensive validation functions for URLs, file paths,
model names, and other user inputs with detailed error reporting.
"""

import re
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, parse_qs
import logging
from utils.url_validator import URLValidator


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class URLValidator:
    """Comprehensive YouTube URL validation with detailed analysis."""
    
    # Comprehensive YouTube URL patterns with capture groups
    YOUTUBE_PATTERNS = [
        # Standard youtube.com/watch URLs
        r'^(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:&.*)?$',
        # Short youtu.be URLs
        r'^(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})(?:\?.*)?$',
        # Embed URLs
        r'^(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})(?:\?.*)?$',
        # Mobile URLs
        r'^(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})(?:&.*)?$',
        # Alternative video URLs
        r'^(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})(?:\?.*)?$',
        # YouTube Shorts
        r'^(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})(?:\?.*)?$',
        # Playlist URLs with video ID
        r'^(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})&list=.*$',
        # Time-specific URLs
        r'^(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})&t=\d+s?(?:&.*)?$'
    ]
    
    def __init__(self):
        """Initialize URL validator."""
        self.logger = logging.getLogger(__name__)
        
    def validate_youtube_url(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate YouTube URL with detailed analysis.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'url': url,
            'is_valid': False,
            'video_id': None,
            'url_type': None,
            'normalized_url': None,
            'errors': [],
            'warnings': []
        }
        
        # Basic input validation
        if not url or not isinstance(url, str):
            validation_info['errors'].append("URL is empty or not a string")
            return False, validation_info
            
        url = url.strip()
        if not url:
            validation_info['errors'].append("URL is empty after trimming whitespace")
            return False, validation_info
            
        # Check URL length (YouTube URLs shouldn't be extremely long)
        if len(url) > 2000:
            validation_info['errors'].append("URL is too long (over 2000 characters)")
            return False, validation_info
            
        # Try to match against YouTube patterns
        video_id = None
        url_type = None
        
        for i, pattern in enumerate(self.YOUTUBE_PATTERNS):
            match = re.match(pattern, url, re.IGNORECASE)
            if match:
                video_id = match.group(1)
                url_type = self._get_url_type(i)
                break
                
        if not video_id:
            validation_info['errors'].append("URL does not match any known YouTube URL pattern")
            return False, validation_info
            
        # Validate video ID format
        if not self._validate_video_id(video_id):
            validation_info['errors'].append(f"Invalid video ID format: {video_id}")
            return False, validation_info
            
        # Check for suspicious patterns
        self._check_suspicious_patterns(url, validation_info)
        
        # Generate normalized URL
        normalized_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Populate validation info
        validation_info.update({
            'is_valid': True,
            'video_id': video_id,
            'url_type': url_type,
            'normalized_url': normalized_url
        })
        
        return True, validation_info
        
    def _get_url_type(self, pattern_index: int) -> str:
        """Get URL type based on pattern index."""
        url_types = [
            'standard',      # youtube.com/watch
            'short',         # youtu.be
            'embed',         # youtube.com/embed
            'mobile',        # m.youtube.com
            'alternative',   # youtube.com/v
            'shorts',        # youtube.com/shorts
            'playlist',      # with playlist
            'timed'          # with timestamp
        ]
        return url_types[pattern_index] if pattern_index < len(url_types) else 'unknown'
        
    def _validate_video_id(self, video_id: str) -> bool:
        """Validate YouTube video ID format.
        
        Args:
            video_id: Video ID to validate
            
        Returns:
            True if valid video ID format
        """
        # YouTube video IDs are exactly 11 characters
        if len(video_id) != 11:
            return False
            
        # Should contain only valid characters
        valid_chars = re.match(r'^[a-zA-Z0-9_-]+$', video_id)
        return valid_chars is not None
        
    def _check_suspicious_patterns(self, url: str, validation_info: Dict[str, Any]) -> None:
        """Check for suspicious patterns in URL.
        
        Args:
            url: URL to check
            validation_info: Validation info dictionary to update
        """
        # Check for suspicious query parameters
        parsed = urlparse(url)
        if parsed.query:
            query_params = parse_qs(parsed.query)
            
            # Check for unusual parameters that might indicate malicious URLs
            suspicious_params = ['redirect', 'goto', 'url', 'link', 'ref']
            for param in suspicious_params:
                if param in query_params:
                    validation_info['warnings'].append(f"URL contains suspicious parameter: {param}")
                    
        # Check for excessively long URLs
        if len(url) > 500:
            validation_info['warnings'].append("URL is unusually long")
            
        # Check for non-standard domains
        domain = parsed.netloc.lower()
        valid_domains = [
            'youtube.com', 'www.youtube.com', 'm.youtube.com',
            'youtu.be', 'gaming.youtube.com'
        ]
        
        if domain and domain not in valid_domains:
            validation_info['warnings'].append(f"Non-standard YouTube domain: {domain}")
            
    def check_url_accessibility(self, url: str, timeout: int = 10) -> Tuple[bool, Dict[str, Any]]:
        """Check if YouTube URL is accessible (not private/deleted).
        
        Args:
            url: YouTube URL to check
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (is_accessible, check_info)
        """
        check_info = {
            'url': url,
            'is_accessible': False,
            'status_code': None,
            'error': None,
            'redirect_url': None
        }
        
        try:
            # Use HEAD request to check accessibility without downloading content
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            check_info['status_code'] = response.status_code
            
            if response.url != url:
                check_info['redirect_url'] = response.url
                
            # Check if URL is accessible
            if response.status_code == 200:
                check_info['is_accessible'] = True
            elif response.status_code == 404:
                check_info['error'] = "Video not found (404)"
            elif response.status_code == 403:
                check_info['error'] = "Video access forbidden (403) - may be private"
            elif response.status_code == 429:
                check_info['error'] = "Too many requests (429) - rate limited"
            else:
                check_info['error'] = f"HTTP {response.status_code}"
                
        except requests.RequestException as e:
            check_info['error'] = f"Network error: {str(e)}"
            
        return check_info['is_accessible'], check_info
        

def validate_youtube_url(url: str) -> bool:
    """Simple video URL validation function for supported platforms.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid video URL (YouTube or Instagram)
    """
    from utils.url_validator import URLValidator
    is_valid, _ = URLValidator.validate_url(url)
    return is_valid


def validate_youtube_url_detailed(url: str) -> Tuple[bool, Dict[str, Any]]:
    """Detailed YouTube URL validation with analysis.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validator = URLValidator()
    return validator.validate_youtube_url(url)


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID string or None if extraction fails
    """
    validator = URLValidator()
    is_valid, info = validator.validate_youtube_url(url)
    return info.get('video_id') if is_valid else None


def normalize_youtube_url(url: str) -> Optional[str]:
    """Normalize YouTube URL to standard format.
    
    Args:
        url: YouTube URL to normalize
        
    Returns:
        Normalized URL or None if invalid
    """
    validator = URLValidator()
    is_valid, info = validator.validate_youtube_url(url)
    return info.get('normalized_url') if is_valid else None


def validate_file_path(path: str, must_exist: bool = False, must_be_writable: bool = False) -> Tuple[bool, Dict[str, Any]]:
    """Validate file path with comprehensive checks.
    
    Args:
        path: File path to validate
        must_exist: Path must exist
        must_be_writable: Path must be writable
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        'path': path,
        'is_valid': False,
        'exists': False,
        'is_file': False,
        'is_directory': False,
        'is_writable': False,
        'is_readable': False,
        'parent_exists': False,
        'errors': [],
        'warnings': []
    }
    
    if not path or not isinstance(path, str):
        validation_info['errors'].append("Path is empty or not a string")
        return False, validation_info
        
    try:
        path_obj = Path(path)
        
        # Check path validity
        validation_info.update({
            'exists': path_obj.exists(),
            'is_file': path_obj.is_file() if path_obj.exists() else False,
            'is_directory': path_obj.is_dir() if path_obj.exists() else False,
            'parent_exists': path_obj.parent.exists()
        })
        
        # Check permissions
        if path_obj.exists():
            validation_info.update({
                'is_readable': os.access(path_obj, os.R_OK),
                'is_writable': os.access(path_obj, os.W_OK)
            })
        elif path_obj.parent.exists():
            validation_info['is_writable'] = os.access(path_obj.parent, os.W_OK)
            
        # Check requirements
        if must_exist and not validation_info['exists']:
            validation_info['errors'].append("Path must exist but does not")
            
        if must_be_writable and not validation_info['is_writable']:
            validation_info['errors'].append("Path must be writable but is not")
            
        # Check for problematic characters
        problematic_chars = ['<', '>', ':', '"', '|', '?', '*']
        if any(char in path for char in problematic_chars):
            validation_info['warnings'].append("Path contains potentially problematic characters")
            
        # Check path length
        if len(path) > 260:  # Windows MAX_PATH limitation
            validation_info['warnings'].append("Path exceeds Windows MAX_PATH limit (260 characters)")
            
        validation_info['is_valid'] = len(validation_info['errors']) == 0
        
    except Exception as e:
        validation_info['errors'].append(f"Path validation error: {str(e)}")
        
    return validation_info['is_valid'], validation_info


def validate_model_name(model: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate Whisper model name with details.
    
    Args:
        model: Model name to validate
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        'model': model,
        'is_valid': False,
        'model_info': None,
        'errors': [],
        'warnings': []
    }
    
    # Valid Whisper models with their characteristics
    valid_models = {
        'tiny': {
            'size': '39 MB',
            'speed': '~32x realtime',
            'memory': '~1 GB',
            'quality': 'Basic',
            'languages': 'English optimized'
        },
        'base': {
            'size': '74 MB', 
            'speed': '~16x realtime',
            'memory': '~1 GB',
            'quality': 'Good',
            'languages': 'Multilingual'
        },
        'small': {
            'size': '244 MB',
            'speed': '~6x realtime', 
            'memory': '~2 GB',
            'quality': 'Better',
            'languages': 'Multilingual'
        },
        'medium': {
            'size': '769 MB',
            'speed': '~2x realtime',
            'memory': '~5 GB', 
            'quality': 'High',
            'languages': 'Multilingual'
        },
        'large': {
            'size': '1550 MB',
            'speed': '~1x realtime',
            'memory': '~10 GB',
            'quality': 'Best',
            'languages': 'Multilingual'
        }
    }
    
    if not model or not isinstance(model, str):
        validation_info['errors'].append("Model name is empty or not a string")
        return False, validation_info
        
    model = model.strip().lower()
    
    if model in valid_models:
        validation_info.update({
            'is_valid': True,
            'model_info': valid_models[model]
        })
    else:
        validation_info['errors'].append(f"Invalid model name. Valid options: {', '.join(valid_models.keys())}")
        
    return validation_info['is_valid'], validation_info


def validate_output_filename(filename: str, extension: str = '.txt') -> Tuple[bool, Dict[str, Any]]:
    """Validate output filename with safety checks.
    
    Args:
        filename: Filename to validate
        extension: Expected file extension
        
    Returns:
        Tuple of (is_valid, validation_info)
    """
    validation_info = {
        'filename': filename,
        'is_valid': False,
        'safe_filename': None,
        'errors': [],
        'warnings': []
    }
    
    if not filename or not isinstance(filename, str):
        validation_info['errors'].append("Filename is empty or not a string")
        return False, validation_info
        
    # Generate safe filename
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'[^\w\s\-_\.]', '', safe_name)
    safe_name = re.sub(r'\s+', '_', safe_name)
    
    # Ensure proper extension
    if not safe_name.endswith(extension):
        safe_name += extension
        
    # Check length
    if len(safe_name) > 255:
        validation_info['warnings'].append("Filename too long, will be truncated")
        safe_name = safe_name[:255-len(extension)] + extension
        
    # Check for reserved names (Windows)
    reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
    base_name = safe_name.split('.')[0].upper()
    if base_name in reserved_names:
        validation_info['warnings'].append(f"Filename uses reserved name: {base_name}")
        safe_name = f"file_{safe_name}"
        
    validation_info.update({
        'is_valid': True,
        'safe_filename': safe_name
    })
    
    return True, validation_info


def validate_disk_space(path: str, required_gb: float = 1.0) -> Tuple[bool, Dict[str, Any]]:
    """Validate available disk space.
    
    Args:
        path: Path to check
        required_gb: Required space in GB
        
    Returns:
        Tuple of (sufficient_space, space_info)
    """
    try:
        from utils.file_manager import FileManager
        manager = FileManager()
        required_bytes = int(required_gb * 1024**3)
        space_info = manager.check_disk_space(Path(path), required_bytes)
        return space_info['sufficient'], space_info
    except Exception as e:
        return False, {'error': str(e), 'sufficient': False}


# Import os for file access checks
import os
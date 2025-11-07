"""URL validation utilities for YouTube Whisper Transcriber.

This module provides comprehensive URL validation for supported platforms
including YouTube and Instagram, with platform detection and format validation.
"""

import re
from typing import Optional, Tuple
from enum import Enum


class SupportedPlatform(Enum):
    """Enumeration of supported video platforms."""
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    UNKNOWN = "unknown"


class URLValidator:
    """Comprehensive URL validator for supported video platforms."""
    
    # YouTube URL patterns
    YOUTUBE_PATTERNS = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)'
    ]
    
    # Instagram URL patterns
    INSTAGRAM_PATTERNS = [
        r'(?:https?://)?(?:www\.)?instagram\.com/reel/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?instagram\.com/p/([a-zA-Z0-9_-]+)',
        r'(?:https?://)?(?:www\.)?instagram\.com/tv/([a-zA-Z0-9_-]+)',
    ]
    
    @classmethod
    def detect_platform(cls, url: str) -> SupportedPlatform:
        """Detect which platform a URL belongs to.
        
        Args:
            url: URL to analyze
            
        Returns:
            SupportedPlatform enum value
        """
        if not url or not isinstance(url, str):
            return SupportedPlatform.UNKNOWN
            
        url_lower = url.lower()
        
        if any('youtube.com' in url_lower or 'youtu.be' in url_lower for _ in [1]):
            return SupportedPlatform.YOUTUBE
        elif 'instagram.com' in url_lower:
            return SupportedPlatform.INSTAGRAM
        else:
            return SupportedPlatform.UNKNOWN
    
    @classmethod
    def validate_url(cls, url: str) -> Tuple[bool, SupportedPlatform]:
        """Validate URL and detect platform.
        
        Args:
            url: URL to validate
            
        Returns:
            Tuple of (is_valid, platform)
        """
        if not url or not isinstance(url, str):
            return False, SupportedPlatform.UNKNOWN
            
        # Check YouTube patterns
        for pattern in cls.YOUTUBE_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True, SupportedPlatform.YOUTUBE
                
        # Check Instagram patterns  
        for pattern in cls.INSTAGRAM_PATTERNS:
            if re.search(pattern, url, re.IGNORECASE):
                return True, SupportedPlatform.INSTAGRAM
                
        return False, SupportedPlatform.UNKNOWN
    
    @classmethod
    def validate_youtube_url(cls, url: str) -> bool:
        """Validate if URL is a valid YouTube URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid YouTube URL
        """
        if not url or not isinstance(url, str):
            return False
            
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in cls.YOUTUBE_PATTERNS)
    
    @classmethod
    def validate_instagram_url(cls, url: str) -> bool:
        """Validate if URL is a valid Instagram URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid Instagram URL
        """
        if not url or not isinstance(url, str):
            return False
            
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in cls.INSTAGRAM_PATTERNS)
    
    @classmethod
    def extract_youtube_id(cls, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID string or None if extraction fails
        """
        for pattern in cls.YOUTUBE_PATTERNS:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def extract_instagram_shortcode(cls, url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL.
        
        Args:
            url: Instagram URL
            
        Returns:
            Shortcode string or None if extraction fails
        """
        for pattern in cls.INSTAGRAM_PATTERNS:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    @classmethod
    def normalize_url(cls, url: str) -> str:
        """Normalize URL by adding protocol if missing.
        
        Args:
            url: URL to normalize
            
        Returns:
            Normalized URL with protocol
        """
        if not url:
            return url
            
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        return url
    
    @classmethod
    def get_platform_name(cls, platform: SupportedPlatform) -> str:
        """Get human-readable platform name.
        
        Args:
            platform: SupportedPlatform enum value
            
        Returns:
            Human-readable platform name
        """
        platform_names = {
            SupportedPlatform.YOUTUBE: "YouTube",
            SupportedPlatform.INSTAGRAM: "Instagram", 
            SupportedPlatform.UNKNOWN: "Unknown"
        }
        return platform_names.get(platform, "Unknown")


# Convenience functions for backward compatibility
def validate_url(url: str) -> Tuple[bool, str]:
    """Validate URL and return platform name.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_valid, platform_name)
    """
    is_valid, platform = URLValidator.validate_url(url)
    return is_valid, URLValidator.get_platform_name(platform)


def detect_platform(url: str) -> str:
    """Detect platform from URL.
    
    Args:
        url: URL to analyze
        
    Returns:
        Platform name string
    """
    platform = URLValidator.detect_platform(url)
    return URLValidator.get_platform_name(platform)


def is_supported_url(url: str) -> bool:
    """Check if URL is from a supported platform.
    
    Args:
        url: URL to check
        
    Returns:
        True if URL is supported
    """
    is_valid, _ = URLValidator.validate_url(url)
    return is_valid
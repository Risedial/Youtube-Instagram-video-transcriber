"""Instagram Reels download functionality for YouTube Whisper Transcriber.

This module handles Instagram Reels downloading using Instaloader with progress tracking,
video extraction optimized for Whisper, and comprehensive error handling.

LEGAL WARNING: This module downloads content from Instagram which may violate Instagram's
Terms of Service. Use only for personal content or with explicit permission from content owners.
Users assume all legal responsibility for the use of this functionality.
"""

from typing import Optional, Callable, Dict, Any, Union
from pathlib import Path
import tempfile
import logging
import threading
import time
import random
import re
import os
from urllib.parse import urlparse

try:
    import instaloader
except ImportError:
    instaloader = None

# Import proxy manager
try:
    from utils.proxy_manager import ProxyManager
    PROXY_MANAGER_AVAILABLE = True
except ImportError:
    PROXY_MANAGER_AVAILABLE = False


class InstagramDownloadError(Exception):
    """Custom exception for Instagram download-related errors."""
    pass


class InstagramDownloader:
    """Instagram Reels downloader with progress tracking and video extraction.
    
    Features:
    - Comprehensive Instagram URL validation for Reels
    - Instaloader integration with video extraction for Whisper
    - Thread-safe progress reporting compatible with Tkinter GUI
    - Robust error handling for network issues, private accounts, unavailable content
    - Download cancellation support with proper cleanup
    - Memory-efficient processing for video files
    - Automatic temporary file management with cleanup
    
    Legal Notice:
    This downloader may violate Instagram's Terms of Service. Use only for:
    - Personal content you own
    - Content with explicit permission from creators
    - Educational or research purposes within legal bounds
    Users assume full legal responsibility for usage.
    """
    
    # Supported video formats for Whisper (in order of preference)
    WHISPER_VIDEO_FORMATS = ['mp4', 'mov', 'avi', 'mkv']
    
    # Maximum video duration in seconds (10 minutes for Reels)
    MAX_VIDEO_DURATION = 600
    
    def __init__(self, temp_dir: Optional[Path] = None, proxy_manager: Optional['ProxyManager'] = None) -> None:
        """Initialize Instagram downloader.
        
        Args:
            temp_dir: Temporary directory for downloads, uses system temp if None
            proxy_manager: ProxyManager instance for proxy support and anti-detection
            
        Raises:
            ImportError: If instaloader is not available
        """
        if instaloader is None:
            raise ImportError("instaloader is required but not installed. Install with: pip install instaloader")
            
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "instagram_whisper"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize Instaloader with proxy support
        self.loader = self._create_instaloader()
        
        # Session management for anti-detection
        self.current_session_requests = 0
        self.session_start_time = time.time()
        self.last_request_time = 0.0
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.status_callback: Optional[Callable[[str, str], None]] = None
        
        # Current download state
        self.current_download: Optional[str] = None
        self.cancel_event = threading.Event()
        self.download_lock = threading.Lock()
        
        # Download statistics
        self.download_stats = {
            'total_items': 0,
            'downloaded_items': 0,
            'current_item': '',
            'status': 'idle'
        }
        
        self.logger.info(f"InstagramDownloader initialized with temp_dir: {self.temp_dir}")
        
        if self.proxy_manager:
            self.logger.info("Proxy support enabled for Instagram downloads")
    
    def _create_instaloader(self) -> 'instaloader.Instaloader':
        """Create Instaloader instance with proxy configuration.
        
        Returns:
            Configured Instaloader instance
        """
        # Base configuration
        loader_config = {
            'download_videos': True,
            'download_video_thumbnails': False,
            'download_geotags': False,
            'download_comments': False,
            'save_metadata': False,
            'dirname_pattern': str(self.temp_dir),
            'filename_pattern': "{shortcode}_{profile}",
            'quiet': True,
        }
        
        # Create loader
        loader = instaloader.Instaloader(**loader_config)
        
        # Configure proxy if available
        if self.proxy_manager:
            proxy = self.proxy_manager.get_healthy_proxy()
            if proxy:
                # Configure requests session with proxy
                session = self.proxy_manager.get_proxy_session("instagram")
                loader.context._session = session
                
                # Set anti-detection headers
                headers = self.proxy_manager.get_randomized_headers("instagram")
                for key, value in headers.items():
                    loader.context._session.headers[key] = value
                
                self.logger.debug(f"Configured Instagram loader with proxy: {proxy.host}:{proxy.port}")
        
        return loader
    
    def _apply_instagram_anti_detection(self):
        """Apply Instagram-specific anti-detection measures."""
        if not self.proxy_manager:
            return
            
        current_time = time.time()
        
        # Check if we need to start a new session (Instagram-specific limits)
        if (self.current_session_requests >= 10 or 
            current_time - self.session_start_time > 300):  # 5 minutes
            
            self._start_new_session()
        
        # Apply delays between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 30:  # Minimum 30 seconds between requests
            delay = 30 - time_since_last + random.uniform(5, 15)  # Add random component
            self.logger.debug(f"Instagram anti-detection delay: {delay:.1f}s")
            time.sleep(delay)
        
        self.last_request_time = time.time()
        self.current_session_requests += 1
    
    def _start_new_session(self):
        """Start a new Instagram session with fresh proxy and headers."""
        if not self.proxy_manager:
            return
            
        self.logger.info("Starting new Instagram session for anti-detection")
        
        # Rotate proxy if available
        new_proxy = self.proxy_manager.rotate_proxy("instagram_session_rotation")
        
        # Create fresh loader with new proxy
        self.loader = self._create_instaloader()
        
        # Reset session counters
        self.current_session_requests = 0
        self.session_start_time = time.time()
        
        # Add cooldown period between sessions
        cooldown = random.uniform(60, 120)  # 1-2 minutes
        self.logger.debug(f"Instagram session cooldown: {cooldown:.1f}s")
        time.sleep(cooldown)
    
    def _is_instagram_error(self, error: Exception) -> bool:
        """Check if error is related to Instagram blocking or proxy issues.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is Instagram/proxy-related
        """
        error_str = str(error).lower()
        instagram_error_indicators = [
            'rate limit', 'too many requests', 'blocked', 'banned', 
            'challenge', 'login required', 'private', 'not found',
            'connection', 'timeout', 'proxy', '403', '429', '502', '503', '504'
        ]
        
        return any(indicator in error_str for indicator in instagram_error_indicators)
        
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function that takes (percentage, status_message)
        """
        self.progress_callback = callback
        
    def set_status_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set status callback function.
        
        Args:
            callback: Function that takes (message, status_type)
        """
        self.status_callback = callback
        
    def validate_url(self, url: str) -> bool:
        """Validate Instagram URL format for Reels.
        
        Args:
            url: Instagram URL to validate
            
        Returns:
            True if URL is valid Instagram Reel URL
        """
        if not url or not isinstance(url, str):
            return False
            
        # Instagram Reel URL patterns
        instagram_patterns = [
            r'(?:https?://)?(?:www\.)?instagram\.com/reel/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?instagram\.com/p/([a-zA-Z0-9_-]+)',  # Posts that might be videos
            r'(?:https?://)?(?:www\.)?instagram\.com/tv/([a-zA-Z0-9_-]+)',  # IGTV
        ]
        
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in instagram_patterns)
        
    def extract_shortcode(self, url: str) -> Optional[str]:
        """Extract shortcode from Instagram URL.
        
        Args:
            url: Instagram URL
            
        Returns:
            Shortcode string or None if extraction fails
        """
        instagram_patterns = [
            r'(?:https?://)?(?:www\.)?instagram\.com/reel/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?instagram\.com/p/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?instagram\.com/tv/([a-zA-Z0-9_-]+)',
        ]
        
        for pattern in instagram_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
        
    def get_reel_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get Instagram Reel information without downloading.
        
        Args:
            url: Instagram URL
            
        Returns:
            Dictionary with reel info (caption, duration, etc.) or None if failed
        """
        if not self.validate_url(url):
            return None
            
        try:
            # Apply anti-detection measures
            self._apply_instagram_anti_detection()
            
            shortcode = self.extract_shortcode(url)
            if not shortcode:
                return None
                
            # Get post information
            post = instaloader.Post.from_shortcode(self.loader.context, shortcode)
            
            return {
                'shortcode': post.shortcode,
                'caption': post.caption or 'No caption',
                'is_video': post.is_video,
                'video_duration': post.video_duration if post.is_video else 0,
                'owner_username': post.owner_username,
                'typename': post.typename,
                'date': post.date_local,
                'likes': post.likes,
                'comments': post.comments,
                'url': post.url,
                'video_url': post.video_url if post.is_video else None
            }
                
        except Exception as e:
            self.logger.error(f"Error getting reel info for {url}: {e}")
            
            # Handle proxy errors
            if self.proxy_manager:
                proxy = self.proxy_manager.get_current_proxy()
                if proxy and self._is_instagram_error(e):
                    self.proxy_manager.mark_proxy_failed(proxy, e)
                    
            return None
            
    def _validate_reel_for_transcription(self, info: Dict[str, Any]) -> None:
        """Validate reel suitability for transcription.
        
        Args:
            info: Reel information dictionary
            
        Raises:
            InstagramDownloadError: If reel is not suitable for transcription
        """
        # Check if it's a video
        if not info.get('is_video'):
            raise InstagramDownloadError("This Instagram post is not a video and cannot be transcribed.")
            
        # Check duration
        duration = info.get('video_duration', 0)
        if duration > self.MAX_VIDEO_DURATION:
            minutes = duration // 60
            seconds = duration % 60
            raise InstagramDownloadError(
                f"Video is too long ({minutes}m {seconds}s). Maximum supported duration is 10 minutes."
            )
            
        # Check if video URL is available
        if not info.get('video_url'):
            raise InstagramDownloadError("Video URL not available. The content may be private or restricted.")
            
    def download_reel(self, url: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Download Instagram Reel video for transcription.
        
        Args:
            url: Instagram URL to download
            output_path: Output directory, uses temp_dir if None
            
        Returns:
            Path to downloaded video file, None if failed
            
        Raises:
            InstagramDownloadError: If download fails
        """
        with self.download_lock:
            if not self.validate_url(url):
                raise InstagramDownloadError(f"Invalid Instagram URL format: {url}")
                
            self.current_download = url
            self.cancel_event.clear()
            
            try:
                # Get reel information first
                if self.progress_callback:
                    self.progress_callback(5.0, "Getting reel information...")
                    
                info = self.get_reel_info(url)
                if not info:
                    raise InstagramDownloadError("Could not retrieve reel information")
                    
                # Validate reel for transcription
                self._validate_reel_for_transcription(info)
                
                if self.cancel_event.is_set():
                    return None
                    
                # Prepare output path
                output_dir = output_path or self.temp_dir
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                if self.progress_callback:
                    self.progress_callback(20.0, "Starting download...")
                    
                if self.status_callback:
                    duration_str = f"{int(info['video_duration'])//60}:{int(info['video_duration'])%60:02d}" if info['video_duration'] else "Unknown"
                    self.status_callback(
                        f"Downloading Instagram Reel\n"
                        f"Duration: {duration_str} | Creator: @{info['owner_username']}\n"
                        f"Caption: {info['caption'][:100]}{'...' if len(info['caption']) > 100 else ''}",
                        "processing"
                    )
                
                # Update download stats
                self.download_stats.update({
                    'total_items': 1,
                    'downloaded_items': 0,
                    'current_item': info['shortcode'],
                    'status': 'downloading'
                })
                
                # Set up progress tracking
                original_dirname = self.loader.dirname_pattern
                self.loader.dirname_pattern = str(output_dir)
                
                try:
                    # Apply anti-detection before download
                    self._apply_instagram_anti_detection()
                    
                    # Download the post
                    shortcode = self.extract_shortcode(url)
                    post = instaloader.Post.from_shortcode(self.loader.context, shortcode)
                    
                    if self.progress_callback:
                        self.progress_callback(50.0, "Downloading video file...")
                    
                    # Download using Instaloader
                    self.loader.download_post(post, target=output_dir)
                    
                    if self.cancel_event.is_set():
                        return None
                        
                    if self.progress_callback:
                        self.progress_callback(80.0, "Processing downloaded file...")
                    
                    # Find the downloaded video file
                    video_files = []
                    for ext in self.WHISPER_VIDEO_FORMATS:
                        video_files.extend(list(output_dir.glob(f"*{shortcode}*.{ext}")))
                    
                    if not video_files:
                        # Try alternative pattern
                        video_files = list(output_dir.glob(f"*{shortcode}*"))
                        video_files = [f for f in video_files if f.suffix.lower() in ['.mp4', '.mov', '.avi', '.mkv']]
                    
                    if not video_files:
                        raise InstagramDownloadError("Video file not found after download")
                        
                    downloaded_file = video_files[0]
                    
                    # Verify file exists and has reasonable size
                    if not downloaded_file.exists() or downloaded_file.stat().st_size < 1024:
                        raise InstagramDownloadError("Downloaded video file is invalid or too small")
                    
                    # Update stats
                    self.download_stats['downloaded_items'] = 1
                    self.download_stats['status'] = 'completed'
                    
                    self.logger.info(f"Successfully downloaded Instagram reel: {downloaded_file}")
                    
                    if self.progress_callback:
                        self.progress_callback(100.0, "Download completed successfully")
                        
                    if self.status_callback:
                        file_size = downloaded_file.stat().st_size / (1024 * 1024)  # MB
                        self.status_callback(
                            f"Download completed successfully!\n"
                            f"Video file: {downloaded_file.name}\n"
                            f"File size: {file_size:.1f} MB",
                            "success"
                        )
                        
                    return downloaded_file
                    
                finally:
                    # Restore original dirname pattern
                    self.loader.dirname_pattern = original_dirname
                    
            except instaloader.exceptions.InstaloaderException as e:
                error_msg = str(e)
                
                # Handle proxy errors
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy and self._is_instagram_error(e):
                        self.proxy_manager.mark_proxy_failed(proxy, e)
                
                if "private" in error_msg.lower():
                    self.logger.error(f"Instagram reel is private: {error_msg}")
                    raise InstagramDownloadError("This Instagram reel is private or requires login to access.")
                elif "not found" in error_msg.lower():
                    self.logger.error(f"Instagram reel not found: {error_msg}")
                    raise InstagramDownloadError("Instagram reel not found. The URL may be invalid or the content may have been deleted.")
                else:
                    self.logger.error(f"Instaloader error: {error_msg}")
                    raise InstagramDownloadError(f"Instagram download failed: {error_msg}")
                    
            except Exception as e:
                # Handle proxy errors for general exceptions
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy and self._is_instagram_error(e):
                        self.proxy_manager.mark_proxy_failed(proxy, e)
                
                if self.cancel_event.is_set():
                    self.logger.info("Instagram download cancelled by user")
                    if self.status_callback:
                        self.status_callback("Download cancelled by user", "warning")
                    return None
                else:
                    self.logger.error(f"Unexpected error during Instagram download: {e}")
                    raise InstagramDownloadError(f"Instagram download failed: {str(e)}")
                    
            finally:
                self.current_download = None
                self.download_stats['status'] = 'idle'
                
    def cancel_download(self) -> None:
        """Cancel current download operation."""
        if self.current_download:
            self.logger.info(f"Cancelling Instagram download: {self.current_download}")
            self.cancel_event.set()
            self.current_download = None
            
    def cleanup_temp_files(self, keep_recent: bool = False) -> None:
        """Clean up temporary download files.
        
        Args:
            keep_recent: If True, keep files modified in the last hour
        """
        try:
            if not self.temp_dir.exists():
                return
                
            current_time = time.time()
            files_removed = 0
            
            for file_path in self.temp_dir.rglob("*"):
                if file_path.is_file():
                    try:
                        # Check if file should be kept
                        if keep_recent:
                            file_age = current_time - file_path.stat().st_mtime
                            if file_age < 3600:  # Less than 1 hour old
                                continue
                                
                        file_path.unlink()
                        files_removed += 1
                        
                    except Exception as e:
                        self.logger.warning(f"Could not remove file {file_path}: {e}")
                        
            # Remove empty directories
            for dir_path in sorted(self.temp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if dir_path.is_dir() and dir_path != self.temp_dir:
                    try:
                        dir_path.rmdir()
                    except OSError:
                        pass  # Directory not empty
                        
            self.logger.info(f"Cleaned up {files_removed} temporary Instagram files")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up Instagram temp files: {e}")
            
    def get_download_progress(self) -> Dict[str, Any]:
        """Get current download progress statistics.
        
        Returns:
            Dictionary with download statistics
        """
        return self.download_stats.copy()
        
    def is_downloading(self) -> bool:
        """Check if a download is currently in progress.
        
        Returns:
            True if download is active
        """
        return self.current_download is not None


# Utility functions for Instagram downloading
def validate_instagram_url(url: str) -> bool:
    """Validate if URL is a valid Instagram Reel/video URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid Instagram video URL
    """
    try:
        downloader = InstagramDownloader()
        return downloader.validate_url(url)
    except ImportError:
        # Fallback validation if instaloader not available
        if not url or not isinstance(url, str):
            return False
        instagram_patterns = [
            r'instagram\.com/reel/',
            r'instagram\.com/p/',
            r'instagram\.com/tv/'
        ]
        return any(re.search(pattern, url.lower()) for pattern in instagram_patterns)


def download_instagram_video(url: str, output_dir: Path, progress_callback: Optional[Callable] = None, 
                           status_callback: Optional[Callable] = None) -> Optional[Path]:
    """Download Instagram reel video for transcription.
    
    Args:
        url: Instagram URL
        output_dir: Output directory
        progress_callback: Progress update function (percentage, message)
        status_callback: Status update function (message, type)
        
    Returns:
        Path to video file or None if failed
        
    Raises:
        InstagramDownloadError: If download fails
    """
    downloader = InstagramDownloader()
    
    if progress_callback:
        downloader.set_progress_callback(progress_callback)
    if status_callback:
        downloader.set_status_callback(status_callback)
        
    return downloader.download_reel(url, output_dir)


def get_instagram_reel_info(url: str) -> Optional[Dict[str, Any]]:
    """Get Instagram reel information without downloading.
    
    Args:
        url: Instagram URL
        
    Returns:
        Dictionary with reel info or None if failed
    """
    try:
        downloader = InstagramDownloader()
        return downloader.get_reel_info(url)
    except Exception:
        return None


def extract_instagram_shortcode(url: str) -> Optional[str]:
    """Extract shortcode from Instagram URL.
    
    Args:
        url: Instagram URL
        
    Returns:
        Shortcode string or None if extraction fails
    """
    try:
        downloader = InstagramDownloader()
        return downloader.extract_shortcode(url)
    except Exception:
        return None
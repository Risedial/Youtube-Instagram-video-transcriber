"""YouTube video download functionality for YouTube Whisper Transcriber.

This module handles YouTube video downloading using yt-dlp with progress tracking,
audio extraction optimized for Whisper, and comprehensive error handling.
"""

from typing import Optional, Callable, Dict, Any, Union
from pathlib import Path
import tempfile
import logging
import threading
import time
import re
import subprocess
import json
import os
from urllib.parse import urlparse, parse_qs

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

# Import proxy manager
try:
    from utils.proxy_manager import ProxyManager
    PROXY_MANAGER_AVAILABLE = True
except ImportError:
    PROXY_MANAGER_AVAILABLE = False
    

class DownloadError(Exception):
    """Custom exception for download-related errors."""
    pass


class VideoDownloader:
    """YouTube video downloader with progress tracking and audio extraction.
    
    Features:
    - Comprehensive YouTube URL validation and format support
    - yt-dlp integration with optimized audio extraction for Whisper
    - Thread-safe progress reporting compatible with Tkinter GUI
    - Robust error handling for network issues, age restrictions, private videos
    - Download cancellation support with proper cleanup
    - Memory-efficient processing for large video files (up to 2 hours)
    - Automatic temporary file management with cleanup
    """
    
    # Supported audio formats for Whisper (in order of preference)
    WHISPER_AUDIO_FORMATS = ['wav', 'flac', 'm4a', 'mp3']
    
    # Maximum video duration in seconds (2 hours)
    MAX_VIDEO_DURATION = 7200
    
    def __init__(self, temp_dir: Optional[Path] = None, proxy_manager: Optional['ProxyManager'] = None) -> None:
        """Initialize video downloader.
        
        Args:
            temp_dir: Temporary directory for downloads, uses system temp if None
            proxy_manager: ProxyManager instance for proxy support
            
        Raises:
            ImportError: If yt-dlp is not available
        """
        if yt_dlp is None:
            raise ImportError("yt-dlp is required but not installed. Install with: pip install yt-dlp")
            
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "youtube_whisper"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.status_callback: Optional[Callable[[str, str], None]] = None
        
        # Current download state
        self.current_download: Optional[str] = None
        self.current_process: Optional[subprocess.Popen] = None
        self.cancel_event = threading.Event()
        self.download_lock = threading.Lock()
        
        # Download statistics
        self.download_stats = {
            'total_bytes': 0,
            'downloaded_bytes': 0,
            'speed': 0,
            'eta': 0
        }
        
        # Setup FFmpeg path for PyInstaller environment
        self.ffmpeg_path = self._find_ffmpeg_path()
        
        self.logger.info(f"VideoDownloader initialized with temp_dir: {self.temp_dir}")
        self.logger.info(f"DEBUG: FFmpeg path: {self.ffmpeg_path}")
        
        if self.proxy_manager:
            self.logger.info("Proxy support enabled for video downloads")
        
    def _find_ffmpeg_path(self) -> str:
        """Find FFmpeg executable path, handling PyInstaller bundle."""
        import sys
        import shutil
        
        # Check if running in PyInstaller bundle
        if hasattr(sys, '_MEIPASS'):
            # Running in PyInstaller bundle - check bundle directory
            bundle_ffmpeg = Path(sys._MEIPASS) / "ffmpeg.exe"
            if bundle_ffmpeg.exists():
                self.logger.info(f"DEBUG: Found bundled FFmpeg: {bundle_ffmpeg}")
                return str(bundle_ffmpeg)
        
        # Check if FFmpeg is in system PATH
        ffmpeg_system = shutil.which('ffmpeg')
        if ffmpeg_system:
            self.logger.info(f"DEBUG: Found system FFmpeg: {ffmpeg_system}")
            return ffmpeg_system
            
        # Fallback to default
        self.logger.warning("DEBUG: FFmpeg not found - using default 'ffmpeg'")
        return "ffmpeg"
    
    def _get_proxy_options(self) -> Dict[str, Any]:
        """Get proxy configuration for yt-dlp.
        
        Returns:
            Dictionary with proxy options for yt-dlp
        """
        proxy_opts = {}
        
        if not self.proxy_manager:
            return proxy_opts
            
        proxy = self.proxy_manager.get_healthy_proxy()
        if not proxy:
            return proxy_opts
            
        # Apply anti-detection delay
        self.proxy_manager.apply_anti_detection_delay()
        
        # Configure proxy URL for yt-dlp
        proxy_opts['proxy'] = proxy.url
        
        # Add randomized headers for anti-detection
        headers = self.proxy_manager.get_randomized_headers("youtube")
        if headers:
            proxy_opts['http_headers'] = headers
            
        # Additional anti-detection options
        proxy_opts.update({
            'sleep_interval': 2,  # Sleep between downloads
            'max_sleep_interval': 5,  # Maximum sleep interval
            'sleep_interval_requests': 1,  # Sleep every N requests
        })
        
        self.logger.debug(f"Configured yt-dlp proxy: {proxy.host}:{proxy.port}")
        
        return proxy_opts
    
    def _is_proxy_error(self, error: Exception) -> bool:
        """Check if error is related to proxy connectivity.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is proxy-related
        """
        error_str = str(error).lower()
        proxy_error_indicators = [
            'proxy', 'connection', 'timeout', 'network', 
            'unreachable', 'refused', 'blocked', 'banned',
            '403', '407', '502', '503', '504'
        ]
        
        return any(indicator in error_str for indicator in proxy_error_indicators)
        
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
        """Validate YouTube URL format comprehensively.
        
        Args:
            url: YouTube URL to validate
            
        Returns:
            True if URL is valid YouTube URL
        """
        if not url or not isinstance(url, str):
            return False
            
        # Comprehensive YouTube URL patterns
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)'
        ]
        
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID string or None if extraction fails
        """
        youtube_patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtu\.be/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]+)',
            r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
                
        return None
        
    def _progress_hook(self, d: Dict[str, Any]) -> None:
        """Progress hook for yt-dlp downloads.
        
        Args:
            d: Progress dictionary from yt-dlp
        """
        if self.cancel_event.is_set():
            raise yt_dlp.DownloadError("Download cancelled by user")
            
        if d['status'] == 'downloading':
            # Update download statistics
            self.download_stats.update({
                'total_bytes': d.get('total_bytes', 0) or d.get('total_bytes_estimate', 0),
                'downloaded_bytes': d.get('downloaded_bytes', 0),
                'speed': d.get('speed', 0),
                'eta': d.get('eta', 0)
            })
            
            # Calculate progress percentage
            if self.download_stats['total_bytes'] > 0:
                percentage = (self.download_stats['downloaded_bytes'] / self.download_stats['total_bytes']) * 100
            else:
                percentage = 0
                
            # Format status message
            speed_str = ""
            if self.download_stats['speed']:
                speed_mb = self.download_stats['speed'] / (1024 * 1024)
                speed_str = f" at {speed_mb:.1f} MB/s"
                
            eta_str = ""
            if self.download_stats['eta']:
                eta_str = f" (ETA: {self.download_stats['eta']}s)"
                
            status_msg = f"Downloading video{speed_str}{eta_str}"
            
            # Call progress callback
            if self.progress_callback:
                self.progress_callback(percentage, status_msg)
                
        elif d['status'] == 'finished':
            if self.progress_callback:
                self.progress_callback(100.0, "Download completed, processing audio...")
                
        elif d['status'] == 'error':
            error_msg = d.get('error', 'Unknown download error')
            self.logger.error(f"Download error: {error_msg}")
            if self.status_callback:
                self.status_callback(f"Download error: {error_msg}", "error")
                
    def get_video_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Get video information without downloading.
        
        Args:
            url: YouTube URL
            
        Returns:
            Dictionary with video info (title, duration, etc.) or None if failed
        """
        if not self.validate_url(url):
            return None
            
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
            }
            
            # Add proxy options if available
            proxy_opts = self._get_proxy_options()
            ydl_opts.update(proxy_opts)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return {
                    'id': info.get('id'),
                    'title': info.get('title', 'Unknown Title'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown Channel'),
                    'description': info.get('description', ''),
                    'view_count': info.get('view_count', 0),
                    'upload_date': info.get('upload_date'),
                    'thumbnail': info.get('thumbnail'),
                    'webpage_url': info.get('webpage_url', url)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting video info for {url}: {e}")
            
            # Mark proxy as failed if proxy-related error
            if self.proxy_manager:
                proxy = self.proxy_manager.get_current_proxy()
                if proxy and self._is_proxy_error(e):
                    self.proxy_manager.mark_proxy_failed(proxy, e)
                    
            return None
            
    def _validate_video_for_transcription(self, info: Dict[str, Any]) -> None:
        """Validate video suitability for transcription.
        
        Args:
            info: Video information dictionary
            
        Raises:
            DownloadError: If video is not suitable for transcription
        """
        # Check duration
        duration = info.get('duration', 0)
        if duration > self.MAX_VIDEO_DURATION:
            hours = duration // 3600
            minutes = (duration % 3600) // 60
            raise DownloadError(
                f"Video is too long ({hours}h {minutes}m). Maximum supported duration is 2 hours."
            )
            
        # Check if video is available
        if info.get('is_live'):
            raise DownloadError("Live streams are not supported for transcription.")
            
        if info.get('availability') == 'private':
            raise DownloadError("Private videos cannot be downloaded.")
            
    def download_video(self, url: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """Download video and extract audio for transcription.
        
        Args:
            url: YouTube URL to download
            output_path: Output directory, uses temp_dir if None
            
        Returns:
            Path to downloaded audio file, None if failed
            
        Raises:
            DownloadError: If download fails
        """
        with self.download_lock:
            if not self.validate_url(url):
                raise DownloadError(f"Invalid YouTube URL format: {url}")
                
            self.current_download = url
            self.cancel_event.clear()
            
            try:
                # Get video information first
                if self.progress_callback:
                    self.progress_callback(5.0, "Getting video information...")
                    
                info = self.get_video_info(url)
                if not info:
                    raise DownloadError("Could not retrieve video information")
                    
                # Validate video for transcription
                self._validate_video_for_transcription(info)
                
                # Prepare output path
                output_dir = output_path or self.temp_dir
                output_dir = Path(output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # Generate safe filename
                safe_title = re.sub(r'[<>:"/\\|?*]', '_', info['title'])
                safe_title = safe_title[:50]  # Limit length
                video_id = self.extract_video_id(url)
                
                # Configure yt-dlp options for audio extraction
                audio_file = output_dir / f"{safe_title}_{video_id}.%(ext)s"
                
                ydl_opts = {
                    'format': 'bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio/best',
                    'outtmpl': str(audio_file),
                    'extractaudio': True,
                    'audioformat': 'wav',  # Best for Whisper
                    'audioquality': '0',   # Best quality
                    'postprocessors': [{
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'wav',
                        'preferredquality': '192',
                    }],
                    'progress_hooks': [self._progress_hook],
                    'quiet': True,
                    'no_warnings': True,
                    'extract_flat': False,
                    'writethumbnail': False,
                    'writesubtitles': False,
                    'writeautomaticsub': False,
                }
                
                # Add proxy options if available
                proxy_opts = self._get_proxy_options()
                ydl_opts.update(proxy_opts)
                
                if self.progress_callback:
                    self.progress_callback(10.0, "Starting download...")
                    
                if self.status_callback:
                    duration_str = f"{info['duration']//60}:{info['duration']%60:02d}" if info['duration'] else "Unknown"
                    self.status_callback(
                        f"Downloading: {info['title']}\n"
                        f"Duration: {duration_str} | Channel: {info['uploader']}",
                        "processing"
                    )
                    
                # Perform download
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                    
                # Find the downloaded audio file
                audio_files = list(output_dir.glob(f"{safe_title}_{video_id}.*"))
                if not audio_files:
                    raise DownloadError("Audio file not found after download")
                    
                downloaded_file = audio_files[0]
                
                # Verify file exists and has reasonable size
                if not downloaded_file.exists() or downloaded_file.stat().st_size < 1024:
                    raise DownloadError("Downloaded audio file is invalid or too small")
                    
                self.logger.info(f"Successfully downloaded audio: {downloaded_file}")
                
                if self.progress_callback:
                    self.progress_callback(100.0, "Download completed successfully")
                    
                if self.status_callback:
                    file_size = downloaded_file.stat().st_size / (1024 * 1024)  # MB
                    self.status_callback(
                        f"Download completed successfully!\n"
                        f"Audio file: {downloaded_file.name}\n"
                        f"File size: {file_size:.1f} MB",
                        "success"
                    )
                    
                return downloaded_file
                
            except yt_dlp.DownloadError as e:
                error_msg = str(e)
                
                # Handle proxy errors
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy and self._is_proxy_error(e):
                        self.proxy_manager.mark_proxy_failed(proxy, e)
                
                if "cancelled" in error_msg.lower():
                    self.logger.info("Download cancelled by user")
                    if self.status_callback:
                        self.status_callback("Download cancelled by user", "warning")
                    return None
                else:
                    self.logger.error(f"yt-dlp download error: {error_msg}")
                    raise DownloadError(f"Download failed: {error_msg}")
                    
            except Exception as e:
                # Handle proxy errors for general exceptions too
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy and self._is_proxy_error(e):
                        self.proxy_manager.mark_proxy_failed(proxy, e)
                
                self.logger.error(f"Unexpected error during download: {e}")
                raise DownloadError(f"Download failed: {str(e)}")
                
            finally:
                self.current_download = None
                
    def cancel_download(self) -> None:
        """Cancel current download operation."""
        if self.current_download:
            self.logger.info(f"Cancelling download: {self.current_download}")
            self.cancel_event.set()
            
            # If there's a current process, terminate it
            if self.current_process:
                try:
                    self.current_process.terminate()
                    self.current_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.current_process.kill()
                except Exception as e:
                    self.logger.warning(f"Error terminating download process: {e}")
                finally:
                    self.current_process = None
                    
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
                        
            self.logger.info(f"Cleaned up {files_removed} temporary files")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")
            
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


# Utility functions for video downloading
def validate_youtube_url(url: str) -> bool:
    """Validate if URL is a valid YouTube URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid YouTube URL
    """
    try:
        downloader = VideoDownloader()
        return downloader.validate_url(url)
    except ImportError:
        # Fallback validation if yt-dlp not available
        if not url or not isinstance(url, str):
            return False
        youtube_patterns = [
            r'youtube\.com/watch\?v=',
            r'youtu\.be/',
            r'youtube\.com/embed/',
            r'm\.youtube\.com/watch'
        ]
        return any(re.search(pattern, url.lower()) for pattern in youtube_patterns)


def download_youtube_audio(url: str, output_dir: Path, progress_callback: Optional[Callable] = None, 
                          status_callback: Optional[Callable] = None) -> Optional[Path]:
    """Download YouTube video audio for transcription.
    
    Args:
        url: YouTube URL
        output_dir: Output directory
        progress_callback: Progress update function (percentage, message)
        status_callback: Status update function (message, type)
        
    Returns:
        Path to audio file or None if failed
        
    Raises:
        DownloadError: If download fails
    """
    downloader = VideoDownloader()
    
    if progress_callback:
        downloader.set_progress_callback(progress_callback)
    if status_callback:
        downloader.set_status_callback(status_callback)
        
    return downloader.download_video(url, output_dir)


def get_youtube_video_info(url: str) -> Optional[Dict[str, Any]]:
    """Get YouTube video information without downloading.
    
    Args:
        url: YouTube URL
        
    Returns:
        Dictionary with video info or None if failed
    """
    try:
        downloader = VideoDownloader()
        return downloader.get_video_info(url)
    except Exception:
        return None


def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL.
    
    Args:
        url: YouTube URL
        
    Returns:
        Video ID string or None if extraction fails
    """
    try:
        downloader = VideoDownloader()
        return downloader.extract_video_id(url)
    except Exception:
        return None
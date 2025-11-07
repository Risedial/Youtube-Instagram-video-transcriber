"""File management utilities for YouTube Whisper Transcriber.

This module handles file operations, temporary file management, cleanup operations,
disk space validation, and Windows-compatible file system operations.
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import tempfile
import shutil
import logging
import os
import time
import psutil
import threading
from contextlib import contextmanager
import uuid


class FileManagerError(Exception):
    """Custom exception for file management errors."""
    pass


class FileManager:
    """Comprehensive file management and cleanup utilities.
    
    Features:
    - Temporary file lifecycle management with automatic cleanup
    - Disk space validation before operations
    - Safe file operations with atomic writes and proper error handling
    - Directory structure creation and validation
    - File permission handling optimized for Windows compatibility
    - Multi-threaded cleanup operations
    - File size and format validation
    """
    
    # Minimum free disk space in bytes (1GB)
    MIN_FREE_SPACE = 1024 * 1024 * 1024
    
    # Maximum single file size in bytes (2GB)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024
    
    def __init__(self, temp_dir: Optional[Path] = None, auto_cleanup: bool = True) -> None:
        """Initialize file manager.
        
        Args:
            temp_dir: Custom temporary directory, uses system temp if None
            auto_cleanup: Enable automatic cleanup of temporary files
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "youtube_whisper"
        self.auto_cleanup = auto_cleanup
        self.logger = logging.getLogger(__name__)
        
        # Track managed files for cleanup
        self.managed_files: List[Path] = []
        self.cleanup_lock = threading.Lock()
        
        # Ensure temp directory exists
        self.ensure_directory(self.temp_dir)
        
        self.logger.info(f"FileManager initialized with temp_dir: {self.temp_dir}")
        
    def ensure_directory(self, path: Path, mode: int = 0o755) -> bool:
        """Ensure directory exists with proper permissions.
        
        Args:
            path: Directory path to create
            mode: Directory permissions (Unix-style)
            
        Returns:
            True if directory exists or was created successfully
            
        Raises:
            FileManagerError: If directory creation fails
        """
        try:
            path = Path(path)
            if path.exists() and path.is_dir():
                return True
                
            path.mkdir(parents=True, exist_ok=True, mode=mode)
            
            # Verify directory was created
            if not path.exists():
                raise FileManagerError(f"Failed to create directory: {path}")
                
            self.logger.debug(f"Directory ensured: {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring directory {path}: {e}")
            raise FileManagerError(f"Could not create directory {path}: {e}")
            
    def check_disk_space(self, path: Path, required_bytes: int = None) -> Dict[str, Any]:
        """Check available disk space at given path.
        
        Args:
            path: Path to check (file or directory)
            required_bytes: Minimum required bytes, uses MIN_FREE_SPACE if None
            
        Returns:
            Dictionary with disk space information
            
        Raises:
            FileManagerError: If insufficient disk space
        """
        try:
            path = Path(path)
            # Use parent directory for disk space check
            check_path = path.parent if path.is_file() else path
            
            if not check_path.exists():
                check_path = path.parents[0] if path.parents else Path.cwd()
                
            # Get disk usage statistics
            usage = shutil.disk_usage(check_path)
            
            # Calculate space information
            total_space = usage.total
            free_space = usage.free
            used_space = usage.used
            
            required = required_bytes or self.MIN_FREE_SPACE
            
            space_info = {
                'total_gb': total_space / (1024**3),
                'free_gb': free_space / (1024**3),
                'used_gb': used_space / (1024**3),
                'free_percent': (free_space / total_space) * 100,
                'sufficient': free_space >= required,
                'required_gb': required / (1024**3),
                'path': str(check_path)
            }
            
            if not space_info['sufficient']:
                raise FileManagerError(
                    f"Insufficient disk space. Available: {space_info['free_gb']:.1f}GB, "
                    f"Required: {space_info['required_gb']:.1f}GB"
                )
                
            return space_info
            
        except Exception as e:
            if isinstance(e, FileManagerError):
                raise
            self.logger.error(f"Error checking disk space for {path}: {e}")
            raise FileManagerError(f"Could not check disk space: {e}")
            
    def create_temp_file(self, suffix: str = ".tmp", prefix: str = "yt_whisper_", 
                        content: Union[str, bytes] = None) -> Path:
        """Create temporary file with optional content.
        
        Args:
            suffix: File extension
            prefix: Filename prefix
            content: Optional initial content (string or bytes)
            
        Returns:
            Path to created temporary file
            
        Raises:
            FileManagerError: If file creation fails
        """
        try:
            # Check disk space first
            self.check_disk_space(self.temp_dir)
            
            # Generate unique filename
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}{timestamp}_{unique_id}{suffix}"
            
            temp_file = self.temp_dir / filename
            
            # Create file with content
            if content is not None:
                mode = 'w' if isinstance(content, str) else 'wb'
                encoding = 'utf-8' if isinstance(content, str) else None
                
                with open(temp_file, mode, encoding=encoding) as f:
                    f.write(content)
            else:
                # Create empty file
                temp_file.touch()
                
            # Track for cleanup
            with self.cleanup_lock:
                self.managed_files.append(temp_file)
                
            self.logger.debug(f"Created temporary file: {temp_file}")
            return temp_file
            
        except Exception as e:
            self.logger.error(f"Error creating temporary file: {e}")
            raise FileManagerError(f"Could not create temporary file: {e}")
            
    def create_temp_directory(self, prefix: str = "yt_whisper_") -> Path:
        """Create temporary directory.
        
        Args:
            prefix: Directory name prefix
            
        Returns:
            Path to created temporary directory
            
        Raises:
            FileManagerError: If directory creation fails
        """
        try:
            # Check disk space first
            self.check_disk_space(self.temp_dir)
            
            # Generate unique directory name
            timestamp = int(time.time())
            unique_id = str(uuid.uuid4())[:8]
            dirname = f"{prefix}{timestamp}_{unique_id}"
            
            temp_dir = self.temp_dir / dirname
            self.ensure_directory(temp_dir)
            
            # Track for cleanup
            with self.cleanup_lock:
                self.managed_files.append(temp_dir)
                
            self.logger.debug(f"Created temporary directory: {temp_dir}")
            return temp_dir
            
        except Exception as e:
            self.logger.error(f"Error creating temporary directory: {e}")
            raise FileManagerError(f"Could not create temporary directory: {e}")
            
    @contextmanager
    def temp_file(self, suffix: str = ".tmp", prefix: str = "yt_whisper_", 
                  content: Union[str, bytes] = None):
        """Context manager for temporary file with automatic cleanup.
        
        Args:
            suffix: File extension
            prefix: Filename prefix
            content: Optional initial content
            
        Yields:
            Path to temporary file
        """
        temp_file = None
        try:
            temp_file = self.create_temp_file(suffix, prefix, content)
            yield temp_file
        finally:
            if temp_file and temp_file.exists():
                self.safe_remove_file(temp_file)
                
    @contextmanager
    def temp_directory(self, prefix: str = "yt_whisper_"):
        """Context manager for temporary directory with automatic cleanup.
        
        Args:
            prefix: Directory name prefix
            
        Yields:
            Path to temporary directory
        """
        temp_dir = None
        try:
            temp_dir = self.create_temp_directory(prefix)
            yield temp_dir
        finally:
            if temp_dir and temp_dir.exists():
                self.safe_remove_directory(temp_dir)
                
    def safe_copy_file(self, source: Path, destination: Path, overwrite: bool = False) -> Path:
        """Safely copy file with validation and error handling.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Allow overwriting existing file
            
        Returns:
            Path to destination file
            
        Raises:
            FileManagerError: If copy operation fails
        """
        try:
            source = Path(source)
            destination = Path(destination)
            
            # Validate source file
            if not source.exists():
                raise FileManagerError(f"Source file does not exist: {source}")
                
            if not source.is_file():
                raise FileManagerError(f"Source is not a file: {source}")
                
            # Check file size
            file_size = source.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                raise FileManagerError(f"File too large: {file_size / (1024**3):.1f}GB")
                
            # Check destination
            if destination.exists() and not overwrite:
                raise FileManagerError(f"Destination file already exists: {destination}")
                
            # Ensure destination directory exists
            self.ensure_directory(destination.parent)
            
            # Check disk space
            self.check_disk_space(destination.parent, file_size)
            
            # Perform copy with progress (for large files)
            if file_size > 100 * 1024 * 1024:  # 100MB
                self._copy_with_progress(source, destination)
            else:
                shutil.copy2(source, destination)
                
            # Verify copy was successful
            if not destination.exists() or destination.stat().st_size != file_size:
                raise FileManagerError("Copy verification failed")
                
            self.logger.info(f"Successfully copied {source} to {destination}")
            return destination
            
        except Exception as e:
            if isinstance(e, FileManagerError):
                raise
            self.logger.error(f"Error copying file {source} to {destination}: {e}")
            raise FileManagerError(f"Could not copy file: {e}")
            
    def _copy_with_progress(self, source: Path, destination: Path, 
                          chunk_size: int = 1024 * 1024) -> None:
        """Copy large file with progress tracking.
        
        Args:
            source: Source file path
            destination: Destination file path
            chunk_size: Copy chunk size in bytes
        """
        total_size = source.stat().st_size
        copied_size = 0
        
        with open(source, 'rb') as src, open(destination, 'wb') as dst:
            while True:
                chunk = src.read(chunk_size)
                if not chunk:
                    break
                    
                dst.write(chunk)
                copied_size += len(chunk)
                
                # Log progress for very large files
                if total_size > 500 * 1024 * 1024:  # 500MB
                    progress = (copied_size / total_size) * 100
                    if copied_size % (50 * 1024 * 1024) == 0:  # Every 50MB
                        self.logger.debug(f"Copy progress: {progress:.1f}%")
                        
    def safe_move_file(self, source: Path, destination: Path, overwrite: bool = False) -> Path:
        """Safely move file with validation and error handling.
        
        Args:
            source: Source file path
            destination: Destination file path
            overwrite: Allow overwriting existing file
            
        Returns:
            Path to destination file
            
        Raises:
            FileManagerError: If move operation fails
        """
        try:
            source = Path(source)
            destination = Path(destination)
            
            # Use copy + delete for cross-filesystem moves
            if source.stat().st_dev != destination.parent.stat().st_dev:
                self.safe_copy_file(source, destination, overwrite)
                self.safe_remove_file(source)
                return destination
            else:
                # Same filesystem - use rename
                if destination.exists() and not overwrite:
                    raise FileManagerError(f"Destination file already exists: {destination}")
                    
                self.ensure_directory(destination.parent)
                source.rename(destination)
                
                self.logger.info(f"Successfully moved {source} to {destination}")
                return destination
                
        except Exception as e:
            if isinstance(e, FileManagerError):
                raise
            self.logger.error(f"Error moving file {source} to {destination}: {e}")
            raise FileManagerError(f"Could not move file: {e}")
            
    def safe_remove_file(self, file_path: Path) -> bool:
        """Safely remove file with error handling.
        
        Args:
            file_path: Path to file to remove
            
        Returns:
            True if file was removed or didn't exist
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return True
                
            if file_path.is_file():
                file_path.unlink()
                self.logger.debug(f"Removed file: {file_path}")
            else:
                self.logger.warning(f"Path is not a file: {file_path}")
                
            # Remove from managed files
            with self.cleanup_lock:
                if file_path in self.managed_files:
                    self.managed_files.remove(file_path)
                    
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not remove file {file_path}: {e}")
            return False
            
    def safe_remove_directory(self, dir_path: Path, recursive: bool = True) -> bool:
        """Safely remove directory with error handling.
        
        Args:
            dir_path: Path to directory to remove
            recursive: Remove directory recursively
            
        Returns:
            True if directory was removed or didn't exist
        """
        try:
            dir_path = Path(dir_path)
            
            if not dir_path.exists():
                return True
                
            if dir_path.is_dir():
                if recursive:
                    shutil.rmtree(dir_path)
                else:
                    dir_path.rmdir()
                self.logger.debug(f"Removed directory: {dir_path}")
            else:
                self.logger.warning(f"Path is not a directory: {dir_path}")
                
            # Remove from managed files
            with self.cleanup_lock:
                if dir_path in self.managed_files:
                    self.managed_files.remove(dir_path)
                    
            return True
            
        except Exception as e:
            self.logger.warning(f"Could not remove directory {dir_path}: {e}")
            return False
            
    def cleanup_temp_files(self, max_age_hours: int = 24, force: bool = False) -> Dict[str, int]:
        """Clean up temporary files based on age and tracking.
        
        Args:
            max_age_hours: Maximum age of files to keep (hours)
            force: Force cleanup of all managed files
            
        Returns:
            Dictionary with cleanup statistics
        """
        stats = {
            'files_removed': 0,
            'directories_removed': 0,
            'bytes_freed': 0,
            'errors': 0
        }
        
        try:
            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            
            with self.cleanup_lock:
                # Clean up tracked managed files
                for file_path in self.managed_files.copy():
                    try:
                        if not file_path.exists():
                            self.managed_files.remove(file_path)
                            continue
                            
                        file_age = current_time - file_path.stat().st_mtime
                        should_remove = force or file_age > max_age_seconds
                        
                        if should_remove:
                            file_size = 0
                            if file_path.is_file():
                                file_size = file_path.stat().st_size
                                
                            if file_path.is_file():
                                if self.safe_remove_file(file_path):
                                    stats['files_removed'] += 1
                                    stats['bytes_freed'] += file_size
                                else:
                                    stats['errors'] += 1
                            elif file_path.is_dir():
                                if self.safe_remove_directory(file_path):
                                    stats['directories_removed'] += 1
                                else:
                                    stats['errors'] += 1
                                    
                    except Exception as e:
                        self.logger.warning(f"Error during cleanup of {file_path}: {e}")
                        stats['errors'] += 1
                        
            # Clean up untracked files in temp directory
            if self.temp_dir.exists():
                for item in self.temp_dir.rglob("*"):
                    try:
                        if item.is_file():
                            file_age = current_time - item.stat().st_mtime
                            if force or file_age > max_age_seconds:
                                file_size = item.stat().st_size
                                if self.safe_remove_file(item):
                                    stats['files_removed'] += 1
                                    stats['bytes_freed'] += file_size
                                    
                    except Exception as e:
                        self.logger.warning(f"Error cleaning untracked file {item}: {e}")
                        stats['errors'] += 1
                        
            # Clean up empty directories
            if self.temp_dir.exists():
                for item in sorted(self.temp_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                    if item.is_dir() and item != self.temp_dir:
                        try:
                            if not any(item.iterdir()):  # Directory is empty
                                if self.safe_remove_directory(item, recursive=False):
                                    stats['directories_removed'] += 1
                        except Exception:
                            pass  # Ignore errors for empty directory cleanup
                            
            self.logger.info(
                f"Cleanup completed: {stats['files_removed']} files, "
                f"{stats['directories_removed']} directories, "
                f"{stats['bytes_freed'] / (1024**2):.1f}MB freed, "
                f"{stats['errors']} errors"
            )
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            stats['errors'] += 1
            
        return stats
        
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive file information.
        
        Args:
            file_path: Path to analyze
            
        Returns:
            Dictionary with file information
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'exists': False, 'path': str(file_path)}
                
            stat = file_path.stat()
            
            return {
                'exists': True,
                'path': str(file_path),
                'name': file_path.name,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024**2),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'accessed': stat.st_atime,
                'is_file': file_path.is_file(),
                'is_dir': file_path.is_dir(),
                'suffix': file_path.suffix,
                'parent': str(file_path.parent),
                'readable': os.access(file_path, os.R_OK),
                'writable': os.access(file_path, os.W_OK),
                'executable': os.access(file_path, os.X_OK)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_path}: {e}")
            return {'exists': False, 'error': str(e), 'path': str(file_path)}
            
    def __del__(self):
        """Cleanup on destruction if auto_cleanup is enabled."""
        if self.auto_cleanup:
            try:
                self.cleanup_temp_files(max_age_hours=0, force=True)
            except Exception:
                pass  # Ignore cleanup errors during destruction


# Utility functions for common file operations
def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure path exists, create if necessary.
    
    Args:
        path: Path to ensure
        
    Returns:
        Path object
        
    Raises:
        FileManagerError: If path cannot be created
    """
    manager = FileManager()
    path = Path(path)
    manager.ensure_directory(path if path.suffix == '' else path.parent)
    return path


def check_available_space(path: Union[str, Path], required_gb: float = 1.0) -> bool:
    """Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        required_gb: Required space in GB
        
    Returns:
        True if sufficient space available
    """
    try:
        manager = FileManager()
        required_bytes = int(required_gb * 1024**3)
        space_info = manager.check_disk_space(Path(path), required_bytes)
        return space_info['sufficient']
    except Exception:
        return False


def safe_filename(filename: str, max_length: int = 100) -> str:
    """Generate safe filename for Windows/Unix compatibility.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Safe filename string
    """
    # Remove/replace invalid characters
    import re
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    safe_name = re.sub(r'[^\w\s\-_\.]', '', safe_name)
    safe_name = re.sub(r'\s+', '_', safe_name)
    
    # Limit length
    if len(safe_name) > max_length:
        name_part = safe_name[:max_length-10]
        ext_part = safe_name[-10:] if '.' in safe_name[-10:] else ''
        safe_name = name_part + ext_part
        
    return safe_name or 'unnamed_file'
"""Comprehensive error handling and recovery system for YouTube Whisper Transcriber.

This module provides error categorization, user-friendly messaging, recovery procedures,
error logging with context, and automated fix application for common issues.
"""

from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
import traceback
import json
import platform
import threading
import time
from datetime import datetime
from pathlib import Path


class ErrorCategory(Enum):
    """Comprehensive error categorization."""
    # Network and connectivity errors
    NETWORK_CONNECTION = "network_connection"
    NETWORK_TIMEOUT = "network_timeout"
    NETWORK_ACCESS_DENIED = "network_access_denied"
    
    # YouTube and video-specific errors
    YOUTUBE_UNAVAILABLE = "youtube_unavailable"
    YOUTUBE_PRIVATE = "youtube_private"
    YOUTUBE_AGE_RESTRICTED = "youtube_age_restricted"
    YOUTUBE_GEOBLOCKED = "youtube_geoblocked"
    VIDEO_NOT_FOUND = "video_not_found"
    VIDEO_TOO_LONG = "video_too_long"
    
    # File system errors
    FILE_NOT_FOUND = "file_not_found"
    FILE_PERMISSION_DENIED = "file_permission_denied"
    DISK_SPACE_INSUFFICIENT = "disk_space_insufficient"
    DIRECTORY_NOT_WRITABLE = "directory_not_writable"
    
    # Processing and AI model errors
    MODEL_DOWNLOAD_FAILED = "model_download_failed"
    MODEL_LOADING_FAILED = "model_loading_failed"
    TRANSCRIPTION_FAILED = "transcription_failed"
    AUDIO_PROCESSING_FAILED = "audio_processing_failed"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    
    # System and hardware errors
    SYSTEM_RESOURCE_EXHAUSTED = "system_resource_exhausted"
    GPU_UNAVAILABLE = "gpu_unavailable"
    DEPENDENCY_MISSING = "dependency_missing"
    
    # User input and validation errors
    INVALID_URL = "invalid_url"
    INVALID_OUTPUT_PATH = "invalid_output_path"
    INVALID_CONFIGURATION = "invalid_configuration"
    
    # Application state errors
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_TIMEOUT = "workflow_timeout"
    STATE_CORRUPTION = "state_corruption"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, recoverable
    MEDIUM = "medium"     # Significant issues, may need user action
    HIGH = "high"         # Critical issues, workflow cannot continue
    CRITICAL = "critical" # System-level issues requiring restart


@dataclass
class RecoveryAction:
    """Recovery action that can be taken for an error."""
    action_id: str
    description: str
    user_friendly_text: str
    automatic: bool = False  # Can be applied automatically
    success_probability: float = 0.5  # Estimated success rate
    side_effects: List[str] = field(default_factory=list)


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    error_message: str
    user_message: str
    technical_details: str
    context: Dict[str, Any]
    traceback_text: str
    recovery_actions: List[RecoveryAction]
    system_info: Dict[str, Any]
    workflow_stage: Optional[str] = None
    can_retry: bool = False
    can_resume: bool = False
    automatic_recovery_attempted: bool = False


class ErrorHandler:
    """Comprehensive error handling and recovery system.
    
    Features:
    - Error categorization and severity assessment
    - User-friendly error messages with actionable guidance
    - Recovery procedures for failed stages with automatic retry
    - Error logging with context and debugging information
    - System health monitoring and diagnostics
    - Automatic fix application for common issues
    """
    
    def __init__(self, log_dir: Optional[Path] = None) -> None:
        """Initialize error handler.
        
        Args:
            log_dir: Directory for error logs
        """
        self.logger = logging.getLogger(__name__)
        self.error_lock = threading.Lock()
        
        # Error callbacks and observers
        self.error_callbacks: List[Callable[[ErrorInfo], None]] = []
        self.recovery_callbacks: List[Callable[[str, bool], None]] = []
        
        # Error tracking and statistics
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_success_rates: Dict[str, float] = {}
        
        # Logging setup
        self.log_dir = log_dir or Path.home() / ".youtube_whisper_transcriber" / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.error_log_file = self.log_dir / "errors.json"
        
        # Load previous error statistics
        self._load_error_statistics()
        
        self.logger.info(f"ErrorHandler initialized with log dir: {self.log_dir}")
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """Handle and categorize error with comprehensive analysis.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            
        Returns:
            ErrorInfo object with categorization and recovery options
        """
        with self.error_lock:
            # Categorize and analyze error
            category = self._categorize_error(error, context)
            severity = self._assess_severity(category, error, context)
            
            # Generate user-friendly message and recovery actions
            user_message = self._generate_user_message(category, error, context)
            recovery_actions = self._generate_recovery_actions(category, error, context)
            
            # Gather system information
            system_info = self._gather_system_info()
            
            # Create comprehensive error info
            error_info = ErrorInfo(
                timestamp=datetime.now(),
                category=category,
                severity=severity,
                error_message=str(error),
                user_message=user_message,
                technical_details=repr(error),
                context=context or {},
                traceback_text=traceback.format_exc(),
                recovery_actions=recovery_actions,
                system_info=system_info,
                workflow_stage=context.get("stage") if context else None,
                can_retry=self._can_retry(category, severity),
                can_resume=self._can_resume(category, severity, context)
            )
            
            # Update statistics
            self.error_counts[category] = self.error_counts.get(category, 0) + 1
            self.error_history.append(error_info)
            
            # Keep only recent history (last 100 errors)
            if len(self.error_history) > 100:
                self.error_history = self.error_history[-100:]
            
            # Attempt automatic recovery if possible
            if severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
                self._attempt_automatic_recovery(error_info)
            
            # Log error details
            self._log_error(error_info)
            
            # Notify callbacks
            self._notify_error_callbacks(error_info)
            
            self.logger.error(f"Error handled: {category.value} - {error}")
            
            return error_info
    
    def _categorize_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorCategory:
        """Categorize error based on type, message, and context."""
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Network errors
        if any(keyword in error_str for keyword in ['network', 'connection', 'timeout', 'unreachable', 'dns']):
            if 'timeout' in error_str:
                return ErrorCategory.NETWORK_TIMEOUT
            elif 'access denied' in error_str or 'forbidden' in error_str:
                return ErrorCategory.NETWORK_ACCESS_DENIED
            else:
                return ErrorCategory.NETWORK_CONNECTION
        
        # YouTube-specific errors
        if any(keyword in error_str for keyword in ['youtube', 'video unavailable', 'video not available']):
            if 'private' in error_str:
                return ErrorCategory.YOUTUBE_PRIVATE
            elif 'age' in error_str or 'restricted' in error_str:
                return ErrorCategory.YOUTUBE_AGE_RESTRICTED
            elif 'geoblocked' in error_str or 'not available in your country' in error_str:
                return ErrorCategory.YOUTUBE_GEOBLOCKED
            elif 'not found' in error_str or '404' in error_str:
                return ErrorCategory.VIDEO_NOT_FOUND
            elif 'too long' in error_str or 'duration' in error_str:
                return ErrorCategory.VIDEO_TOO_LONG
            else:
                return ErrorCategory.YOUTUBE_UNAVAILABLE
        
        # File system errors
        if error_type in ['FileNotFoundError', 'OSError', 'IOError', 'PermissionError']:
            if 'not found' in error_str or error_type == 'FileNotFoundError':
                return ErrorCategory.FILE_NOT_FOUND
            elif 'permission' in error_str or error_type == 'PermissionError':
                return ErrorCategory.FILE_PERMISSION_DENIED
            elif 'no space' in error_str or 'disk full' in error_str:
                return ErrorCategory.DISK_SPACE_INSUFFICIENT
            elif 'read-only' in error_str or 'not writable' in error_str:
                return ErrorCategory.DIRECTORY_NOT_WRITABLE
        
        # Model and processing errors
        if any(keyword in error_str for keyword in ['whisper', 'model', 'torch', 'cuda']):
            if 'download' in error_str or 'fetch' in error_str:
                return ErrorCategory.MODEL_DOWNLOAD_FAILED
            elif 'load' in error_str or 'loading' in error_str:
                return ErrorCategory.MODEL_LOADING_FAILED
            elif 'memory' in error_str or 'out of memory' in error_str:
                return ErrorCategory.MEMORY_INSUFFICIENT
            elif 'transcrib' in error_str:
                return ErrorCategory.TRANSCRIPTION_FAILED
            elif 'audio' in error_str:
                return ErrorCategory.AUDIO_PROCESSING_FAILED
        
        # System resource errors
        if error_type in ['MemoryError', 'ResourceWarning']:
            return ErrorCategory.SYSTEM_RESOURCE_EXHAUSTED
        
        # GPU errors
        if 'cuda' in error_str or 'gpu' in error_str:
            return ErrorCategory.GPU_UNAVAILABLE
        
        # Dependency errors
        if error_type in ['ImportError', 'ModuleNotFoundError']:
            return ErrorCategory.DEPENDENCY_MISSING
        
        # URL validation errors
        if context and context.get("url"):
            if 'invalid' in error_str or 'malformed' in error_str:
                return ErrorCategory.INVALID_URL
        
        # Workflow state errors
        if 'cancel' in error_str:
            return ErrorCategory.WORKFLOW_CANCELLED
        elif 'timeout' in error_str:
            return ErrorCategory.WORKFLOW_TIMEOUT
        
        # Default to unknown
        return ErrorCategory.UNKNOWN_ERROR
    
    def _assess_severity(self, category: ErrorCategory, error: Exception, 
                        context: Optional[Dict[str, Any]]) -> ErrorSeverity:
        """Assess error severity based on category and context."""
        # Critical errors that require restart
        critical_categories = [
            ErrorCategory.STATE_CORRUPTION,
            ErrorCategory.SYSTEM_RESOURCE_EXHAUSTED,
            ErrorCategory.DEPENDENCY_MISSING
        ]
        
        # High severity errors that stop workflow
        high_severity_categories = [
            ErrorCategory.MODEL_LOADING_FAILED,
            ErrorCategory.DISK_SPACE_INSUFFICIENT,
            ErrorCategory.MEMORY_INSUFFICIENT,
            ErrorCategory.FILE_PERMISSION_DENIED
        ]
        
        # Medium severity errors that may be recoverable
        medium_severity_categories = [
            ErrorCategory.NETWORK_CONNECTION,
            ErrorCategory.MODEL_DOWNLOAD_FAILED,
            ErrorCategory.TRANSCRIPTION_FAILED,
            ErrorCategory.AUDIO_PROCESSING_FAILED,
            ErrorCategory.YOUTUBE_UNAVAILABLE
        ]
        
        if category in critical_categories:
            return ErrorSeverity.CRITICAL
        elif category in high_severity_categories:
            return ErrorSeverity.HIGH
        elif category in medium_severity_categories:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _generate_user_message(self, category: ErrorCategory, error: Exception, 
                             context: Optional[Dict[str, Any]]) -> str:
        """Generate user-friendly error message with actionable guidance."""
        messages = {
            # Network errors
            ErrorCategory.NETWORK_CONNECTION: (
                "Network connection problem detected. "
                "Please check your internet connection and try again."
            ),
            ErrorCategory.NETWORK_TIMEOUT: (
                "Network request timed out. "
                "Please check your connection speed and try again."
            ),
            ErrorCategory.NETWORK_ACCESS_DENIED: (
                "Network access was denied. "
                "Please check your firewall settings or VPN configuration."
            ),
            
            # YouTube errors
            ErrorCategory.YOUTUBE_UNAVAILABLE: (
                "YouTube video is currently unavailable. "
                "Please try a different video or check the URL."
            ),
            ErrorCategory.YOUTUBE_PRIVATE: (
                "This YouTube video is private and cannot be accessed. "
                "Please use a public video URL."
            ),
            ErrorCategory.YOUTUBE_AGE_RESTRICTED: (
                "This YouTube video is age-restricted. "
                "Age-restricted videos cannot be processed."
            ),
            ErrorCategory.YOUTUBE_GEOBLOCKED: (
                "This YouTube video is not available in your region. "
                "Please try a different video or use a VPN."
            ),
            ErrorCategory.VIDEO_NOT_FOUND: (
                "YouTube video not found. "
                "Please check the URL and ensure the video exists."
            ),
            ErrorCategory.VIDEO_TOO_LONG: (
                "Video is too long for processing. "
                "Please use videos shorter than 2 hours."
            ),
            
            # File system errors
            ErrorCategory.FILE_NOT_FOUND: (
                "Required file not found. "
                "Please check the file path and ensure the file exists."
            ),
            ErrorCategory.FILE_PERMISSION_DENIED: (
                "File access permission denied. "
                "Please check folder permissions or run as administrator."
            ),
            ErrorCategory.DISK_SPACE_INSUFFICIENT: (
                "Insufficient disk space for processing. "
                "Please free up disk space and try again."
            ),
            ErrorCategory.DIRECTORY_NOT_WRITABLE: (
                "Output directory is not writable. "
                "Please select a different directory or check permissions."
            ),
            
            # Model and processing errors
            ErrorCategory.MODEL_DOWNLOAD_FAILED: (
                "Failed to download Whisper model. "
                "Please check your internet connection and try again."
            ),
            ErrorCategory.MODEL_LOADING_FAILED: (
                "Failed to load Whisper model. "
                "Please try restarting the application or using a smaller model."
            ),
            ErrorCategory.TRANSCRIPTION_FAILED: (
                "Audio transcription failed. "
                "Please try a different model or check the audio quality."
            ),
            ErrorCategory.AUDIO_PROCESSING_FAILED: (
                "Audio processing failed. "
                "Please ensure the video has clear audio content."
            ),
            ErrorCategory.MEMORY_INSUFFICIENT: (
                "Insufficient memory for processing. "
                "Please close other applications or use a smaller Whisper model."
            ),
            
            # System errors
            ErrorCategory.SYSTEM_RESOURCE_EXHAUSTED: (
                "System resources exhausted. "
                "Please close other applications and restart the program."
            ),
            ErrorCategory.GPU_UNAVAILABLE: (
                "GPU processing unavailable. "
                "Processing will continue using CPU (slower)."
            ),
            ErrorCategory.DEPENDENCY_MISSING: (
                "Required software component missing. "
                "Please reinstall the application or check dependencies."
            ),
            
            # User input errors
            ErrorCategory.INVALID_URL: (
                "Invalid YouTube URL format. "
                "Please enter a valid YouTube video URL."
            ),
            ErrorCategory.INVALID_OUTPUT_PATH: (
                "Invalid output directory. "
                "Please select a valid folder for saving files."
            ),
            ErrorCategory.INVALID_CONFIGURATION: (
                "Invalid configuration detected. "
                "Please check your settings and try again."
            ),
            
            # Workflow errors
            ErrorCategory.WORKFLOW_CANCELLED: (
                "Processing was cancelled by user request."
            ),
            ErrorCategory.WORKFLOW_TIMEOUT: (
                "Processing timed out. "
                "Please try again or use a shorter video."
            ),
            
            # Default
            ErrorCategory.UNKNOWN_ERROR: (
                "An unexpected error occurred. "
                "Please try again or contact support if the issue persists."
            )
        }
        
        base_message = messages.get(category, messages[ErrorCategory.UNKNOWN_ERROR])
        
        # Add context-specific information
        if context:
            if context.get("url"):
                base_message += f"\n\nURL: {context['url'][:100]}{'...' if len(context['url']) > 100 else ''}"
            if context.get("file_path"):
                base_message += f"\nFile: {context['file_path']}"
        
        return base_message
    
    def _generate_recovery_actions(self, category: ErrorCategory, error: Exception, 
                                 context: Optional[Dict[str, Any]]) -> List[RecoveryAction]:
        """Generate context-appropriate recovery actions."""
        actions = []
        
        # Network-related recovery actions
        if category in [ErrorCategory.NETWORK_CONNECTION, ErrorCategory.NETWORK_TIMEOUT]:
            actions.extend([
                RecoveryAction(
                    "retry_with_delay",
                    "Wait and retry",
                    "Wait 30 seconds and try again",
                    automatic=True,
                    success_probability=0.7
                ),
                RecoveryAction(
                    "check_connection",
                    "Check network connection",
                    "Verify your internet connection is working",
                    automatic=False,
                    success_probability=0.9
                )
            ])
        
        # YouTube-specific recovery actions
        elif category in [ErrorCategory.YOUTUBE_UNAVAILABLE, ErrorCategory.VIDEO_NOT_FOUND]:
            actions.extend([
                RecoveryAction(
                    "validate_url",
                    "Validate URL",
                    "Check the YouTube URL is correct",
                    automatic=True,
                    success_probability=0.3
                ),
                RecoveryAction(
                    "try_different_url",
                    "Try different video",
                    "Use a different YouTube video URL",
                    automatic=False,
                    success_probability=0.9
                )
            ])
        
        # Model-related recovery actions
        elif category in [ErrorCategory.MODEL_LOADING_FAILED, ErrorCategory.MEMORY_INSUFFICIENT]:
            actions.extend([
                RecoveryAction(
                    "use_smaller_model",
                    "Use smaller model",
                    "Switch to 'tiny' model for faster processing",
                    automatic=True,
                    success_probability=0.8
                ),
                RecoveryAction(
                    "clear_memory",
                    "Clear memory",
                    "Free up system memory",
                    automatic=True,
                    success_probability=0.6
                ),
                RecoveryAction(
                    "restart_application",
                    "Restart application",
                    "Close and restart the application",
                    automatic=False,
                    success_probability=0.9
                )
            ])
        
        # File system recovery actions
        elif category in [ErrorCategory.FILE_PERMISSION_DENIED, ErrorCategory.DISK_SPACE_INSUFFICIENT]:
            actions.extend([
                RecoveryAction(
                    "change_output_directory",
                    "Change output directory",
                    "Select a different output folder",
                    automatic=False,
                    success_probability=0.8
                ),
                RecoveryAction(
                    "check_permissions",
                    "Check permissions",
                    "Verify folder write permissions",
                    automatic=True,
                    success_probability=0.5
                )
            ])
        
        # General recovery actions
        actions.append(
            RecoveryAction(
                "retry_operation",
                "Retry operation",
                "Try the operation again",
                automatic=False,
                success_probability=0.4
            )
        )
        
        return actions
    
    def _gather_system_info(self) -> Dict[str, Any]:
        """Gather comprehensive system information for debugging."""
        try:
            # Basic system info
            info = {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Try to get system info with psutil if available
            try:
                import psutil
                info.update({
                    "cpu_count": psutil.cpu_count(),
                    "memory_total": psutil.virtual_memory().total,
                    "memory_available": psutil.virtual_memory().available,
                    "disk_free": psutil.disk_usage('/').free if platform.system() != 'Windows' else psutil.disk_usage('C:').free,
                })
            except ImportError:
                # Fallback without psutil
                import os
                info.update({
                    "cpu_count": os.cpu_count() or 1,
                    "memory_total": "unknown",
                    "memory_available": "unknown", 
                    "disk_free": "unknown"
                })
            
            # GPU information if available
            try:
                import torch
                if torch.cuda.is_available():
                    info["gpu_available"] = True
                    info["gpu_count"] = torch.cuda.device_count()
                    info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
                else:
                    info["gpu_available"] = False
            except ImportError:
                info["gpu_available"] = False
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to gather system info: {e}"}
    
    def _can_retry(self, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if operation can be retried."""
        non_retryable = [
            ErrorCategory.YOUTUBE_PRIVATE,
            ErrorCategory.YOUTUBE_AGE_RESTRICTED,
            ErrorCategory.VIDEO_NOT_FOUND,
            ErrorCategory.INVALID_URL,
            ErrorCategory.DEPENDENCY_MISSING,
            ErrorCategory.WORKFLOW_CANCELLED
        ]
        
        return category not in non_retryable and severity != ErrorSeverity.CRITICAL
    
    def _can_resume(self, category: ErrorCategory, severity: ErrorSeverity, 
                   context: Optional[Dict[str, Any]]) -> bool:
        """Determine if workflow can be resumed."""
        resumable_categories = [
            ErrorCategory.NETWORK_CONNECTION,
            ErrorCategory.NETWORK_TIMEOUT,
            ErrorCategory.MODEL_DOWNLOAD_FAILED,
            ErrorCategory.TRANSCRIPTION_FAILED
        ]
        
        return (category in resumable_categories and 
                severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM] and
                context and context.get("stage") not in ["cleanup", "completed"])
    
    def _attempt_automatic_recovery(self, error_info: ErrorInfo) -> None:
        """Attempt automatic recovery for recoverable errors."""
        error_info.automatic_recovery_attempted = True
        
        for action in error_info.recovery_actions:
            if action.automatic and action.success_probability > 0.5:
                try:
                    success = self._execute_recovery_action(action, error_info)
                    if success:
                        self.logger.info(f"Automatic recovery successful: {action.action_id}")
                        self._notify_recovery_callbacks(action.action_id, True)
                        return
                except Exception as e:
                    self.logger.warning(f"Automatic recovery failed: {action.action_id} - {e}")
        
        self.logger.warning("No automatic recovery options successful")
    
    def _execute_recovery_action(self, action: RecoveryAction, error_info: ErrorInfo) -> bool:
        """Execute a specific recovery action."""
        action_id = action.action_id
        
        if action_id == "retry_with_delay":
            time.sleep(30)
            return True
            
        elif action_id == "use_smaller_model":
            # This would need to be implemented with callback to change model
            return False
            
        elif action_id == "clear_memory":
            try:
                import gc
                gc.collect()
                # Try to clear GPU memory if available
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                return True
            except Exception:
                return False
        
        elif action_id == "validate_url":
            # URL validation would need external context
            return False
        
        elif action_id == "check_permissions":
            # Permission checking would need file system access
            return False
        
        return False
    
    def _log_error(self, error_info: ErrorInfo) -> None:
        """Log error information to file."""
        try:
            # Prepare log entry
            log_entry = {
                "timestamp": error_info.timestamp.isoformat(),
                "category": error_info.category.value,
                "severity": error_info.severity.value,
                "error_message": error_info.error_message,
                "user_message": error_info.user_message,
                "context": error_info.context,
                "system_info": error_info.system_info,
                "workflow_stage": error_info.workflow_stage,
                "recovery_actions": [
                    {
                        "action_id": action.action_id,
                        "description": action.description,
                        "automatic": action.automatic,
                        "success_probability": action.success_probability
                    }
                    for action in error_info.recovery_actions
                ],
                "can_retry": error_info.can_retry,
                "can_resume": error_info.can_resume,
                "automatic_recovery_attempted": error_info.automatic_recovery_attempted
            }
            
            # Load existing logs
            logs = []
            if self.error_log_file.exists():
                with open(self.error_log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            
            # Add new entry and keep only recent ones (last 1000)
            logs.append(log_entry)
            if len(logs) > 1000:
                logs = logs[-1000:]
            
            # Save updated logs
            with open(self.error_log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.warning(f"Failed to log error: {e}")
    
    def _load_error_statistics(self) -> None:
        """Load error statistics from previous sessions."""
        try:
            stats_file = self.log_dir / "error_statistics.json"
            if stats_file.exists():
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    
                # Load error counts
                for category_str, count in stats.get("error_counts", {}).items():
                    try:
                        category = ErrorCategory(category_str)
                        self.error_counts[category] = count
                    except ValueError:
                        pass  # Skip invalid categories
                
                # Load recovery success rates
                self.recovery_success_rates = stats.get("recovery_success_rates", {})
                
                self.logger.info("Error statistics loaded")
                
        except Exception as e:
            self.logger.warning(f"Failed to load error statistics: {e}")
    
    def _save_error_statistics(self) -> None:
        """Save error statistics for future sessions."""
        try:
            stats = {
                "error_counts": {cat.value: count for cat, count in self.error_counts.items()},
                "recovery_success_rates": self.recovery_success_rates,
                "last_updated": datetime.now().isoformat()
            }
            
            stats_file = self.log_dir / "error_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save error statistics: {e}")
    
    def _notify_error_callbacks(self, error_info: ErrorInfo) -> None:
        """Notify all registered error callbacks."""
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                self.logger.warning(f"Error callback failed: {e}")
    
    def _notify_recovery_callbacks(self, action_id: str, success: bool) -> None:
        """Notify recovery callbacks about action results."""
        for callback in self.recovery_callbacks:
            try:
                callback(action_id, success)
            except Exception as e:
                self.logger.warning(f"Recovery callback failed: {e}")
    
    def add_error_callback(self, callback: Callable[[ErrorInfo], None]) -> None:
        """Add error notification callback.
        
        Args:
            callback: Function to call when errors occur
        """
        self.error_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Add recovery notification callback.
        
        Args:
            callback: Function to call when recovery actions are attempted
        """
        self.recovery_callbacks.append(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and diagnostics.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            "total_errors": total_errors,
            "error_counts": {cat.value: count for cat, count in self.error_counts.items()},
            "most_common_errors": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5],
            "recovery_success_rates": self.recovery_success_rates,
            "recent_errors": len([e for e in self.error_history if 
                                (datetime.now() - e.timestamp).total_seconds() < 3600])
        }
    
    def get_user_guidance(self, category: ErrorCategory) -> Dict[str, Any]:
        """Get detailed user guidance for specific error category.
        
        Args:
            category: Error category to get guidance for
            
        Returns:
            Dictionary with detailed guidance information
        """
        guidance = {
            ErrorCategory.NETWORK_CONNECTION: {
                "title": "Network Connection Issues",
                "description": "Problems connecting to the internet or YouTube",
                "steps": [
                    "Check your internet connection",
                    "Try opening a website in your browser",
                    "Restart your router if needed",
                    "Disable VPN temporarily",
                    "Check firewall settings"
                ],
                "prevention": "Ensure stable internet connection before starting"
            },
            ErrorCategory.YOUTUBE_UNAVAILABLE: {
                "title": "YouTube Video Issues",
                "description": "The requested YouTube video cannot be accessed",
                "steps": [
                    "Verify the video URL is correct",
                    "Check if the video is publicly available",
                    "Try a different video",
                    "Wait and try again later"
                ],
                "prevention": "Use public YouTube videos that are not region-blocked"
            },
            ErrorCategory.MEMORY_INSUFFICIENT: {
                "title": "Memory Issues",
                "description": "Not enough memory available for processing",
                "steps": [
                    "Close other applications",
                    "Use a smaller Whisper model (tiny instead of large)",
                    "Restart the application",
                    "Consider upgrading your computer's RAM"
                ],
                "prevention": "Close unnecessary applications before processing large videos"
            }
        }
        
        return guidance.get(category, {
            "title": "General Error",
            "description": "An error occurred during processing",
            "steps": [
                "Try the operation again",
                "Restart the application",
                "Check your settings",
                "Contact support if the issue persists"
            ],
            "prevention": "Ensure all requirements are met before starting"
        })
    
    def cleanup(self) -> None:
        """Clean up error handler resources."""
        self._save_error_statistics()
        self.logger.info("ErrorHandler cleanup completed")


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    """Setup application-wide logging configuration.
    
    Args:
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create log directory if not provided
    if log_dir is None:
        log_dir = Path.home() / ".youtube_whisper_transcriber" / "logs"
    
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Setup logging configuration
    log_file = log_dir / "application.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create application logger
    app_logger = logging.getLogger("youtube_whisper_transcriber")
    app_logger.info(f"Logging initialized - Log file: {log_file}")
    
    return app_logger


def log_exception(error: Exception, context: Optional[str] = None, logger: Optional[logging.Logger] = None) -> None:
    """Log exception with context information.
    
    Args:
        error: Exception to log
        context: Additional context information
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = logging.getLogger("youtube_whisper_transcriber")
    
    # Format exception message
    error_msg = f"Exception occurred: {type(error).__name__}: {error}"
    if context:
        error_msg = f"{context} - {error_msg}"
    
    # Log with full traceback
    logger.exception(error_msg)
    
    # Also log system information for debugging
    try:
        import platform
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        }
        logger.error(f"System info: {system_info}")
    except Exception:
        pass  # Don't let logging errors break the application

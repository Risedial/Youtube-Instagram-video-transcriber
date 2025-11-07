"""Application controller for YouTube Whisper Transcriber.

This module coordinates the complete workflow from YouTube URL input to final
transcription output, managing the pipeline of download → audio extraction → transcription.
"""

from typing import Optional, Callable, Dict, Any
from pathlib import Path
import logging
import threading
import time
from enum import Enum
from dataclasses import dataclass

from processing.video_downloader import VideoDownloader, DownloadError
from processing.instagram_downloader import InstagramDownloader, InstagramDownloadError
from processing.whisper_transcriber import WhisperTranscriber, TranscriptionError, TranscriptionResult
from utils.state_manager import ApplicationState, StateManager
from utils.error_handler import ErrorHandler, ErrorCategory
from utils.url_validator import URLValidator, SupportedPlatform
from utils.proxy_manager import ProxyManager
from config.settings import get_settings


class WorkflowStage(Enum):
    """Enumeration of workflow stages."""
    IDLE = "idle"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    SAVING = "saving"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class WorkflowProgress:
    """Progress information for the complete workflow."""
    stage: WorkflowStage
    overall_percentage: float
    stage_percentage: float
    status_message: str
    estimated_time_remaining: Optional[float] = None


class TranscriptionWorkflow:
    """Complete transcription workflow coordinator.
    
    This class manages the entire process from YouTube/Instagram URL to transcription file:
    1. URL validation and platform detection
    2. Video/audio download and extraction (YouTube/Instagram)
    3. Audio transcription with Whisper
    4. File output and cleanup
    
    Features:
    - Multi-platform support (YouTube, Instagram Reels)
    - Coordinated progress tracking across all stages
    - Comprehensive error handling and recovery
    - User cancellation support with proper cleanup
    - Resource management and optimization
    - Thread-safe operation with GUI integration
    
    Legal Notice:
    Instagram functionality may violate Instagram's Terms of Service.
    Users assume full legal responsibility for usage.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None, whisper_model: str = "tiny", 
                 whisper_device: str = "auto") -> None:
        """Initialize transcription workflow.
        
        Args:
            temp_dir: Temporary directory for intermediate files
            whisper_model: Whisper model to use for transcription
            whisper_device: Device for Whisper processing ('auto', 'cpu', 'cuda')
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize proxy manager
        self.settings = get_settings()
        self.proxy_manager = ProxyManager(self.settings) if self.settings else None
        
        # Initialize components with proxy support
        self.youtube_downloader = VideoDownloader(temp_dir, self.proxy_manager)
        self.instagram_downloader = InstagramDownloader(temp_dir, self.proxy_manager)
        self.transcriber = WhisperTranscriber(whisper_model, whisper_device, self.proxy_manager)
        self.state_manager = StateManager()
        self.error_handler = ErrorHandler()
        
        # Kill switch monitoring
        self.kill_switch_active = False
        if self.proxy_manager and self.settings and self.settings._settings.enable_kill_switch:
            self._setup_kill_switch_monitoring()
        
        # Workflow state
        self.current_stage = WorkflowStage.IDLE
        self.current_url: Optional[str] = None
        self.current_output_dir: Optional[Path] = None
        self.cancel_event = threading.Event()
        self.workflow_lock = threading.Lock()
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[WorkflowProgress], None]] = None
        self.status_callback: Optional[Callable[[str, str], None]] = None
        
        # Stage weights for overall progress calculation
        self.stage_weights = {
            WorkflowStage.VALIDATING: 5,
            WorkflowStage.DOWNLOADING: 40,
            WorkflowStage.TRANSCRIBING: 50,
            WorkflowStage.SAVING: 5
        }
        
        # Current progress tracking
        self.stage_progress = {stage: 0.0 for stage in self.stage_weights.keys()}
        
        # Setup callbacks for components
        self.youtube_downloader.set_progress_callback(self._download_progress_callback)
        self.youtube_downloader.set_status_callback(self._component_status_callback)
        self.instagram_downloader.set_progress_callback(self._download_progress_callback)
        self.instagram_downloader.set_status_callback(self._component_status_callback)
        self.transcriber.set_progress_callback(self._transcription_progress_callback)
        self.transcriber.set_status_callback(self._component_status_callback)
        
        self.logger.info(f"TranscriptionWorkflow initialized: model={whisper_model}, device={whisper_device}")
        
    def set_progress_callback(self, callback: Callable[[WorkflowProgress], None]) -> None:
        """Set workflow progress callback.
        
        Args:
            callback: Function that receives WorkflowProgress updates
        """
        self.progress_callback = callback
        
    def set_status_callback(self, callback: Callable[[str, str], None]) -> None:
        """Set status message callback.
        
        Args:
            callback: Function that receives (message, status_type) updates
        """
        self.status_callback = callback
        
    def _download_progress_callback(self, percentage: float, message: str) -> None:
        """Handle progress updates from video downloader."""
        self.stage_progress[WorkflowStage.DOWNLOADING] = percentage
        self._update_workflow_progress(message)
        
    def _transcription_progress_callback(self, percentage: float, message: str) -> None:
        """Handle progress updates from transcriber."""
        self.stage_progress[WorkflowStage.TRANSCRIBING] = percentage
        self._update_workflow_progress(message)
        
    def _component_status_callback(self, message: str, status_type: str) -> None:
        """Handle status updates from components."""
        if self.status_callback:
            self.status_callback(message, status_type)
            
    def _update_workflow_progress(self, status_message: str) -> None:
        """Update overall workflow progress."""
        if not self.progress_callback:
            return
            
        # Calculate overall percentage
        total_weight = sum(self.stage_weights.values())
        weighted_progress = 0.0
        
        for stage, weight in self.stage_weights.items():
            stage_progress = self.stage_progress.get(stage, 0.0)
            weighted_progress += (stage_progress * weight) / total_weight
            
        # Get current stage percentage
        current_stage_progress = self.stage_progress.get(self.current_stage, 0.0)
        
        progress = WorkflowProgress(
            stage=self.current_stage,
            overall_percentage=weighted_progress,
            stage_percentage=current_stage_progress,
            status_message=status_message
        )
        
        self.progress_callback(progress)
        
    def _transition_to_stage(self, stage: WorkflowStage, message: str) -> None:
        """Transition to a new workflow stage."""
        self.logger.info(f"Workflow stage: {self.current_stage.value} → {stage.value}")
        self.current_stage = stage
        
        # Reset progress for new stage
        if stage in self.stage_progress:
            self.stage_progress[stage] = 0.0
            
        self._update_workflow_progress(message)
        
    def transcribe_from_url(self, url: str, output_directory: Path,
                          output_filename: Optional[str] = None,
                          timestamp_format: Optional[str] = None) -> Optional[TranscriptionResult]:
        """Perform complete transcription workflow from supported platform URL.

        Args:
            url: YouTube or Instagram URL to transcribe
            output_directory: Directory to save transcription file
            output_filename: Custom filename for output (optional)
            timestamp_format: Timestamp format - "include" or "exclude" (optional, defaults to settings)

        Returns:
            TranscriptionResult if successful, None if failed or cancelled
        """
        with self.workflow_lock:
            try:
                self.current_url = url
                self.current_output_dir = Path(output_directory)
                self.cancel_event.clear()
                
                # Reset all stage progress
                for stage in self.stage_progress:
                    self.stage_progress[stage] = 0.0
                    
                self.logger.info(f"Starting transcription workflow for: {url}")
                
                # Stage 1: Validation and Platform Detection
                self._transition_to_stage(WorkflowStage.VALIDATING, "Validating URL and detecting platform...")
                
                # Detect platform
                is_valid, platform = URLValidator.validate_url(url)
                if not is_valid:
                    raise ValueError("Invalid URL format. Please provide a valid YouTube or Instagram URL.")
                
                platform_name = URLValidator.get_platform_name(platform)
                self.logger.info(f"Detected platform: {platform_name}")
                
                # Get content information based on platform
                if platform == SupportedPlatform.YOUTUBE:
                    content_info = self.youtube_downloader.get_video_info(url)
                    content_type = "video"
                elif platform == SupportedPlatform.INSTAGRAM:
                    content_info = self.instagram_downloader.get_reel_info(url)
                    content_type = "reel"
                else:
                    raise ValueError(f"Unsupported platform: {platform_name}")
                
                if not content_info:
                    raise ValueError(f"Could not retrieve {content_type} information")
                    
                self.stage_progress[WorkflowStage.VALIDATING] = 100.0
                content_title = content_info.get('title') or content_info.get('caption', 'Unknown')[:50]
                self._update_workflow_progress(f"{platform_name} {content_type} found: {content_title}")
                
                # Check for cancellation
                if self.cancel_event.is_set():
                    return self._handle_cancellation()
                    
                # Stage 2: Download
                download_msg = f"Starting {content_type} download from {platform_name}..."
                self._transition_to_stage(WorkflowStage.DOWNLOADING, download_msg)
                
                # Download based on platform
                if platform == SupportedPlatform.YOUTUBE:
                    media_file = self.youtube_downloader.download_video(url)
                    if not media_file:
                        raise DownloadError("YouTube video download failed")
                elif platform == SupportedPlatform.INSTAGRAM:
                    media_file = self.instagram_downloader.download_reel(url)
                    if not media_file:
                        raise InstagramDownloadError("Instagram reel download failed")
                else:
                    raise ValueError(f"Unsupported platform for download: {platform_name}")
                    
                # Check for cancellation
                if self.cancel_event.is_set():
                    return self._handle_cancellation()
                    
                # Stage 3: Transcription
                self._transition_to_stage(WorkflowStage.TRANSCRIBING, "Starting audio transcription...")

                # Perform transcription (without saving yet)
                result = self.transcriber.transcribe_audio(media_file, output_path=None)
                if not result:
                    raise TranscriptionError("Audio transcription failed")
                    
                # Check for cancellation
                if self.cancel_event.is_set():
                    return self._handle_cancellation()

                # Stage 4: Saving and cleanup
                self._transition_to_stage(WorkflowStage.SAVING, "Saving transcription outputs...")

                # Get format preference (from parameter or settings)
                if timestamp_format is None:
                    # Read from settings if not provided
                    format_choice = self.settings._settings.timestamp_format if self.settings else "include"
                else:
                    # Use provided parameter
                    format_choice = timestamp_format

                # Validate format choice
                if format_choice not in ["include", "exclude"]:
                    self.logger.warning(f"Invalid timestamp format '{format_choice}', defaulting to 'include'")
                    format_choice = "include"

                # Convert to boolean flags for transcriber
                include_timestamps = (format_choice == "include")
                exclude_timestamps = (format_choice == "exclude")

                # Generate base output filename if not provided
                if not output_filename:
                    # Get safe title from content info
                    raw_title = content_info.get('title') or content_info.get('caption', 'transcription')
                    safe_title = raw_title.replace(' ', '_')[:50]
                    safe_title = ''.join(c for c in safe_title if c.isalnum() or c in '_-')
                    platform_prefix = platform_name.lower()
                    output_filename = f"{platform_prefix}_{safe_title}_transcript.txt"

                base_output_path = self.current_output_dir / output_filename

                # Determine status message based on format selection
                if format_choice == "include":
                    format_msg = "Generating transcription with timestamps..."
                else:  # format_choice == "exclude"
                    format_msg = "Generating transcription without timestamps..."

                self._update_workflow_progress(format_msg)

                # Save transcription in selected format
                saved_files = self.transcriber.save_transcription_outputs(
                    result,
                    base_output_path,
                    include_timestamps=include_timestamps,
                    exclude_timestamps=exclude_timestamps
                )

                if not saved_files:
                    raise TranscriptionError("Failed to save transcription files")

                # Cleanup temporary media file
                try:
                    if media_file.exists():
                        media_file.unlink()
                        self.logger.info(f"Cleaned up temporary media file: {media_file}")
                except Exception as e:
                    self.logger.warning(f"Could not clean up media file: {e}")

                self.stage_progress[WorkflowStage.SAVING] = 100.0

                # Stage 5: Completed
                self._transition_to_stage(WorkflowStage.COMPLETED, "Transcription completed successfully!")

                self.logger.info(f"Workflow completed successfully. Saved {len(saved_files)} file(s)")

                if self.status_callback:
                    # Build status message with saved file(s)
                    if len(saved_files) > 1:
                        file_list = "\n".join([f"  - {path.name}" for path in saved_files.values()])
                        file_msg = f"Saved files:\n{file_list}"
                    elif len(saved_files) == 1:
                        file_path = list(saved_files.values())[0]
                        file_msg = f"Saved file: {file_path.name}"
                    else:
                        file_msg = "No files saved"

                    self.status_callback(
                        f"Transcription completed successfully!\n"
                        f"{file_msg}\n"
                        f"Language: {result.language}\n"
                        f"Word count: {result.word_count:,}",
                        "success"
                    )
                    
                return result
                
            except Exception as e:
                return self._handle_error(e)
                
            finally:
                self.current_url = None
                self.current_output_dir = None
                
    def _handle_cancellation(self) -> None:
        """Handle workflow cancellation."""
        self._transition_to_stage(WorkflowStage.CANCELLED, "Workflow cancelled by user")
        
        # Cancel component operations
        self.youtube_downloader.cancel_download()
        self.instagram_downloader.cancel_download()
        self.transcriber.cancel_transcription()
        
        # Cleanup temporary files
        self._cleanup_temporary_files()
        
        self.logger.info("Workflow cancelled by user")
        
        if self.status_callback:
            self.status_callback("Transcription cancelled by user", "warning")
            
        return None
        
    def _handle_error(self, error: Exception) -> None:
        """Handle workflow errors."""
        self._transition_to_stage(WorkflowStage.ERROR, f"Error: {str(error)}")
        
        error_info = self.error_handler.handle_error(error, {
            "url": self.current_url,
            "stage": self.current_stage.value,
            "output_dir": str(self.current_output_dir) if self.current_output_dir else None
        })
        
        self.logger.error(f"Workflow error in stage {self.current_stage.value}: {error}")
        
        if self.status_callback:
            self.status_callback(
                f"Transcription failed in {self.current_stage.value} stage\n"
                f"Error: {str(error)}\n"
                f"Please check the URL and try again",
                "error"
            )
            
        # Cleanup on error
        self._cleanup_temporary_files()
        
        return None
        
    def _cleanup_temporary_files(self) -> None:
        """Clean up temporary files from failed or cancelled operations."""
        try:
            self.youtube_downloader.cleanup_temp_files(keep_recent=False)
            self.instagram_downloader.cleanup_temp_files(keep_recent=False)
            self.transcriber.cleanup_resources()
            self.logger.info("Temporary files cleaned up")
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {e}")
            
    def cancel_workflow(self) -> None:
        """Cancel the current workflow operation."""
        if self.current_url and self.current_stage not in [WorkflowStage.IDLE, WorkflowStage.COMPLETED, WorkflowStage.ERROR]:
            self.logger.info(f"Cancelling workflow for: {self.current_url}")
            self.cancel_event.set()
            
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status information.
        
        Returns:
            Dictionary with current workflow status
        """
        return {
            "stage": self.current_stage.value,
            "url": self.current_url,
            "output_dir": str(self.current_output_dir) if self.current_output_dir else None,
            "is_active": self.current_stage not in [WorkflowStage.IDLE, WorkflowStage.COMPLETED, WorkflowStage.ERROR, WorkflowStage.CANCELLED],
            "can_cancel": self.current_stage in [WorkflowStage.DOWNLOADING, WorkflowStage.TRANSCRIBING],
            "stage_progress": dict(self.stage_progress),
            "whisper_model": self.transcriber.model_name,
            "whisper_device": self.transcriber.device
        }
        
    def change_whisper_model(self, model_name: str) -> bool:
        """Change the Whisper model for transcription.
        
        Args:
            model_name: New model name
            
        Returns:
            True if model changed successfully
        """
        if self.current_stage in [WorkflowStage.DOWNLOADING, WorkflowStage.TRANSCRIBING]:
            self.logger.warning("Cannot change model during active workflow")
            return False
            
        return self.transcriber.change_model(model_name)
        
    def get_available_models(self) -> Dict[str, Dict[str, str]]:
        """Get information about available Whisper models.
        
        Returns:
            Dictionary with model information
        """
        return self.transcriber.model_info
        
    def estimate_workflow_time(self, url: str) -> Optional[float]:
        """Estimate total time for workflow completion.
        
        Args:
            url: YouTube or Instagram URL to analyze
            
        Returns:
            Estimated time in seconds, or None if estimation fails
        """
        try:
            # Detect platform and get content info
            is_valid, platform = URLValidator.validate_url(url)
            if not is_valid:
                return None
                
            # Get content info based on platform
            if platform == SupportedPlatform.YOUTUBE:
                content_info = self.youtube_downloader.get_video_info(url)
                duration_key = 'duration'
            elif platform == SupportedPlatform.INSTAGRAM:
                content_info = self.instagram_downloader.get_reel_info(url)
                duration_key = 'video_duration'
            else:
                return None
                
            if not content_info or not content_info.get(duration_key):
                return None
                
            media_duration = content_info[duration_key]
            
            # Estimate components (Instagram typically faster downloads)
            if platform == SupportedPlatform.YOUTUBE:
                download_time = media_duration * 0.1  # Rough estimate based on connection
            else:  # Instagram
                download_time = media_duration * 0.05  # Usually smaller files
                
            transcription_time = self.transcriber.estimate_processing_time(media_duration)
            overhead_time = 30  # Setup, validation, file operations
            
            total_time = download_time + transcription_time + overhead_time
            return total_time
            
        except Exception as e:
            self.logger.warning(f"Could not estimate workflow time: {e}")
            return None
    
    def _setup_kill_switch_monitoring(self) -> None:
        """Setup kill switch monitoring for proxy failures."""
        if not self.proxy_manager:
            return
            
        # Override the proxy manager's kill switch trigger to stop our workflow
        original_trigger = self.proxy_manager._trigger_kill_switch
        
        def enhanced_kill_switch():
            """Enhanced kill switch that stops the transcription workflow."""
            self.logger.critical("Kill switch activated - stopping all operations")
            
            # Set cancel event to stop current workflow
            self.cancel_event.set()
            
            # Update status
            if self.status_callback:
                self.status_callback(
                    "KILL SWITCH ACTIVATED: All network operations stopped due to proxy failure.\n"
                    "Please check your proxy settings and try again.",
                    "error"
                )
            
            # Transition to error state
            self._transition_to_stage(
                WorkflowStage.ERROR, 
                "Kill switch activated - proxy connection failed"
            )
            
            # Call original trigger
            original_trigger()
        
        # Replace the trigger method
        self.proxy_manager._trigger_kill_switch = enhanced_kill_switch
        
        self.logger.info("Kill switch monitoring enabled for transcription workflow")
    
    def get_proxy_status(self) -> Dict[str, Any]:
        """Get current proxy status information.
        
        Returns:
            Dictionary with proxy status information
        """
        if not self.proxy_manager:
            return {"enabled": False, "status": "disabled"}
            
        stats = self.proxy_manager.get_statistics()
        return {
            "enabled": True,
            "total_proxies": stats["total_proxies"],
            "healthy_proxies": stats["healthy_proxies"],
            "current_proxy_index": stats["current_proxy_index"],
            "successful_requests": stats["stats"]["successful_requests"],
            "failed_requests": stats["stats"]["failed_requests"],
            "kill_switch_activations": stats["stats"]["kill_switch_activations"]
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'proxy_manager') and self.proxy_manager:
            self.proxy_manager.shutdown()


# Factory function for easy instantiation
def create_transcription_workflow(temp_dir: Optional[Path] = None, whisper_model: str = "tiny", 
                                whisper_device: str = "auto") -> TranscriptionWorkflow:
    """Create a new transcription workflow instance.
    
    Args:
        temp_dir: Temporary directory for intermediate files
        whisper_model: Whisper model to use
        whisper_device: Device for Whisper processing
        
    Returns:
        Configured TranscriptionWorkflow instance
    """
    return TranscriptionWorkflow(temp_dir, whisper_model, whisper_device)
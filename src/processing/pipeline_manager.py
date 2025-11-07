"""Sequential processing pipeline manager for YouTube Whisper Transcriber.

This module coordinates the complete workflow pipeline with dependency validation,
error recovery, rollback procedures, partial completion handling, and performance monitoring.
"""

from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
import threading
import time
import traceback
from pathlib import Path
from abc import ABC, abstractmethod

from utils.state_manager import (
    StateManager, ProcessingStage, ApplicationState, 
    StageProgress, WorkflowSession
)
from utils.error_handler import ErrorHandler, ErrorCategory, ErrorInfo
from processing.video_downloader import VideoDownloader, DownloadError
from processing.whisper_transcriber import WhisperTranscriber, TranscriptionError, TranscriptionResult


class PipelineStage(Enum):
    """Pipeline stages with execution order."""
    URL_VALIDATION = 1
    VIDEO_INFO_RETRIEVAL = 2
    VIDEO_DOWNLOAD = 3
    AUDIO_EXTRACTION = 4
    MODEL_LOADING = 5
    AUDIO_TRANSCRIPTION = 6
    FILE_OUTPUT = 7
    CLEANUP = 8


class StageStatus(Enum):
    """Status of individual pipeline stages."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


@dataclass
class PipelineContext:
    """Context data passed between pipeline stages."""
    youtube_url: str
    output_directory: Path
    whisper_model: str
    whisper_device: str = "auto"
    
    # Stage outputs
    video_info: Optional[Dict[str, Any]] = None
    audio_file: Optional[Path] = None
    transcription_result: Optional[TranscriptionResult] = None
    output_file: Optional[Path] = None
    
    # Processing metadata
    start_time: float = field(default_factory=time.time)
    stage_times: Dict[PipelineStage, float] = field(default_factory=dict)
    total_bytes_processed: int = 0
    
    # Configuration
    include_timestamps: bool = True
    cleanup_temp_files: bool = True
    max_video_duration: int = 7200  # 2 hours


@dataclass
class StageResult:
    """Result of pipeline stage execution."""
    stage: PipelineStage
    status: StageStatus
    success: bool
    output_data: Any = None
    error_info: Optional[ErrorInfo] = None
    execution_time: float = 0.0
    memory_used: int = 0
    can_retry: bool = False
    can_rollback: bool = False


class PipelineStageBase(ABC):
    """Abstract base class for pipeline stages."""
    
    def __init__(self, stage: PipelineStage, name: str):
        self.stage = stage
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Execute the pipeline stage.
        
        Args:
            context: Pipeline context with input data
            progress_callback: Progress update callback
            
        Returns:
            StageResult with execution outcome
        """
        pass
    
    @abstractmethod
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Validate that all dependencies for this stage are met.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if dependencies are satisfied
        """
        pass
    
    @abstractmethod
    def rollback(self, context: PipelineContext) -> bool:
        """Rollback changes made by this stage.
        
        Args:
            context: Pipeline context
            
        Returns:
            True if rollback was successful
        """
        pass
    
    def estimate_duration(self, context: PipelineContext) -> float:
        """Estimate stage execution duration in seconds.
        
        Args:
            context: Pipeline context
            
        Returns:
            Estimated duration in seconds
        """
        return 30.0  # Default estimate


class URLValidationStage(PipelineStageBase):
    """URL validation and format checking stage."""
    
    def __init__(self):
        super().__init__(PipelineStage.URL_VALIDATION, "URLValidation")
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Validate YouTube URL format."""
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(0.0, "Validating YouTube URL...")
            
            # Import here to avoid circular imports
            from utils.validators import validate_youtube_url
            
            if not validate_youtube_url(context.youtube_url):
                raise ValueError(f"Invalid YouTube URL format: {context.youtube_url}")
            
            if progress_callback:
                progress_callback(50.0, "URL format validated")
            
            # Additional URL checks
            if len(context.youtube_url) > 1000:
                raise ValueError("URL is too long")
            
            if not any(domain in context.youtube_url.lower() for domain in ['youtube.com', 'youtu.be']):
                raise ValueError("URL does not appear to be a YouTube URL")
            
            if progress_callback:
                progress_callback(100.0, "URL validation completed")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data={"validated_url": context.youtube_url},
                execution_time=execution_time,
                can_retry=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=ErrorCategory.INVALID_URL,
                    severity="medium",
                    error_message=str(e),
                    user_message=f"URL validation failed: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={"url": context.youtube_url},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time,
                can_retry=True
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Check that URL is provided."""
        return bool(context.youtube_url)
    
    def rollback(self, context: PipelineContext) -> bool:
        """No rollback needed for validation."""
        return True


class VideoInfoStage(PipelineStageBase):
    """Video information retrieval stage."""
    
    def __init__(self, downloader: VideoDownloader):
        super().__init__(PipelineStage.VIDEO_INFO_RETRIEVAL, "VideoInfo")
        self.downloader = downloader
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Retrieve video information from YouTube."""
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(0.0, "Retrieving video information...")
            
            video_info = self.downloader.get_video_info(context.youtube_url)
            if not video_info:
                raise ValueError("Could not retrieve video information")
            
            if progress_callback:
                progress_callback(50.0, f"Found video: {video_info.get('title', 'Unknown')}")
            
            # Validate video suitability
            duration = video_info.get('duration', 0)
            if duration > context.max_video_duration:
                hours = duration // 3600
                minutes = (duration % 3600) // 60
                raise ValueError(f"Video too long ({hours}h {minutes}m). Maximum: 2 hours")
            
            if video_info.get('is_live'):
                raise ValueError("Live streams are not supported")
            
            context.video_info = video_info
            
            if progress_callback:
                progress_callback(100.0, "Video information retrieved")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data=video_info,
                execution_time=execution_time,
                can_retry=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=ErrorCategory.YOUTUBE_UNAVAILABLE,
                    severity="medium",
                    error_message=str(e),
                    user_message=f"Failed to get video information: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={"url": context.youtube_url},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time,
                can_retry=True
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Check that URL is validated."""
        return bool(context.youtube_url)
    
    def rollback(self, context: PipelineContext) -> bool:
        """Clear video info from context."""
        context.video_info = None
        return True
    
    def estimate_duration(self, context: PipelineContext) -> float:
        """Estimate duration for video info retrieval."""
        return 10.0


class VideoDownloadStage(PipelineStageBase):
    """Video download and audio extraction stage."""
    
    def __init__(self, downloader: VideoDownloader):
        super().__init__(PipelineStage.VIDEO_DOWNLOAD, "VideoDownload")
        self.downloader = downloader
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Download video and extract audio."""
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(0.0, "Starting video download...")
            
            # Set up progress callback for downloader
            if progress_callback:
                self.downloader.set_progress_callback(progress_callback)
            
            audio_file = self.downloader.download_video(context.youtube_url)
            if not audio_file or not audio_file.exists():
                raise DownloadError("Video download failed")
            
            context.audio_file = audio_file
            context.total_bytes_processed = audio_file.stat().st_size
            
            if progress_callback:
                progress_callback(100.0, "Video download completed")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data={"audio_file": str(audio_file)},
                execution_time=execution_time,
                memory_used=audio_file.stat().st_size,
                can_retry=True,
                can_rollback=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Determine error category
            category = ErrorCategory.NETWORK_CONNECTION
            if "private" in str(e).lower():
                category = ErrorCategory.YOUTUBE_PRIVATE
            elif "not found" in str(e).lower():
                category = ErrorCategory.VIDEO_NOT_FOUND
            elif "age" in str(e).lower():
                category = ErrorCategory.YOUTUBE_AGE_RESTRICTED
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=category,
                    severity="high",
                    error_message=str(e),
                    user_message=f"Video download failed: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={"url": context.youtube_url},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time,
                can_retry=True
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Check that video info is available."""
        return context.video_info is not None
    
    def rollback(self, context: PipelineContext) -> bool:
        """Remove downloaded audio file."""
        if context.audio_file and context.audio_file.exists():
            try:
                context.audio_file.unlink()
                context.audio_file = None
                return True
            except Exception as e:
                self.logger.warning(f"Failed to rollback audio file: {e}")
                return False
        return True
    
    def estimate_duration(self, context: PipelineContext) -> float:
        """Estimate download duration based on video length."""
        if context.video_info:
            duration = context.video_info.get('duration', 300)
            # Rough estimate: 0.1x video duration for download
            return max(duration * 0.1, 30.0)
        return 120.0


class TranscriptionStage(PipelineStageBase):
    """Audio transcription stage."""
    
    def __init__(self, transcriber: WhisperTranscriber):
        super().__init__(PipelineStage.AUDIO_TRANSCRIPTION, "Transcription")
        self.transcriber = transcriber
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Transcribe audio to text."""
        start_time = time.time()
        
        try:
            if not context.audio_file or not context.audio_file.exists():
                raise ValueError("Audio file not available for transcription")
            
            if progress_callback:
                progress_callback(0.0, "Starting audio transcription...")
            
            # Set up progress callback for transcriber
            if progress_callback:
                self.transcriber.set_progress_callback(progress_callback)
            
            # Perform transcription
            result = self.transcriber.transcribe_audio(context.audio_file)
            if not result:
                raise TranscriptionError("Audio transcription failed")
            
            context.transcription_result = result
            
            if progress_callback:
                progress_callback(100.0, "Audio transcription completed")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data={
                    "text": result.text,
                    "language": result.language,
                    "word_count": result.word_count,
                    "confidence": result.confidence
                },
                execution_time=execution_time,
                can_retry=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Determine error category
            category = ErrorCategory.TRANSCRIPTION_FAILED
            if "memory" in str(e).lower():
                category = ErrorCategory.MEMORY_INSUFFICIENT
            elif "model" in str(e).lower():
                category = ErrorCategory.MODEL_LOADING_FAILED
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=category,
                    severity="high",
                    error_message=str(e),
                    user_message=f"Transcription failed: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={"audio_file": str(context.audio_file) if context.audio_file else None},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time,
                can_retry=True
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Check that audio file is available."""
        return context.audio_file is not None and context.audio_file.exists()
    
    def rollback(self, context: PipelineContext) -> bool:
        """Clear transcription result."""
        context.transcription_result = None
        return True
    
    def estimate_duration(self, context: PipelineContext) -> float:
        """Estimate transcription duration based on model and audio length."""
        if context.video_info:
            duration = context.video_info.get('duration', 300)
            # Estimate based on model type
            model_ratios = {
                "tiny": 0.15,
                "base": 0.25,
                "small": 0.4,
                "medium": 0.7,
                "large": 1.0
            }
            ratio = model_ratios.get(context.whisper_model, 0.5)
            return max(duration * ratio, 60.0)
        return 300.0


class FileOutputStage(PipelineStageBase):
    """Transcription file output stage."""
    
    def __init__(self):
        super().__init__(PipelineStage.FILE_OUTPUT, "FileOutput")
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Save transcription to file."""
        start_time = time.time()
        
        try:
            if not context.transcription_result:
                raise ValueError("No transcription result available")
            
            if progress_callback:
                progress_callback(0.0, "Saving transcription file...")
            
            # Generate output filename
            video_title = context.video_info.get('title', 'transcription') if context.video_info else 'transcription'
            safe_title = ''.join(c for c in video_title if c.isalnum() or c in ' -_')[:50].strip()
            safe_title = safe_title.replace(' ', '_')
            
            output_file = context.output_directory / f"{safe_title}_transcript.txt"
            
            if progress_callback:
                progress_callback(50.0, f"Writing to {output_file.name}")
            
            # Save transcription using transcriber's save method
            from processing.whisper_transcriber import WhisperTranscriber
            temp_transcriber = WhisperTranscriber()
            success = temp_transcriber.save_transcription(
                context.transcription_result, 
                output_file, 
                include_metadata=context.include_timestamps
            )
            
            if not success:
                raise IOError("Failed to save transcription file")
            
            context.output_file = output_file
            
            if progress_callback:
                progress_callback(100.0, "Transcription file saved")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data={"output_file": str(output_file)},
                execution_time=execution_time,
                can_rollback=True
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=ErrorCategory.FILE_PERMISSION_DENIED,
                    severity="high",
                    error_message=str(e),
                    user_message=f"Failed to save transcription: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={"output_dir": str(context.output_directory)},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time,
                can_retry=True
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Check that transcription result is available."""
        return (context.transcription_result is not None and 
                context.output_directory.exists())
    
    def rollback(self, context: PipelineContext) -> bool:
        """Remove output file."""
        if context.output_file and context.output_file.exists():
            try:
                context.output_file.unlink()
                context.output_file = None
                return True
            except Exception as e:
                self.logger.warning(f"Failed to rollback output file: {e}")
                return False
        return True


class CleanupStage(PipelineStageBase):
    """Temporary file cleanup stage."""
    
    def __init__(self):
        super().__init__(PipelineStage.CLEANUP, "Cleanup")
    
    def execute(self, context: PipelineContext, 
                progress_callback: Optional[Callable[[float, str], None]] = None) -> StageResult:
        """Clean up temporary files."""
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(0.0, "Cleaning up temporary files...")
            
            cleaned_files = 0
            
            # Clean up audio file if requested
            if context.cleanup_temp_files and context.audio_file and context.audio_file.exists():
                try:
                    context.audio_file.unlink()
                    cleaned_files += 1
                    self.logger.info(f"Cleaned up audio file: {context.audio_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up audio file: {e}")
            
            if progress_callback:
                progress_callback(100.0, f"Cleanup completed ({cleaned_files} files)")
            
            execution_time = time.time() - start_time
            
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=True,
                output_data={"files_cleaned": cleaned_files},
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Cleanup failures are usually not critical
            return StageResult(
                stage=self.stage,
                status=StageStatus.COMPLETED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=ErrorCategory.FILE_NOT_FOUND,
                    severity="low",
                    error_message=str(e),
                    user_message=f"Cleanup warning: {str(e)}",
                    technical_details=traceback.format_exc(),
                    context={},
                    traceback_text=traceback.format_exc(),
                    recovery_actions=[],
                    system_info={}
                ),
                execution_time=execution_time
            )
    
    def validate_dependencies(self, context: PipelineContext) -> bool:
        """Cleanup can always run."""
        return True
    
    def rollback(self, context: PipelineContext) -> bool:
        """No rollback needed for cleanup."""
        return True


class PipelineManager:
    """Sequential processing pipeline coordinator with comprehensive functionality.
    
    Features:
    - Sequential processing pipeline coordination with dependency validation
    - Error recovery and rollback procedures for failed stages
    - Partial completion handling and workflow resumption capability
    - Resource allocation and cleanup between stages
    - Performance monitoring and optimization across entire pipeline
    """
    
    def __init__(self, state_manager: StateManager, error_handler: ErrorHandler,
                 downloader: VideoDownloader, transcriber: WhisperTranscriber):
        """Initialize pipeline manager.
        
        Args:
            state_manager: State management system
            error_handler: Error handling system
            downloader: Video downloader instance
            transcriber: Whisper transcriber instance
        """
        self.state_manager = state_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        self.pipeline_lock = threading.Lock()
        
        # Initialize pipeline stages
        self.stages: Dict[PipelineStage, PipelineStageBase] = {
            PipelineStage.URL_VALIDATION: URLValidationStage(),
            PipelineStage.VIDEO_INFO_RETRIEVAL: VideoInfoStage(downloader),
            PipelineStage.VIDEO_DOWNLOAD: VideoDownloadStage(downloader),
            PipelineStage.AUDIO_TRANSCRIPTION: TranscriptionStage(transcriber),
            PipelineStage.FILE_OUTPUT: FileOutputStage(),
            PipelineStage.CLEANUP: CleanupStage()
        }
        
        # Pipeline state
        self.current_context: Optional[PipelineContext] = None
        self.stage_results: Dict[PipelineStage, StageResult] = {}
        self.cancel_event = threading.Event()
        
        # Performance tracking
        self.pipeline_metrics = {
            "total_pipelines": 0,
            "successful_pipelines": 0,
            "stage_success_rates": {stage: 0.0 for stage in PipelineStage},
            "average_stage_times": {stage: 0.0 for stage in PipelineStage}
        }
        
        self.logger.info("PipelineManager initialized with all stages")
    
    def execute_pipeline(self, youtube_url: str, output_directory: Path,
                        whisper_model: str = "tiny", whisper_device: str = "auto",
                        progress_callback: Optional[Callable[[float, str], None]] = None,
                        status_callback: Optional[Callable[[str, str], None]] = None) -> Optional[Path]:
        """Execute complete processing pipeline.
        
        Args:
            youtube_url: YouTube URL to process
            output_directory: Directory for output file
            whisper_model: Whisper model to use
            whisper_device: Device for processing
            progress_callback: Progress update callback
            status_callback: Status message callback
            
        Returns:
            Path to output file if successful, None if failed
        """
        with self.pipeline_lock:
            try:
                # Initialize pipeline context
                context = PipelineContext(
                    youtube_url=youtube_url,
                    output_directory=Path(output_directory),
                    whisper_model=whisper_model,
                    whisper_device=whisper_device
                )
                
                self.current_context = context
                self.stage_results.clear()
                self.cancel_event.clear()
                
                # Start workflow session in state manager
                session_id = self.state_manager.start_workflow_session(
                    youtube_url, str(output_directory), whisper_model
                )
                
                self.logger.info(f"Starting pipeline execution: {session_id}")
                
                if status_callback:
                    status_callback(
                        f"Starting transcription pipeline...\n"
                        f"URL: {youtube_url[:100]}{'...' if len(youtube_url) > 100 else ''}\n"
                        f"Model: {whisper_model}\n"
                        f"Output: {output_directory}",
                        "processing"
                    )
                
                # Execute stages in order
                total_stages = len(self.stages)
                
                for i, (stage_enum, stage_impl) in enumerate(self.stages.items()):
                    if self.cancel_event.is_set():
                        self.logger.info("Pipeline cancelled by user")
                        self._handle_cancellation(context)
                        return None
                    
                    # Update overall progress
                    overall_progress = (i / total_stages) * 100
                    
                    # Execute stage
                    result = self._execute_stage(stage_enum, stage_impl, context, progress_callback)
                    self.stage_results[stage_enum] = result
                    
                    # Record stage completion time
                    context.stage_times[stage_enum] = result.execution_time
                    
                    if not result.success:
                        self.logger.error(f"Stage {stage_enum.name} failed: {result.error_info}")
                        
                        # Handle stage failure
                        recovery_success = self._handle_stage_failure(stage_enum, result, context)
                        
                        if not recovery_success:
                            self._handle_pipeline_failure(context, result)
                            return None
                
                # Pipeline completed successfully
                self._handle_pipeline_success(context)
                
                if status_callback:
                    status_callback(
                        f"Pipeline completed successfully!\n"
                        f"Output file: {context.output_file}\n"
                        f"Processing time: {time.time() - context.start_time:.1f}s",
                        "success"
                    )
                
                return context.output_file
                
            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}")
                error_info = self.error_handler.handle_error(e, {
                    "stage": "pipeline_execution",
                    "url": youtube_url,
                    "output_dir": str(output_directory)
                })
                
                if status_callback:
                    status_callback(f"Pipeline failed: {error_info.user_message}", "error")
                
                return None
                
            finally:
                self.current_context = None
                self._update_pipeline_metrics()
    
    def _execute_stage(self, stage_enum: PipelineStage, stage_impl: PipelineStageBase,
                      context: PipelineContext, 
                      progress_callback: Optional[Callable[[float, str], None]]) -> StageResult:
        """Execute individual pipeline stage with validation and monitoring."""
        self.logger.info(f"Starting stage: {stage_enum.name}")
        
        # Update state manager
        processing_stage = self._map_to_processing_stage(stage_enum)
        if processing_stage:
            estimated_duration = stage_impl.estimate_duration(context)
            self.state_manager.start_stage(processing_stage, 
                                         f"Starting {stage_impl.name}...",
                                         estimated_duration)
        
        # Validate dependencies
        if not stage_impl.validate_dependencies(context):
            error_msg = f"Dependencies not met for stage {stage_enum.name}"
            self.logger.error(error_msg)
            
            return StageResult(
                stage=stage_enum,
                status=StageStatus.FAILED,
                success=False,
                error_info=ErrorInfo(
                    timestamp=time.time(),
                    category=ErrorCategory.INVALID_CONFIGURATION,
                    severity="high",
                    error_message=error_msg,
                    user_message=error_msg,
                    technical_details="Stage dependencies validation failed",
                    context={"stage": stage_enum.name},
                    traceback_text="",
                    recovery_actions=[],
                    system_info={}
                )
            )
        
        # Create stage-specific progress callback
        def stage_progress_callback(percentage: float, message: str):
            if processing_stage:
                self.state_manager.update_stage_progress(processing_stage, percentage, message)
            if progress_callback:
                progress_callback(percentage, message)
        
        # Execute stage
        result = stage_impl.execute(context, stage_progress_callback)
        
        # Update state manager based on result
        if processing_stage:
            if result.success:
                self.state_manager.complete_stage(processing_stage, f"{stage_impl.name} completed")
            else:
                recovery_actions = result.error_info.recovery_actions if result.error_info else []
                self.state_manager.fail_stage(processing_stage, result.error_info.error_message,
                                            [action.user_friendly_text for action in recovery_actions])
        
        self.logger.info(f"Stage {stage_enum.name} {'completed' if result.success else 'failed'} "
                        f"in {result.execution_time:.1f}s")
        
        return result
    
    def _map_to_processing_stage(self, pipeline_stage: PipelineStage) -> Optional[ProcessingStage]:
        """Map pipeline stage to processing stage for state manager."""
        mapping = {
            PipelineStage.URL_VALIDATION: ProcessingStage.URL_VALIDATION,
            PipelineStage.VIDEO_INFO_RETRIEVAL: ProcessingStage.VIDEO_INFO,
            PipelineStage.VIDEO_DOWNLOAD: ProcessingStage.VIDEO_DOWNLOAD,
            PipelineStage.AUDIO_TRANSCRIPTION: ProcessingStage.AUDIO_TRANSCRIPTION,
            PipelineStage.FILE_OUTPUT: ProcessingStage.FILE_OUTPUT,
            PipelineStage.CLEANUP: ProcessingStage.CLEANUP
        }
        return mapping.get(pipeline_stage)
    
    def _handle_stage_failure(self, stage: PipelineStage, result: StageResult,
                            context: PipelineContext) -> bool:
        """Handle individual stage failure with recovery attempts."""
        if not result.can_retry:
            self.logger.error(f"Stage {stage.name} failed and cannot be retried")
            return False
        
        # Attempt stage-specific recovery
        if result.error_info and result.error_info.recovery_actions:
            for action in result.error_info.recovery_actions:
                if action.automatic and action.success_probability > 0.6:
                    self.logger.info(f"Attempting automatic recovery: {action.action_id}")
                    
                    # Simple retry logic
                    if action.action_id == "retry_with_delay":
                        time.sleep(10)  # Shorter delay for pipeline
                        
                        # Retry the stage
                        retry_result = self.stages[stage].execute(context)
                        if retry_result.success:
                            self.stage_results[stage] = retry_result
                            self.logger.info(f"Stage {stage.name} recovery successful")
                            return True
        
        self.logger.error(f"Stage {stage.name} recovery failed")
        return False
    
    def _handle_pipeline_failure(self, context: PipelineContext, failed_result: StageResult) -> None:
        """Handle complete pipeline failure with rollback."""
        self.logger.error("Pipeline failed, initiating rollback")
        
        # Transition to error state
        self.state_manager.transition_to_state(
            ApplicationState.ERROR, 
            "Pipeline failed", 
            f"Failed at stage: {failed_result.stage.name}"
        )
        
        # Rollback completed stages in reverse order
        for stage_enum in reversed(list(self.stage_results.keys())):
            if self.stage_results[stage_enum].success and self.stage_results[stage_enum].can_rollback:
                try:
                    self.stages[stage_enum].rollback(context)
                    self.logger.info(f"Rolled back stage: {stage_enum.name}")
                except Exception as e:
                    self.logger.warning(f"Rollback failed for stage {stage_enum.name}: {e}")
        
        # End workflow session as failed
        self.state_manager.end_workflow_session(success=False)
    
    def _handle_pipeline_success(self, context: PipelineContext) -> None:
        """Handle successful pipeline completion."""
        self.logger.info("Pipeline completed successfully")
        
        # Transition to completed state
        self.state_manager.transition_to_state(
            ApplicationState.COMPLETED,
            "Pipeline completed successfully",
            f"Output file: {context.output_file}"
        )
        
        # End workflow session as successful
        self.state_manager.end_workflow_session(success=True)
    
    def _handle_cancellation(self, context: PipelineContext) -> None:
        """Handle pipeline cancellation with cleanup."""
        self.logger.info("Handling pipeline cancellation")
        
        # Transition to cancelled state
        self.state_manager.transition_to_state(
            ApplicationState.CANCELLED,
            "Pipeline cancelled by user"
        )
        
        # Cleanup partial results
        for stage_enum in reversed(list(self.stage_results.keys())):
            if self.stage_results[stage_enum].can_rollback:
                try:
                    self.stages[stage_enum].rollback(context)
                except Exception as e:
                    self.logger.warning(f"Cleanup failed for stage {stage_enum.name}: {e}")
        
        # End workflow session as cancelled
        self.state_manager.end_workflow_session(success=False)
    
    def _update_pipeline_metrics(self) -> None:
        """Update pipeline performance metrics."""
        self.pipeline_metrics["total_pipelines"] += 1
        
        # Check if pipeline was successful
        if (self.stage_results and 
            all(result.success for result in self.stage_results.values())):
            self.pipeline_metrics["successful_pipelines"] += 1
        
        # Update stage success rates and timing
        for stage, result in self.stage_results.items():
            current_rate = self.pipeline_metrics["stage_success_rates"][stage]
            total_pipelines = self.pipeline_metrics["total_pipelines"]
            
            # Update success rate
            if result.success:
                new_rate = (current_rate * (total_pipelines - 1) + 1) / total_pipelines
            else:
                new_rate = (current_rate * (total_pipelines - 1)) / total_pipelines
            
            self.pipeline_metrics["stage_success_rates"][stage] = new_rate
            
            # Update average timing
            current_avg = self.pipeline_metrics["average_stage_times"][stage]
            new_avg = (current_avg * (total_pipelines - 1) + result.execution_time) / total_pipelines
            self.pipeline_metrics["average_stage_times"][stage] = new_avg
    
    def cancel_pipeline(self) -> None:
        """Cancel currently running pipeline."""
        if self.current_context:
            self.logger.info("Cancelling pipeline execution")
            self.cancel_event.set()
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics.
        
        Returns:
            Dictionary with pipeline status information
        """
        return {
            "is_running": self.current_context is not None,
            "current_context": {
                "url": self.current_context.youtube_url,
                "output_dir": str(self.current_context.output_directory),
                "model": self.current_context.whisper_model,
                "start_time": self.current_context.start_time,
                "elapsed_time": time.time() - self.current_context.start_time
            } if self.current_context else None,
            "stage_results": {
                stage.name: {
                    "status": result.status.value,
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "can_retry": result.can_retry,
                    "can_rollback": result.can_rollback
                } for stage, result in self.stage_results.items()
            },
            "metrics": self.pipeline_metrics
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report.
        
        Returns:
            Dictionary with performance metrics and analysis
        """
        total_pipelines = self.pipeline_metrics["total_pipelines"]
        success_rate = (self.pipeline_metrics["successful_pipelines"] / total_pipelines) if total_pipelines > 0 else 0
        
        return {
            "overall_success_rate": success_rate,
            "total_pipelines_executed": total_pipelines,
            "successful_pipelines": self.pipeline_metrics["successful_pipelines"],
            "stage_performance": {
                stage.name: {
                    "success_rate": self.pipeline_metrics["stage_success_rates"][stage],
                    "average_time": self.pipeline_metrics["average_stage_times"][stage],
                    "estimated_time": self.stages[stage].estimate_duration(
                        PipelineContext("", Path("."), "tiny")  # Default context for estimation
                    )
                } for stage in PipelineStage
            },
            "bottleneck_analysis": self._analyze_bottlenecks(),
            "optimization_suggestions": self._get_optimization_suggestions()
        }
    
    def _analyze_bottlenecks(self) -> List[Dict[str, Any]]:
        """Analyze pipeline bottlenecks based on timing data."""
        bottlenecks = []
        
        # Find stages that take significantly longer than estimated
        for stage in PipelineStage:
            avg_time = self.pipeline_metrics["average_stage_times"][stage]
            estimated_time = self.stages[stage].estimate_duration(
                PipelineContext("", Path("."), "tiny")
            )
            
            if avg_time > estimated_time * 1.5:  # 50% longer than estimated
                bottlenecks.append({
                    "stage": stage.name,
                    "average_time": avg_time,
                    "estimated_time": estimated_time,
                    "slowdown_factor": avg_time / estimated_time if estimated_time > 0 else float('inf'),
                    "impact": "high" if avg_time > 300 else "medium"  # 5+ minutes is high impact
                })
        
        return sorted(bottlenecks, key=lambda x: x["slowdown_factor"], reverse=True)
    
    def _get_optimization_suggestions(self) -> List[str]:
        """Get optimization suggestions based on performance data."""
        suggestions = []
        
        # Analyze success rates
        for stage in PipelineStage:
            success_rate = self.pipeline_metrics["stage_success_rates"][stage]
            if success_rate < 0.9:  # Less than 90% success rate
                suggestions.append(f"Improve reliability of {stage.name} stage (current success: {success_rate:.1%})")
        
        # Analyze timing
        bottlenecks = self._analyze_bottlenecks()
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            if bottleneck["stage"] == "VIDEO_DOWNLOAD":
                suggestions.append("Consider implementing parallel download segments or connection optimization")
            elif bottleneck["stage"] == "AUDIO_TRANSCRIPTION":
                suggestions.append("Consider using smaller Whisper models or GPU acceleration")
        
        # General suggestions
        total_pipelines = self.pipeline_metrics["total_pipelines"]
        if total_pipelines > 10:
            overall_success = self.pipeline_metrics["successful_pipelines"] / total_pipelines
            if overall_success < 0.8:
                suggestions.append("Overall pipeline reliability needs improvement - focus on error handling")
        
        return suggestions
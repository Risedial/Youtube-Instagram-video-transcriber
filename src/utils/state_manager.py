"""Centralized state management for YouTube Whisper Transcriber.

This module handles application state, configuration persistence, state transitions,
progress tracking across multiple processing stages, and session state management.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
import threading
import time
import json
from pathlib import Path
import logging


class ApplicationState(Enum):
    """Application processing states with comprehensive coverage."""
    IDLE = "idle"
    VALIDATING = "validating"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class ProcessingStage(Enum):
    """Individual processing stages for granular tracking."""
    URL_VALIDATION = "url_validation"
    VIDEO_INFO = "video_info"
    VIDEO_DOWNLOAD = "video_download"
    AUDIO_EXTRACTION = "audio_extraction"
    MODEL_LOADING = "model_loading"
    AUDIO_TRANSCRIPTION = "audio_transcription"
    FILE_OUTPUT = "file_output"
    CLEANUP = "cleanup"


@dataclass
class StageProgress:
    """Progress information for individual processing stage."""
    stage: ProcessingStage
    percentage: float = 0.0
    status_message: str = ""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    estimated_duration: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get stage duration if completed."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def elapsed_time(self) -> Optional[float]:
        """Get elapsed time for current stage."""
        if self.start_time:
            end = self.end_time or time.time()
            return end - self.start_time
        return None


@dataclass
class WorkflowSession:
    """Session information for workflow resumption."""
    session_id: str
    youtube_url: str
    output_directory: str
    whisper_model: str
    start_time: float
    last_stage: ProcessingStage
    completed_stages: List[ProcessingStage] = field(default_factory=list)
    failed_stages: List[ProcessingStage] = field(default_factory=list)
    can_resume: bool = False


@dataclass
class StateData:
    """Comprehensive application state data."""
    # Core application state
    current_state: ApplicationState = ApplicationState.IDLE
    previous_state: Optional[ApplicationState] = None
    
    # Progress tracking
    overall_progress: float = 0.0
    stage_progress: Dict[ProcessingStage, StageProgress] = field(default_factory=dict)
    current_stage: Optional[ProcessingStage] = None
    
    # Status and messaging
    status_message: str = "Ready"
    detailed_status: str = ""
    last_error: Optional[str] = None
    
    # Current operation context
    current_url: Optional[str] = None
    output_directory: Optional[str] = None
    selected_model: str = "tiny"
    selected_device: str = "auto"
    
    # Session management
    current_session: Optional[WorkflowSession] = None
    
    # Performance metrics
    total_processing_time: float = 0.0
    bytes_processed: int = 0
    estimated_completion_time: Optional[float] = None
    
    # Recovery options
    recovery_options: List[str] = field(default_factory=list)
    can_retry: bool = False
    can_resume: bool = False


class StateObserver:
    """Observer interface for state changes."""
    
    def on_state_changed(self, old_state: ApplicationState, new_state: ApplicationState) -> None:
        """Called when application state changes."""
        pass
    
    def on_progress_updated(self, stage: Optional[ProcessingStage], progress: float) -> None:
        """Called when progress is updated."""
        pass
    
    def on_stage_changed(self, old_stage: Optional[ProcessingStage], new_stage: Optional[ProcessingStage]) -> None:
        """Called when processing stage changes."""
        pass
    
    def on_error_occurred(self, error_message: str, recovery_options: List[str]) -> None:
        """Called when an error occurs."""
        pass


class StateManager:
    """Centralized application state management with comprehensive functionality.
    
    Features:
    - Application state enum with complete workflow coverage
    - Progress percentage calculation across all processing stages with weighted timing
    - Status message updates for detailed user feedback
    - Configuration persistence for user preferences and settings
    - Error state management with recovery options and user guidance
    - Session state tracking for workflow resumption
    """
    
    # Stage weights for overall progress calculation
    STAGE_WEIGHTS = {
        ProcessingStage.URL_VALIDATION: 5,
        ProcessingStage.VIDEO_INFO: 5,
        ProcessingStage.VIDEO_DOWNLOAD: 35,
        ProcessingStage.AUDIO_EXTRACTION: 10,
        ProcessingStage.MODEL_LOADING: 10,
        ProcessingStage.AUDIO_TRANSCRIPTION: 30,
        ProcessingStage.FILE_OUTPUT: 3,
        ProcessingStage.CLEANUP: 2
    }
    
    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize state manager.
        
        Args:
            config_dir: Directory for configuration persistence
        """
        self.state = StateData()
        self.logger = logging.getLogger(__name__)
        self.state_lock = threading.Lock()
        
        # Observer pattern for state changes
        self.observers: List[StateObserver] = []
        
        # Configuration persistence
        self.config_dir = config_dir or Path.home() / ".youtube_whisper_transcriber"
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.state_file = self.config_dir / "session_state.json"
        
        # Load previous session if available
        self._load_session_state()
        
        self.logger.info(f"StateManager initialized with config dir: {self.config_dir}")
    
    def add_observer(self, observer: StateObserver) -> None:
        """Add state change observer.
        
        Args:
            observer: Observer to add
        """
        if observer not in self.observers:
            self.observers.append(observer)
    
    def remove_observer(self, observer: StateObserver) -> None:
        """Remove state change observer.
        
        Args:
            observer: Observer to remove
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def get_state(self) -> StateData:
        """Get current application state.
        
        Returns:
            Copy of current state data
        """
        with self.state_lock:
            # Return a copy to prevent external modification
            return StateData(
                current_state=self.state.current_state,
                previous_state=self.state.previous_state,
                overall_progress=self.state.overall_progress,
                stage_progress=dict(self.state.stage_progress),
                current_stage=self.state.current_stage,
                status_message=self.state.status_message,
                detailed_status=self.state.detailed_status,
                last_error=self.state.last_error,
                current_url=self.state.current_url,
                output_directory=self.state.output_directory,
                selected_model=self.state.selected_model,
                selected_device=self.state.selected_device,
                current_session=self.state.current_session,
                total_processing_time=self.state.total_processing_time,
                bytes_processed=self.state.bytes_processed,
                estimated_completion_time=self.state.estimated_completion_time,
                recovery_options=list(self.state.recovery_options),
                can_retry=self.state.can_retry,
                can_resume=self.state.can_resume
            )
    
    def transition_to_state(self, new_state: ApplicationState, message: str = "", 
                          detailed_message: str = "", save_session: bool = True) -> None:
        """Transition to new application state with comprehensive tracking.
        
        Args:
            new_state: New state to transition to
            message: Brief status message
            detailed_message: Detailed status information
            save_session: Whether to save session state
        """
        with self.state_lock:
            old_state = self.state.current_state
            
            if old_state != new_state:
                self.state.previous_state = old_state
                self.state.current_state = new_state
                
                if message:
                    self.state.status_message = message
                if detailed_message:
                    self.state.detailed_status = detailed_message
                
                # Clear error state when transitioning out of error
                if old_state == ApplicationState.ERROR and new_state != ApplicationState.ERROR:
                    self.state.last_error = None
                    self.state.recovery_options.clear()
                
                # Update retry/resume capabilities
                self._update_recovery_capabilities()
                
                self.logger.info(f"State transition: {old_state.value} → {new_state.value}")
                
                # Notify observers
                for observer in self.observers:
                    try:
                        observer.on_state_changed(old_state, new_state)
                    except Exception as e:
                        self.logger.warning(f"Observer notification failed: {e}")
                
                # Save session state if requested
                if save_session:
                    self._save_session_state()
    
    def start_stage(self, stage: ProcessingStage, message: str = "", 
                   estimated_duration: Optional[float] = None) -> None:
        """Start a new processing stage.
        
        Args:
            stage: Processing stage to start
            message: Status message for the stage
            estimated_duration: Estimated stage duration in seconds
        """
        with self.state_lock:
            old_stage = self.state.current_stage
            self.state.current_stage = stage
            
            # Initialize stage progress
            stage_progress = StageProgress(
                stage=stage,
                percentage=0.0,
                status_message=message,
                start_time=time.time(),
                estimated_duration=estimated_duration
            )
            self.state.stage_progress[stage] = stage_progress
            
            # Update overall progress
            self._calculate_overall_progress()
            
            self.logger.info(f"Started stage: {stage.value}")
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_stage_changed(old_stage, stage)
                except Exception as e:
                    self.logger.warning(f"Observer notification failed: {e}")
    
    def update_stage_progress(self, stage: ProcessingStage, percentage: float, 
                            message: str = "") -> None:
        """Update progress for current processing stage.
        
        Args:
            stage: Processing stage to update
            percentage: Progress percentage (0-100)
            message: Updated status message
        """
        with self.state_lock:
            if stage in self.state.stage_progress:
                stage_progress = self.state.stage_progress[stage]
                stage_progress.percentage = min(max(percentage, 0.0), 100.0)
                
                if message:
                    stage_progress.status_message = message
                    self.state.status_message = message
                
                # Update overall progress
                self._calculate_overall_progress()
                
                # Notify observers
                for observer in self.observers:
                    try:
                        observer.on_progress_updated(stage, percentage)
                    except Exception as e:
                        self.logger.warning(f"Observer notification failed: {e}")
    
    def complete_stage(self, stage: ProcessingStage, message: str = "") -> None:
        """Mark a processing stage as completed.
        
        Args:
            stage: Processing stage that completed
            message: Completion message
        """
        with self.state_lock:
            if stage in self.state.stage_progress:
                stage_progress = self.state.stage_progress[stage]
                stage_progress.percentage = 100.0
                stage_progress.end_time = time.time()
                
                if message:
                    stage_progress.status_message = message
                    self.state.status_message = message
                
                # Add to completed stages in session
                if self.state.current_session:
                    if stage not in self.state.current_session.completed_stages:
                        self.state.current_session.completed_stages.append(stage)
                
                # Update overall progress
                self._calculate_overall_progress()
                
                self.logger.info(f"Completed stage: {stage.value} in {stage_progress.duration:.1f}s")
    
    def fail_stage(self, stage: ProcessingStage, error_message: str, 
                  recovery_options: List[str] = None) -> None:
        """Mark a processing stage as failed.
        
        Args:
            stage: Processing stage that failed
            error_message: Error description
            recovery_options: Available recovery options
        """
        with self.state_lock:
            if stage in self.state.stage_progress:
                stage_progress = self.state.stage_progress[stage]
                stage_progress.end_time = time.time()
                stage_progress.status_message = f"Failed: {error_message}"
            
            # Update error state
            self.state.last_error = error_message
            self.state.recovery_options = recovery_options or []
            
            # Add to failed stages in session
            if self.state.current_session:
                if stage not in self.state.current_session.failed_stages:
                    self.state.current_session.failed_stages.append(stage)
            
            # Update recovery capabilities
            self._update_recovery_capabilities()
            
            self.logger.error(f"Stage failed: {stage.value} - {error_message}")
            
            # Notify observers
            for observer in self.observers:
                try:
                    observer.on_error_occurred(error_message, self.state.recovery_options)
                except Exception as e:
                    self.logger.warning(f"Observer notification failed: {e}")
    
    def start_workflow_session(self, youtube_url: str, output_directory: str, 
                             whisper_model: str) -> str:
        """Start a new workflow session.
        
        Args:
            youtube_url: YouTube URL being processed
            output_directory: Output directory path
            whisper_model: Selected Whisper model
            
        Returns:
            Session ID
        """
        with self.state_lock:
            session_id = f"session_{int(time.time())}"
            
            self.state.current_session = WorkflowSession(
                session_id=session_id,
                youtube_url=youtube_url,
                output_directory=output_directory,
                whisper_model=whisper_model,
                start_time=time.time(),
                last_stage=ProcessingStage.URL_VALIDATION
            )
            
            # Update context
            self.state.current_url = youtube_url
            self.state.output_directory = output_directory
            self.state.selected_model = whisper_model
            
            # Reset progress tracking
            self.state.stage_progress.clear()
            self.state.overall_progress = 0.0
            self.state.total_processing_time = 0.0
            self.state.bytes_processed = 0
            
            self.logger.info(f"Started workflow session: {session_id}")
            
            self._save_session_state()
            return session_id
    
    def end_workflow_session(self, success: bool = True) -> None:
        """End the current workflow session.
        
        Args:
            success: Whether the session completed successfully
        """
        with self.state_lock:
            if self.state.current_session:
                session = self.state.current_session
                
                # Calculate total processing time
                total_time = time.time() - session.start_time
                self.state.total_processing_time = total_time
                
                self.logger.info(
                    f"Ended workflow session: {session.session_id} "
                    f"({'success' if success else 'failure'}) in {total_time:.1f}s"
                )
                
                # Clear session if successful, keep for resumption if failed
                if success:
                    self.state.current_session = None
                else:
                    session.can_resume = True
                
                self._save_session_state()
    
    def _calculate_overall_progress(self) -> None:
        """Calculate overall progress based on stage weights and progress."""
        total_weight = sum(self.STAGE_WEIGHTS.values())
        weighted_progress = 0.0
        
        for stage, weight in self.STAGE_WEIGHTS.items():
            if stage in self.state.stage_progress:
                stage_progress = self.state.stage_progress[stage].percentage
                weighted_progress += (stage_progress * weight) / total_weight
        
        self.state.overall_progress = min(weighted_progress, 100.0)
        
        # Update estimated completion time
        if self.state.overall_progress > 0 and self.state.current_session:
            elapsed = time.time() - self.state.current_session.start_time
            if self.state.overall_progress > 5:  # Avoid division by very small numbers
                estimated_total = elapsed * (100.0 / self.state.overall_progress)
                self.state.estimated_completion_time = estimated_total - elapsed
    
    def _update_recovery_capabilities(self) -> None:
        """Update recovery capabilities based on current state."""
        # Can retry if in error state and not in a critical stage
        self.state.can_retry = (
            self.state.current_state == ApplicationState.ERROR and
            self.state.current_stage not in [ProcessingStage.CLEANUP]
        )
        
        # Can resume if session exists and workflow was interrupted
        self.state.can_resume = (
            self.state.current_session is not None and
            self.state.current_session.can_resume and
            self.state.current_state in [ApplicationState.ERROR, ApplicationState.CANCELLED]
        )
    
    def _save_session_state(self) -> None:
        """Save current session state to disk for resumption."""
        try:
            state_data = {
                "current_state": self.state.current_state.value,
                "current_url": self.state.current_url,
                "output_directory": self.state.output_directory,
                "selected_model": self.state.selected_model,
                "selected_device": self.state.selected_device,
                "overall_progress": self.state.overall_progress,
                "total_processing_time": self.state.total_processing_time,
                "bytes_processed": self.state.bytes_processed,
                "current_session": {
                    "session_id": self.state.current_session.session_id,
                    "youtube_url": self.state.current_session.youtube_url,
                    "output_directory": self.state.current_session.output_directory,
                    "whisper_model": self.state.current_session.whisper_model,
                    "start_time": self.state.current_session.start_time,
                    "last_stage": self.state.current_session.last_stage.value,
                    "completed_stages": [s.value for s in self.state.current_session.completed_stages],
                    "failed_stages": [s.value for s in self.state.current_session.failed_stages],
                    "can_resume": self.state.current_session.can_resume
                } if self.state.current_session else None,
                "timestamp": time.time()
            }
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save session state: {e}")
    
    def _load_session_state(self) -> None:
        """Load previous session state from disk if available."""
        try:
            if not self.state_file.exists():
                return
                
            with open(self.state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # Check if state is recent (within last 24 hours)
            state_age = time.time() - state_data.get("timestamp", 0)
            if state_age > 86400:  # 24 hours
                self.logger.info("Previous session state too old, starting fresh")
                return
            
            # Restore basic state
            if "current_state" in state_data:
                try:
                    self.state.current_state = ApplicationState(state_data["current_state"])
                except ValueError:
                    pass  # Invalid state, keep default
            
            self.state.current_url = state_data.get("current_url")
            self.state.output_directory = state_data.get("output_directory")
            self.state.selected_model = state_data.get("selected_model", "tiny")
            self.state.selected_device = state_data.get("selected_device", "auto")
            self.state.overall_progress = state_data.get("overall_progress", 0.0)
            self.state.total_processing_time = state_data.get("total_processing_time", 0.0)
            self.state.bytes_processed = state_data.get("bytes_processed", 0)
            
            # Restore session if available
            session_data = state_data.get("current_session")
            if session_data:
                try:
                    self.state.current_session = WorkflowSession(
                        session_id=session_data["session_id"],
                        youtube_url=session_data["youtube_url"],
                        output_directory=session_data["output_directory"],
                        whisper_model=session_data["whisper_model"],
                        start_time=session_data["start_time"],
                        last_stage=ProcessingStage(session_data["last_stage"]),
                        completed_stages=[ProcessingStage(s) for s in session_data.get("completed_stages", [])],
                        failed_stages=[ProcessingStage(s) for s in session_data.get("failed_stages", [])],
                        can_resume=session_data.get("can_resume", False)
                    )
                    
                    self.logger.info(f"Restored session: {self.state.current_session.session_id}")
                    
                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Failed to restore session: {e}")
                    self.state.current_session = None
            
            # Update recovery capabilities
            self._update_recovery_capabilities()
            
        except Exception as e:
            self.logger.warning(f"Failed to load session state: {e}")
    
    def clear_session_state(self) -> None:
        """Clear saved session state."""
        try:
            if self.state_file.exists():
                self.state_file.unlink()
            self.logger.info("Session state cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear session state: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for current session.
        
        Returns:
            Dictionary with performance information
        """
        with self.state_lock:
            metrics = {
                "overall_progress": self.state.overall_progress,
                "total_processing_time": self.state.total_processing_time,
                "bytes_processed": self.state.bytes_processed,
                "estimated_completion_time": self.state.estimated_completion_time,
                "stage_durations": {},
                "current_stage": self.state.current_stage.value if self.state.current_stage else None
            }
            
            # Add stage durations
            for stage, progress in self.state.stage_progress.items():
                if progress.duration:
                    metrics["stage_durations"][stage.value] = progress.duration
                elif progress.elapsed_time:
                    metrics["stage_durations"][stage.value] = progress.elapsed_time
            
            return metrics

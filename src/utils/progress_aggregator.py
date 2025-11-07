"""Progress aggregation and calculation across multiple processing stages.

This module provides progress calculation with weighted timing, progress callback aggregation,
smooth progress transitions, and time estimation for complete workflow coordination.
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
import threading
import time
import logging
from collections import deque

from .state_manager import ProcessingStage, StateManager


class ProgressPriority(Enum):
    """Priority levels for progress updates."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProgressUpdate:
    """Individual progress update event."""
    stage: ProcessingStage
    percentage: float
    message: str
    timestamp: float
    priority: ProgressPriority = ProgressPriority.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageWeightConfig:
    """Configuration for stage weight and timing."""
    weight: float  # Relative weight in overall progress (0-100)
    base_duration: float  # Base duration estimate in seconds
    duration_multiplier: float = 1.0  # Multiplier based on input size
    min_duration: float = 5.0  # Minimum duration in seconds
    max_duration: float = 1800.0  # Maximum duration in seconds (30 minutes)


class ProgressAggregator:
    """Progress calculation and aggregation across multiple processing stages.
    
    Features:
    - Progress calculation across all processing stages with weighted timing system
    - Progress callback aggregation from download and transcription components
    - Smooth progress transitions between workflow stages with interpolation
    - Time estimation for total workflow completion with adaptive algorithms
    - Real-time progress monitoring with performance optimization
    """
    
    # Default stage weights and timing configurations
    DEFAULT_STAGE_WEIGHTS = {
        ProcessingStage.URL_VALIDATION: StageWeightConfig(
            weight=2.0, base_duration=5.0, min_duration=2.0, max_duration=30.0
        ),
        ProcessingStage.VIDEO_INFO: StageWeightConfig(
            weight=3.0, base_duration=8.0, min_duration=3.0, max_duration=60.0
        ),
        ProcessingStage.VIDEO_DOWNLOAD: StageWeightConfig(
            weight=35.0, base_duration=120.0, duration_multiplier=0.1, min_duration=30.0, max_duration=600.0
        ),
        ProcessingStage.AUDIO_EXTRACTION: StageWeightConfig(
            weight=8.0, base_duration=15.0, duration_multiplier=0.02, min_duration=5.0, max_duration=120.0
        ),
        ProcessingStage.MODEL_LOADING: StageWeightConfig(
            weight=12.0, base_duration=30.0, min_duration=10.0, max_duration=300.0
        ),
        ProcessingStage.AUDIO_TRANSCRIPTION: StageWeightConfig(
            weight=35.0, base_duration=180.0, duration_multiplier=0.5, min_duration=30.0, max_duration=1200.0
        ),
        ProcessingStage.FILE_OUTPUT: StageWeightConfig(
            weight=3.0, base_duration=5.0, min_duration=2.0, max_duration=30.0
        ),
        ProcessingStage.CLEANUP: StageWeightConfig(
            weight=2.0, base_duration=3.0, min_duration=1.0, max_duration=15.0
        )
    }
    
    def __init__(self, stage_weights: Optional[Dict[ProcessingStage, StageWeightConfig]] = None,
                 smoothing_factor: float = 0.1, update_frequency: float = 0.5):
        """Initialize progress aggregator.
        
        Args:
            stage_weights: Custom stage weight configuration
            smoothing_factor: Factor for progress smoothing (0-1, lower = smoother)
            update_frequency: Minimum time between progress updates in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.progress_lock = threading.Lock()
        
        # Configuration
        self.stage_weights = stage_weights or self.DEFAULT_STAGE_WEIGHTS
        self.smoothing_factor = max(0.01, min(1.0, smoothing_factor))
        self.update_frequency = max(0.1, update_frequency)
        
        # Progress tracking
        self.stage_progress: Dict[ProcessingStage, float] = {}
        self.stage_start_times: Dict[ProcessingStage, float] = {}
        self.stage_durations: Dict[ProcessingStage, float] = {}
        self.stage_estimated_durations: Dict[ProcessingStage, float] = {}
        
        # Smoothing and interpolation
        self.current_overall_progress = 0.0
        self.target_overall_progress = 0.0
        self.last_update_time = 0.0
        self.progress_history: deque = deque(maxlen=100)  # Keep recent progress points
        
        # Callbacks and observers
        self.progress_callbacks: List[Callable[[float, str], None]] = []
        self.stage_callbacks: List[Callable[[ProcessingStage, float, str], None]] = []
        self.completion_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Performance metrics
        self.update_count = 0
        self.total_processing_time = 0.0
        self.workflow_start_time: Optional[float] = None
        
        # Context information
        self.current_context: Dict[str, Any] = {}
        self.video_duration: Optional[float] = None
        self.whisper_model: str = "tiny"
        
        self.logger.info("ProgressAggregator initialized with weighted timing system")
    
    def start_workflow(self, context: Dict[str, Any]) -> None:
        """Start progress tracking for new workflow.
        
        Args:
            context: Workflow context (URL, model, video_duration, etc.)
        """
        with self.progress_lock:
            self.current_context = context.copy()
            self.video_duration = context.get("video_duration")
            self.whisper_model = context.get("whisper_model", "tiny")
            
            # Reset all tracking data
            self.stage_progress.clear()
            self.stage_start_times.clear()
            self.stage_durations.clear()
            self.stage_estimated_durations.clear()
            
            self.current_overall_progress = 0.0
            self.target_overall_progress = 0.0
            self.progress_history.clear()
            
            self.workflow_start_time = time.time()
            self.last_update_time = self.workflow_start_time
            
            # Calculate estimated durations for all stages
            self._calculate_estimated_durations()
            
            self.logger.info(f"Started workflow progress tracking: {context.get('url', 'Unknown')}")
            
            # Notify callbacks of workflow start
            self._notify_progress_callbacks(0.0, "Workflow started")
    
    def start_stage(self, stage: ProcessingStage, message: str = "") -> None:
        """Start tracking progress for a specific stage.
        
        Args:
            stage: Processing stage that's starting
            message: Initial status message
        """
        with self.progress_lock:
            current_time = time.time()
            
            self.stage_start_times[stage] = current_time
            self.stage_progress[stage] = 0.0
            
            # Update target progress to reflect stage start
            self._calculate_target_progress()
            
            self.logger.info(f"Started stage: {stage.value}")
            
            # Notify stage callbacks
            for callback in self.stage_callbacks:
                try:
                    callback(stage, 0.0, message or f"Starting {stage.value}")
                except Exception as e:
                    self.logger.warning(f"Stage callback failed: {e}")
    
    def update_stage_progress(self, stage: ProcessingStage, percentage: float, 
                            message: str = "", priority: ProgressPriority = ProgressPriority.MEDIUM,
                            metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update progress for a specific stage.
        
        Args:
            stage: Processing stage being updated
            percentage: Progress percentage (0-100)
            message: Status message
            priority: Update priority level
            metadata: Additional metadata
        """
        with self.progress_lock:
            current_time = time.time()
            
            # Validate and clamp percentage
            percentage = max(0.0, min(100.0, percentage))
            
            # Check update frequency for non-critical updates
            if (priority != ProgressPriority.CRITICAL and 
                current_time - self.last_update_time < self.update_frequency):
                return
            
            # Update stage progress
            old_progress = self.stage_progress.get(stage, 0.0)
            self.stage_progress[stage] = percentage
            
            # Create progress update record
            progress_update = ProgressUpdate(
                stage=stage,
                percentage=percentage,
                message=message,
                timestamp=current_time,
                priority=priority,
                metadata=metadata or {}
            )
            
            # Add to history
            self.progress_history.append(progress_update)
            
            # Calculate new target overall progress
            self._calculate_target_progress()
            
            # Update timing estimates if stage is progressing
            if percentage > old_progress and stage in self.stage_start_times:
                self._update_timing_estimates(stage, percentage)
            
            self.last_update_time = current_time
            self.update_count += 1
            
            # Notify callbacks
            self._notify_callbacks(progress_update)
    
    def complete_stage(self, stage: ProcessingStage, message: str = "") -> None:
        """Mark a stage as completed.
        
        Args:
            stage: Processing stage that completed
            message: Completion message
        """
        with self.progress_lock:
            current_time = time.time()
            
            # Set stage to 100%
            self.stage_progress[stage] = 100.0
            
            # Record completion time
            if stage in self.stage_start_times:
                duration = current_time - self.stage_start_times[stage]
                self.stage_durations[stage] = duration
                
                self.logger.info(f"Completed stage {stage.value} in {duration:.1f}s")
            
            # Update target progress
            self._calculate_target_progress()
            
            # Notify callbacks
            self.update_stage_progress(stage, 100.0, message or f"Completed {stage.value}", 
                                     ProgressPriority.HIGH)
    
    def get_overall_progress(self, smooth: bool = True) -> float:
        """Get current overall progress percentage.
        
        Args:
            smooth: Whether to return smoothed progress
            
        Returns:
            Overall progress percentage (0-100)
        """
        with self.progress_lock:
            if smooth:
                self._update_smoothed_progress()
                return self.current_overall_progress
            else:
                return self.target_overall_progress
    
    def get_stage_progress(self, stage: ProcessingStage) -> float:
        """Get progress for specific stage.
        
        Args:
            stage: Processing stage
            
        Returns:
            Stage progress percentage (0-100)
        """
        with self.progress_lock:
            return self.stage_progress.get(stage, 0.0)
    
    def get_estimated_completion_time(self) -> Optional[float]:
        """Get estimated time to completion in seconds.
        
        Returns:
            Estimated seconds to completion, None if cannot estimate
        """
        with self.progress_lock:
            if not self.workflow_start_time:
                return None
            
            current_progress = self.target_overall_progress
            if current_progress <= 0:
                return None
            
            elapsed_time = time.time() - self.workflow_start_time
            
            # Calculate based on current progress rate
            if current_progress >= 5.0:  # Need at least 5% progress for reliable estimate
                estimated_total_time = elapsed_time * (100.0 / current_progress)
                remaining_time = estimated_total_time - elapsed_time
                return max(0.0, remaining_time)
            
            # Fallback to stage-based estimation
            return self._estimate_remaining_time_by_stages()
    
    def get_progress_velocity(self) -> float:
        """Get current progress velocity (percentage per second).
        
        Returns:
            Progress velocity in percentage per second
        """
        with self.progress_lock:
            if len(self.progress_history) < 2:
                return 0.0
            
            # Calculate velocity based on recent progress points
            recent_updates = list(self.progress_history)[-10:]  # Last 10 updates
            
            if len(recent_updates) < 2:
                return 0.0
            
            time_span = recent_updates[-1].timestamp - recent_updates[0].timestamp
            if time_span <= 0:
                return 0.0
            
            progress_span = self._calculate_progress_from_update(recent_updates[-1]) - \
                          self._calculate_progress_from_update(recent_updates[0])
            
            return progress_span / time_span
    
    def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive progress status information.
        
        Returns:
            Dictionary with detailed progress information
        """
        with self.progress_lock:
            current_time = time.time()
            elapsed_time = current_time - self.workflow_start_time if self.workflow_start_time else 0
            
            return {
                "overall_progress": self.get_overall_progress(smooth=True),
                "target_progress": self.target_overall_progress,
                "stage_progress": dict(self.stage_progress),
                "estimated_completion_time": self.get_estimated_completion_time(),
                "progress_velocity": self.get_progress_velocity(),
                "elapsed_time": elapsed_time,
                "total_updates": self.update_count,
                "active_stages": [stage for stage, progress in self.stage_progress.items() 
                                if 0 < progress < 100],
                "completed_stages": [stage for stage, progress in self.stage_progress.items() 
                                   if progress >= 100],
                "stage_durations": dict(self.stage_durations),
                "stage_estimates": dict(self.stage_estimated_durations),
                "performance_metrics": self._get_performance_metrics()
            }
    
    def _calculate_estimated_durations(self) -> None:
        """Calculate estimated durations for all stages based on context."""
        for stage, config in self.stage_weights.items():
            base_duration = config.base_duration
            
            # Apply multipliers based on context
            if self.video_duration and config.duration_multiplier > 0:
                duration = base_duration + (self.video_duration * config.duration_multiplier)
            else:
                duration = base_duration
            
            # Apply model-specific adjustments for transcription
            if stage == ProcessingStage.AUDIO_TRANSCRIPTION:
                model_multipliers = {
                    "tiny": 0.6,
                    "base": 0.8,
                    "small": 1.0,
                    "medium": 1.5,
                    "large": 2.0
                }
                duration *= model_multipliers.get(self.whisper_model, 1.0)
            
            # Apply constraints
            duration = max(config.min_duration, min(config.max_duration, duration))
            
            self.stage_estimated_durations[stage] = duration
    
    def _calculate_target_progress(self) -> None:
        """Calculate target overall progress based on stage weights and progress."""
        total_weight = sum(config.weight for config in self.stage_weights.values())
        weighted_progress = 0.0
        
        for stage, config in self.stage_weights.items():
            stage_progress = self.stage_progress.get(stage, 0.0)
            weighted_progress += (stage_progress * config.weight) / total_weight
        
        self.target_overall_progress = min(100.0, weighted_progress)
    
    def _update_smoothed_progress(self) -> None:
        """Update smoothed progress using interpolation."""
        # Simple exponential smoothing
        progress_diff = self.target_overall_progress - self.current_overall_progress
        
        # Apply smoothing factor
        adjustment = progress_diff * self.smoothing_factor
        
        # Ensure we don't overshoot and progress doesn't go backwards
        if progress_diff > 0:
            self.current_overall_progress += adjustment
        else:
            # Allow immediate backwards movement for corrections
            self.current_overall_progress = self.target_overall_progress
        
        # Ensure bounds
        self.current_overall_progress = max(0.0, min(100.0, self.current_overall_progress))
    
    def _update_timing_estimates(self, stage: ProcessingStage, percentage: float) -> None:
        """Update timing estimates based on actual progress."""
        if stage not in self.stage_start_times or percentage <= 0:
            return
        
        elapsed_time = time.time() - self.stage_start_times[stage]
        
        if percentage >= 10.0:  # Need reasonable progress for estimate
            estimated_total_time = elapsed_time * (100.0 / percentage)
            
            # Update estimate with smoothing
            current_estimate = self.stage_estimated_durations.get(stage, estimated_total_time)
            smoothed_estimate = (current_estimate * 0.7) + (estimated_total_time * 0.3)
            
            # Apply constraints
            config = self.stage_weights[stage]
            smoothed_estimate = max(config.min_duration, 
                                  min(config.max_duration, smoothed_estimate))
            
            self.stage_estimated_durations[stage] = smoothed_estimate
    
    def _estimate_remaining_time_by_stages(self) -> float:
        """Estimate remaining time based on individual stage estimates."""
        remaining_time = 0.0
        
        for stage, config in self.stage_weights.items():
            stage_progress = self.stage_progress.get(stage, 0.0)
            estimated_duration = self.stage_estimated_durations.get(stage, config.base_duration)
            
            if stage_progress >= 100.0:
                continue  # Stage completed
            elif stage_progress > 0:
                # Stage in progress
                elapsed = time.time() - self.stage_start_times.get(stage, time.time())
                remaining_stage_time = max(0, estimated_duration - elapsed)
                remaining_time += remaining_stage_time
            else:
                # Stage not started
                remaining_time += estimated_duration
        
        return remaining_time
    
    def _calculate_progress_from_update(self, update: ProgressUpdate) -> float:
        """Calculate overall progress at the time of a specific update."""
        # This is a simplified calculation - in practice, you'd need to reconstruct
        # the state at that point in time
        stage_weight = self.stage_weights[update.stage].weight
        total_weight = sum(config.weight for config in self.stage_weights.values())
        
        return (update.percentage * stage_weight) / total_weight
    
    def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the progress tracking system."""
        current_time = time.time()
        elapsed_time = current_time - self.workflow_start_time if self.workflow_start_time else 0
        
        return {
            "updates_per_second": self.update_count / elapsed_time if elapsed_time > 0 else 0,
            "average_update_frequency": elapsed_time / self.update_count if self.update_count > 0 else 0,
            "smoothing_factor": self.smoothing_factor,
            "progress_variance": self._calculate_progress_variance(),
            "timing_accuracy": self._calculate_timing_accuracy()
        }
    
    def _calculate_progress_variance(self) -> float:
        """Calculate variance in progress updates."""
        if len(self.progress_history) < 2:
            return 0.0
        
        progress_values = [self._calculate_progress_from_update(update) 
                         for update in self.progress_history]
        
        if not progress_values:
            return 0.0
        
        mean_progress = sum(progress_values) / len(progress_values)
        variance = sum((p - mean_progress) ** 2 for p in progress_values) / len(progress_values)
        
        return variance
    
    def _calculate_timing_accuracy(self) -> float:
        """Calculate accuracy of timing estimates."""
        accurate_estimates = 0
        total_estimates = 0
        
        for stage in self.stage_durations:
            if stage in self.stage_estimated_durations:
                actual_duration = self.stage_durations[stage]
                estimated_duration = self.stage_estimated_durations[stage]
                
                # Consider estimate accurate if within 25% of actual
                if estimated_duration > 0:
                    accuracy_ratio = min(actual_duration, estimated_duration) / max(actual_duration, estimated_duration)
                    if accuracy_ratio >= 0.75:
                        accurate_estimates += 1
                    total_estimates += 1
        
        return accurate_estimates / total_estimates if total_estimates > 0 else 0.0
    
    def _notify_callbacks(self, update: ProgressUpdate) -> None:
        """Notify all registered callbacks about progress update."""
        # Update smoothed progress
        self._update_smoothed_progress()
        
        # Notify progress callbacks
        self._notify_progress_callbacks(self.current_overall_progress, update.message)
        
        # Notify stage callbacks
        for callback in self.stage_callbacks:
            try:
                callback(update.stage, update.percentage, update.message)
            except Exception as e:
                self.logger.warning(f"Stage callback failed: {e}")
    
    def _notify_progress_callbacks(self, percentage: float, message: str) -> None:
        """Notify progress callbacks."""
        for callback in self.progress_callbacks:
            try:
                callback(percentage, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")
    
    def add_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Add overall progress callback.
        
        Args:
            callback: Function that takes (percentage, message)
        """
        self.progress_callbacks.append(callback)
    
    def add_stage_callback(self, callback: Callable[[ProcessingStage, float, str], None]) -> None:
        """Add stage-specific progress callback.
        
        Args:
            callback: Function that takes (stage, percentage, message)
        """
        self.stage_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add workflow completion callback.
        
        Args:
            callback: Function that takes completion report
        """
        self.completion_callbacks.append(callback)
    
    def complete_workflow(self, success: bool = True) -> Dict[str, Any]:
        """Mark workflow as completed and generate final report.
        
        Args:
            success: Whether workflow completed successfully
            
        Returns:
            Completion report with performance metrics
        """
        with self.progress_lock:
            current_time = time.time()
            
            if self.workflow_start_time:
                self.total_processing_time = current_time - self.workflow_start_time
            
            # Generate completion report
            completion_report = {
                "success": success,
                "total_processing_time": self.total_processing_time,
                "final_progress": self.target_overall_progress,
                "stage_durations": dict(self.stage_durations),
                "stage_estimates": dict(self.stage_estimated_durations),
                "timing_accuracy": self._calculate_timing_accuracy(),
                "total_updates": self.update_count,
                "average_velocity": self.get_progress_velocity(),
                "context": self.current_context.copy(),
                "performance_metrics": self._get_performance_metrics()
            }
            
            self.logger.info(f"Workflow completed: success={success}, "
                           f"time={self.total_processing_time:.1f}s")
            
            # Notify completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(completion_report)
                except Exception as e:
                    self.logger.warning(f"Completion callback failed: {e}")
            
            return completion_report
    
    def reset(self) -> None:
        """Reset progress aggregator for new workflow."""
        with self.progress_lock:
            self.stage_progress.clear()
            self.stage_start_times.clear()
            self.stage_durations.clear()
            self.stage_estimated_durations.clear()
            
            self.current_overall_progress = 0.0
            self.target_overall_progress = 0.0
            self.progress_history.clear()
            
            self.update_count = 0
            self.total_processing_time = 0.0
            self.workflow_start_time = None
            self.last_update_time = 0.0
            
            self.current_context.clear()
            self.video_duration = None
            self.whisper_model = "tiny"
            
            self.logger.info("Progress aggregator reset")


# Utility functions for progress management
def create_progress_aggregator(smoothing_factor: float = 0.1, 
                             update_frequency: float = 0.5) -> ProgressAggregator:
    """Create a new progress aggregator instance.
    
    Args:
        smoothing_factor: Progress smoothing factor (0-1)
        update_frequency: Minimum update frequency in seconds
        
    Returns:
        Configured ProgressAggregator instance
    """
    return ProgressAggregator(smoothing_factor=smoothing_factor, 
                            update_frequency=update_frequency)


def estimate_workflow_duration(video_duration: float, whisper_model: str = "tiny") -> float:
    """Estimate total workflow duration for given parameters.
    
    Args:
        video_duration: Video length in seconds
        whisper_model: Whisper model to use
        
    Returns:
        Estimated total duration in seconds
    """
    aggregator = ProgressAggregator()
    
    # Set context for estimation
    context = {
        "video_duration": video_duration,
        "whisper_model": whisper_model
    }
    
    aggregator.start_workflow(context)
    
    # Sum all estimated stage durations
    total_duration = sum(aggregator.stage_estimated_durations.values())
    
    return total_duration


def get_progress_weights() -> Dict[ProcessingStage, float]:
    """Get default progress weights for all stages.
    
    Returns:
        Dictionary mapping stages to their progress weights
    """
    return {stage: config.weight for stage, config in 
            ProgressAggregator.DEFAULT_STAGE_WEIGHTS.items()}
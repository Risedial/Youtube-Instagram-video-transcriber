"""Whisper transcription functionality for YouTube Whisper Transcriber.

This module handles audio transcription using OpenAI Whisper models with
automatic model downloading, caching, progress tracking, and memory optimization.
"""

from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import logging
import time
import threading
import tempfile
from dataclasses import dataclass
import json

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

# Import proxy manager
try:
    from utils.proxy_manager import ProxyManager
    PROXY_MANAGER_AVAILABLE = True
except ImportError:
    PROXY_MANAGER_AVAILABLE = False


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str
    confidence: float
    segments: List[Dict[str, Any]]
    processing_time: float
    word_count: int
    audio_duration: float


class TranscriptionError(Exception):
    """Custom exception for transcription-related errors."""
    pass


class WhisperTranscriber:
    """Whisper-based audio transcription with model management.
    
    Features:
    - Automatic Whisper model downloading and caching
    - Audio file transcription with multiple model options (tiny through large)
    - Thread-safe progress tracking for transcription processing
    - Memory optimization for different model sizes
    - Text output formatting and file writing
    - GPU/CPU device selection with automatic fallback
    - Comprehensive error handling and recovery
    """
    
    def __init__(self, model_name: str = "tiny", device: str = "auto", proxy_manager: Optional['ProxyManager'] = None) -> None:
        """Initialize Whisper transcriber.
        
        Args:
            model_name: Whisper model to use (tiny, base, small, medium, large)
            device: Device to run on ('auto', 'cpu', or 'cuda')
            proxy_manager: ProxyManager instance for model download proxy support
            
        Raises:
            ImportError: If required dependencies are not available
        """
        if whisper is None:
            raise ImportError("whisper is required but not installed. Install with: pip install openai-whisper")
        if torch is None:
            raise ImportError("torch is required but not installed. Install with: pip install torch")
            
        self.model_name = model_name
        self.device = self._select_device(device)
        self.model = None
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self.status_callback: Optional[Callable[[str, str], None]] = None
        
        # Processing state
        self.cancel_event = threading.Event()
        self.transcription_lock = threading.Lock()
        self.current_transcription: Optional[str] = None
        
        # Model information
        self.model_info = {
            "tiny": {"size": "39MB", "memory": "1GB", "speed": "fastest", "accuracy": "good"},
            "base": {"size": "74MB", "memory": "1GB", "speed": "fast", "accuracy": "better"},
            "small": {"size": "244MB", "memory": "2GB", "speed": "medium", "accuracy": "good"},
            "medium": {"size": "769MB", "memory": "5GB", "speed": "slow", "accuracy": "very good"},
            "large": {"size": "1550MB", "memory": "10GB", "speed": "slowest", "accuracy": "best"}
        }
        
        self.logger.info(f"WhisperTranscriber initialized: model={model_name}, device={self.device}")
        
        if self.proxy_manager:
            self.logger.info("Proxy support enabled for Whisper model downloads")
        
    def _setup_model_cache(self) -> None:
        """Setup model cache directory for PyInstaller environment."""
        import sys
        import os
        
        try:
            # Check if running in PyInstaller bundle
            if hasattr(sys, '_MEIPASS'):
                # Running in PyInstaller bundle - use application directory for cache
                app_dir = Path(sys.executable).parent
                cache_dir = app_dir / "whisper_models"
            else:
                # Development environment - use default cache
                cache_dir = Path.home() / ".cache" / "whisper"
            
            # Create cache directory if it doesn't exist
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Set environment variable for Whisper to use our cache
            os.environ['WHISPER_CACHE'] = str(cache_dir)
            
            self.logger.info(f"DEBUG: Whisper cache directory: {cache_dir}")
            self.logger.info(f"DEBUG: Cache directory exists: {cache_dir.exists()}")
            self.logger.info(f"DEBUG: Cache directory writable: {os.access(cache_dir, os.W_OK)}")
            
        except Exception as e:
            self.logger.error(f"DEBUG: Failed to setup model cache: {e}")
            # Continue without custom cache - let Whisper use default
    
    def _configure_proxy_for_model_download(self):
        """Configure proxy settings for Whisper model downloads.
        
        This method monkey-patches urllib to use proxy settings during model downloads.
        """
        if not self.proxy_manager:
            return
            
        try:
            import urllib.request
            import urllib.error
            
            proxy = self.proxy_manager.get_healthy_proxy()
            if not proxy:
                return
                
            # Create proxy handler
            if proxy.proxy_type.value in ['http', 'https']:
                proxy_handler = urllib.request.ProxyHandler({
                    'http': proxy.url,
                    'https': proxy.url
                })
            else:
                # For SOCKS proxies, we need additional setup
                self.logger.warning("SOCKS proxy detected for Whisper downloads - may require additional configuration")
                return
            
            # Build opener with proxy
            opener = urllib.request.build_opener(proxy_handler)
            
            # Add headers for anti-detection
            headers = self.proxy_manager.get_randomized_headers("general")
            for key, value in headers.items():
                opener.addheaders.append((key, value))
            
            # Install the opener globally (Whisper will use this)
            urllib.request.install_opener(opener)
            
            self.logger.debug(f"Configured proxy for Whisper model downloads: {proxy.host}:{proxy.port}")
            
        except Exception as e:
            self.logger.warning(f"Failed to configure proxy for model downloads: {e}")
    
    def _restore_default_urllib(self):
        """Restore default urllib configuration after model download."""
        try:
            import urllib.request
            # Install default opener to restore normal behavior
            urllib.request.install_opener(urllib.request.build_opener())
        except Exception as e:
            self.logger.warning(f"Failed to restore default urllib configuration: {e}")
    
    def _is_model_download_error(self, error: Exception) -> bool:
        """Check if error is related to model download network issues.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is network/proxy-related
        """
        error_str = str(error).lower()
        download_error_indicators = [
            'connection', 'timeout', 'network', 'proxy', 'ssl', 'certificate',
            'unreachable', 'refused', 'blocked', 'download', 'url', 'http'
        ]
        
        return any(indicator in error_str for indicator in download_error_indicators)
        
    def _select_device(self, device: str) -> str:
        """Select the best available device for processing.
        
        Args:
            device: Requested device ('auto', 'cpu', or 'cuda')
            
        Returns:
            Selected device string
        """
        if device == "auto":
            if torch and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        elif device == "cuda":
            if torch and torch.cuda.is_available():
                return "cuda"
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        else:
            return "cpu"
        
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
        
    def load_model(self) -> bool:
        """Load Whisper model into memory.
        
        Returns:
            True if model loaded successfully
        """
        try:
            if self.model is not None:
                return True  # Already loaded
                
            # DEBUG: Log environment information
            import sys
            import os
            self.logger.info(f"DEBUG: Python executable: {sys.executable}")
            self.logger.info(f"DEBUG: Working directory: {os.getcwd()}")
            self.logger.info(f"DEBUG: PyInstaller bundle: {hasattr(sys, '_MEIPASS')}")
            if hasattr(sys, '_MEIPASS'):
                self.logger.info(f"DEBUG: Bundle directory: {sys._MEIPASS}")
            
            # Set up model cache directory for PyInstaller
            self._setup_model_cache()
                
            if self.progress_callback:
                self.progress_callback(0.0, f"Loading {self.model_name} model...")
                
            if self.status_callback:
                model_info = self.model_info.get(self.model_name, {})
                self.status_callback(
                    f"Loading Whisper model: {self.model_name}\n"
                    f"Model size: {model_info.get('size', 'Unknown')}\n"
                    f"Memory required: {model_info.get('memory', 'Unknown')}\n"
                    f"This may take a few minutes for first-time download...",
                    "processing"
                )
                
            self.logger.info(f"Loading Whisper model: {self.model_name} on device: {self.device}")
            
            # Load model with proper device configuration
            if self.progress_callback:
                self.progress_callback(25.0, "Downloading/loading model...")
                
            # Configure proxy for model download if needed
            if self.proxy_manager:
                self._configure_proxy_for_model_download()
            
            try:
                # Load model (will download if not cached)
                self.logger.info(f"DEBUG: About to load Whisper model {self.model_name}")
                self.model = whisper.load_model(self.model_name, device=self.device)
                self.logger.info(f"DEBUG: Whisper model loaded successfully")
                
                # Mark proxy as successful if used
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy:
                        self.proxy_manager.mark_proxy_success(proxy, 1.0)  # Dummy response time
                        
            except Exception as model_error:
                # Handle proxy errors during model download
                if self.proxy_manager:
                    proxy = self.proxy_manager.get_current_proxy()
                    if proxy and self._is_model_download_error(model_error):
                        self.proxy_manager.mark_proxy_failed(proxy, model_error)
                raise model_error
            finally:
                # Always restore default urllib configuration
                if self.proxy_manager:
                    self._restore_default_urllib()
            
            if self.progress_callback:
                self.progress_callback(75.0, "Optimizing model for device...")
                
            # Warm up the model with a small test
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear CUDA cache
                
            if self.progress_callback:
                self.progress_callback(100.0, f"Model {self.model_name} loaded successfully")
                
            if self.status_callback:
                self.status_callback(
                    f"Model {self.model_name} loaded successfully!\n"
                    f"Device: {self.device}\n"
                    f"Ready for transcription",
                    "success"
                )
                
            self.logger.info(f"Whisper model {self.model_name} loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            error_msg = f"Error loading model {self.model_name}: {str(e)}"
            self.logger.error(error_msg)
            
            if self.status_callback:
                self.status_callback(
                    f"Failed to load model {self.model_name}\n"
                    f"Error: {str(e)}\n"
                    f"Try using a smaller model or check your internet connection",
                    "error"
                )
                
            return False
            
    def transcribe_audio(self, audio_path: Path, output_path: Optional[Path] = None) -> Optional[TranscriptionResult]:
        """Transcribe audio file to text.
        
        Args:
            audio_path: Path to audio file
            output_path: Path to save transcription, if None returns result only
            
        Returns:
            TranscriptionResult object or None if failed
        """
        with self.transcription_lock:
            if not audio_path.exists():
                error_msg = f"Audio file not found: {audio_path}"
                self.logger.error(error_msg)
                if self.status_callback:
                    self.status_callback(f"Error: {error_msg}", "error")
                return None
                
            self.current_transcription = str(audio_path)
            self.cancel_event.clear()
            
            try:
                # Load model if not already loaded
                if not self.model:
                    if not self.load_model():
                        return None
                        
                self.logger.info(f"Starting transcription of: {audio_path}")
                start_time = time.time()
                
                if self.progress_callback:
                    self.progress_callback(0.0, "Preparing audio for transcription...")
                    
                if self.status_callback:
                    file_size = audio_path.stat().st_size / (1024 * 1024)  # MB
                    self.status_callback(
                        f"Starting transcription...\n"
                        f"Audio file: {audio_path.name}\n"
                        f"File size: {file_size:.1f} MB\n"
                        f"Model: {self.model_name} ({self.device})",
                        "processing"
                    )
                    
                # Check for cancellation
                if self.cancel_event.is_set():
                    return None
                    
                if self.progress_callback:
                    self.progress_callback(10.0, "Loading audio...")
                    
                # Transcribe with Whisper
                self.logger.info(f"Transcribing with model {self.model_name} on {self.device}")
                
                if self.progress_callback:
                    self.progress_callback(20.0, "Processing audio...")
                    
                # Perform transcription
                whisper_result = self.model.transcribe(
                    str(audio_path),
                    verbose=False,
                    language=None,  # Auto-detect language
                    task="transcribe",
                    fp16=(self.device == "cuda")  # Use FP16 for GPU
                )
                
                if self.progress_callback:
                    self.progress_callback(80.0, "Processing transcription results...")
                    
                # Check for cancellation
                if self.cancel_event.is_set():
                    return None
                    
                # Process results
                processing_time = time.time() - start_time
                text = whisper_result.get("text", "").strip()
                language = whisper_result.get("language", "unknown")
                segments = whisper_result.get("segments", [])
                
                # Calculate confidence (average from segments)
                confidence = 0.0
                if segments:
                    confidences = [seg.get("confidence", 0.0) for seg in segments if "confidence" in seg]
                    confidence = sum(confidences) / len(confidences) if confidences else 0.0
                else:
                    confidence = 0.95  # Default confidence when no segments
                    
                # Calculate word count and audio duration
                word_count = len(text.split()) if text else 0
                audio_duration = whisper_result.get("duration", 0.0)
                
                result = TranscriptionResult(
                    text=text,
                    language=language,
                    confidence=confidence,
                    segments=segments,
                    processing_time=processing_time,
                    word_count=word_count,
                    audio_duration=audio_duration
                )
                
                if self.progress_callback:
                    self.progress_callback(90.0, "Saving transcription...")
                    
                # Save to file if output path provided
                if output_path:
                    if not self.save_transcription(result, output_path):
                        return None
                        
                if self.progress_callback:
                    self.progress_callback(100.0, "Transcription completed successfully")
                    
                if self.status_callback:
                    processing_ratio = processing_time / audio_duration if audio_duration > 0 else 0
                    self.status_callback(
                        f"Transcription completed successfully!\n"
                        f"Language detected: {language}\n"
                        f"Word count: {word_count:,}\n"
                        f"Processing time: {processing_time:.1f}s\n"
                        f"Audio duration: {audio_duration:.1f}s\n"
                        f"Processing ratio: {processing_ratio:.2f}x",
                        "success"
                    )
                    
                self.logger.info(
                    f"Transcription completed: {word_count} words, "
                    f"{processing_time:.1f}s processing time, "
                    f"language: {language}"
                )
                
                return result
                
            except Exception as e:
                error_msg = f"Error during transcription: {str(e)}"
                self.logger.error(error_msg)
                
                if self.status_callback:
                    self.status_callback(
                        f"Transcription failed\n"
                        f"Error: {str(e)}\n"
                        f"Try using a smaller model or check the audio file",
                        "error"
                    )
                    
                return None
                
            finally:
                self.current_transcription = None
                
                # Clear CUDA cache if using GPU
                if self.device == "cuda" and torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
    def cancel_transcription(self) -> None:
        """Cancel current transcription operation."""
        if self.current_transcription:
            self.logger.info(f"Cancelling transcription: {self.current_transcription}")
            self.cancel_event.set()

            if self.status_callback:
                self.status_callback("Transcription cancelled by user", "warning")

            self.current_transcription = None

    def format_transcription_with_timestamps(self, result: TranscriptionResult) -> str:
        """Format transcription result with timestamps.

        Args:
            result: TranscriptionResult to format

        Returns:
            Formatted transcription with timestamps
        """
        lines = []

        # Add metadata header
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# Transcription Generated by YouTube/Instagram Video Transcriber")
        lines.append(f"# Generated on: {timestamp}")
        lines.append(f"# Language: {result.language}")
        lines.append(f"# Model: {self.model_name}")
        lines.append(f"# Processing time: {result.processing_time:.1f} seconds")
        lines.append(f"# Audio duration: {result.audio_duration:.1f} seconds")
        lines.append(f"# Word count: {result.word_count:,}")
        if result.confidence > 0:
            lines.append(f"# Confidence: {result.confidence:.2%}")
        lines.append(f"\n{'-'*50}\n")

        # Add plain text version first
        lines.append("# Full Transcription\n")
        lines.append(result.text)

        # Add timestamped segments
        if result.segments:
            lines.append(f"\n\n{'-'*50}\n")
            lines.append("# Transcript with Timestamps\n")

            for segment in result.segments:
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()

                # Format timestamp as MM:SS
                start_mins, start_secs = divmod(int(start_time), 60)
                end_mins, end_secs = divmod(int(end_time), 60)

                lines.append(f"[{start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}] {text}")

        return '\n'.join(lines)

    def format_transcription_without_timestamps(self, result: TranscriptionResult) -> str:
        """Format transcription result without timestamps (plain text only).

        Args:
            result: TranscriptionResult to format

        Returns:
            Formatted transcription without timestamps
        """
        lines = []

        # Add minimal metadata header
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"# Transcription Generated by YouTube/Instagram Video Transcriber")
        lines.append(f"# Generated on: {timestamp}")
        lines.append(f"# Language: {result.language}")
        lines.append(f"# Model: {self.model_name}")
        lines.append(f"# Word count: {result.word_count:,}")
        lines.append(f"\n{'-'*50}\n")

        # Add plain text only
        lines.append(result.text)

        return '\n'.join(lines)

    def save_transcription_outputs(
        self,
        result: TranscriptionResult,
        base_output_path: Path,
        include_timestamps: bool = True,
        exclude_timestamps: bool = False
    ) -> Dict[str, Path]:
        """Save transcription result in multiple formats based on preferences.

        Args:
            result: TranscriptionResult to save
            base_output_path: Base path for output files (without format suffix)
            include_timestamps: Whether to save version with timestamps
            exclude_timestamps: Whether to save version without timestamps

        Returns:
            Dictionary mapping format type to saved file path
        """
        saved_files = {}

        # If both are False, default to timestamps version
        if not include_timestamps and not exclude_timestamps:
            include_timestamps = True

        # Ensure parent directory exists
        base_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate base filename (without extension)
        base_name = base_output_path.stem
        parent_dir = base_output_path.parent

        try:
            # Save version with timestamps
            if include_timestamps:
                timestamped_content = self.format_transcription_with_timestamps(result)
                timestamped_path = parent_dir / f"{base_name}_with_timestamps.txt"

                # Ensure unique filename
                counter = 1
                while timestamped_path.exists():
                    timestamped_path = parent_dir / f"{base_name}_with_timestamps_{counter}.txt"
                    counter += 1

                with open(timestamped_path, "w", encoding="utf-8") as f:
                    f.write(timestamped_content)

                saved_files["with_timestamps"] = timestamped_path
                self.logger.info(f"Saved transcription with timestamps: {timestamped_path}")

            # Save version without timestamps
            if exclude_timestamps:
                plain_content = self.format_transcription_without_timestamps(result)
                plain_path = parent_dir / f"{base_name}_without_timestamps.txt"

                # Ensure unique filename
                counter = 1
                while plain_path.exists():
                    plain_path = parent_dir / f"{base_name}_without_timestamps_{counter}.txt"
                    counter += 1

                with open(plain_path, "w", encoding="utf-8") as f:
                    f.write(plain_content)

                saved_files["without_timestamps"] = plain_path
                self.logger.info(f"Saved transcription without timestamps: {plain_path}")

            return saved_files

        except Exception as e:
            error_msg = f"Error saving transcription outputs: {str(e)}"
            self.logger.error(error_msg)

            if self.status_callback:
                self.status_callback(f"Failed to save transcription: {str(e)}", "error")

            return {}

    def save_transcription(self, result: TranscriptionResult, output_path: Path, include_metadata: bool = True) -> bool:
        """Save transcription result to file.
        
        Args:
            result: TranscriptionResult to save
            output_path: Path to save file
            include_metadata: Whether to include metadata in the output
            
        Returns:
            True if saved successfully
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp for file
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            with open(output_path, "w", encoding="utf-8") as f:
                if include_metadata:
                    # Write metadata header
                    f.write(f"# Transcription Generated by YouTube/Instagram Video Transcriber\n")
                    f.write(f"# Generated on: {timestamp}\n")
                    f.write(f"# Language: {result.language}\n")
                    f.write(f"# Model: {self.model_name}\n")
                    f.write(f"# Processing time: {result.processing_time:.1f} seconds\n")
                    f.write(f"# Audio duration: {result.audio_duration:.1f} seconds\n")
                    f.write(f"# Word count: {result.word_count:,}\n")
                    if result.confidence > 0:
                        f.write(f"# Confidence: {result.confidence:.2%}\n")
                    f.write(f"\n{'-'*50}\n\n")
                
                # Write the transcription text
                f.write(result.text)
                
                # Optionally write segments with timestamps
                if include_metadata and result.segments:
                    f.write(f"\n\n{'-'*50}\n")
                    f.write("# Transcript with Timestamps\n\n")
                    
                    for segment in result.segments:
                        start_time = segment.get('start', 0)
                        end_time = segment.get('end', 0)
                        text = segment.get('text', '').strip()
                        
                        # Format timestamp as MM:SS
                        start_mins, start_secs = divmod(int(start_time), 60)
                        end_mins, end_secs = divmod(int(end_time), 60)
                        
                        f.write(f"[{start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}] {text}\n")
                        
            self.logger.info(f"Transcription saved to: {output_path}")
            return True
            
        except Exception as e:
            error_msg = f"Error saving transcription: {str(e)}"
            self.logger.error(error_msg)
            
            if self.status_callback:
                self.status_callback(f"Failed to save transcription: {str(e)}", "error")
                
            return False
            
    def get_model_info(self) -> Dict[str, str]:
        """Get information about current model.
        
        Returns:
            Dictionary with model information
        """
        return self.model_info.get(self.model_name, {})
        
    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models.
        
        Returns:
            List of model names
        """
        return list(self.model_info.keys())
        
    def change_model(self, model_name: str) -> bool:
        """Change to different Whisper model.
        
        Args:
            model_name: New model to use
            
        Returns:
            True if model changed successfully
        """
        if model_name not in self.model_info:
            self.logger.error(f"Invalid model name: {model_name}")
            return False
            
        old_model = self.model_name
        self.model_name = model_name
        self.model = None  # Force reload on next transcription
        
        self.logger.info(f"Changed model from {old_model} to {model_name}")
        
        if self.status_callback:
            model_info = self.model_info.get(model_name, {})
            self.status_callback(
                f"Model changed to {model_name}\n"
                f"Size: {model_info.get('size', 'Unknown')}\n"
                f"Memory: {model_info.get('memory', 'Unknown')}\n"
                f"Speed: {model_info.get('speed', 'Unknown')}",
                "info"
            )
            
        return True
        
    def estimate_processing_time(self, audio_duration_seconds: float) -> float:
        """Estimate processing time for audio.
        
        Args:
            audio_duration_seconds: Duration of audio in seconds
            
        Returns:
            Estimated processing time in seconds
        """
        # Processing ratios (approximate, varies by hardware)
        ratios = {
            "tiny": 0.15,    # Very fast
            "base": 0.25,    # Fast  
            "small": 0.4,    # Medium
            "medium": 0.7,   # Slow
            "large": 1.0     # Slowest, but best quality
        }
        
        ratio = ratios.get(self.model_name, 0.5)
        base_time = audio_duration_seconds * ratio
        
        # Adjust for device
        if self.device == "cuda":
            base_time *= 0.3  # GPU is much faster
            
        return max(base_time, 5.0)  # Minimum 5 seconds for overhead
        
    def is_transcribing(self) -> bool:
        """Check if transcription is currently in progress.
        
        Returns:
            True if transcription is active
        """
        return self.current_transcription is not None
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information.
        
        Returns:
            Dictionary with memory usage info
        """
        memory_info = {"cpu_memory": 0.0, "gpu_memory": 0.0}
        
        try:
            import psutil
            process = psutil.Process()
            memory_info["cpu_memory"] = process.memory_info().rss / (1024 * 1024)  # MB
        except ImportError:
            pass
            
        if self.device == "cuda" and torch and torch.cuda.is_available():
            try:
                memory_info["gpu_memory"] = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            except Exception:
                pass
                
        return memory_info
        
    def cleanup_resources(self) -> None:
        """Clean up model resources and free memory."""
        try:
            self.model = None
            
            if self.device == "cuda" and torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Resources cleaned up successfully")
            
            if self.status_callback:
                self.status_callback("Model resources cleaned up", "info")
                
        except Exception as e:
            self.logger.warning(f"Error cleaning up resources: {e}")


# Utility functions for transcription
def transcribe_audio_file(audio_path: Path, model_name: str = "tiny", output_path: Optional[Path] = None, 
                         progress_callback: Optional[Callable] = None, status_callback: Optional[Callable] = None) -> Optional[TranscriptionResult]:
    """Transcribe an audio file using Whisper.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model to use
        output_path: Path to save transcription
        progress_callback: Progress update function (percentage, message)
        status_callback: Status update function (message, type)
        
    Returns:
        TranscriptionResult or None if failed
    """
    try:
        transcriber = WhisperTranscriber(model_name)
        if progress_callback:
            transcriber.set_progress_callback(progress_callback)
        if status_callback:
            transcriber.set_status_callback(status_callback)
        return transcriber.transcribe_audio(audio_path, output_path)
    except Exception as e:
        if status_callback:
            status_callback(f"Transcription failed: {str(e)}", "error")
        return None


def get_model_information() -> Dict[str, Dict[str, str]]:
    """Get information about all available Whisper models.
    
    Returns:
        Dictionary mapping model names to their information
    """
    try:
        transcriber = WhisperTranscriber()
        return transcriber.model_info
    except Exception:
        return {}


def estimate_transcription_time(audio_duration: float, model_name: str = "tiny") -> float:
    """Estimate processing time for transcription.
    
    Args:
        audio_duration: Duration of audio in seconds
        model_name: Whisper model to use
        
    Returns:
        Estimated processing time in seconds
    """
    try:
        transcriber = WhisperTranscriber(model_name)
        return transcriber.estimate_processing_time(audio_duration)
    except Exception:
        return audio_duration * 0.5  # Default estimate


def validate_whisper_dependencies() -> Dict[str, bool]:
    """Validate that Whisper dependencies are available.
    
    Returns:
        Dictionary with availability status of dependencies
    """
    dependencies = {
        "whisper": whisper is not None,
        "torch": torch is not None,
        "cuda": torch is not None and torch.cuda.is_available() if torch else False
    }
    
    return dependencies
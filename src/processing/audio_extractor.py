"""Audio extraction and optimization for Whisper transcription.

This module handles audio file processing, format conversion, and optimization
specifically for OpenAI Whisper speech recognition models.
"""

from typing import Optional, Callable, Dict, Any, Tuple
from pathlib import Path
import logging
import subprocess
import tempfile
import shutil
import os

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


class AudioExtractionError(Exception):
    """Custom exception for audio extraction errors."""
    pass


class AudioExtractor:
    """Audio extraction and optimization for Whisper processing.
    
    Features:
    - Audio format conversion optimized for Whisper models
    - Sample rate and bit depth optimization
    - Audio quality enhancement and noise reduction
    - File size optimization while preserving transcription accuracy
    - Progress tracking for long audio files
    - Multiple audio format support (MP4, MKV, AVI, etc.)
    """
    
    # Optimal audio settings for Whisper
    WHISPER_SAMPLE_RATE = 16000  # Hz - Whisper's native sample rate
    WHISPER_CHANNELS = 1         # Mono audio
    WHISPER_BIT_DEPTH = 16       # 16-bit depth
    WHISPER_FORMAT = 'wav'       # Preferred format for Whisper
    
    # Supported input audio/video formats
    SUPPORTED_FORMATS = [
        'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm',  # Video formats
        'mp3', 'wav', 'aac', 'm4a', 'ogg', 'flac', 'wma'   # Audio formats
    ]
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """Initialize audio extractor.
        
        Args:
            temp_dir: Temporary directory for processing, uses system temp if None
            
        Raises:
            AudioExtractionError: If required dependencies are missing
        """
        if ffmpeg is None:
            raise AudioExtractionError("ffmpeg-python is required but not installed. Install with: pip install ffmpeg-python")
            
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "youtube_whisper_audio"
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        
        # Verify FFmpeg installation
        self._verify_ffmpeg()
        
        self.logger.info(f"AudioExtractor initialized with temp_dir: {self.temp_dir}")
        
    def _verify_ffmpeg(self) -> None:
        """Verify FFmpeg is installed and accessible.
        
        Raises:
            AudioExtractionError: If FFmpeg is not found
        """
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise AudioExtractionError("FFmpeg is not working properly")
                
            self.logger.debug("FFmpeg verification successful")
            
        except FileNotFoundError:
            raise AudioExtractionError(
                "FFmpeg is not installed or not in PATH. "
                "Please install FFmpeg from https://ffmpeg.org/"
            )
        except subprocess.TimeoutExpired:
            raise AudioExtractionError("FFmpeg verification timed out")
            
    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        """Set progress callback function.
        
        Args:
            callback: Function that takes (percentage, status_message)
        """
        self.progress_callback = callback
        
    def get_audio_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed audio information from file.
        
        Args:
            file_path: Path to audio/video file
            
        Returns:
            Dictionary with audio information
            
        Raises:
            AudioExtractionError: If file analysis fails
        """
        try:
            probe = ffmpeg.probe(str(file_path))
            
            # Find audio stream
            audio_stream = None
            for stream in probe['streams']:
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
                    
            if not audio_stream:
                raise AudioExtractionError("No audio stream found in file")
                
            # Extract relevant information
            info = {
                'duration': float(probe.get('format', {}).get('duration', 0)),
                'format_name': probe.get('format', {}).get('format_name', 'unknown'),
                'codec': audio_stream.get('codec_name', 'unknown'),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'channels': int(audio_stream.get('channels', 0)),
                'bit_rate': int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else None,
                'file_size': int(probe.get('format', {}).get('size', 0)),
                'is_optimized': self._is_whisper_optimized(audio_stream)
            }
            
            return info
            
        except ffmpeg.Error as e:
            raise AudioExtractionError(f"Failed to analyze audio file: {str(e)}")
        except Exception as e:
            raise AudioExtractionError(f"Unexpected error analyzing file: {str(e)}")
            
    def _is_whisper_optimized(self, audio_stream: Dict[str, Any]) -> bool:
        """Check if audio stream is already optimized for Whisper.
        
        Args:
            audio_stream: Audio stream information
            
        Returns:
            True if already optimized
        """
        sample_rate = int(audio_stream.get('sample_rate', 0))
        channels = int(audio_stream.get('channels', 0))
        
        return (sample_rate == self.WHISPER_SAMPLE_RATE and 
                channels == self.WHISPER_CHANNELS)
                
    def extract_audio(self, input_path: Path, output_path: Optional[Path] = None,
                     optimize_for_whisper: bool = True) -> Path:
        """Extract and optimize audio for Whisper transcription.
        
        Args:
            input_path: Path to input audio/video file
            output_path: Path for output audio file, auto-generated if None
            optimize_for_whisper: Apply Whisper-specific optimizations
            
        Returns:
            Path to extracted audio file
            
        Raises:
            AudioExtractionError: If extraction fails
        """
        if not input_path.exists():
            raise AudioExtractionError(f"Input file does not exist: {input_path}")
            
        # Validate input format
        input_suffix = input_path.suffix.lower().lstrip('.')
        if input_suffix not in self.SUPPORTED_FORMATS:
            raise AudioExtractionError(f"Unsupported format: {input_suffix}")
            
        # Generate output path if not provided
        if output_path is None:
            output_name = f"{input_path.stem}_whisper_optimized.{self.WHISPER_FORMAT}"
            output_path = self.temp_dir / output_name
        else:
            output_path = Path(output_path)
            
        try:
            # Get input audio information
            if self.progress_callback:
                self.progress_callback(10.0, "Analyzing input audio...")
                
            audio_info = self.get_audio_info(input_path)
            self.logger.info(f"Input audio: {audio_info['duration']:.1f}s, "
                           f"{audio_info['sample_rate']}Hz, {audio_info['channels']} channels")
            
            # Check if optimization is needed
            if optimize_for_whisper and audio_info['is_optimized']:
                self.logger.info("Audio is already optimized for Whisper")
                if input_path.suffix.lower() == f'.{self.WHISPER_FORMAT}':
                    # Just copy the file
                    shutil.copy2(input_path, output_path)
                    if self.progress_callback:
                        self.progress_callback(100.0, "Audio copied (already optimized)")
                    return output_path
                    
            # Prepare FFmpeg stream
            if self.progress_callback:
                self.progress_callback(20.0, "Setting up audio processing...")
                
            input_stream = ffmpeg.input(str(input_path))
            
            # Configure output parameters
            output_params = {}
            
            if optimize_for_whisper:
                output_params.update({
                    'acodec': 'pcm_s16le',  # 16-bit PCM for WAV
                    'ar': self.WHISPER_SAMPLE_RATE,  # Sample rate
                    'ac': self.WHISPER_CHANNELS,     # Mono
                    'format': self.WHISPER_FORMAT    # WAV format
                })
                
                # Apply audio filters for better transcription
                audio_filters = []
                
                # Normalize audio levels
                audio_filters.append('loudnorm=I=-16:LRA=11:TP=-1.5')
                
                # High-pass filter to remove low-frequency noise
                audio_filters.append('highpass=f=80')
                
                # Low-pass filter to remove high-frequency noise
                audio_filters.append('lowpass=f=8000')
                
                if audio_filters:
                    input_stream = input_stream.filter('af', ','.join(audio_filters))
            else:
                # Basic extraction without optimization
                output_params.update({
                    'acodec': 'copy' if input_suffix in ['wav', 'flac'] else 'pcm_s16le',
                    'format': self.WHISPER_FORMAT
                })
                
            # Set up progress tracking
            if self.progress_callback and audio_info['duration'] > 0:
                def progress_handler(progress_info):
                    if 'out_time_ms' in progress_info:
                        progress_ms = progress_info['out_time_ms']
                        progress_s = progress_ms / 1000000  # Convert to seconds
                        percentage = min(95.0, 20.0 + (progress_s / audio_info['duration']) * 75.0)
                        self.progress_callback(percentage, f"Processing audio... {progress_s:.1f}s/{audio_info['duration']:.1f}s")
                        
                # Create output stream with progress
                output_stream = ffmpeg.output(input_stream, str(output_path), **output_params)
                ffmpeg.run(output_stream, overwrite_output=True, quiet=True, 
                          progress=progress_handler if self.progress_callback else None)
            else:
                # Create output stream without progress
                output_stream = ffmpeg.output(input_stream, str(output_path), **output_params)
                ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
                
            # Verify output file
            if not output_path.exists():
                raise AudioExtractionError("Output file was not created")
                
            output_size = output_path.stat().st_size
            if output_size < 1024:  # Less than 1KB
                raise AudioExtractionError("Output file is too small, extraction likely failed")
                
            # Get output audio information
            output_info = self.get_audio_info(output_path)
            
            self.logger.info(f"Audio extraction completed: {output_path}")
            self.logger.info(f"Output audio: {output_info['duration']:.1f}s, "
                           f"{output_info['sample_rate']}Hz, {output_info['channels']} channels, "
                           f"{output_size / (1024*1024):.1f}MB")
                           
            if self.progress_callback:
                self.progress_callback(100.0, f"Audio extraction completed ({output_size / (1024*1024):.1f}MB)")
                
            return output_path
            
        except ffmpeg.Error as e:
            error_msg = f"FFmpeg error during audio extraction: {str(e)}"
            if e.stderr:
                error_msg += f"\nFFmpeg stderr: {e.stderr.decode()}"
            self.logger.error(error_msg)
            raise AudioExtractionError(error_msg)
            
        except Exception as e:
            error_msg = f"Unexpected error during audio extraction: {str(e)}"
            self.logger.error(error_msg)
            raise AudioExtractionError(error_msg)
            
    def validate_audio_for_whisper(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate audio file for Whisper transcription.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, validation_info)
        """
        validation_info = {
            'file_path': str(file_path),
            'is_valid': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            if not file_path.exists():
                validation_info['issues'].append("File does not exist")
                return False, validation_info
                
            audio_info = self.get_audio_info(file_path)
            validation_info.update(audio_info)
            
            # Check duration
            if audio_info['duration'] <= 0:
                validation_info['issues'].append("Audio has no duration")
            elif audio_info['duration'] > 7200:  # 2 hours
                validation_info['issues'].append("Audio is longer than 2 hours (may cause processing issues)")
                
            # Check sample rate
            if audio_info['sample_rate'] < 8000:
                validation_info['issues'].append("Sample rate too low (< 8kHz)")
            elif audio_info['sample_rate'] != self.WHISPER_SAMPLE_RATE:
                validation_info['recommendations'].append(f"Resample to {self.WHISPER_SAMPLE_RATE}Hz for optimal performance")
                
            # Check channels
            if audio_info['channels'] > 2:
                validation_info['recommendations'].append("Convert to mono for optimal processing")
            elif audio_info['channels'] > 1:
                validation_info['recommendations'].append("Convert to mono for optimal processing")
                
            # Check file size
            if audio_info['file_size'] > 500 * 1024 * 1024:  # 500MB
                validation_info['recommendations'].append("File is very large, consider compression")
                
            # Overall validation
            validation_info['is_valid'] = len(validation_info['issues']) == 0
            
        except Exception as e:
            validation_info['issues'].append(f"Validation error: {str(e)}")
            
        return validation_info['is_valid'], validation_info
        
    def get_optimal_settings(self, input_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get optimal extraction settings based on input audio.
        
        Args:
            input_info: Input audio information
            
        Returns:
            Dictionary with optimal settings
        """
        settings = {
            'sample_rate': self.WHISPER_SAMPLE_RATE,
            'channels': self.WHISPER_CHANNELS,
            'format': self.WHISPER_FORMAT,
            'apply_filters': True,
            'estimated_output_size_mb': 0
        }
        
        # Estimate output file size
        duration = input_info.get('duration', 0)
        if duration > 0:
            # WAV file size calculation: sample_rate * channels * bit_depth/8 * duration
            bytes_per_second = self.WHISPER_SAMPLE_RATE * self.WHISPER_CHANNELS * (self.WHISPER_BIT_DEPTH // 8)
            estimated_size = bytes_per_second * duration
            settings['estimated_output_size_mb'] = estimated_size / (1024 * 1024)
            
        # Adjust settings based on input characteristics
        if input_info.get('sample_rate', 0) < self.WHISPER_SAMPLE_RATE:
            settings['apply_upsampling'] = True
            
        if input_info.get('duration', 0) > 3600:  # > 1 hour
            settings['apply_compression'] = True
            
        return settings
        
    def cleanup_temp_files(self) -> None:
        """Clean up temporary audio files."""
        try:
            if self.temp_dir.exists():
                for file_path in self.temp_dir.glob("*"):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                        except Exception as e:
                            self.logger.warning(f"Could not remove temp file {file_path}: {e}")
                            
                self.logger.info("Temporary audio files cleaned up")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up temp files: {e}")


# Utility functions for audio processing
def extract_audio_for_whisper(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Extract and optimize audio for Whisper transcription.
    
    Args:
        input_path: Path to input audio/video file
        output_path: Path for output audio file, auto-generated if None
        
    Returns:
        Path to extracted audio file
        
    Raises:
        AudioExtractionError: If extraction fails
    """
    extractor = AudioExtractor()
    return extractor.extract_audio(input_path, output_path)


def validate_audio_file(file_path: Path) -> bool:
    """Quick validation of audio file for Whisper.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if valid for Whisper processing
    """
    try:
        extractor = AudioExtractor()
        is_valid, _ = extractor.validate_audio_for_whisper(file_path)
        return is_valid
    except Exception:
        return False


def get_audio_duration(file_path: Path) -> float:
    """Get audio file duration in seconds.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Duration in seconds, 0 if error
    """
    try:
        extractor = AudioExtractor()
        info = extractor.get_audio_info(file_path)
        return info.get('duration', 0)
    except Exception:
        return 0
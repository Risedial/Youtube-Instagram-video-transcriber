"""Application settings and configuration management.

This module will handle application settings, preferences, and configuration.
To be implemented in Phase 6: Application Integration & State Management.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict, field
import json
import logging
import os


@dataclass
class ApplicationSettings:
    """Application settings data structure."""
    
    # Whisper Configuration
    default_model: str = "tiny"
    device: str = "cpu"
    enable_gpu: bool = False
    
    # Download Settings
    download_timeout: int = 300
    max_video_length_hours: int = 2
    temp_directory: Optional[str] = None
    
    # Output Settings
    default_output_directory: Optional[str] = None
    auto_open_output_folder: bool = True
    timestamp_format: str = "include"  # "include" or "exclude"
    
    # GUI Settings
    window_width: int = 800
    window_height: int = 800
    theme: str = "default"
    enable_progress_sound: bool = False
    
    # Performance Settings
    max_memory_gb: int = 8
    concurrent_downloads: int = 1
    preserve_temp_files: bool = False
    
    # Logging Settings
    log_level: str = "INFO"
    log_to_file: bool = True
    max_log_size_mb: int = 10
    
    # Advanced Settings
    enable_crash_reporting: bool = False
    auto_update_check: bool = True
    sanitize_filenames: bool = True
    
    # Proxy/VPN Settings
    enable_proxy: bool = False
    proxy_type: str = "http"  # http, https, socks4, socks5
    proxy_host: str = ""
    proxy_port: int = 8080
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    proxy_rotate_enabled: bool = False
    proxy_rotation_interval: int = 10  # requests before rotation
    proxy_list: List[str] = field(default_factory=list)  # List of proxy servers
    
    # Kill Switch Settings
    enable_kill_switch: bool = True
    kill_switch_check_interval: int = 30  # seconds
    auto_reconnect_attempts: int = 3
    
    # Anti-Detection Settings
    enable_random_delays: bool = True
    min_request_delay: int = 30  # seconds
    max_request_delay: int = 120  # seconds
    rotate_user_agents: bool = True
    enable_header_randomization: bool = True
    
    # Instagram-Specific Settings
    instagram_session_preserve: bool = True
    instagram_max_requests_per_session: int = 10
    instagram_cool_down_period: int = 300  # seconds between sessions


class Settings:
    """Application settings manager.
    
    This class will implement:
    - Settings persistence (JSON files)
    - Environment variable override
    - Default value management
    - Settings validation
    - Migration between versions
    
    To be fully implemented in Phase 6.
    """
    
    def __init__(self, config_dir: Optional[Path] = None) -> None:
        """Initialize settings manager.
        
        Args:
            config_dir: Configuration directory, uses default if None
        """
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "settings.json"
        self.logger = logging.getLogger(__name__)
        
        # Load settings
        self._settings = self._load_settings()
        
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory."""
        if os.name == 'nt':  # Windows
            config_base = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
        else:  # Unix-like
            config_base = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))
            
        return config_base / 'youtube_whisper_transcriber'
        
    def _load_settings(self) -> ApplicationSettings:
        """Load settings from file with environment overrides.
        
        Returns:
            ApplicationSettings instance
        """
        # Start with defaults
        settings = ApplicationSettings()
        
        # Load from file if exists
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    saved_settings = json.load(f)

                # Migrate legacy settings if present (BEFORE applying to dataclass)
                saved_settings = self._migrate_legacy_settings_dict(saved_settings)

                # Update settings with saved values
                for key, value in saved_settings.items():
                    if hasattr(settings, key):
                        setattr(settings, key, value)

                self.logger.info(f"Loaded settings from: {self.config_file}")
                
            except Exception as e:
                self.logger.warning(f"Error loading settings file: {e}")
                self.logger.info("Using default settings")
                
        # Apply environment variable overrides
        self._apply_environment_overrides(settings)

        return settings
        
    def _apply_environment_overrides(self, settings: ApplicationSettings) -> None:
        """Apply environment variable overrides to settings.
        
        Args:
            settings: Settings object to modify
        """
        # Environment variable mapping
        env_mapping = {
            'DEFAULT_WHISPER_MODEL': 'default_model',
            'WHISPER_DEVICE': 'device',
            'DEFAULT_OUTPUT_DIRECTORY': 'default_output_directory',
            'TEMP_DIRECTORY': 'temp_directory',
            'MAX_VIDEO_LENGTH_HOURS': ('max_video_length_hours', int),
            'DOWNLOAD_TIMEOUT_SECONDS': ('download_timeout', int),
            'WINDOW_WIDTH': ('window_width', int),
            'WINDOW_HEIGHT': ('window_height', int),
            'THEME': 'theme',
            'LOG_LEVEL': 'log_level',
            'MAX_MEMORY_GB': ('max_memory_gb', int),
            'ENABLE_GPU_ACCELERATION': ('enable_gpu', bool),
            'AUTO_OPEN_OUTPUT_FOLDER': ('auto_open_output_folder', bool),
            'LOG_TO_FILE': ('log_to_file', bool),
            
            # Proxy environment variables
            'ENABLE_PROXY': ('enable_proxy', bool),
            'PROXY_TYPE': 'proxy_type',
            'PROXY_HOST': 'proxy_host',
            'PROXY_PORT': ('proxy_port', int),
            'PROXY_USERNAME': 'proxy_username',
            'PROXY_PASSWORD': 'proxy_password',
            'ENABLE_KILL_SWITCH': ('enable_kill_switch', bool),
            'ENABLE_RANDOM_DELAYS': ('enable_random_delays', bool),
            'MIN_REQUEST_DELAY': ('min_request_delay', int),
            'MAX_REQUEST_DELAY': ('max_request_delay', int),
        }
        
        for env_var, setting_info in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    if isinstance(setting_info, tuple):
                        setting_name, type_converter = setting_info
                        if type_converter == bool:
                            value = env_value.lower() in ('true', '1', 'yes', 'on')
                        else:
                            value = type_converter(env_value)
                    else:
                        setting_name = setting_info
                        value = env_value
                        
                    setattr(settings, setting_name, value)
                    self.logger.debug(f"Applied environment override: {env_var}={value}")
                    
                except Exception as e:
                    self.logger.warning(f"Error applying environment override {env_var}: {e}")

    def _migrate_legacy_settings_dict(self, settings_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate legacy settings dictionary to new format.

        This migrates old include_timestamps/exclude_timestamps boolean fields
        to the new timestamp_format string field.

        Args:
            settings_dict: Dictionary loaded from settings JSON file

        Returns:
            Migrated settings dictionary
        """
        # Check if legacy fields exist
        if 'include_timestamps' in settings_dict or 'exclude_timestamps' in settings_dict:
            include_ts = settings_dict.get('include_timestamps', True)
            exclude_ts = settings_dict.get('exclude_timestamps', False)

            # Determine new format based on legacy settings
            if include_ts and not exclude_ts:
                new_format = "include"
            elif exclude_ts and not include_ts:
                new_format = "exclude"
            elif include_ts and exclude_ts:
                # Both enabled (contradictory) - default to include
                new_format = "include"
                self.logger.warning("Legacy settings had both formats enabled. Defaulting to 'include'.")
            else:
                # Both disabled (invalid) - default to include
                new_format = "include"
                self.logger.warning("Legacy settings had neither format enabled. Defaulting to 'include'.")

            # Add new field
            settings_dict['timestamp_format'] = new_format

            # Remove legacy fields
            settings_dict.pop('include_timestamps', None)
            settings_dict.pop('exclude_timestamps', None)

            self.logger.info(f"Migrated legacy format settings to: {new_format}")

        return settings_dict

    def save_settings(self) -> bool:
        """Save current settings to file.
        
        Returns:
            True if saved successfully
        """
        try:
            settings_dict = asdict(self._settings)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"Saved settings to: {self.config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            return False
            
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get setting value.
        
        Args:
            key: Setting key
            default: Default value if key not found
            
        Returns:
            Setting value
        """
        return getattr(self._settings, key, default)
        
    def set_setting(self, key: str, value: Any) -> bool:
        """Set setting value.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if set successfully
        """
        if hasattr(self._settings, key):
            setattr(self._settings, key, value)
            return True
        else:
            self.logger.warning(f"Unknown setting key: {key}")
            return False
            
    def get_all_settings(self) -> Dict[str, Any]:
        """Get all settings as dictionary.
        
        Returns:
            Dictionary of all settings
        """
        return asdict(self._settings)
        
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults."""
        self._settings = ApplicationSettings()
        self.logger.info("Reset settings to defaults")
        
    def validate_settings(self) -> bool:
        """Validate current settings.
        
        Returns:
            True if all settings are valid
        """
        try:
            # Validate model name
            valid_models = ["tiny", "base", "small", "medium", "large"]
            if self._settings.default_model not in valid_models:
                self.logger.error(f"Invalid default model: {self._settings.default_model}")
                return False
                
            # Validate device
            if self._settings.device not in ["cpu", "cuda"]:
                self.logger.error(f"Invalid device: {self._settings.device}")
                return False
                
            # Validate directories if specified
            if self._settings.default_output_directory:
                output_dir = Path(self._settings.default_output_directory)
                if not output_dir.parent.exists():
                    self.logger.error(f"Invalid output directory: {self._settings.default_output_directory}")
                    return False
                    
            # Validate numeric values
            if self._settings.max_video_length_hours <= 0:
                self.logger.error("Invalid max video length")
                return False
                
            if self._settings.download_timeout <= 0:
                self.logger.error("Invalid download timeout")
                return False

            # Validate timestamp format
            if self._settings.timestamp_format not in ["include", "exclude"]:
                self.logger.error(f"Invalid timestamp format: {self._settings.timestamp_format}")
                return False

            # Validate proxy settings
            if self._settings.enable_proxy:
                if not self._settings.proxy_host:
                    self.logger.error("Proxy enabled but no host specified")
                    return False
                    
                if self._settings.proxy_type not in ["http", "https", "socks4", "socks5"]:
                    self.logger.error(f"Invalid proxy type: {self._settings.proxy_type}")
                    return False
                    
                if not (1 <= self._settings.proxy_port <= 65535):
                    self.logger.error(f"Invalid proxy port: {self._settings.proxy_port}")
                    return False
                    
            # Validate delay settings
            if self._settings.min_request_delay < 0 or self._settings.max_request_delay < 0:
                self.logger.error("Invalid request delay settings")
                return False
                
            if self._settings.min_request_delay > self._settings.max_request_delay:
                self.logger.error("Min request delay cannot be greater than max request delay")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating settings: {e}")
            return False


# Global settings instance
_global_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance.
    
    Returns:
        Settings instance
    """
    global _global_settings
    if _global_settings is None:
        _global_settings = Settings()
    return _global_settings


def load_settings(config_dir: Optional[Path] = None) -> ApplicationSettings:
    """Load application settings.
    
    Args:
        config_dir: Configuration directory
        
    Returns:
        ApplicationSettings instance
    """
    settings_manager = Settings(config_dir)
    return settings_manager._settings


def save_settings(settings: ApplicationSettings, config_dir: Optional[Path] = None) -> bool:
    """Save application settings.
    
    Args:
        settings: Settings to save
        config_dir: Configuration directory
        
    Returns:
        True if saved successfully
    """
    settings_manager = Settings(config_dir)
    settings_manager._settings = settings
    return settings_manager.save_settings()


def get_default_settings() -> ApplicationSettings:
    """Get default application settings.
    
    Returns:
        Default ApplicationSettings instance
    """
    return ApplicationSettings()
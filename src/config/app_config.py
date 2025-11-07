"""Application configuration and user preferences management for YouTube Whisper Transcriber.

This module provides configuration persistence, user preference management, validation,
default value handling, and configuration migration for version updates.
"""

from typing import Any, Dict, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import logging
import threading
from pathlib import Path
import platform
import os
from datetime import datetime


class ConfigCategory(Enum):
    """Configuration categories for organization."""
    UI = "ui"
    PROCESSING = "processing"
    OUTPUT = "output"
    PERFORMANCE = "performance"
    ADVANCED = "advanced"
    SYSTEM = "system"


class ConfigType(Enum):
    """Configuration value types for validation."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    PATH = "path"
    LIST = "list"
    DICT = "dict"


@dataclass
class ConfigItem:
    """Configuration item definition with validation and metadata."""
    key: str
    default_value: Any
    config_type: ConfigType
    category: ConfigCategory
    description: str
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[Any]] = None
    required: bool = False
    hidden: bool = False
    restart_required: bool = False
    validation_function: Optional[Callable[[Any], bool]] = None
    display_name: Optional[str] = None


@dataclass
class ConfigSection:
    """Group of related configuration items."""
    name: str
    category: ConfigCategory
    items: Dict[str, ConfigItem] = field(default_factory=dict)
    description: str = ""
    icon: str = ""


class AppConfig:
    """Application configuration and user preferences management system.
    
    Features:
    - Configuration persistence with automatic saving and loading
    - User preference categories (UI, processing, output, performance)
    - Configuration validation with type checking and constraints
    - Default value handling with fallback mechanisms
    - Configuration migration for version updates and compatibility
    - Real-time configuration change notifications
    """
    
    # Configuration schema version for migration
    CONFIG_VERSION = "1.0.0"
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.logger = logging.getLogger(__name__)
        self.config_lock = threading.Lock()
        
        # Configuration directory setup
        self.config_dir = config_dir or self._get_default_config_dir()
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        # Configuration files
        self.config_file = self.config_dir / "config.json"
        self.user_preferences_file = self.config_dir / "user_preferences.json"
        self.cache_file = self.config_dir / "cache.json"
        
        # Configuration storage
        self.config_values: Dict[str, Any] = {}
        self.config_schema: Dict[str, ConfigItem] = {}
        self.config_sections: Dict[str, ConfigSection] = {}
        
        # Change notifications
        self.change_callbacks: List[Callable[[str, Any, Any], None]] = []
        
        # Performance cache
        self.cache: Dict[str, Any] = {}
        self.cache_dirty = False
        
        # Initialize default configuration schema
        self._initialize_schema()
        
        # Load existing configuration
        self._load_configuration()
        
        self.logger.info(f"AppConfig initialized with config dir: {self.config_dir}")
    
    def _get_default_config_dir(self) -> Path:
        """Get default configuration directory based on platform."""
        if platform.system() == "Windows":
            # Use AppData for Windows
            appdata = os.getenv("APPDATA")
            if appdata:
                return Path(appdata) / "YouTube Whisper Transcriber"
            else:
                return Path.home() / ".youtube_whisper_transcriber"
        else:
            # Use XDG_CONFIG_HOME or ~/.config for Unix-like systems
            config_home = os.getenv("XDG_CONFIG_HOME")
            if config_home:
                return Path(config_home) / "youtube_whisper_transcriber"
            else:
                return Path.home() / ".config" / "youtube_whisper_transcriber"
    
    def _initialize_schema(self) -> None:
        """Initialize configuration schema with all available settings."""
        
        # UI Configuration Section
        ui_section = ConfigSection(
            name="User Interface",
            category=ConfigCategory.UI,
            description="User interface preferences and appearance settings",
            icon="🎨"
        )
        
        ui_items = [
            ConfigItem(
                key="ui.window_width",
                default_value=800,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.UI,
                description="Main window width in pixels",
                min_value=600,
                max_value=2000,
                display_name="Window Width"
            ),
            ConfigItem(
                key="ui.window_height",
                default_value=700,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.UI,
                description="Main window height in pixels",
                min_value=500,
                max_value=1500,
                display_name="Window Height"
            ),
            ConfigItem(
                key="ui.theme",
                default_value="light",
                config_type=ConfigType.STRING,
                category=ConfigCategory.UI,
                description="Application color theme",
                valid_values=["light", "dark", "auto"],
                display_name="Theme"
            ),
            ConfigItem(
                key="ui.font_size",
                default_value=10,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.UI,
                description="Font size for UI elements",
                min_value=8,
                max_value=16,
                display_name="Font Size"
            ),
            ConfigItem(
                key="ui.auto_open_output_folder",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.UI,
                description="Automatically open output folder after completion",
                display_name="Auto-open Output Folder"
            ),
            ConfigItem(
                key="ui.show_progress_details",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.UI,
                description="Show detailed progress information",
                display_name="Show Progress Details"
            ),
            ConfigItem(
                key="ui.minimize_to_tray",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.UI,
                description="Minimize to system tray instead of taskbar",
                display_name="Minimize to Tray"
            )
        ]
        
        for item in ui_items:
            ui_section.items[item.key] = item
            self.config_schema[item.key] = item
        
        self.config_sections["ui"] = ui_section
        
        # Processing Configuration Section
        processing_section = ConfigSection(
            name="Processing",
            category=ConfigCategory.PROCESSING,
            description="Audio processing and transcription settings",
            icon="⚙️"
        )
        
        processing_items = [
            ConfigItem(
                key="processing.default_whisper_model",
                default_value="tiny",
                config_type=ConfigType.STRING,
                category=ConfigCategory.PROCESSING,
                description="Default Whisper model for transcription",
                valid_values=["tiny", "base", "small", "medium", "large"],
                display_name="Default Whisper Model"
            ),
            ConfigItem(
                key="processing.whisper_device",
                default_value="auto",
                config_type=ConfigType.STRING,
                category=ConfigCategory.PROCESSING,
                description="Device for Whisper processing",
                valid_values=["auto", "cpu", "cuda"],
                display_name="Processing Device"
            ),
            ConfigItem(
                key="processing.audio_quality",
                default_value="high",
                config_type=ConfigType.STRING,
                category=ConfigCategory.PROCESSING,
                description="Audio extraction quality",
                valid_values=["low", "medium", "high"],
                display_name="Audio Quality"
            ),
            ConfigItem(
                key="processing.max_video_duration",
                default_value=7200,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.PROCESSING,
                description="Maximum video duration in seconds (2 hours default)",
                min_value=300,
                max_value=14400,
                display_name="Max Video Duration (seconds)"
            ),
            ConfigItem(
                key="processing.cleanup_temp_files",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.PROCESSING,
                description="Automatically clean up temporary files",
                display_name="Cleanup Temp Files"
            ),
            ConfigItem(
                key="processing.parallel_downloads",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.PROCESSING,
                description="Enable parallel segment downloads (experimental)",
                display_name="Parallel Downloads"
            )
        ]
        
        for item in processing_items:
            processing_section.items[item.key] = item
            self.config_schema[item.key] = item
        
        self.config_sections["processing"] = processing_section
        
        # Output Configuration Section
        output_section = ConfigSection(
            name="Output",
            category=ConfigCategory.OUTPUT,
            description="Output file settings and formatting options",
            icon="📁"
        )
        
        output_items = [
            ConfigItem(
                key="output.default_directory",
                default_value=str(Path.home() / "Downloads" / "Transcriptions"),
                config_type=ConfigType.PATH,
                category=ConfigCategory.OUTPUT,
                description="Default directory for transcription files",
                display_name="Default Output Directory"
            ),
            ConfigItem(
                key="output.filename_format",
                default_value="{title}_transcript",
                config_type=ConfigType.STRING,
                category=ConfigCategory.OUTPUT,
                description="Format for output filenames ({title}, {date}, {model})",
                display_name="Filename Format"
            ),
            ConfigItem(
                key="output.include_timestamps",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.OUTPUT,
                description="Generate transcription file with timestamps",
                display_name="Include Timestamps"
            ),
            ConfigItem(
                key="output.exclude_timestamps",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.OUTPUT,
                description="Generate transcription file without timestamps",
                display_name="Exclude Timestamps"
            ),
            ConfigItem(
                key="output.include_metadata",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.OUTPUT,
                description="Include metadata header in output files",
                display_name="Include Metadata"
            ),
            ConfigItem(
                key="output.text_encoding",
                default_value="utf-8",
                config_type=ConfigType.STRING,
                category=ConfigCategory.OUTPUT,
                description="Text encoding for output files",
                valid_values=["utf-8", "ascii", "latin-1"],
                display_name="Text Encoding"
            ),
            ConfigItem(
                key="output.backup_files",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.OUTPUT,
                description="Create backup copies of output files",
                display_name="Backup Files"
            )
        ]
        
        for item in output_items:
            output_section.items[item.key] = item
            self.config_schema[item.key] = item
        
        self.config_sections["output"] = output_section
        
        # Performance Configuration Section
        performance_section = ConfigSection(
            name="Performance",
            category=ConfigCategory.PERFORMANCE,
            description="Performance optimization and resource management",
            icon="🚀"
        )
        
        performance_items = [
            ConfigItem(
                key="performance.max_memory_usage",
                default_value=4096,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.PERFORMANCE,
                description="Maximum memory usage in MB",
                min_value=1024,
                max_value=16384,
                display_name="Max Memory Usage (MB)"
            ),
            ConfigItem(
                key="performance.thread_count",
                default_value=0,  # 0 = auto-detect
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.PERFORMANCE,
                description="Number of processing threads (0 = auto)",
                min_value=0,
                max_value=16,
                display_name="Thread Count"
            ),
            ConfigItem(
                key="performance.gpu_acceleration",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.PERFORMANCE,
                description="Use GPU acceleration when available",
                display_name="GPU Acceleration"
            ),
            ConfigItem(
                key="performance.cache_models",
                default_value=True,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.PERFORMANCE,
                description="Cache Whisper models in memory",
                display_name="Cache Models"
            ),
            ConfigItem(
                key="performance.progress_update_frequency",
                default_value=0.5,
                config_type=ConfigType.FLOAT,
                category=ConfigCategory.PERFORMANCE,
                description="Progress update frequency in seconds",
                min_value=0.1,
                max_value=5.0,
                display_name="Progress Update Frequency"
            )
        ]
        
        for item in performance_items:
            performance_section.items[item.key] = item
            self.config_schema[item.key] = item
        
        self.config_sections["performance"] = performance_section
        
        # Advanced Configuration Section
        advanced_section = ConfigSection(
            name="Advanced",
            category=ConfigCategory.ADVANCED,
            description="Advanced settings for power users",
            icon="🔧"
        )
        
        advanced_items = [
            ConfigItem(
                key="advanced.debug_mode",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.ADVANCED,
                description="Enable debug logging and verbose output",
                restart_required=True,
                display_name="Debug Mode"
            ),
            ConfigItem(
                key="advanced.log_level",
                default_value="INFO",
                config_type=ConfigType.STRING,
                category=ConfigCategory.ADVANCED,
                description="Logging level for application",
                valid_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                restart_required=True,
                display_name="Log Level"
            ),
            ConfigItem(
                key="advanced.temp_directory",
                default_value="",  # Empty = use system default
                config_type=ConfigType.PATH,
                category=ConfigCategory.ADVANCED,
                description="Custom temporary directory (empty = use system default)",
                display_name="Temp Directory"
            ),
            ConfigItem(
                key="advanced.network_timeout",
                default_value=30,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.ADVANCED,
                description="Network timeout in seconds",
                min_value=10,
                max_value=300,
                display_name="Network Timeout"
            ),
            ConfigItem(
                key="advanced.retry_attempts",
                default_value=3,
                config_type=ConfigType.INTEGER,
                category=ConfigCategory.ADVANCED,
                description="Number of retry attempts for failed operations",
                min_value=1,
                max_value=10,
                display_name="Retry Attempts"
            ),
            ConfigItem(
                key="advanced.experimental_features",
                default_value=False,
                config_type=ConfigType.BOOLEAN,
                category=ConfigCategory.ADVANCED,
                description="Enable experimental features (may be unstable)",
                restart_required=True,
                display_name="Experimental Features"
            )
        ]
        
        for item in advanced_items:
            advanced_section.items[item.key] = item
            self.config_schema[item.key] = item
        
        self.config_sections["advanced"] = advanced_section
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get configuration setting value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        with self.config_lock:
            # Check if value exists in current config
            if key in self.config_values:
                return self.config_values[key]
            
            # Check schema for default
            if key in self.config_schema:
                return self.config_schema[key].default_value
            
            # Return provided default
            return default
    
    def set_setting(self, key: str, value: Any, validate: bool = True, 
                   notify: bool = True) -> bool:
        """Set configuration setting value.
        
        Args:
            key: Configuration key
            value: Value to set
            validate: Whether to validate the value
            notify: Whether to notify change callbacks
            
        Returns:
            True if setting was successfully set
        """
        with self.config_lock:
            # Validate if requested
            if validate and not self._validate_setting(key, value):
                self.logger.warning(f"Validation failed for setting {key}={value}")
                return False
            
            # Get old value for change notification
            old_value = self.config_values.get(key)
            
            # Set new value
            self.config_values[key] = value
            
            # Mark for saving
            self._schedule_save()
            
            self.logger.info(f"Setting updated: {key}={value}")
            
            # Notify callbacks if requested
            if notify and old_value != value:
                self._notify_change_callbacks(key, old_value, value)
            
            return True
    
    def get_section_settings(self, section_name: str) -> Dict[str, Any]:
        """Get all settings for a specific section.
        
        Args:
            section_name: Name of configuration section
            
        Returns:
            Dictionary of section settings
        """
        if section_name not in self.config_sections:
            return {}
        
        section = self.config_sections[section_name]
        settings = {}
        
        for key in section.items:
            settings[key] = self.get_setting(key)
        
        return settings
    
    def reset_section(self, section_name: str) -> bool:
        """Reset all settings in a section to defaults.
        
        Args:
            section_name: Name of section to reset
            
        Returns:
            True if section was reset successfully
        """
        if section_name not in self.config_sections:
            return False
        
        section = self.config_sections[section_name]
        
        with self.config_lock:
            for key, item in section.items.items():
                self.config_values[key] = item.default_value
            
            self._schedule_save()
        
        self.logger.info(f"Reset section to defaults: {section_name}")
        return True
    
    def reset_to_defaults(self) -> None:
        """Reset all configuration to default values."""
        with self.config_lock:
            self.config_values.clear()
            
            # Set all defaults
            for key, item in self.config_schema.items():
                self.config_values[key] = item.default_value
            
            self._schedule_save()
        
        self.logger.info("All configuration reset to defaults")
    
    def export_configuration(self, file_path: Path) -> bool:
        """Export current configuration to file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if export was successful
        """
        try:
            export_data = {
                "version": self.CONFIG_VERSION,
                "timestamp": datetime.now().isoformat(),
                "configuration": dict(self.config_values)
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def import_configuration(self, file_path: Path, merge: bool = True) -> bool:
        """Import configuration from file.
        
        Args:
            file_path: Path to import file
            merge: Whether to merge with existing config or replace
            
        Returns:
            True if import was successful
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # Validate format
            if "configuration" not in import_data:
                self.logger.error("Invalid configuration file format")
                return False
            
            imported_config = import_data["configuration"]
            
            with self.config_lock:
                if not merge:
                    self.config_values.clear()
                
                # Import settings with validation
                imported_count = 0
                for key, value in imported_config.items():
                    if self._validate_setting(key, value):
                        self.config_values[key] = value
                        imported_count += 1
                    else:
                        self.logger.warning(f"Skipped invalid setting: {key}={value}")
                
                self._schedule_save()
            
            self.logger.info(f"Imported {imported_count} settings from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {e}")
            return False
    
    def _validate_setting(self, key: str, value: Any) -> bool:
        """Validate a configuration setting.
        
        Args:
            key: Configuration key
            value: Value to validate
            
        Returns:
            True if value is valid
        """
        if key not in self.config_schema:
            return False
        
        item = self.config_schema[key]
        
        # Type validation
        if not self._validate_type(value, item.config_type):
            return False
        
        # Range validation
        if item.min_value is not None and value < item.min_value:
            return False
        
        if item.max_value is not None and value > item.max_value:
            return False
        
        # Valid values validation
        if item.valid_values is not None and value not in item.valid_values:
            return False
        
        # Custom validation function
        if item.validation_function and not item.validation_function(value):
            return False
        
        # Path validation
        if item.config_type == ConfigType.PATH and value:
            try:
                path = Path(value)
                # Check if parent directory exists for new files
                if not path.exists() and not path.parent.exists():
                    return False
            except Exception:
                return False
        
        return True
    
    def _validate_type(self, value: Any, config_type: ConfigType) -> bool:
        """Validate value type."""
        type_validators = {
            ConfigType.STRING: lambda v: isinstance(v, str),
            ConfigType.INTEGER: lambda v: isinstance(v, int),
            ConfigType.FLOAT: lambda v: isinstance(v, (int, float)),
            ConfigType.BOOLEAN: lambda v: isinstance(v, bool),
            ConfigType.PATH: lambda v: isinstance(v, str),
            ConfigType.LIST: lambda v: isinstance(v, list),
            ConfigType.DICT: lambda v: isinstance(v, dict)
        }
        
        validator = type_validators.get(config_type)
        return validator(value) if validator else False
    
    def _load_configuration(self) -> None:
        """Load configuration from files."""
        # Load main configuration
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle version migration
                config_version = data.get("version", "0.0.0")
                if config_version != self.CONFIG_VERSION:
                    self._migrate_configuration(data, config_version)
                
                # Load settings with validation
                loaded_config = data.get("configuration", {})
                for key, value in loaded_config.items():
                    if self._validate_setting(key, value):
                        self.config_values[key] = value
                
                self.logger.info(f"Loaded configuration: {len(self.config_values)} settings")
                
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
        
        # Load cache
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
        
        # Set defaults for missing settings
        for key, item in self.config_schema.items():
            if key not in self.config_values:
                self.config_values[key] = item.default_value
    
    def _migrate_configuration(self, data: Dict[str, Any], from_version: str) -> None:
        """Migrate configuration from older version.
        
        Args:
            data: Configuration data to migrate
            from_version: Source version
        """
        self.logger.info(f"Migrating configuration from version {from_version} to {self.CONFIG_VERSION}")
        
        # Example migration logic - extend as needed
        if from_version < "1.0.0":
            # Migrate old setting names
            config = data.get("configuration", {})
            
            # Example: rename old settings
            if "window_width" in config:
                config["ui.window_width"] = config.pop("window_width")
            if "window_height" in config:
                config["ui.window_height"] = config.pop("window_height")
            if "default_model" in config:
                config["processing.default_whisper_model"] = config.pop("default_model")
            
            data["configuration"] = config
        
        # Update version
        data["version"] = self.CONFIG_VERSION
    
    def _schedule_save(self) -> None:
        """Schedule configuration save (immediate for now, could be delayed)."""
        self._save_configuration()
    
    def _save_configuration(self) -> None:
        """Save configuration to file."""
        try:
            # Prepare data
            save_data = {
                "version": self.CONFIG_VERSION,
                "timestamp": datetime.now().isoformat(),
                "configuration": dict(self.config_values)
            }
            
            # Save main configuration
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # Save cache if dirty
            if self.cache_dirty:
                with open(self.cache_file, 'w', encoding='utf-8') as f:
                    json.dump(self.cache, f, indent=2)
                self.cache_dirty = False
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def _notify_change_callbacks(self, key: str, old_value: Any, new_value: Any) -> None:
        """Notify registered callbacks about configuration changes."""
        for callback in self.change_callbacks:
            try:
                callback(key, old_value, new_value)
            except Exception as e:
                self.logger.warning(f"Configuration change callback failed: {e}")
    
    def add_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Add configuration change callback.
        
        Args:
            callback: Function that takes (key, old_value, new_value)
        """
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable[[str, Any, Any], None]) -> None:
        """Remove configuration change callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def get_schema_info(self, key: str) -> Optional[ConfigItem]:
        """Get schema information for a configuration key.
        
        Args:
            key: Configuration key
            
        Returns:
            ConfigItem with schema information or None
        """
        return self.config_schema.get(key)
    
    def get_all_sections(self) -> Dict[str, ConfigSection]:
        """Get all configuration sections.
        
        Returns:
            Dictionary of all configuration sections
        """
        return dict(self.config_sections)
    
    def validate_all_settings(self) -> Dict[str, List[str]]:
        """Validate all current settings.
        
        Returns:
            Dictionary with validation errors by section
        """
        errors = {}
        
        for section_name, section in self.config_sections.items():
            section_errors = []
            
            for key, item in section.items.items():
                current_value = self.get_setting(key)
                if not self._validate_setting(key, current_value):
                    section_errors.append(f"{item.display_name or key}: Invalid value '{current_value}'")
            
            if section_errors:
                errors[section_name] = section_errors
        
        return errors
    
    def get_cache_value(self, key: str, default: Any = None) -> Any:
        """Get value from performance cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        return self.cache.get(key, default)
    
    def set_cache_value(self, key: str, value: Any) -> None:
        """Set value in performance cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self.cache[key] = value
        self.cache_dirty = True
    
    def clear_cache(self) -> None:
        """Clear performance cache."""
        self.cache.clear()
        self.cache_dirty = True
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration.
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            "version": self.CONFIG_VERSION,
            "total_settings": len(self.config_values),
            "sections": len(self.config_sections),
            "config_file": str(self.config_file),
            "config_dir": str(self.config_dir),
            "cache_size": len(self.cache),
            "change_callbacks": len(self.change_callbacks),
            "last_modified": self.config_file.stat().st_mtime if self.config_file.exists() else None
        }


# Global configuration instance
_app_config: Optional[AppConfig] = None


def get_app_config() -> AppConfig:
    """Get global application configuration instance.
    
    Returns:
        AppConfig singleton instance
    """
    global _app_config
    if _app_config is None:
        _app_config = AppConfig()
    return _app_config


def initialize_config(config_dir: Optional[Path] = None) -> AppConfig:
    """Initialize global configuration with custom directory.
    
    Args:
        config_dir: Custom configuration directory
        
    Returns:
        Initialized AppConfig instance
    """
    global _app_config
    _app_config = AppConfig(config_dir)
    return _app_config
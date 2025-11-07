"""Main entry point for YouTube Whisper Transcriber application.

This module provides the main application launcher with proper Windows integration,
error handling, and configuration setup.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import messagebox
import argparse

# Add src directory to path for imports
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from ui.main_window import MainWindow
from config.settings import get_settings
from utils.error_handler import setup_logging, log_exception


def setup_application() -> bool:
    """Setup application environment and logging.
    
    Returns:
        True if setup successful, False otherwise
    """
    try:
        # Get settings and setup logging
        settings = get_settings()
        # Enable debug logging for debugging transcription issues
        setup_logging(
            level=getattr(logging, settings.get_setting('log_level', 'DEBUG').upper(), logging.DEBUG)
        )
        
        # Add console logging for PyInstaller debug
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
        
        # Create required directories
        config_dir = settings.config_dir
        if not config_dir.exists():
            config_dir.mkdir(parents=True, exist_ok=True)
            
        # Validate settings
        if not settings.validate_settings():
            logging.error("Invalid application settings")
            return False
            
        logging.info("Application setup completed successfully")
        return True
        
    except Exception as e:
        log_exception(e, "Failed to setup application")
        return False


def setup_windows_integration():
    """Setup Windows-specific application behavior."""
    if os.name == 'nt':  # Windows
        try:
            # Set process DPI awareness for high-DPI displays
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass  # Not critical if this fails
        
        try:
            # Set application ID for taskbar grouping
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "YouTubeWhisperTranscriber.Desktop.1.0"
            )
        except Exception:
            pass  # Not critical if this fails


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="YouTube Whisper Transcriber - Convert YouTube videos to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Launch GUI application
  %(prog)s --version          # Show version information
  %(prog)s --debug            # Enable debug logging
        """
    )
    
    parser.add_argument(
        '--version', 
        action='version', 
        version='YouTube Whisper Transcriber 1.0.0'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        help='Custom configuration directory path'
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run in console mode (future implementation)'
    )
    
    return parser.parse_args()


def show_error_dialog(title: str, message: str):
    """Show error dialog to user.
    
    Args:
        title: Dialog title
        message: Error message
    """
    try:
        root = tk.Tk()
        root.withdraw()  # Hide main window
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        # Fallback to console output
        print(f"ERROR: {title}\n{message}")


def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Override log level if debug mode
        if args.debug:
            logging.basicConfig(level=logging.DEBUG)
            
        # Setup Windows integration
        setup_windows_integration()
        
        # Setup application environment
        if not setup_application():
            show_error_dialog(
                "Startup Error",
                "Failed to initialize application. Please check the log files for details."
            )
            sys.exit(1)
            
        # Future: Handle console mode
        if args.no_gui:
            show_error_dialog(
                "Not Implemented",
                "Console mode is not yet implemented. Please use the GUI interface."
            )
            sys.exit(1)
            
        # Launch GUI application
        logging.info("Starting YouTube Whisper Transcriber GUI")
        
        # Create and configure main window
        root = tk.Tk()
        app = MainWindow(root)
        
        # Setup global exception handler
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
                
            log_exception(
                exc_value, 
                "Unhandled exception in main application",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
            
            show_error_dialog(
                "Unexpected Error",
                f"An unexpected error occurred: {exc_value}\n\n"
                "Please check the log files for details."
            )
            
        sys.excepthook = handle_exception
        
        # Start GUI event loop
        try:
            app.run()
        except KeyboardInterrupt:
            logging.info("Application interrupted by user")
        except Exception as e:
            log_exception(e, "Error in main application loop")
            show_error_dialog(
                "Application Error",
                f"Application encountered an error: {e}"
            )
            
    except Exception as e:
        # Critical startup error
        error_msg = f"Critical startup error: {e}"
        logging.critical(error_msg)
        show_error_dialog("Startup Error", error_msg)
        sys.exit(1)
        
    finally:
        logging.info("YouTube Whisper Transcriber shutdown complete")


if __name__ == "__main__":
    main()
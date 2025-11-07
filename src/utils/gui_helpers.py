"""GUI utility functions and helpers for YouTube Whisper Transcriber.

This module provides thread-safe GUI update mechanisms, Windows integration helpers,
and utility functions for the desktop application.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Any, Optional, Dict, Union
import threading
import queue
import logging
import os
import subprocess
from pathlib import Path
from functools import wraps


class ThreadSafeGUI:
    """Thread-safe GUI update manager.
    
    Provides methods to safely update GUI components from background threads
    using the tkinter after() method.
    """
    
    def __init__(self, root: tk.Tk) -> None:
        """Initialize thread-safe GUI manager.
        
        Args:
            root: Root tkinter window
        """
        self.root = root
        self.update_queue = queue.Queue()
        self.logger = logging.getLogger(__name__)
        
    def schedule_update(self, callback: Callable, *args, **kwargs) -> None:
        """Schedule a GUI update from any thread.
        
        Args:
            callback: Function to call on main thread
            *args: Arguments for callback
            **kwargs: Keyword arguments for callback
        """
        def update_wrapper():
            try:
                callback(*args, **kwargs)
            except Exception as e:
                self.logger.error(f"Error in GUI update: {e}")
                
        self.root.after(0, update_wrapper)
        
    def safe_update(self, callback: Callable) -> Callable:
        """Decorator to make any function thread-safe for GUI updates.
        
        Args:
            callback: Function to make thread-safe
            
        Returns:
            Thread-safe wrapper function
        """
        @wraps(callback)
        def wrapper(*args, **kwargs):
            self.schedule_update(callback, *args, **kwargs)
        return wrapper


class ProgressTracker:
    """Thread-safe progress tracking with GUI updates."""
    
    def __init__(self, gui_manager: ThreadSafeGUI, progress_callback: Callable[[float, str], None]) -> None:
        """Initialize progress tracker.
        
        Args:
            gui_manager: Thread-safe GUI manager
            progress_callback: Function to call with (percentage, status)
        """
        self.gui_manager = gui_manager
        self.progress_callback = progress_callback
        self._current_progress = 0.0
        self._lock = threading.Lock()
        
    def update_progress(self, percentage: float, status: str = None) -> None:
        """Update progress from any thread.
        
        Args:
            percentage: Progress percentage (0-100)
            status: Optional status message
        """
        with self._lock:
            self._current_progress = max(0.0, min(100.0, percentage))
            
        self.gui_manager.schedule_update(self.progress_callback, self._current_progress, status)
        
    def increment_progress(self, amount: float, status: str = None) -> None:
        """Increment progress by amount.
        
        Args:
            amount: Amount to increment
            status: Optional status message
        """
        with self._lock:
            new_progress = self._current_progress + amount
            
        self.update_progress(new_progress, status)
        
    def get_current_progress(self) -> float:
        """Get current progress value.
        
        Returns:
            Current progress percentage
        """
        with self._lock:
            return self._current_progress
            
    def reset(self) -> None:
        """Reset progress to 0."""
        self.update_progress(0.0, "Ready")


class WindowsIntegration:
    """Windows-specific integration helpers."""
    
    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows.
        
        Returns:
            True if Windows
        """
        return os.name == 'nt'
        
    @staticmethod
    def open_folder(path: Union[str, Path]) -> bool:
        """Open folder in Windows Explorer.
        
        Args:
            path: Path to folder
            
        Returns:
            True if successful
        """
        try:
            if WindowsIntegration.is_windows():
                subprocess.run(['explorer', str(path)], check=True)
                return True
            else:
                # Unix-like systems
                subprocess.run(['xdg-open', str(path)], check=True)
                return True
        except Exception:
            return False
            
    @staticmethod
    def open_file(file_path: Union[str, Path]) -> bool:
        """Open file with default application.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if successful
        """
        try:
            if WindowsIntegration.is_windows():
                os.startfile(str(file_path))
                return True
            else:
                # Unix-like systems
                subprocess.run(['xdg-open', str(file_path)], check=True)
                return True
        except Exception:
            return False
            
    @staticmethod
    def set_window_icon(window: tk.Tk, icon_path: Union[str, Path]) -> bool:
        """Set window icon.
        
        Args:
            window: Tkinter window
            icon_path: Path to icon file
            
        Returns:
            True if successful
        """
        try:
            if Path(icon_path).exists():
                window.iconbitmap(str(icon_path))
                return True
        except Exception:
            pass
        return False
        
    @staticmethod
    def center_window(window: tk.Tk, width: int = None, height: int = None) -> None:
        """Center window on screen.
        
        Args:
            window: Tkinter window
            width: Window width (uses current if None)
            height: Window height (uses current if None)
        """
        window.update_idletasks()
        
        # Get current or specified dimensions
        if width is None or height is None:
            current_geo = window.geometry()
            if 'x' in current_geo:
                size_part = current_geo.split('+')[0]
                current_width, current_height = map(int, size_part.split('x'))
                width = width or current_width
                height = height or current_height
            else:
                width = width or 800
                height = height or 600
                
        # Calculate center position
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        # Set geometry
        window.geometry(f"{width}x{height}+{x}+{y}")
        
    @staticmethod
    def setup_high_dpi_awareness() -> bool:
        """Setup high DPI awareness on Windows.
        
        Returns:
            True if successful
        """
        if not WindowsIntegration.is_windows():
            return False
            
        try:
            import ctypes
            # Set DPI awareness
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
            return True
        except Exception:
            return False
            
    @staticmethod
    def set_app_user_model_id(app_id: str) -> bool:
        """Set application user model ID for taskbar grouping.
        
        Args:
            app_id: Application ID string
            
        Returns:
            True if successful
        """
        if not WindowsIntegration.is_windows():
            return False
            
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
            return True
        except Exception:
            return False


class DialogHelpers:
    """Helper functions for dialogs and user interaction."""
    
    @staticmethod
    def show_error(title: str, message: str, parent: tk.Widget = None) -> None:
        """Show error dialog.
        
        Args:
            title: Dialog title
            message: Error message
            parent: Parent widget
        """
        from tkinter import messagebox
        messagebox.showerror(title, message, parent=parent)
        
    @staticmethod
    def show_warning(title: str, message: str, parent: tk.Widget = None) -> None:
        """Show warning dialog.
        
        Args:
            title: Dialog title
            message: Warning message
            parent: Parent widget
        """
        from tkinter import messagebox
        messagebox.showwarning(title, message, parent=parent)
        
    @staticmethod
    def show_info(title: str, message: str, parent: tk.Widget = None) -> None:
        """Show info dialog.
        
        Args:
            title: Dialog title
            message: Info message
            parent: Parent widget
        """
        from tkinter import messagebox
        messagebox.showinfo(title, message, parent=parent)
        
    @staticmethod
    def ask_yes_no(title: str, message: str, parent: tk.Widget = None) -> bool:
        """Ask yes/no question.
        
        Args:
            title: Dialog title
            message: Question message
            parent: Parent widget
            
        Returns:
            True if yes selected
        """
        from tkinter import messagebox
        return messagebox.askyesno(title, message, parent=parent)
        
    @staticmethod
    def ask_ok_cancel(title: str, message: str, parent: tk.Widget = None) -> bool:
        """Ask OK/Cancel question.
        
        Args:
            title: Dialog title
            message: Question message
            parent: Parent widget
            
        Returns:
            True if OK selected
        """
        from tkinter import messagebox
        return messagebox.askokcancel(title, message, parent=parent)


class ValidationHelpers:
    """Helper functions for input validation with visual feedback."""
    
    @staticmethod
    def validate_and_style_entry(entry: ttk.Entry, validator: Callable[[str], bool], 
                                valid_style: str = "TEntry", invalid_style: str = "Invalid.TEntry") -> bool:
        """Validate entry and apply styling.
        
        Args:
            entry: Entry widget to validate
            validator: Function that returns True if valid
            valid_style: Style for valid input
            invalid_style: Style for invalid input
            
        Returns:
            True if valid
        """
        value = entry.get().strip()
        is_valid = validator(value) if value else True  # Empty is neutral
        
        try:
            style = valid_style if is_valid else invalid_style
            entry.config(style=style)
        except tk.TclError:
            # Style doesn't exist, ignore
            pass
            
        return is_valid
        
    @staticmethod
    def create_validation_indicator(parent: tk.Widget, text: str = "") -> ttk.Label:
        """Create a validation indicator label.
        
        Args:
            parent: Parent widget
            text: Initial text
            
        Returns:
            Label widget for validation indicator
        """
        label = ttk.Label(parent, text=text, font=('Arial', 9, 'bold'))
        return label
        
    @staticmethod
    def update_validation_indicator(label: ttk.Label, is_valid: bool, 
                                  valid_text: str = "✓", invalid_text: str = "✗",
                                  valid_color: str = "green", invalid_color: str = "red") -> None:
        """Update validation indicator.
        
        Args:
            label: Label widget
            is_valid: Whether input is valid
            valid_text: Text for valid state
            invalid_text: Text for invalid state
            valid_color: Color for valid state
            invalid_color: Color for invalid state
        """
        if is_valid:
            label.config(text=valid_text, foreground=valid_color)
        else:
            label.config(text=invalid_text, foreground=invalid_color)


class KeyboardShortcuts:
    """Keyboard shortcut management."""
    
    def __init__(self, root: tk.Tk) -> None:
        """Initialize keyboard shortcuts manager.
        
        Args:
            root: Root window
        """
        self.root = root
        self.shortcuts: Dict[str, Callable] = {}
        
    def add_shortcut(self, key_sequence: str, callback: Callable, description: str = "") -> None:
        """Add keyboard shortcut.
        
        Args:
            key_sequence: Key sequence (e.g., '<Control-s>')
            callback: Function to call
            description: Description for help
        """
        self.shortcuts[key_sequence] = {
            'callback': callback,
            'description': description
        }
        self.root.bind(key_sequence, lambda e: callback())
        
    def remove_shortcut(self, key_sequence: str) -> None:
        """Remove keyboard shortcut.
        
        Args:
            key_sequence: Key sequence to remove
        """
        if key_sequence in self.shortcuts:
            self.root.unbind(key_sequence)
            del self.shortcuts[key_sequence]
            
    def get_shortcuts_help(self) -> str:
        """Get help text for all shortcuts.
        
        Returns:
            Formatted help text
        """
        if not self.shortcuts:
            return "No keyboard shortcuts defined."
            
        lines = ["Keyboard Shortcuts:", ""]
        for key_seq, info in self.shortcuts.items():
            desc = info.get('description', 'No description')
            # Clean up key sequence for display
            display_key = key_seq.replace('<', '').replace('>', '').replace('Control', 'Ctrl')
            lines.append(f"• {display_key}: {desc}")
            
        return "\n".join(lines)


class StyleManager:
    """Manage TTK styles for consistent appearance."""
    
    def __init__(self) -> None:
        """Initialize style manager."""
        self.style = ttk.Style()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self) -> None:
        """Setup custom TTK styles."""
        try:
            # Invalid entry style (red border would require theme modification)
            self.style.configure('Invalid.TEntry', fieldbackground='#ffe6e6')
            
            # Accent button style
            self.style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
            
            # Success style
            self.style.configure('Success.TLabel', foreground='green')
            
            # Error style
            self.style.configure('Error.TLabel', foreground='red')
            
            # Warning style
            self.style.configure('Warning.TLabel', foreground='orange')
            
        except Exception:
            # Style configuration failed, continue with defaults
            pass
            
    def apply_theme(self, theme_name: str) -> bool:
        """Apply TTK theme.
        
        Args:
            theme_name: Theme name
            
        Returns:
            True if successful
        """
        try:
            available_themes = self.style.theme_names()
            if theme_name in available_themes:
                self.style.theme_use(theme_name)
                self._setup_custom_styles()  # Reapply custom styles
                return True
        except Exception:
            pass
        return False
        
    def get_available_themes(self) -> list:
        """Get list of available themes.
        
        Returns:
            List of theme names
        """
        try:
            return list(self.style.theme_names())
        except Exception:
            return ['default']


# Global instances for easy access
_thread_safe_gui: Optional[ThreadSafeGUI] = None
_style_manager: Optional[StyleManager] = None


def get_thread_safe_gui(root: tk.Tk = None) -> ThreadSafeGUI:
    """Get global thread-safe GUI manager.
    
    Args:
        root: Root window (required on first call)
        
    Returns:
        Thread-safe GUI manager instance
    """
    global _thread_safe_gui
    if _thread_safe_gui is None:
        if root is None:
            raise ValueError("Root window required for first call")
        _thread_safe_gui = ThreadSafeGUI(root)
    return _thread_safe_gui


def get_style_manager() -> StyleManager:
    """Get global style manager.
    
    Returns:
        Style manager instance
    """
    global _style_manager
    if _style_manager is None:
        _style_manager = StyleManager()
    return _style_manager


# Utility functions for common tasks
def safe_gui_call(root: tk.Tk, callback: Callable, *args, **kwargs) -> None:
    """Safely call GUI function from any thread.
    
    Args:
        root: Root window
        callback: Function to call
        *args: Arguments
        **kwargs: Keyword arguments
    """
    gui_manager = get_thread_safe_gui(root)
    gui_manager.schedule_update(callback, *args, **kwargs)


def setup_windows_app(root: tk.Tk, app_name: str = "YouTubeWhisperTranscriber", 
                     icon_path: str = None) -> None:
    """Setup Windows-specific application features.
    
    Args:
        root: Root window
        app_name: Application name for taskbar
        icon_path: Path to icon file
    """
    if WindowsIntegration.is_windows():
        # Setup high DPI awareness
        WindowsIntegration.setup_high_dpi_awareness()
        
        # Set app ID for taskbar grouping
        WindowsIntegration.set_app_user_model_id(f"{app_name}.Desktop.1.0")
        
        # Set icon if provided
        if icon_path:
            WindowsIntegration.set_window_icon(root, icon_path)


def create_progress_tracker(root: tk.Tk, progress_callback: Callable[[float, str], None]) -> ProgressTracker:
    """Create a thread-safe progress tracker.
    
    Args:
        root: Root window
        progress_callback: Function to call with progress updates
        
    Returns:
        Progress tracker instance
    """
    gui_manager = get_thread_safe_gui(root)
    return ProgressTracker(gui_manager, progress_callback)
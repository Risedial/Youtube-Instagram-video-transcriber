"""Reusable UI components for YouTube Whisper Transcriber.

This module contains professional GUI components and widgets with comprehensive
functionality for the desktop application.
"""

from typing import Optional, Callable, List, Any, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import re
import threading
from utils.url_validator import URLValidator, SupportedPlatform


class ProgressBar:
    """Custom progress bar component with percentage display and status text."""
    
    def __init__(self, parent: tk.Widget, width: int = 400) -> None:
        """Initialize progress bar component.
        
        Args:
            parent: Parent widget
            width: Width of progress bar in pixels
        """
        self.parent = parent
        self.width = width
        self._progress_var = tk.DoubleVar(value=0.0)
        self._status_var = tk.StringVar(value="Ready")
        self._percentage_var = tk.StringVar(value="0%")
        
        self._create_widgets()
        
    def _create_widgets(self) -> None:
        """Create progress bar widgets."""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        
        # Progress bar
        self.progressbar = ttk.Progressbar(
            self.frame,
            length=self.width,
            mode='determinate',
            variable=self._progress_var
        )
        self.progressbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Status frame for percentage and message
        status_frame = ttk.Frame(self.frame)
        status_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # Percentage label
        self.percentage_label = ttk.Label(
            status_frame,
            textvariable=self._percentage_var,
            font=('Arial', 9, 'bold')
        )
        self.percentage_label.pack(side=tk.RIGHT)
        
        # Status message label
        self.status_label = ttk.Label(
            status_frame,
            textvariable=self._status_var,
            font=('Arial', 9)
        )
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def pack(self, **kwargs) -> None:
        """Pack the progress bar frame."""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs) -> None:
        """Grid the progress bar frame."""
        self.frame.grid(**kwargs)
        
    def set_progress(self, percentage: float, status: str = None) -> None:
        """Update progress bar percentage and status.
        
        Args:
            percentage: Progress percentage (0-100)
            status: Optional status message
        """
        # Clamp percentage to 0-100
        percentage = max(0.0, min(100.0, percentage))
        
        self._progress_var.set(percentage)
        self._percentage_var.set(f"{percentage:.1f}%")
        
        if status is not None:
            self._status_var.set(status)
            
    def set_status(self, status: str) -> None:
        """Set status message without changing progress.
        
        Args:
            status: Status message
        """
        self._status_var.set(status)
        
    def reset(self) -> None:
        """Reset progress bar to 0%."""
        self.set_progress(0.0, "Ready")
        
    def set_indeterminate(self, status: str = "Processing...") -> None:
        """Set progress bar to indeterminate mode.
        
        Args:
            status: Status message
        """
        self.progressbar.config(mode='indeterminate')
        self.progressbar.start(10)
        self._percentage_var.set("...")
        self._status_var.set(status)
        
    def set_determinate(self) -> None:
        """Set progress bar back to determinate mode."""
        self.progressbar.stop()
        self.progressbar.config(mode='determinate')
        self.reset()


class FileSelector:
    """File/directory selector component with browse button."""
    
    def __init__(self, parent: tk.Widget, mode: str = "directory", label: str = "Select") -> None:
        """Initialize file selector component.
        
        Args:
            parent: Parent widget
            mode: Selection mode - 'directory', 'file', or 'save'
            label: Label for the selector
        """
        self.parent = parent
        self.mode = mode
        self.label = label
        self._path_var = tk.StringVar()
        self._validation_callback: Optional[Callable[[str], bool]] = None
        
        self._create_widgets()
        
    def _create_widgets(self) -> None:
        """Create file selector widgets."""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        
        # Label
        if self.label:
            label = ttk.Label(self.frame, text=self.label + ":")
            label.pack(side=tk.TOP, anchor=tk.W, pady=(0, 2))
            
        # Path selection frame
        path_frame = ttk.Frame(self.frame)
        path_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Path entry
        self.path_entry = ttk.Entry(
            path_frame,
            textvariable=self._path_var,
            state='readonly',
            font=('Arial', 9)
        )
        self.path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        # Browse button
        self.browse_button = ttk.Button(
            path_frame,
            text="Browse...",
            command=self._browse_path,
            width=10
        )
        self.browse_button.pack(side=tk.RIGHT)
        
        # Clear button
        self.clear_button = ttk.Button(
            path_frame,
            text="Clear",
            command=self.clear,
            width=8
        )
        self.clear_button.pack(side=tk.RIGHT, padx=(0, 5))
        
    def pack(self, **kwargs) -> None:
        """Pack the file selector frame."""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs) -> None:
        """Grid the file selector frame."""
        self.frame.grid(**kwargs)
        
    def _browse_path(self) -> None:
        """Open file/directory browser."""
        try:
            if self.mode == "directory":
                path = filedialog.askdirectory(
                    title=f"Select {self.label}",
                    initialdir=self.get_path() or Path.home()
                )
            elif self.mode == "file":
                path = filedialog.askopenfilename(
                    title=f"Select {self.label}",
                    initialdir=(Path(self.get_path()).parent if self.get_path() else Path.home())
                )
            elif self.mode == "save":
                path = filedialog.asksaveasfilename(
                    title=f"Save {self.label}",
                    initialdir=(Path(self.get_path()).parent if self.get_path() else Path.home())
                )
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
                
            if path:
                self.set_path(path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to browse for path: {e}")
            
    def get_path(self) -> Optional[str]:
        """Get currently selected path.
        
        Returns:
            Selected path or None if empty
        """
        path = self._path_var.get().strip()
        return path if path else None
        
    def set_path(self, path: str) -> None:
        """Set path value.
        
        Args:
            path: Path to set
        """
        if path:
            self._path_var.set(str(Path(path)))
            if self._validation_callback:
                self._validation_callback(path)
        else:
            self.clear()
            
    def clear(self) -> None:
        """Clear path selection."""
        self._path_var.set("")
        
    def set_validation_callback(self, callback: Callable[[str], bool]) -> None:
        """Set validation callback for path changes.
        
        Args:
            callback: Function that takes path and returns validity
        """
        self._validation_callback = callback
        
    def is_valid(self) -> bool:
        """Check if current path is valid.
        
        Returns:
            True if path is valid
        """
        path = self.get_path()
        if not path:
            return False
            
        path_obj = Path(path)
        
        if self.mode == "directory":
            return path_obj.is_dir()
        elif self.mode == "file":
            return path_obj.is_file()
        elif self.mode == "save":
            return path_obj.parent.is_dir()
            
        return False


class ModelSelector:
    """Whisper model selection dropdown component with performance information."""
    
    MODEL_INFO = {
        "tiny": {
            "size": "39 MB",
            "speed": "~32x realtime",
            "memory": "~1 GB",
            "description": "Fastest, lowest accuracy"
        },
        "base": {
            "size": "74 MB", 
            "speed": "~16x realtime",
            "memory": "~1 GB",
            "description": "Good speed, decent accuracy"
        },
        "small": {
            "size": "244 MB",
            "speed": "~6x realtime", 
            "memory": "~2 GB",
            "description": "Balanced speed and accuracy"
        },
        "medium": {
            "size": "769 MB",
            "speed": "~2x realtime",
            "memory": "~5 GB", 
            "description": "Good accuracy, slower"
        },
        "large": {
            "size": "1550 MB",
            "speed": "~1x realtime",
            "memory": "~10 GB",
            "description": "Best accuracy, slowest"
        }
    }
    
    def __init__(self, parent: tk.Widget, label: str = "Whisper Model") -> None:
        """Initialize model selector component.
        
        Args:
            parent: Parent widget
            label: Label for the selector
        """
        self.parent = parent
        self.label = label
        self.models = list(self.MODEL_INFO.keys())
        self._model_var = tk.StringVar(value="tiny")
        self._change_callback: Optional[Callable[[str], None]] = None
        
        self._create_widgets()
        
    def _create_widgets(self) -> None:
        """Create model selector widgets."""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        
        # Label
        if self.label:
            label = ttk.Label(self.frame, text=self.label + ":")
            label.pack(side=tk.TOP, anchor=tk.W, pady=(0, 2))
            
        # Selection frame
        selection_frame = ttk.Frame(self.frame)
        selection_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Model dropdown
        self.model_combo = ttk.Combobox(
            selection_frame,
            textvariable=self._model_var,
            values=self.models,
            state='readonly',
            width=15
        )
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self._on_model_changed)
        
        # Info label
        self.info_label = ttk.Label(
            selection_frame,
            text=self._get_model_info("tiny"),
            font=('Arial', 9),
            foreground='gray'
        )
        self.info_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
    def pack(self, **kwargs) -> None:
        """Pack the model selector frame."""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs) -> None:
        """Grid the model selector frame."""
        self.frame.grid(**kwargs)
        
    def _get_model_info(self, model: str) -> str:
        """Get formatted model information.
        
        Args:
            model: Model name
            
        Returns:
            Formatted info string
        """
        if model in self.MODEL_INFO:
            info = self.MODEL_INFO[model]
            return f"{info['size']} | {info['speed']} | {info['memory']} | {info['description']}"
        return ""
        
    def _on_model_changed(self, event=None) -> None:
        """Handle model selection change."""
        model = self.get_selected_model()
        self.info_label.config(text=self._get_model_info(model))
        
        if self._change_callback:
            self._change_callback(model)
            
    def get_selected_model(self) -> str:
        """Get currently selected model.
        
        Returns:
            Selected model name
        """
        return self._model_var.get()
        
    def set_model(self, model: str) -> None:
        """Set selected model.
        
        Args:
            model: Model name to select
        """
        if model in self.models:
            self._model_var.set(model)
            self._on_model_changed()
        else:
            raise ValueError(f"Invalid model: {model}")
            
    def set_change_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for model changes.
        
        Args:
            callback: Function that takes model name
        """
        self._change_callback = callback
        
    def get_model_info(self, model: str = None) -> Dict[str, str]:
        """Get model information dictionary.
        
        Args:
            model: Model name, uses current selection if None
            
        Returns:
            Model information dictionary
        """
        model = model or self.get_selected_model()
        return self.MODEL_INFO.get(model, {})


class URLInput:
    """Video URL input component with real-time validation for YouTube and Instagram."""
    
    def __init__(self, parent: tk.Widget, label: str = "Video URL") -> None:
        """Initialize URL input component.
        
        Args:
            parent: Parent widget
            label: Label for the input
        """
        self.parent = parent
        self.label = label
        self._url_var = tk.StringVar()
        self._validation_callback: Optional[Callable[[str, bool], None]] = None
        self._is_valid = False
        
        self._create_widgets()
        self._setup_validation()
        
    def _create_widgets(self) -> None:
        """Create URL input widgets."""
        # Main frame
        self.frame = ttk.Frame(self.parent)
        
        # Label frame
        label_frame = ttk.Frame(self.frame)
        label_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        
        # Label
        if self.label:
            label = ttk.Label(label_frame, text=self.label + ":")
            label.pack(side=tk.LEFT)
            
        # Validation indicator
        self.validation_label = ttk.Label(
            label_frame,
            text="",
            font=('Arial', 9, 'bold')
        )
        self.validation_label.pack(side=tk.RIGHT)
        
        # URL entry
        self.url_entry = ttk.Entry(
            self.frame,
            textvariable=self._url_var,
            font=('Arial', 10)
        )
        self.url_entry.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        # Buttons frame
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Paste button
        self.paste_button = ttk.Button(
            buttons_frame,
            text="Paste",
            command=self._paste_url,
            width=8
        )
        self.paste_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Clear button
        self.clear_button = ttk.Button(
            buttons_frame,
            text="Clear",
            command=self.clear,
            width=8
        )
        self.clear_button.pack(side=tk.LEFT)
        
        # Example text
        example_label = ttk.Label(
            self.frame,
            text="Example: https://www.youtube.com/watch?v=dQw4w9WgXcQ or https://www.instagram.com/reel/ABC123/",
            font=('Arial', 8),
            foreground='gray'
        )
        example_label.pack(side=tk.TOP, anchor=tk.W, pady=(2, 0))
        
    def pack(self, **kwargs) -> None:
        """Pack the URL input frame."""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs) -> None:
        """Grid the URL input frame."""
        self.frame.grid(**kwargs)
        
    def _setup_validation(self) -> None:
        """Setup real-time URL validation."""
        def validate(*args):
            self._validate_url()
            
        self._url_var.trace('w', validate)
        
    def _validate_url(self) -> None:
        """Validate current URL and update indicators."""
        url = self.get_url()
        
        if not url:
            # Empty URL
            self._is_valid = False
            self.validation_label.config(text="", foreground="black")
            self.url_entry.config(style="TEntry")
        else:
            # Check URL validity
            self._is_valid = self._is_supported_url(url)
            
            if self._is_valid:
                self.validation_label.config(text="✓ Valid", foreground="green")
                self.url_entry.config(style="TEntry")
            else:
                self.validation_label.config(text="✗ Invalid", foreground="red")
                # Note: Would need custom style for red border
                
        # Call validation callback
        if self._validation_callback:
            self._validation_callback(url, self._is_valid)
            
    def _is_supported_url(self, url: str) -> bool:
        """Check if URL is a valid supported platform URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if valid supported URL (YouTube or Instagram)
        """
        if not url:
            return False
            
        # Use comprehensive validator
        is_valid, platform = URLValidator.validate_url(url)
        return is_valid
        
    def _paste_url(self) -> None:
        """Paste URL from clipboard."""
        try:
            clipboard_text = self.parent.clipboard_get()
            if clipboard_text:
                self.set_url(clipboard_text.strip())
        except tk.TclError:
            pass  # Clipboard empty or unavailable
            
    def get_url(self) -> str:
        """Get entered URL.
        
        Returns:
            Current URL
        """
        return self._url_var.get().strip()
        
    def set_url(self, url: str) -> None:
        """Set URL value.
        
        Args:
            url: URL to set
        """
        self._url_var.set(url)
        
    def clear(self) -> None:
        """Clear URL input."""
        self._url_var.set("")
        
    def validate(self) -> bool:
        """Validate current URL.
        
        Returns:
            True if URL is valid
        """
        return self._is_valid
        
    def set_validation_callback(self, callback: Callable[[str, bool], None]) -> None:
        """Set validation callback.
        
        Args:
            callback: Function that takes (url, is_valid)
        """
        self._validation_callback = callback
        
    def focus(self) -> None:
        """Focus the URL entry."""
        self.url_entry.focus_set()


class StatusDisplay:
    """Status message display component with color-coded messages."""
    
    STATUS_COLORS = {
        'info': '#0066CC',      # Blue
        'success': '#009900',   # Green
        'warning': '#FF9900',   # Orange
        'error': '#CC0000',     # Red
        'processing': '#6600CC' # Purple
    }
    
    def __init__(self, parent: tk.Widget, height: int = 3) -> None:
        """Initialize status display component.
        
        Args:
            parent: Parent widget
            height: Height in lines
        """
        self.parent = parent
        self.height = height
        self._current_type = 'info'
        
        self._create_widgets()
        
    def _create_widgets(self) -> None:
        """Create status display widgets."""
        # Main frame with border
        self.frame = ttk.LabelFrame(self.parent, text="Status", padding=5)
        
        # Status text widget
        self.status_text = tk.Text(
            self.frame,
            height=self.height,
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            borderwidth=0,
            font=('Arial', 9)
        )
        self.status_text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for long messages
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configure text tags for different status types
        for status_type, color in self.STATUS_COLORS.items():
            self.status_text.tag_configure(status_type, foreground=color)
            
    def pack(self, **kwargs) -> None:
        """Pack the status display frame."""
        self.frame.pack(**kwargs)
        
    def grid(self, **kwargs) -> None:
        """Grid the status display frame."""
        self.frame.grid(**kwargs)
        
    def set_status(self, message: str, status_type: str = "info") -> None:
        """Set status message.
        
        Args:
            message: Status message to display
            status_type: Type of status - 'info', 'warning', 'error', 'success', 'processing'
        """
        self._current_type = status_type
        
        # Enable text widget for editing
        self.status_text.config(state=tk.NORMAL)
        
        # Clear existing content
        self.status_text.delete(1.0, tk.END)
        
        # Insert new message with appropriate tag
        self.status_text.insert(tk.END, message, status_type)
        
        # Disable text widget
        self.status_text.config(state=tk.DISABLED)
        
        # Auto-scroll to bottom
        self.status_text.see(tk.END)
        
    def append_status(self, message: str, status_type: str = None) -> None:
        """Append message to existing status.
        
        Args:
            message: Message to append
            status_type: Type of status, uses current if None
        """
        status_type = status_type or self._current_type
        
        # Enable text widget for editing
        self.status_text.config(state=tk.NORMAL)
        
        # Add newline and message
        self.status_text.insert(tk.END, f"\n{message}", status_type)
        
        # Disable text widget
        self.status_text.config(state=tk.DISABLED)
        
        # Auto-scroll to bottom
        self.status_text.see(tk.END)
        
    def clear(self) -> None:
        """Clear status display."""
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state=tk.DISABLED)
        self._current_type = 'info'
        
    def get_text(self) -> str:
        """Get current status text.
        
        Returns:
            Current status text
        """
        return self.status_text.get(1.0, tk.END).strip()


# Component factory functions for easier usage
def create_progress_bar(parent: tk.Widget, width: int = 400) -> ProgressBar:
    """Create a new progress bar component."""
    return ProgressBar(parent, width)


def create_file_selector(parent: tk.Widget, mode: str = "directory", label: str = "Select") -> FileSelector:
    """Create a new file selector component."""
    return FileSelector(parent, mode, label)


def create_model_selector(parent: tk.Widget, label: str = "Whisper Model") -> ModelSelector:
    """Create a new model selector component."""
    return ModelSelector(parent, label)


def create_url_input(parent: tk.Widget, label: str = "Video URL") -> URLInput:
    """Create a new URL input component."""
    return URLInput(parent, label)


def create_status_display(parent: tk.Widget, height: int = 3) -> StatusDisplay:
    """Create a new status display component."""
    return StatusDisplay(parent, height)
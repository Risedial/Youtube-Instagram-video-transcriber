"""Main application window for YouTube Whisper Transcriber.

This module contains the primary GUI implementation using Tkinter with professional
Windows desktop application behavior and comprehensive functionality.
"""

from typing import Optional, Callable, Dict, Any
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import logging
import threading
from enum import Enum

from .components import (
    URLInput, FileSelector, ModelSelector, ProgressBar, StatusDisplay,
    create_url_input, create_file_selector, create_model_selector, 
    create_progress_bar, create_status_display
)
from .proxy_settings_dialog import show_proxy_settings
from config.settings import get_settings
from utils.validators import validate_youtube_url, validate_file_path
from controller.app_controller import TranscriptionWorkflow, WorkflowStage, WorkflowProgress


class ApplicationState(Enum):
    """Application processing states."""
    IDLE = "idle"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class MainWindow:
    """Main application window class.
    
    Implements a professional Windows desktop application with:
    - YouTube URL input and real-time validation
    - Output directory selection with browse dialog
    - Whisper model selection with performance information
    - Progress tracking with detailed status updates
    - Start/stop controls with proper state management
    - Professional appearance and behavior
    """
    
    def __init__(self, master: Optional[tk.Tk] = None) -> None:
        """Initialize the main window.
        
        Args:
            master: Parent Tkinter window, if None creates new root window
        """
        self.master = master or tk.Tk()
        self.logger = logging.getLogger(__name__)
        self.settings = get_settings()
        
        # Application state
        self.current_state = ApplicationState.IDLE
        self.processing_thread: Optional[threading.Thread] = None
        self.cancel_event = threading.Event()
        
        # Transcription workflow
        self.workflow: Optional[TranscriptionWorkflow] = None
        self._initialize_workflow()
        
        # GUI components
        self.url_input: Optional[URLInput] = None
        self.output_selector: Optional[FileSelector] = None
        self.model_selector: Optional[ModelSelector] = None
        self.progress_bar: Optional[ProgressBar] = None
        self.status_display: Optional[StatusDisplay] = None
        self.start_button: Optional[ttk.Button] = None
        self.stop_button: Optional[ttk.Button] = None

        # Format selection radio button (mutually exclusive choice)
        # Values: "include" = with timestamps, "exclude" = without timestamps
        self.timestamp_format_var = tk.StringVar(value="include")
        
        # Processing callbacks
        self.start_callback: Optional[Callable] = None
        self.stop_callback: Optional[Callable] = None
        
        self._setup_window()
        self._create_menu_bar()
        self._create_widgets()
        self._load_user_preferences()
        self._update_ui_state()
        
    def _initialize_workflow(self) -> None:
        """Initialize the transcription workflow."""
        try:
            # Get default model from settings
            default_model = self.settings.get_setting('default_whisper_model', 'tiny')
            default_device = self.settings.get_setting('whisper_device', 'auto')
            
            # Create workflow instance
            self.workflow = TranscriptionWorkflow(
                whisper_model=default_model,
                whisper_device=default_device
            )
            
            # Set up callbacks
            self.workflow.set_progress_callback(self._on_workflow_progress)
            self.workflow.set_status_callback(self._on_workflow_status)
            
            self.logger.info(f"Workflow initialized with model: {default_model}, device: {default_device}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize workflow: {e}")
            # We'll handle this gracefully and show an error to the user
            self.workflow = None
        
    def _setup_window(self) -> None:
        """Configure main window properties."""
        self.master.title("Video Whisper Transcriber v1.0")
        self.master.geometry("800x800")
        self.master.minsize(600, 500)
        
        # Set window icon (placeholder for future implementation)
        try:
            # Future: Add application icon
            pass
        except Exception:
            pass
            
        # Center window on screen
        self._center_window()
        
        # Configure window behavior
        self.master.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.master.resizable(True, True)
        
        # Configure grid weights for responsive layout
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)
        
    def _center_window(self) -> None:
        """Center window on screen."""
        self.master.update_idletasks()
        
        # Get window dimensions
        width = self.settings.get_setting('window_width', 800)
        height = self.settings.get_setting('window_height', 800)
        
        # Calculate center position
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        
        # Set geometry
        self.master.geometry(f"{width}x{height}+{x}+{y}")
    
    def _create_menu_bar(self) -> None:
        """Create menu bar with settings and help options."""
        menubar = tk.Menu(self.master)
        self.master.config(menu=menubar)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Proxy/VPN Settings...", command=self._show_proxy_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self._show_help)
        help_menu.add_command(label="Keyboard Shortcuts", command=self._show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._show_about)
        
    def _create_widgets(self) -> None:
        """Create and layout GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.master, padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Status display expands
        
        # Title section
        self._create_title_section(main_frame)
        
        # Input section
        self._create_input_section(main_frame)
        
        # Progress section
        self._create_progress_section(main_frame)
        
        # Status section
        self._create_status_section(main_frame)
        
        # Control buttons section
        self._create_controls_section(main_frame)
        
    def _create_title_section(self, parent: ttk.Frame) -> None:
        """Create title and description section.
        
        Args:
            parent: Parent frame
        """
        title_frame = ttk.Frame(parent)
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(0, weight=1)
        
        # Application title
        title_label = ttk.Label(
            title_frame,
            text="YouTube/Instagram Video Transcriber",
            font=("Arial", 18, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 5))
        
        # Description
        desc_label = ttk.Label(
            title_frame,
            text="Convert YouTube and Instagram videos to text files for free",
            font=("Arial", 11),
            foreground="gray"
        )
        desc_label.grid(row=1, column=0)
        
    def _create_input_section(self, parent: ttk.Frame) -> None:
        """Create input controls section.
        
        Args:
            parent: Parent frame
        """
        input_frame = ttk.LabelFrame(parent, text="Input Settings", padding=10)
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(0, weight=1)
        
        # Video URL input
        self.url_input = create_url_input(input_frame, "Video URL")
        self.url_input.pack(fill=tk.X, pady=(0, 10))
        self.url_input.set_validation_callback(self._on_url_changed)
        
        # Output directory selector
        self.output_selector = create_file_selector(
            input_frame, mode="directory", label="Output Directory"
        )
        self.output_selector.pack(fill=tk.X, pady=(0, 10))
        self.output_selector.set_validation_callback(self._on_output_changed)
        
        # Whisper model selector
        self.model_selector = create_model_selector(input_frame, "Whisper Model")
        self.model_selector.pack(fill=tk.X, pady=(0, 10))
        self.model_selector.set_change_callback(self._on_model_changed)

        # Output format selection radio buttons (mutually exclusive)
        format_label = ttk.Label(input_frame, text="Timestamp Format:")
        format_label.pack(anchor=tk.W, pady=(0, 5))

        format_frame = ttk.Frame(input_frame)
        format_frame.pack(fill=tk.X)

        # Include timestamps radio button
        self.include_timestamps_radio = ttk.Radiobutton(
            format_frame,
            text="Include Timestamps",
            variable=self.timestamp_format_var,
            value="include",
            command=self._on_format_change
        )
        self.include_timestamps_radio.pack(side=tk.LEFT, padx=(0, 15))

        # Exclude timestamps radio button
        self.exclude_timestamps_radio = ttk.Radiobutton(
            format_frame,
            text="Exclude Timestamps",
            variable=self.timestamp_format_var,
            value="exclude",
            command=self._on_format_change
        )
        self.exclude_timestamps_radio.pack(side=tk.LEFT)
        
    def _create_progress_section(self, parent: ttk.Frame) -> None:
        """Create progress tracking section.
        
        Args:
            parent: Parent frame
        """
        progress_frame = ttk.LabelFrame(parent, text="Progress", padding=10)
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        progress_frame.columnconfigure(0, weight=1)
        
        # Progress bar with status
        self.progress_bar = create_progress_bar(progress_frame, width=500)
        self.progress_bar.pack(fill=tk.X)
        
    def _create_status_section(self, parent: ttk.Frame) -> None:
        """Create status display section.
        
        Args:
            parent: Parent frame
        """
        # Status display (expandable)
        self.status_display = create_status_display(parent, height=6)
        self.status_display.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 15))
        
        # Initial status message
        self.status_display.set_status(
            "Welcome to YouTube/Instagram Video Transcriber!\n\n"
            "Instructions:\n"
            "1. Enter a YouTube or Instagram video URL\n"
            "2. Select an output directory\n"
            "3. Choose a Whisper model\n"
            "4. Click 'Start Transcription' to begin\n\n"
            "Ready to process your first video.",
            "info"
        )
        
    def _create_controls_section(self, parent: ttk.Frame) -> None:
        """Create control buttons section.
        
        Args:
            parent: Parent frame
        """
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=4, column=0, sticky=(tk.W, tk.E))
        controls_frame.columnconfigure(0, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side=tk.RIGHT)
        
        # Start button
        self.start_button = ttk.Button(
            button_frame,
            text="Start Transcription",
            command=self._on_start_clicked,
            style="Accent.TButton",
            width=18
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Stop button
        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self._on_stop_clicked,
            state="disabled",
            width=12
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(
            button_frame,
            text="Clear URL",
            command=self._on_clear_clicked,
            width=12
        )
        clear_button.pack(side=tk.LEFT)
        
        # Keyboard shortcuts info
        shortcuts_label = ttk.Label(
            controls_frame,
            text="Shortcuts: Ctrl+Return (Start) | Escape (Stop) | Ctrl+Delete (Clear URL)",
            font=("Arial", 8),
            foreground="gray"
        )
        shortcuts_label.pack(side=tk.LEFT, anchor=tk.W)
        
        # Setup keyboard shortcuts
        self._setup_keyboard_shortcuts()
        
    def _setup_keyboard_shortcuts(self) -> None:
        """Setup keyboard shortcuts."""
        self.master.bind('<Control-Return>', lambda e: self._on_start_clicked())
        self.master.bind('<Escape>', lambda e: self._on_stop_clicked())
        self.master.bind('<Control-Delete>', lambda e: self._on_clear_clicked())
        self.master.bind('<F1>', lambda e: self._show_help())
        
    def _load_user_preferences(self) -> None:
        """Load user preferences from settings."""
        try:
            # Load default output directory
            default_output = self.settings.get_setting('default_output_directory')
            if default_output and Path(default_output).exists():
                self.output_selector.set_path(default_output)

            # Load default model
            default_model = self.settings.get_setting('default_model', 'tiny')
            self.model_selector.set_model(default_model)

            # Load timestamp format preference
            timestamp_format = self.settings.get_setting('timestamp_format', 'include')
            self.timestamp_format_var.set(timestamp_format)

            self.logger.info("User preferences loaded successfully")

        except Exception as e:
            self.logger.warning(f"Failed to load user preferences: {e}")
            
    def _save_user_preferences(self) -> None:
        """Save current user preferences."""
        try:
            # Save output directory
            output_path = self.output_selector.get_path()
            if output_path:
                self.settings.set_setting('default_output_directory', output_path)

            # Save model selection
            model = self.model_selector.get_selected_model()
            self.settings.set_setting('default_model', model)

            # Save timestamp format preference
            self.settings.set_setting('timestamp_format', self.timestamp_format_var.get())

            # Save window size
            geometry = self.master.geometry()
            if 'x' in geometry:
                size_part = geometry.split('+')[0]
                width, height = map(int, size_part.split('x'))
                self.settings.set_setting('window_width', width)
                self.settings.set_setting('window_height', height)

            self.settings.save_settings()
            self.logger.info("User preferences saved successfully")

        except Exception as e:
            self.logger.warning(f"Failed to save user preferences: {e}")
            
    def _update_ui_state(self) -> None:
        """Update UI state based on current application state."""
        if self.current_state == ApplicationState.IDLE:
            self.start_button.config(state="normal" if self._can_start() else "disabled")
            self.stop_button.config(state="disabled")
            self.progress_bar.reset()
            
        elif self.current_state in [ApplicationState.DOWNLOADING, ApplicationState.TRANSCRIBING]:
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            
        elif self.current_state == ApplicationState.COMPLETED:
            self.start_button.config(state="normal" if self._can_start() else "disabled")
            self.stop_button.config(state="disabled")
            
        elif self.current_state == ApplicationState.ERROR:
            self.start_button.config(state="normal" if self._can_start() else "disabled")
            self.stop_button.config(state="disabled")
            
        elif self.current_state == ApplicationState.CANCELLED:
            self.start_button.config(state="normal" if self._can_start() else "disabled")
            self.stop_button.config(state="disabled")
            self.progress_bar.reset()
            
    def _can_start(self) -> bool:
        """Check if transcription can be started.
        
        Returns:
            True if all requirements are met
        """
        return (
            self.url_input.validate() and
            self.output_selector.is_valid() and
            self.current_state == ApplicationState.IDLE
        )
        
    def _on_url_changed(self, url: str, is_valid: bool) -> None:
        """Handle URL input changes.
        
        Args:
            url: Current URL
            is_valid: Whether URL is valid
        """
        self._update_ui_state()
        
        if url and is_valid:
            self.status_display.set_status(
                f"Valid video URL detected: {url[:50]}{'...' if len(url) > 50 else ''}",
                "success"
            )
        elif url and not is_valid:
            self.status_display.set_status(
                "Invalid video URL format. Please enter a valid YouTube or Instagram video URL.",
                "error"
            )
            
    def _on_output_changed(self, path: str) -> None:
        """Handle output directory changes.
        
        Args:
            path: Selected output path
        """
        self._update_ui_state()
        
        if path:
            self.status_display.set_status(
                f"Output directory set: {path}",
                "success"
            )
            
    def _on_model_changed(self, model: str) -> None:
        """Handle model selection changes.

        Args:
            model: Selected model name
        """
        model_info = self.model_selector.get_model_info(model)
        self.status_display.set_status(
            f"Whisper model '{model}' selected\n"
            f"Size: {model_info.get('size', 'Unknown')} | "
            f"Speed: {model_info.get('speed', 'Unknown')} | "
            f"Memory: {model_info.get('memory', 'Unknown')}",
            "info"
        )

        # Update workflow model if not currently processing
        if self.workflow and self.current_state == ApplicationState.IDLE:
            try:
                self.workflow.change_whisper_model(model)
                self.logger.info(f"Updated workflow to use model: {model}")
            except Exception as e:
                self.logger.warning(f"Failed to update workflow model: {e}")

    def _on_format_change(self) -> None:
        """Handle timestamp format radio button changes."""
        format_choice = self.timestamp_format_var.get()

        # Save format preference immediately
        self.settings.set_setting('timestamp_format', format_choice)
        self.settings.save_settings()

        # Update status display
        if format_choice == "include":
            format_desc = "with timestamps"
        else:  # format_choice == "exclude"
            format_desc = "without timestamps"

        self.status_display.set_status(
            f"Timestamp format updated: {format_desc}",
            "info"
        )
        
    def _on_start_clicked(self) -> None:
        """Handle start button click."""
        if not self._can_start():
            messagebox.showwarning(
                "Cannot Start",
                "Please ensure all fields are filled correctly before starting."
            )
            return

        # Get timestamp format (no validation needed - radio buttons are mutually exclusive)
        timestamp_format = self.timestamp_format_var.get()

        # Get current settings
        url = self.url_input.get_url()
        output_dir = self.output_selector.get_path()
        model = self.model_selector.get_selected_model()

        # Update state
        self.current_state = ApplicationState.DOWNLOADING
        self._update_ui_state()

        # Save preferences
        self._save_user_preferences()

        # Build format description for status display
        if timestamp_format == "include":
            format_desc = "with timestamps"
        else:  # timestamp_format == "exclude"
            format_desc = "without timestamps"

        # Start processing
        self.progress_bar.set_progress(0, "Initializing...")
        self.status_display.set_status(
            f"Starting transcription process...\n"
            f"URL: {url}\n"
            f"Output: {output_dir}\n"
            f"Model: {model}\n"
            f"Format: {format_desc}",
            "processing"
        )

        # Start transcription workflow in background thread
        if self.workflow:
            self.processing_thread = threading.Thread(
                target=self._run_transcription_workflow,
                args=(url, output_dir, model, timestamp_format),
                daemon=True
            )
            self.processing_thread.start()
        else:
            self._on_error("Transcription workflow not available. Please restart the application.")
                
    def _on_stop_clicked(self) -> None:
        """Handle stop button click."""
        if self.current_state in [ApplicationState.DOWNLOADING, ApplicationState.TRANSCRIBING]:
            # Cancel workflow
            if self.workflow:
                self.workflow.cancel_workflow()
            
            # Set cancel event
            self.cancel_event.set()
            
            # Update state
            self.current_state = ApplicationState.CANCELLED
            self._update_ui_state()
            
            self.progress_bar.set_progress(0, "Cancelling...")
            self.status_display.set_status("Transcription cancelled by user.", "warning")
                    
    def _on_clear_clicked(self) -> None:
        """Handle clear button click."""
        if self.current_state not in [ApplicationState.DOWNLOADING, ApplicationState.TRANSCRIBING]:
            # Only clear the URL input field
            self.url_input.clear()
            self.status_display.set_status("URL field cleared. Ready for new transcription.", "info")
            self.current_state = ApplicationState.IDLE
            self._update_ui_state()
            
    def _show_help(self) -> None:
        """Show help dialog."""
        help_text = """Video Whisper Transcriber Help

GETTING STARTED:
1. Enter a YouTube or Instagram video URL in the first field
2. Select an output directory where the transcription will be saved
3. Choose a Whisper model (tiny is fastest, large is most accurate)
4. Choose timestamp format (with or without timestamps)
5. Click 'Start Transcription' to begin

WHISPER MODELS:
• tiny: Fastest processing, basic accuracy (~32x realtime)
• base: Good balance of speed and accuracy (~16x realtime)
• small: Better accuracy, moderate speed (~6x realtime)
• medium: High accuracy, slower processing (~2x realtime)
• large: Best accuracy, slowest processing (~1x realtime)

KEYBOARD SHORTCUTS:
• Ctrl+Enter: Start transcription
• Escape: Stop processing
• Ctrl+Delete: Clear URL field
• F1: Show this help

SUPPORTED PLATFORMS:
• YouTube: All video formats including Shorts
• Instagram: Reels and video posts
The application will automatically extract audio for transcription.

For more information, visit the project documentation."""

        messagebox.showinfo("Help - Video Whisper Transcriber", help_text)
    
    def _show_proxy_settings(self) -> None:
        """Show proxy settings dialog."""
        def on_settings_saved():
            """Handle proxy settings saved callback."""
            # Reinitialize workflow with new proxy settings
            self._initialize_workflow()
            self.status_display.add_message(
                "Proxy settings have been updated and applied.", 
                "success"
            )
        
        show_proxy_settings(
            parent=self.master,
            settings_manager=self.settings,
            callback=on_settings_saved
        )
    
    def _show_app_settings(self) -> None:
        """Show application settings dialog (placeholder)."""
        messagebox.showinfo(
            "Application Settings", 
            "Application settings dialog will be implemented in a future version.\n\n"
            "Currently available settings:\n"
            "• Proxy/VPN Settings (available in Settings menu)\n"
            "• Model selection (available in main window)\n"
            "• Output directory (available in main window)"
        )
    
    def _show_shortcuts(self) -> None:
        """Show keyboard shortcuts dialog."""
        shortcut_text = """Keyboard Shortcuts

MAIN CONTROLS:
• Ctrl+Enter: Start transcription
• Escape: Stop processing
• Ctrl+Delete: Clear URL field
• F1: Show help

NAVIGATION:
• Tab: Move between input fields
• Enter: Activate focused button
• Alt+F4: Close application

SETTINGS:
• Access proxy settings via Settings menu
• Model and timestamp format selection available in main window"""

        messagebox.showinfo("Keyboard Shortcuts", shortcut_text)
    
    def _show_about(self) -> None:
        """Show about dialog."""
        about_text = """Video Whisper Transcriber v1.0

A professional desktop application for converting YouTube and Instagram videos to text transcriptions using OpenAI's Whisper speech recognition models.

FEATURES:
• 100% local processing for complete privacy
• Support for YouTube and Instagram videos
• Multiple Whisper model options
• Professional Windows desktop integration
• Proxy/VPN support for IP protection
• No ongoing costs or subscriptions

POWERED BY:
• OpenAI Whisper for speech recognition
• yt-dlp for video downloading
• Python and Tkinter for the interface

This application processes everything locally on your computer - no data is sent to external servers.

© 2024 - Open Source Software"""

        messagebox.showinfo("About Video Whisper Transcriber", about_text)
        
    def _on_closing(self) -> None:
        """Handle window closing event."""
        if self.current_state in [ApplicationState.DOWNLOADING, ApplicationState.TRANSCRIBING]:
            if messagebox.askokcancel("Quit", "Processing is in progress. Do you want to quit?"):
                self.cancel_event.set()
                self._save_user_preferences()
                self.master.destroy()
        else:
            self._save_user_preferences()
            self.master.destroy()
            
    def _run_transcription_workflow(self, url: str, output_dir: str, model: str,
                                    timestamp_format: str) -> None:
        """Run the transcription workflow in background thread.

        Args:
            url: YouTube URL
            output_dir: Output directory path
            model: Whisper model name
            timestamp_format: Timestamp format ("include" or "exclude")
        """
        try:
            # Update workflow model if changed
            if self.workflow.transcriber.model_name != model:
                self.workflow.change_whisper_model(model)

            # Log format preference
            self.logger.info(f"Timestamp format: {timestamp_format}")

            # Run the complete workflow
            # NOTE: This passes the format preference to the backend
            # The backend (modified in Backend Implementation Plan) will handle this parameter
            try:
                result = self.workflow.transcribe_from_url(
                    url,
                    Path(output_dir),
                    timestamp_format=timestamp_format
                )
            except TypeError:
                # Backend doesn't support format parameter yet, use default call
                self.logger.warning("Backend does not support timestamp_format parameter yet. Using default transcription.")
                result = self.workflow.transcribe_from_url(url, Path(output_dir))

            if result and not self.cancel_event.is_set():
                # Success - workflow completed
                # Determine output file path based on format
                if timestamp_format == "include":
                    output_file = Path(output_dir) / f"{result.language}_transcript_with_timestamps.txt"
                else:  # timestamp_format == "exclude"
                    output_file = Path(output_dir) / f"{result.language}_transcript.txt"

                # Show completion with file path
                self.master.after(0, lambda: self._on_completion(str(output_file)))
            elif self.cancel_event.is_set():
                # Cancelled
                self.master.after(0, lambda: self._on_cancellation())
            else:
                # Failed
                self.master.after(0, lambda: self._on_error("Transcription workflow failed"))

        except Exception as e:
            self.logger.error(f"Workflow error: {e}")
            error_msg = f"Workflow error: {str(e)}"
            self.master.after(0, lambda: self._on_error(error_msg))
            
    def _on_workflow_progress(self, progress: WorkflowProgress) -> None:
        """Handle workflow progress updates.
        
        Args:
            progress: WorkflowProgress object with current status
        """
        # Update UI state based on workflow stage
        if progress.stage == WorkflowStage.DOWNLOADING:
            if self.current_state != ApplicationState.DOWNLOADING:
                self.current_state = ApplicationState.DOWNLOADING
                self.master.after(0, self._update_ui_state)
        elif progress.stage == WorkflowStage.TRANSCRIBING:
            if self.current_state != ApplicationState.TRANSCRIBING:
                self.current_state = ApplicationState.TRANSCRIBING
                self.master.after(0, self._update_ui_state)
                
        # Update progress bar
        self.master.after(0, lambda: self.progress_bar.set_progress(
            progress.overall_percentage, 
            progress.status_message
        ))
        
    def _on_workflow_status(self, message: str, status_type: str) -> None:
        """Handle workflow status updates.
        
        Args:
            message: Status message
            status_type: Type of status (info, error, warning, success)
        """
        self.master.after(0, lambda: self.status_display.set_status(message, status_type))
        
    def _on_cancellation(self) -> None:
        """Handle workflow cancellation."""
        self.current_state = ApplicationState.CANCELLED
        self._update_ui_state()
        self.progress_bar.reset()
        self.status_display.set_status("Transcription cancelled by user.", "warning")
            
    def _on_error(self, error_message: str) -> None:
        """Handle processing errors.
        
        Args:
            error_message: Error message to display
        """
        self.current_state = ApplicationState.ERROR
        self._update_ui_state()
        self.progress_bar.reset()
        self.status_display.set_status(f"Error: {error_message}", "error")
        
    def _on_completion(self, output_files: str) -> None:
        """Handle successful completion.

        Args:
            output_files: Path(s) to generated transcription file(s) (newline-separated if multiple)
        """
        self.current_state = ApplicationState.COMPLETED
        self._update_ui_state()
        self.progress_bar.set_progress(100, "Completed successfully")

        # Parse output files
        file_paths = output_files.split('\n')
        if len(file_paths) > 1:
            files_display = "Output files:\n" + "\n".join([f"  • {f}" for f in file_paths])
        else:
            files_display = f"Output file: {file_paths[0]}"

        self.status_display.set_status(
            f"Transcription completed successfully!\n\n"
            f"{files_display}\n\n"
            f"You can now:\n"
            f"• Open the output files to view the transcriptions\n"
            f"• Start a new transcription with different settings\n"
            f"• Clear the URL field to start fresh",
            "success"
        )

        # Ask user if they want to open the output folder
        if self.settings.get_setting('auto_open_output_folder', True):
            if messagebox.askyesno("Open Output Folder", "Would you like to open the output folder?"):
                try:
                    import subprocess
                    import os
                    if os.name == 'nt':  # Windows
                        subprocess.run(['explorer', Path(file_paths[0]).parent])
                    else:  # Unix-like
                        subprocess.run(['xdg-open', Path(file_paths[0]).parent])
                except Exception as e:
                    self.logger.warning(f"Failed to open output folder: {e}")
                    
    # Public interface for external processing integration
    def set_progress(self, percentage: float, status: str = None) -> None:
        """Update progress from external process.
        
        Args:
            percentage: Progress percentage (0-100)
            status: Optional status message
        """
        self.master.after(0, lambda: self.progress_bar.set_progress(percentage, status))
        
    def set_status(self, message: str, status_type: str = "info") -> None:
        """Update status from external process.
        
        Args:
            message: Status message
            status_type: Type of status message
        """
        self.master.after(0, lambda: self.status_display.set_status(message, status_type))
        
    def append_status(self, message: str, status_type: str = None) -> None:
        """Append to status from external process.
        
        Args:
            message: Message to append
            status_type: Type of status message
        """
        self.master.after(0, lambda: self.status_display.append_status(message, status_type))
        
    def set_start_callback(self, callback: Callable[[str, str, str], None]) -> None:
        """Set callback for start processing.
        
        Args:
            callback: Function that takes (url, output_dir, model)
        """
        self.start_callback = callback
        
    def set_stop_callback(self, callback: Callable[[], None]) -> None:
        """Set callback for stop processing.
        
        Args:
            callback: Function to call when stopping
        """
        self.stop_callback = callback
        
    def on_processing_complete(self, output_file: str) -> None:
        """Called when processing completes successfully.
        
        Args:
            output_file: Path to output file
        """
        self.master.after(0, lambda: self._on_completion(output_file))
        
    def on_processing_error(self, error_message: str) -> None:
        """Called when processing encounters an error.
        
        Args:
            error_message: Error message
        """
        self.master.after(0, lambda: self._on_error(error_message))
        
    def get_cancel_event(self) -> threading.Event:
        """Get the cancellation event for background processing.
        
        Returns:
            Threading event for cancellation
        """
        return self.cancel_event
        
    def run(self) -> None:
        """Start the GUI event loop."""
        try:
            # Focus URL input for immediate use
            self.url_input.focus()
            
            # Start main loop
            self.master.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error in main GUI loop: {e}")
            raise


def main() -> None:
    """Main entry point for GUI application."""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
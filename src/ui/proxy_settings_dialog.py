"""Proxy settings configuration dialog for YouTube Whisper Transcriber.

This module provides a GUI dialog for configuring proxy settings including
proxy type, server details, authentication, and anti-detection features.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any, Callable
import logging
from pathlib import Path

try:
    from config.settings import ApplicationSettings, get_settings
    from utils.proxy_manager import ProxyManager, ProxyType
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False


class ProxySettingsDialog:
    """Proxy configuration dialog with comprehensive settings.
    
    Features:
    - Proxy type selection (HTTP/HTTPS/SOCKS4/SOCKS5)
    - Server and authentication configuration
    - Anti-detection settings
    - Connection testing
    - Proxy list management
    - Kill switch configuration
    """
    
    def __init__(self, parent: tk.Toplevel = None, settings_manager=None, callback: Optional[Callable] = None):
        """Initialize proxy settings dialog.
        
        Args:
            parent: Parent window
            settings_manager: Settings manager instance
            callback: Callback function called when settings are saved
        """
        self.parent = parent
        self.settings_manager = settings_manager or get_settings()
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        
        # Get current settings
        self.current_settings = self.settings_manager._settings if self.settings_manager else ApplicationSettings()
        
        # Create dialog
        self.dialog = tk.Toplevel(parent) if parent else tk.Tk()
        self.dialog.title("Proxy Settings")
        self.dialog.geometry("600x700")
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        if parent:
            self.dialog.transient(parent)
            self.dialog.grab_set()
        
        # Center dialog
        self._center_dialog()
        
        # Create UI
        self._create_widgets()
        self._load_current_settings()
        
        # Bind events
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
    
    def _center_dialog(self):
        """Center the dialog on screen or parent."""
        self.dialog.update_idletasks()
        
        if self.parent:
            # Center on parent
            parent_x = self.parent.winfo_rootx()
            parent_y = self.parent.winfo_rooty()
            parent_width = self.parent.winfo_width()
            parent_height = self.parent.winfo_height()
            
            x = parent_x + (parent_width - 600) // 2
            y = parent_y + (parent_height - 700) // 2
        else:
            # Center on screen
            screen_width = self.dialog.winfo_screenwidth()
            screen_height = self.dialog.winfo_screenheight()
            x = (screen_width - 600) // 2
            y = (screen_height - 700) // 2
        
        self.dialog.geometry(f"600x700+{x}+{y}")
    
    def _create_widgets(self):
        """Create dialog widgets."""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Title
        title_label = ttk.Label(main_frame, text="Proxy Configuration", font=('Arial', 14, 'bold'))
        title_label.grid(row=row, column=0, columnspan=2, pady=(0, 20))
        row += 1
        
        # Enable proxy checkbox
        self.enable_proxy_var = tk.BooleanVar()
        enable_check = ttk.Checkbutton(
            main_frame, 
            text="Enable Proxy/VPN Protection", 
            variable=self.enable_proxy_var,
            command=self._on_enable_proxy_changed
        )
        enable_check.grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Proxy settings frame
        self.proxy_frame = ttk.LabelFrame(main_frame, text="Proxy Server Settings", padding="10")
        self.proxy_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.proxy_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Proxy type
        ttk.Label(self.proxy_frame, text="Proxy Type:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.proxy_type_var = tk.StringVar(value="http")
        proxy_type_combo = ttk.Combobox(
            self.proxy_frame, 
            textvariable=self.proxy_type_var,
            values=["http", "https", "socks4", "socks5"],
            state="readonly",
            width=15
        )
        proxy_type_combo.grid(row=0, column=1, sticky=tk.W)
        
        # Proxy host
        ttk.Label(self.proxy_frame, text="Host:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.proxy_host_var = tk.StringVar()
        host_entry = ttk.Entry(self.proxy_frame, textvariable=self.proxy_host_var, width=30)
        host_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Proxy port
        ttk.Label(self.proxy_frame, text="Port:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.proxy_port_var = tk.StringVar(value="8080")
        port_entry = ttk.Entry(self.proxy_frame, textvariable=self.proxy_port_var, width=10)
        port_entry.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))
        
        # Authentication frame
        auth_frame = ttk.LabelFrame(main_frame, text="Authentication (Optional)", padding="10")
        auth_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        auth_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Username
        ttk.Label(auth_frame, text="Username:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.proxy_username_var = tk.StringVar()
        username_entry = ttk.Entry(auth_frame, textvariable=self.proxy_username_var, width=30)
        username_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Password
        ttk.Label(auth_frame, text="Password:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        self.proxy_password_var = tk.StringVar()
        password_entry = ttk.Entry(auth_frame, textvariable=self.proxy_password_var, show="*", width=30)
        password_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Anti-detection frame
        detection_frame = ttk.LabelFrame(main_frame, text="Anti-Detection Settings", padding="10")
        detection_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        detection_frame.columnconfigure(1, weight=1)
        row += 1
        
        # Random delays
        self.random_delays_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            detection_frame, 
            text="Enable random delays between requests", 
            variable=self.random_delays_var
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Delay range
        ttk.Label(detection_frame, text="Delay Range (seconds):").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        delay_frame = ttk.Frame(detection_frame)
        delay_frame.grid(row=1, column=1, sticky=tk.W, pady=(10, 0))
        
        self.min_delay_var = tk.StringVar(value="30")
        ttk.Entry(delay_frame, textvariable=self.min_delay_var, width=8).grid(row=0, column=0)
        ttk.Label(delay_frame, text=" to ").grid(row=0, column=1)
        self.max_delay_var = tk.StringVar(value="120")
        ttk.Entry(delay_frame, textvariable=self.max_delay_var, width=8).grid(row=0, column=2)
        
        # User agent rotation
        self.rotate_user_agents_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            detection_frame, 
            text="Rotate user agents", 
            variable=self.rotate_user_agents_var
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(10, 0))
        
        # Header randomization
        self.randomize_headers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            detection_frame, 
            text="Randomize HTTP headers", 
            variable=self.randomize_headers_var
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Kill switch frame
        killswitch_frame = ttk.LabelFrame(main_frame, text="Kill Switch Settings", padding="10")
        killswitch_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        killswitch_frame.columnconfigure(1, weight=1)
        row += 1
        
        self.enable_killswitch_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            killswitch_frame, 
            text="Enable kill switch (stop all operations if proxy fails)", 
            variable=self.enable_killswitch_var
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        
        # Connection testing frame
        test_frame = ttk.Frame(main_frame)
        test_frame.grid(row=row, column=0, columnspan=2, pady=(0, 20))
        row += 1
        
        self.test_button = ttk.Button(test_frame, text="Test Connection", command=self._test_connection)
        self.test_button.grid(row=0, column=0, padx=(0, 10))
        
        self.test_status_var = tk.StringVar(value="Not tested")
        test_status_label = ttk.Label(test_frame, textvariable=self.test_status_var)
        test_status_label.grid(row=0, column=1)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=row, column=0, columnspan=2)
        
        ttk.Button(buttons_frame, text="Save", command=self._on_save).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(buttons_frame, text="Cancel", command=self._on_cancel).grid(row=0, column=1)
        
        # Initially disable proxy settings
        self._on_enable_proxy_changed()
    
    def _load_current_settings(self):
        """Load current settings into the dialog."""
        if not self.current_settings:
            return
            
        # Load proxy settings
        self.enable_proxy_var.set(self.current_settings.enable_proxy)
        self.proxy_type_var.set(self.current_settings.proxy_type)
        self.proxy_host_var.set(self.current_settings.proxy_host)
        self.proxy_port_var.set(str(self.current_settings.proxy_port))
        
        if self.current_settings.proxy_username:
            self.proxy_username_var.set(self.current_settings.proxy_username)
        if self.current_settings.proxy_password:
            self.proxy_password_var.set(self.current_settings.proxy_password)
        
        # Load anti-detection settings
        self.random_delays_var.set(self.current_settings.enable_random_delays)
        self.min_delay_var.set(str(self.current_settings.min_request_delay))
        self.max_delay_var.set(str(self.current_settings.max_request_delay))
        self.rotate_user_agents_var.set(self.current_settings.rotate_user_agents)
        self.randomize_headers_var.set(self.current_settings.enable_header_randomization)
        
        # Load kill switch settings
        self.enable_killswitch_var.set(self.current_settings.enable_kill_switch)
    
    def _on_enable_proxy_changed(self):
        """Handle enable proxy checkbox change."""
        enabled = self.enable_proxy_var.get()
        
        # Enable/disable all child widgets in proxy frame
        for child in self.proxy_frame.winfo_children():
            if isinstance(child, (ttk.Entry, ttk.Combobox)):
                child.configure(state="normal" if enabled else "disabled")
    
    def _test_connection(self):
        """Test proxy connection."""
        if not self.enable_proxy_var.get():
            self.test_status_var.set("Proxy disabled")
            return
            
        if not self.proxy_host_var.get().strip():
            self.test_status_var.set("Host required")
            return
        
        self.test_status_var.set("Testing...")
        self.test_button.configure(state="disabled")
        
        # Run test in background thread to avoid blocking UI
        import threading
        
        def test_thread():
            try:
                # Create temporary proxy manager for testing
                from config.settings import ApplicationSettings
                temp_settings = ApplicationSettings()
                temp_settings.enable_proxy = True
                temp_settings.proxy_type = self.proxy_type_var.get()
                temp_settings.proxy_host = self.proxy_host_var.get().strip()
                temp_settings.proxy_port = int(self.proxy_port_var.get())
                temp_settings.proxy_username = self.proxy_username_var.get().strip() or None
                temp_settings.proxy_password = self.proxy_password_var.get().strip() or None
                
                # Create settings manager mock
                class TempSettingsManager:
                    def __init__(self, settings):
                        self._settings = settings
                
                temp_manager = TempSettingsManager(temp_settings)
                proxy_manager = ProxyManager(temp_manager)
                
                # Test connection
                if proxy_manager.proxies:
                    proxy = proxy_manager.proxies[0]
                    success = proxy_manager.check_proxy_health(proxy)
                    
                    if success:
                        self.dialog.after(0, lambda: self.test_status_var.set("✓ Connection successful"))
                    else:
                        self.dialog.after(0, lambda: self.test_status_var.set("✗ Connection failed"))
                else:
                    self.dialog.after(0, lambda: self.test_status_var.set("✗ Invalid configuration"))
                    
            except Exception as e:
                self.dialog.after(0, lambda: self.test_status_var.set(f"✗ Error: {str(e)[:30]}"))
            finally:
                self.dialog.after(0, lambda: self.test_button.configure(state="normal"))
        
        threading.Thread(target=test_thread, daemon=True).start()
    
    def _validate_settings(self) -> bool:
        """Validate proxy settings.
        
        Returns:
            True if settings are valid
        """
        if not self.enable_proxy_var.get():
            return True  # No validation needed if disabled
            
        # Validate host
        if not self.proxy_host_var.get().strip():
            messagebox.showerror("Invalid Settings", "Proxy host is required when proxy is enabled.")
            return False
        
        # Validate port
        try:
            port = int(self.proxy_port_var.get())
            if not (1 <= port <= 65535):
                raise ValueError()
        except ValueError:
            messagebox.showerror("Invalid Settings", "Proxy port must be a number between 1 and 65535.")
            return False
        
        # Validate delay range
        try:
            min_delay = int(self.min_delay_var.get())
            max_delay = int(self.max_delay_var.get())
            if min_delay < 0 or max_delay < 0:
                raise ValueError("Delays must be positive")
            if min_delay > max_delay:
                raise ValueError("Minimum delay cannot be greater than maximum delay")
        except ValueError as e:
            messagebox.showerror("Invalid Settings", f"Invalid delay settings: {str(e)}")
            return False
        
        return True
    
    def _on_save(self):
        """Save proxy settings."""
        if not self._validate_settings():
            return
        
        if not self.settings_manager:
            messagebox.showerror("Error", "Settings manager not available.")
            return
        
        try:
            # Update settings
            settings = self.settings_manager._settings
            
            settings.enable_proxy = self.enable_proxy_var.get()
            settings.proxy_type = self.proxy_type_var.get()
            settings.proxy_host = self.proxy_host_var.get().strip()
            settings.proxy_port = int(self.proxy_port_var.get())
            settings.proxy_username = self.proxy_username_var.get().strip() or None
            settings.proxy_password = self.proxy_password_var.get().strip() or None
            
            settings.enable_random_delays = self.random_delays_var.get()
            settings.min_request_delay = int(self.min_delay_var.get())
            settings.max_request_delay = int(self.max_delay_var.get())
            settings.rotate_user_agents = self.rotate_user_agents_var.get()
            settings.enable_header_randomization = self.randomize_headers_var.get()
            
            settings.enable_kill_switch = self.enable_killswitch_var.get()
            
            # Save settings
            if self.settings_manager.save_settings():
                messagebox.showinfo("Settings Saved", "Proxy settings have been saved successfully.")
                
                # Call callback if provided
                if self.callback:
                    self.callback()
                
                self.dialog.destroy()
            else:
                messagebox.showerror("Error", "Failed to save settings.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
    
    def _on_cancel(self):
        """Cancel dialog."""
        self.dialog.destroy()
    
    def show(self):
        """Show the dialog (for non-modal use)."""
        self.dialog.mainloop()


def show_proxy_settings(parent=None, settings_manager=None, callback=None):
    """Show proxy settings dialog.
    
    Args:
        parent: Parent window
        settings_manager: Settings manager instance
        callback: Callback function called when settings are saved
    """
    dialog = ProxySettingsDialog(parent, settings_manager, callback)
    
    if parent:
        parent.wait_window(dialog.dialog)
    else:
        dialog.show()


if __name__ == "__main__":
    # Test the dialog
    show_proxy_settings()
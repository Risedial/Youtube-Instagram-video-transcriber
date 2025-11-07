"""Styling and theme management for YouTube Whisper Transcriber.

This module provides consistent styling, themes, and visual appearance
configuration for the desktop application.
"""

from typing import Dict, Any, Optional, Tuple
import tkinter as tk
from tkinter import ttk
import logging


class AppColors:
    """Application color palette."""
    
    # Primary colors
    PRIMARY_BLUE = "#0066CC"
    PRIMARY_DARK = "#004499"
    PRIMARY_LIGHT = "#3388DD"
    
    # Status colors
    SUCCESS_GREEN = "#009900"
    WARNING_ORANGE = "#FF9900"
    ERROR_RED = "#CC0000"
    INFO_BLUE = "#0066CC"
    PROCESSING_PURPLE = "#6600CC"
    
    # Neutral colors
    DARK_GRAY = "#333333"
    MEDIUM_GRAY = "#666666"
    LIGHT_GRAY = "#999999"
    VERY_LIGHT_GRAY = "#CCCCCC"
    
    # Background colors
    WHITE = "#FFFFFF"
    OFF_WHITE = "#FAFAFA"
    LIGHT_BACKGROUND = "#F5F5F5"
    
    # Input validation colors
    VALID_BACKGROUND = "#E8F5E8"
    INVALID_BACKGROUND = "#FFE6E6"
    NEUTRAL_BACKGROUND = "#FFFFFF"


class AppFonts:
    """Application font definitions."""
    
    # Font families
    DEFAULT_FAMILY = "Arial"
    MONOSPACE_FAMILY = "Consolas"
    
    # Font sizes
    TITLE_SIZE = 18
    SUBTITLE_SIZE = 14
    NORMAL_SIZE = 10
    SMALL_SIZE = 9
    TINY_SIZE = 8
    
    # Font weights
    NORMAL = "normal"
    BOLD = "bold"
    
    @classmethod
    def title_font(cls) -> Tuple[str, int, str]:
        """Get title font configuration."""
        return (cls.DEFAULT_FAMILY, cls.TITLE_SIZE, cls.BOLD)
        
    @classmethod
    def subtitle_font(cls) -> Tuple[str, int, str]:
        """Get subtitle font configuration."""
        return (cls.DEFAULT_FAMILY, cls.SUBTITLE_SIZE, cls.BOLD)
        
    @classmethod
    def normal_font(cls) -> Tuple[str, int, str]:
        """Get normal font configuration."""
        return (cls.DEFAULT_FAMILY, cls.NORMAL_SIZE, cls.NORMAL)
        
    @classmethod
    def small_font(cls) -> Tuple[str, int, str]:
        """Get small font configuration."""
        return (cls.DEFAULT_FAMILY, cls.SMALL_SIZE, cls.NORMAL)
        
    @classmethod
    def tiny_font(cls) -> Tuple[str, int, str]:
        """Get tiny font configuration."""
        return (cls.DEFAULT_FAMILY, cls.TINY_SIZE, cls.NORMAL)
        
    @classmethod
    def monospace_font(cls) -> Tuple[str, int, str]:
        """Get monospace font configuration."""
        return (cls.MONOSPACE_FAMILY, cls.NORMAL_SIZE, cls.NORMAL)


class StyleTheme:
    """Theme configuration for the application."""
    
    def __init__(self, name: str, base_theme: str = "default") -> None:
        """Initialize theme.
        
        Args:
            name: Theme name
            base_theme: Base TTK theme to extend
        """
        self.name = name
        self.base_theme = base_theme
        self.colors = AppColors()
        self.fonts = AppFonts()
        
    def get_widget_config(self, widget_type: str) -> Dict[str, Any]:
        """Get configuration for specific widget type.
        
        Args:
            widget_type: Type of widget
            
        Returns:
            Configuration dictionary
        """
        configs = {
            'title_label': {
                'font': self.fonts.title_font(),
                'foreground': self.colors.DARK_GRAY
            },
            'subtitle_label': {
                'font': self.fonts.subtitle_font(),
                'foreground': self.colors.MEDIUM_GRAY
            },
            'normal_label': {
                'font': self.fonts.normal_font(),
                'foreground': self.colors.DARK_GRAY
            },
            'small_label': {
                'font': self.fonts.small_font(),
                'foreground': self.colors.LIGHT_GRAY
            },
            'success_label': {
                'font': self.fonts.normal_font(),
                'foreground': self.colors.SUCCESS_GREEN
            },
            'error_label': {
                'font': self.fonts.normal_font(),
                'foreground': self.colors.ERROR_RED
            },
            'warning_label': {
                'font': self.fonts.normal_font(),
                'foreground': self.colors.WARNING_ORANGE
            },
            'info_label': {
                'font': self.fonts.normal_font(),
                'foreground': self.colors.INFO_BLUE
            },
            'entry': {
                'font': self.fonts.normal_font(),
                'background': self.colors.WHITE
            },
            'button': {
                'font': self.fonts.normal_font()
            },
            'accent_button': {
                'font': (self.fonts.DEFAULT_FAMILY, self.fonts.NORMAL_SIZE, self.fonts.BOLD)
            }
        }
        
        return configs.get(widget_type, {})


class StyleManager:
    """Manages application styling and themes."""
    
    def __init__(self) -> None:
        """Initialize style manager."""
        self.logger = logging.getLogger(__name__)
        self.style = ttk.Style()
        self.current_theme: Optional[StyleTheme] = None
        self.custom_styles_configured = False
        
        # Initialize default theme
        self.setup_default_theme()
        
    def setup_default_theme(self) -> None:
        """Setup the default application theme."""
        try:
            # Create default theme
            default_theme = StyleTheme("default", "default")
            
            # Apply theme
            self.apply_theme(default_theme)
            
        except Exception as e:
            self.logger.error(f"Failed to setup default theme: {e}")
            
    def apply_theme(self, theme: StyleTheme) -> bool:
        """Apply a theme to the application.
        
        Args:
            theme: Theme to apply
            
        Returns:
            True if successful
        """
        try:
            # Set base TTK theme
            available_themes = self.style.theme_names()
            if theme.base_theme in available_themes:
                self.style.theme_use(theme.base_theme)
            else:
                self.logger.warning(f"Base theme '{theme.base_theme}' not available")
                
            # Configure custom styles
            self._configure_custom_styles(theme)
            
            self.current_theme = theme
            self.logger.info(f"Applied theme: {theme.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply theme: {e}")
            return False
            
    def _configure_custom_styles(self, theme: StyleTheme) -> None:
        """Configure custom TTK styles for the theme.
        
        Args:
            theme: Theme to configure styles for
        """
        try:
            # Configure label styles
            self.style.configure(
                'Title.TLabel',
                font=theme.fonts.title_font(),
                foreground=theme.colors.DARK_GRAY
            )
            
            self.style.configure(
                'Subtitle.TLabel', 
                font=theme.fonts.subtitle_font(),
                foreground=theme.colors.MEDIUM_GRAY
            )
            
            self.style.configure(
                'Small.TLabel',
                font=theme.fonts.small_font(),
                foreground=theme.colors.LIGHT_GRAY
            )
            
            # Status label styles
            self.style.configure(
                'Success.TLabel',
                font=theme.fonts.normal_font(),
                foreground=theme.colors.SUCCESS_GREEN
            )
            
            self.style.configure(
                'Error.TLabel',
                font=theme.fonts.normal_font(),
                foreground=theme.colors.ERROR_RED
            )
            
            self.style.configure(
                'Warning.TLabel',
                font=theme.fonts.normal_font(),
                foreground=theme.colors.WARNING_ORANGE
            )
            
            self.style.configure(
                'Info.TLabel',
                font=theme.fonts.normal_font(),
                foreground=theme.colors.INFO_BLUE
            )
            
            # Entry styles
            self.style.configure(
                'TEntry',
                font=theme.fonts.normal_font(),
                fieldbackground=theme.colors.WHITE
            )
            
            self.style.configure(
                'Valid.TEntry',
                font=theme.fonts.normal_font(),
                fieldbackground=theme.colors.VALID_BACKGROUND
            )
            
            self.style.configure(
                'Invalid.TEntry',
                font=theme.fonts.normal_font(),
                fieldbackground=theme.colors.INVALID_BACKGROUND
            )
            
            # Button styles
            self.style.configure(
                'TButton',
                font=theme.fonts.normal_font()
            )
            
            self.style.configure(
                'Accent.TButton',
                font=(theme.fonts.DEFAULT_FAMILY, theme.fonts.NORMAL_SIZE, theme.fonts.BOLD)
            )
            
            # Combobox styles
            self.style.configure(
                'TCombobox',
                font=theme.fonts.normal_font()
            )
            
            # Progress bar styles
            self.style.configure(
                'TProgressbar',
                background=theme.colors.PRIMARY_BLUE,
                troughcolor=theme.colors.LIGHT_BACKGROUND
            )
            
            # Notebook styles
            self.style.configure(
                'TNotebook.Tab',
                font=theme.fonts.normal_font()
            )
            
            # Frame styles
            self.style.configure(
                'TLabelFrame.Label',
                font=theme.fonts.normal_font(),
                foreground=theme.colors.DARK_GRAY
            )
            
            self.custom_styles_configured = True
            
        except Exception as e:
            self.logger.error(f"Failed to configure custom styles: {e}")
            
    def get_color(self, color_name: str) -> str:
        """Get color value by name.
        
        Args:
            color_name: Name of color
            
        Returns:
            Color hex value
        """
        if self.current_theme:
            return getattr(self.current_theme.colors, color_name.upper(), "#000000")
        return "#000000"
        
    def get_font(self, font_name: str) -> Tuple[str, int, str]:
        """Get font configuration by name.
        
        Args:
            font_name: Name of font
            
        Returns:
            Font tuple (family, size, weight)
        """
        if self.current_theme:
            font_method = getattr(self.current_theme.fonts, f"{font_name}_font", None)
            if font_method:
                return font_method()
        return AppFonts.normal_font()
        
    def apply_widget_style(self, widget: tk.Widget, style_name: str) -> None:
        """Apply style to a widget.
        
        Args:
            widget: Widget to style
            style_name: Name of style to apply
        """
        if not self.current_theme:
            return
            
        try:
            config = self.current_theme.get_widget_config(style_name)
            if config:
                widget.configure(**config)
                
        except Exception as e:
            self.logger.warning(f"Failed to apply style '{style_name}' to widget: {e}")
            
    def get_status_color(self, status_type: str) -> str:
        """Get color for status type.
        
        Args:
            status_type: Type of status (success, error, warning, info, processing)
            
        Returns:
            Color hex value
        """
        if not self.current_theme:
            return "#000000"
            
        status_colors = {
            'success': self.current_theme.colors.SUCCESS_GREEN,
            'error': self.current_theme.colors.ERROR_RED,
            'warning': self.current_theme.colors.WARNING_ORANGE,
            'info': self.current_theme.colors.INFO_BLUE,
            'processing': self.current_theme.colors.PROCESSING_PURPLE
        }
        
        return status_colors.get(status_type, self.current_theme.colors.DARK_GRAY)
        
    def create_themed_text_tags(self, text_widget: tk.Text) -> None:
        """Create themed tags for Text widget.
        
        Args:
            text_widget: Text widget to configure
        """
        if not self.current_theme:
            return
            
        try:
            # Configure tags for different status types
            text_widget.tag_configure(
                'success',
                foreground=self.current_theme.colors.SUCCESS_GREEN,
                font=self.current_theme.fonts.normal_font()
            )
            
            text_widget.tag_configure(
                'error',
                foreground=self.current_theme.colors.ERROR_RED,
                font=self.current_theme.fonts.normal_font()
            )
            
            text_widget.tag_configure(
                'warning',
                foreground=self.current_theme.colors.WARNING_ORANGE,
                font=self.current_theme.fonts.normal_font()
            )
            
            text_widget.tag_configure(
                'info',
                foreground=self.current_theme.colors.INFO_BLUE,
                font=self.current_theme.fonts.normal_font()
            )
            
            text_widget.tag_configure(
                'processing',
                foreground=self.current_theme.colors.PROCESSING_PURPLE,
                font=self.current_theme.fonts.normal_font()
            )
            
            # Title and emphasis tags
            text_widget.tag_configure(
                'title',
                font=self.current_theme.fonts.title_font(),
                foreground=self.current_theme.colors.DARK_GRAY
            )
            
            text_widget.tag_configure(
                'bold',
                font=(self.current_theme.fonts.DEFAULT_FAMILY, 
                     self.current_theme.fonts.NORMAL_SIZE, 
                     self.current_theme.fonts.BOLD)
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create themed text tags: {e}")
            
    def get_available_themes(self) -> list:
        """Get list of available TTK themes.
        
        Returns:
            List of theme names
        """
        try:
            return list(self.style.theme_names())
        except Exception:
            return ['default']
            
    def set_ttk_theme(self, theme_name: str) -> bool:
        """Set base TTK theme.
        
        Args:
            theme_name: TTK theme name
            
        Returns:
            True if successful
        """
        try:
            available_themes = self.get_available_themes()
            if theme_name in available_themes:
                self.style.theme_use(theme_name)
                if self.current_theme:
                    self.current_theme.base_theme = theme_name
                    self._configure_custom_styles(self.current_theme)
                return True
        except Exception as e:
            self.logger.error(f"Failed to set TTK theme: {e}")
        return False


# Global style manager instance
_style_manager: Optional[StyleManager] = None


def get_style_manager() -> StyleManager:
    """Get global style manager instance.
    
    Returns:
        Style manager instance
    """
    global _style_manager
    if _style_manager is None:
        _style_manager = StyleManager()
    return _style_manager


def apply_app_styling(root: tk.Tk) -> StyleManager:
    """Apply application styling to root window.
    
    Args:
        root: Root window
        
    Returns:
        Style manager instance
    """
    style_manager = get_style_manager()
    
    # Configure root window styling
    try:
        root.configure(bg=style_manager.current_theme.colors.LIGHT_BACKGROUND)
    except Exception:
        pass
        
    return style_manager


def create_styled_widget(widget_class, parent: tk.Widget, style_name: str = None, **kwargs):
    """Create a widget with automatic styling applied.
    
    Args:
        widget_class: Widget class to create
        parent: Parent widget
        style_name: Style name to apply
        **kwargs: Additional widget arguments
        
    Returns:
        Styled widget instance
    """
    style_manager = get_style_manager()
    
    # Create widget
    widget = widget_class(parent, **kwargs)
    
    # Apply styling if specified
    if style_name and style_manager.current_theme:
        style_manager.apply_widget_style(widget, style_name)
        
    return widget


# Convenience functions for common styled widgets
def create_title_label(parent: tk.Widget, text: str, **kwargs) -> ttk.Label:
    """Create a styled title label."""
    return create_styled_widget(ttk.Label, parent, 'title_label', text=text, **kwargs)


def create_subtitle_label(parent: tk.Widget, text: str, **kwargs) -> ttk.Label:
    """Create a styled subtitle label."""
    return create_styled_widget(ttk.Label, parent, 'subtitle_label', text=text, **kwargs)


def create_info_label(parent: tk.Widget, text: str, **kwargs) -> ttk.Label:
    """Create a styled info label."""
    return create_styled_widget(ttk.Label, parent, 'info_label', text=text, **kwargs)


def create_success_label(parent: tk.Widget, text: str, **kwargs) -> ttk.Label:
    """Create a styled success label."""
    return create_styled_widget(ttk.Label, parent, 'success_label', text=text, **kwargs)


def create_error_label(parent: tk.Widget, text: str, **kwargs) -> ttk.Label:
    """Create a styled error label."""
    return create_styled_widget(ttk.Label, parent, 'error_label', text=text, **kwargs)


def create_accent_button(parent: tk.Widget, text: str, **kwargs) -> ttk.Button:
    """Create a styled accent button."""
    kwargs['style'] = 'Accent.TButton'
    return ttk.Button(parent, text=text, **kwargs)
"""
Application styling and theming

Provides stylesheet and styling utilities for the Farsi Transcriber app.
"""

# Modern, professional dark-themed stylesheet
MAIN_STYLESHEET = """
QMainWindow {
    background-color: #f5f5f5;
}

QLabel {
    color: #333333;
}

QLineEdit, QTextEdit {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    padding: 5px;
    font-size: 11pt;
}

QLineEdit:focus, QTextEdit:focus {
    border: 2px solid #4CAF50;
    background-color: #fafafa;
}

QPushButton {
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
    font-size: 11pt;
    min-height: 32px;
}

QPushButton:hover {
    background-color: #45a049;
}

QPushButton:pressed {
    background-color: #3d8b40;
}

QPushButton:disabled {
    background-color: #cccccc;
    color: #999999;
}

QProgressBar {
    border: 1px solid #d0d0d0;
    border-radius: 4px;
    text-align: center;
    background-color: #ffffff;
    height: 20px;
}

QProgressBar::chunk {
    background-color: #4CAF50;
    border-radius: 3px;
}

QMessageBox QLabel {
    color: #333333;
}

QMessageBox QPushButton {
    min-width: 60px;
}
"""

# Color palette
COLORS = {
    "primary": "#4CAF50",
    "primary_hover": "#45a049",
    "primary_active": "#3d8b40",
    "background": "#f5f5f5",
    "text": "#333333",
    "text_secondary": "#666666",
    "border": "#d0d0d0",
    "success": "#4CAF50",
    "error": "#f44336",
    "warning": "#ff9800",
    "info": "#2196F3",
}

# Font settings
FONTS = {
    "default_size": 11,
    "title_size": 16,
    "mono_family": "Courier New",
}


def get_stylesheet() -> str:
    """Get the main stylesheet for the application"""
    return MAIN_STYLESHEET


def get_color(color_name: str) -> str:
    """Get a color from the palette"""
    return COLORS.get(color_name, "#000000")

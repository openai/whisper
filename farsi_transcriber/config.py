"""
Configuration settings for Farsi Transcriber application

Manages model selection, device settings, and other configuration options.
"""

from pathlib import Path

# Application metadata
APP_NAME = "Farsi Transcriber"
APP_VERSION = "0.1.0"
APP_DESCRIPTION = "A desktop application for transcribing Farsi audio and video files"

# Model settings
DEFAULT_MODEL = "medium"  # Options: tiny, base, small, medium, large
AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large"]
MODEL_DESCRIPTIONS = {
    "tiny": "Smallest model (39M params) - Fastest, ~1GB VRAM required",
    "base": "Small model (74M params) - Fast, ~1GB VRAM required",
    "small": "Medium model (244M params) - Balanced, ~2GB VRAM required",
    "medium": "Large model (769M params) - Good accuracy, ~5GB VRAM required",
    "large": "Largest model (1550M params) - Best accuracy, ~10GB VRAM required",
}

# Language settings
LANGUAGE_CODE = "fa"  # Farsi/Persian
LANGUAGE_NAME = "Farsi"

# Audio/Video settings
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv"}

# UI settings
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700
WINDOW_MIN_WIDTH = 800
WINDOW_MIN_HEIGHT = 600

# Output settings
OUTPUT_DIR = Path.home() / "FarsiTranscriber" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORT_FORMATS = {
    "txt": "Plain Text",
    "srt": "SRT Subtitles",
    "vtt": "WebVTT Subtitles",
    "json": "JSON Format",
    "tsv": "Tab-Separated Values",
}

# Device settings (auto-detect CUDA if available)
try:
    import torch

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "transcriber.log"


def get_model_info(model_name: str) -> str:
    """Get description for a model"""
    return MODEL_DESCRIPTIONS.get(model_name, "Unknown model")


def get_supported_formats() -> set:
    """Get all supported audio and video formats"""
    return SUPPORTED_AUDIO_FORMATS | SUPPORTED_VIDEO_FORMATS

# Whisper Language Processing Module
"""
This module provides enhanced language-aware processing for OpenAI Whisper.
Includes language detection, accent adaptation, confidence scoring, and
multilingual processing improvements.
"""

from .language_detector import LanguageDetector, AccentClassifier
from .confidence_calibration import ConfidenceCalibrator, LanguageSpecificScorer
from .multilingual_processor import MultilingualProcessor, CodeSwitchingDetector
from .accent_adaptation import AccentAdaptationEngine, RegionalVariantHandler

__all__ = [
    'LanguageDetector',
    'AccentClassifier',
    'ConfidenceCalibrator',
    'LanguageSpecificScorer',
    'MultilingualProcessor',
    'CodeSwitchingDetector',
    'AccentAdaptationEngine',
    'RegionalVariantHandler'
]

# Version info
__version__ = "1.0.0"
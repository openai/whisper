# Whisper Enhancements Module
"""
This module contains enhanced functionality for the OpenAI Whisper speech recognition system.
These enhancements provide additional features while maintaining backward compatibility with the core Whisper API.
"""

from .hallucination_detector import HallucinationDetector, detect_hallucinations
from .confidence_scorer import ConfidenceScorer, calculate_confidence_score

__all__ = [
    'HallucinationDetector',
    'detect_hallucinations',
    'ConfidenceScorer',
    'calculate_confidence_score'
]
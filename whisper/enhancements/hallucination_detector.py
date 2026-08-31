"""
Advanced Hallucination Detection for OpenAI Whisper

This module implements sophisticated hallucination detection and mitigation techniques
based on research findings and community-reported patterns.
"""

import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class HallucinationResult:
    """Result of hallucination detection analysis."""
    is_hallucination: bool
    confidence_score: float
    detected_patterns: List[str]
    risk_factors: Dict[str, float]
    recommended_action: str


class HallucinationDetector:
    """
    Advanced hallucination detection system for Whisper transcriptions.

    Based on research showing that hallucinations occur in ~80% of transcriptions
    in certain conditions, this detector uses multiple detection strategies:

    1. Pattern-based detection (YouTube artifacts, common phrases)
    2. Repetition analysis (looping behavior detection)
    3. Statistical analysis (compression ratios, log probabilities)
    4. Temporal analysis (silence periods, timing consistency)
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the hallucination detector.

        Args:
            language: Target language code for language-specific patterns
        """
        self.language = language
        self.patterns = self._load_hallucination_patterns()
        self.repetition_threshold = 3  # Number of repetitions to flag
        self.silence_threshold = 2.0   # Seconds of silence before text

    def _load_hallucination_patterns(self) -> Dict[str, List[str]]:
        """Load language-specific hallucination patterns."""
        patterns = {
            "en": [
                # YouTube artifacts (most common)
                "thank you for watching",
                "thanks for watching",
                "please subscribe",
                "like and subscribe",
                "don't forget to subscribe",
                "hit the bell icon",
                "ring that notification bell",
                "check out my other videos",
                "leave a comment below",
                "see you in the next video",
                "until next time",

                # Generic endings
                "that's all for now",
                "that's it for today",
                "catch you later",
                "peace out",
                "see you soon",

                # Advertisement artifacts
                "this video is sponsored by",
                "thanks to our sponsor",
                "use code",
                "get % off",
                "limited time offer",
                "act now",
                "call now",

                # Repetitive/looping indicators
                "and then and then",
                "so so so",
                "the the the",
                "i mean i mean",
                "you know you know",

                # Non-speech sounds interpreted as speech
                "hmm hmm hmm",
                "uh uh uh",
                "ah ah ah",
                "mm mm mm",
            ],
            "es": [
                "gracias por ver",
                "suscríbete",
                "dale like",
                "no olvides suscribirte",
                "hasta la próxima",
                "nos vemos",
            ],
            "fr": [
                "merci de regarder",
                "abonnez-vous",
                "n'oubliez pas de vous abonner",
                "à bientôt",
                "merci d'avoir regardé",
            ]
        }
        return patterns.get(self.language, patterns["en"])

    def detect_pattern_hallucinations(self, text: str) -> Tuple[bool, List[str], float]:
        """
        Detect hallucinations based on known patterns.

        Args:
            text: Transcribed text to analyze

        Returns:
            Tuple of (is_hallucination, detected_patterns, confidence_score)
        """
        text_lower = text.lower().strip()
        detected_patterns = []
        penalty_score = 0.0

        for pattern in self.patterns:
            if pattern in text_lower:
                detected_patterns.append(pattern)
                # Weight penalties by pattern severity
                if any(keyword in pattern for keyword in ["subscribe", "like", "sponsor"]):
                    penalty_score += 0.4  # High penalty for YouTube artifacts
                elif any(keyword in pattern for keyword in ["thank", "watch", "video"]):
                    penalty_score += 0.3  # Medium penalty for video endings
                else:
                    penalty_score += 0.2  # Lower penalty for other patterns

        is_hallucination = penalty_score > 0.3 or len(detected_patterns) > 1
        confidence_score = max(0.0, 1.0 - penalty_score)

        return is_hallucination, detected_patterns, confidence_score

    def detect_repetition_hallucinations(self, text: str) -> Tuple[bool, float]:
        """
        Detect hallucinations based on repetitive patterns.

        Args:
            text: Transcribed text to analyze

        Returns:
            Tuple of (is_hallucination, confidence_score)
        """
        words = text.lower().split()
        if len(words) < 4:
            return False, 1.0

        # Check for immediate repetitions
        repetition_count = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetition_count += 1

        repetition_ratio = repetition_count / len(words)

        # Check for phrase repetitions
        phrase_repetitions = 0
        for i in range(len(words) - 5):
            phrase = " ".join(words[i:i+3])
            remaining_text = " ".join(words[i+3:])
            if phrase in remaining_text:
                phrase_repetitions += 1

        phrase_ratio = phrase_repetitions / max(1, len(words) - 5)

        # Combine metrics
        total_repetition_score = repetition_ratio + (phrase_ratio * 2)
        is_hallucination = total_repetition_score > 0.3
        confidence_score = max(0.0, 1.0 - total_repetition_score * 2)

        return is_hallucination, confidence_score

    def detect_statistical_hallucinations(
        self,
        text: str,
        avg_logprob: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        no_speech_prob: Optional[float] = None
    ) -> Tuple[bool, float]:
        """
        Detect hallucinations using statistical metrics from Whisper.

        Args:
            text: Transcribed text
            avg_logprob: Average log probability of tokens
            compression_ratio: Compression ratio from Whisper
            no_speech_prob: No-speech probability from Whisper

        Returns:
            Tuple of (is_hallucination, confidence_score)
        """
        risk_score = 0.0

        # Text length analysis
        if len(text.strip()) < 10:
            risk_score += 0.2  # Very short text is suspicious
        elif len(text.strip()) > 500:
            risk_score += 0.1  # Very long text without breaks

        # Log probability analysis
        if avg_logprob is not None:
            if avg_logprob < -1.0:
                risk_score += 0.3  # Low confidence from model
            elif avg_logprob < -0.5:
                risk_score += 0.1

        # Compression ratio analysis
        if compression_ratio is not None:
            if compression_ratio > 2.4:
                risk_score += 0.4  # High compression ratio indicates repetitive text
            elif compression_ratio > 2.0:
                risk_score += 0.2

        # No-speech probability analysis
        if no_speech_prob is not None:
            if no_speech_prob > 0.6:
                risk_score += 0.3  # High probability of no speech
            elif no_speech_prob > 0.4:
                risk_score += 0.1

        is_hallucination = risk_score > 0.4
        confidence_score = max(0.0, 1.0 - risk_score)

        return is_hallucination, confidence_score

    def analyze_segment(
        self,
        text: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        avg_logprob: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        no_speech_prob: Optional[float] = None,
        preceding_silence: Optional[float] = None
    ) -> HallucinationResult:
        """
        Comprehensive hallucination analysis for a text segment.

        Args:
            text: Transcribed text segment
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
            avg_logprob: Average log probability
            compression_ratio: Text compression ratio
            no_speech_prob: No-speech probability
            preceding_silence: Duration of silence before this segment

        Returns:
            HallucinationResult with analysis details
        """
        # Pattern-based detection
        pattern_halluc, detected_patterns, pattern_confidence = self.detect_pattern_hallucinations(text)

        # Repetition-based detection
        repetition_halluc, repetition_confidence = self.detect_repetition_hallucinations(text)

        # Statistical detection
        statistical_halluc, statistical_confidence = self.detect_statistical_hallucinations(
            text, avg_logprob, compression_ratio, no_speech_prob
        )

        # Temporal analysis (silence-based detection)
        silence_risk = 0.0
        if preceding_silence is not None and preceding_silence > self.silence_threshold:
            silence_risk = min(0.4, preceding_silence / 10.0)  # More silence = higher risk

        # Combine all detection methods
        risk_factors = {
            "pattern_risk": 1.0 - pattern_confidence,
            "repetition_risk": 1.0 - repetition_confidence,
            "statistical_risk": 1.0 - statistical_confidence,
            "silence_risk": silence_risk
        }

        # Weighted combination (patterns are strongest indicator)
        combined_confidence = (
            pattern_confidence * 0.4 +
            repetition_confidence * 0.3 +
            statistical_confidence * 0.2 +
            (1.0 - silence_risk) * 0.1
        )

        is_hallucination = (
            pattern_halluc or
            repetition_halluc or
            statistical_halluc or
            silence_risk > 0.3
        )

        # Determine recommended action
        if is_hallucination:
            if combined_confidence < 0.3:
                recommended_action = "reject_segment"
            elif combined_confidence < 0.6:
                recommended_action = "flag_for_review"
            else:
                recommended_action = "accept_with_warning"
        else:
            recommended_action = "accept"

        return HallucinationResult(
            is_hallucination=is_hallucination,
            confidence_score=combined_confidence,
            detected_patterns=detected_patterns,
            risk_factors=risk_factors,
            recommended_action=recommended_action
        )


def detect_hallucinations(
    text: str,
    language: str = "en",
    **whisper_metrics
) -> HallucinationResult:
    """
    Convenience function for quick hallucination detection.

    Args:
        text: Text to analyze
        language: Language code
        **whisper_metrics: Additional metrics from Whisper (avg_logprob, etc.)

    Returns:
        HallucinationResult
    """
    detector = HallucinationDetector(language)
    return detector.analyze_segment(text, **whisper_metrics)


def filter_hallucinations(
    segments: List[dict],
    language: str = "en",
    strict_mode: bool = False
) -> List[dict]:
    """
    Filter out likely hallucinated segments from transcription results.

    Args:
        segments: List of segment dictionaries from Whisper
        language: Language code
        strict_mode: If True, use stricter filtering criteria

    Returns:
        Filtered list of segments
    """
    detector = HallucinationDetector(language)
    filtered_segments = []

    for segment in segments:
        text = segment.get('text', '')

        # Extract metrics if available
        kwargs = {}
        if 'avg_logprob' in segment:
            kwargs['avg_logprob'] = segment['avg_logprob']
        if 'compression_ratio' in segment:
            kwargs['compression_ratio'] = segment['compression_ratio']
        if 'no_speech_prob' in segment:
            kwargs['no_speech_prob'] = segment['no_speech_prob']
        if 'start' in segment:
            kwargs['start_time'] = segment['start']
        if 'end' in segment:
            kwargs['end_time'] = segment['end']

        result = detector.analyze_segment(text, **kwargs)

        # Apply filtering based on mode
        if strict_mode:
            # In strict mode, only accept high-confidence segments
            if result.recommended_action == "accept" and result.confidence_score > 0.7:
                filtered_segments.append(segment)
        else:
            # In normal mode, reject only obvious hallucinations
            if result.recommended_action != "reject_segment":
                # Add confidence metadata
                segment['hallucination_analysis'] = {
                    'confidence_score': result.confidence_score,
                    'risk_factors': result.risk_factors,
                    'detected_patterns': result.detected_patterns
                }
                filtered_segments.append(segment)

    return filtered_segments
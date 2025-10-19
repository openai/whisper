"""
Confidence Scoring System for Whisper Transcriptions

This module provides enhanced confidence scoring beyond the basic metrics
provided by Whisper, incorporating multiple factors for more accurate
assessment of transcription quality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConfidenceMetrics:
    """Container for confidence scoring metrics."""
    overall_score: float
    component_scores: Dict[str, float]
    quality_indicators: Dict[str, float]
    recommended_threshold: float


class ConfidenceScorer:
    """
    Advanced confidence scoring system for Whisper transcriptions.

    Combines multiple factors to provide more accurate confidence assessment:
    - Whisper's internal metrics (log probabilities, compression ratios)
    - Temporal consistency (timing patterns, speech rate)
    - Linguistic coherence (grammar, vocabulary consistency)
    - Audio quality indicators (signal-to-noise estimations)
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the confidence scorer.

        Args:
            language: Target language for language-specific scoring
        """
        self.language = language

    def score_segment_confidence(
        self,
        text: str,
        avg_logprob: Optional[float] = None,
        compression_ratio: Optional[float] = None,
        no_speech_prob: Optional[float] = None,
        word_timestamps: Optional[List[dict]] = None,
        segment_duration: Optional[float] = None
    ) -> ConfidenceMetrics:
        """
        Calculate comprehensive confidence score for a transcription segment.

        Args:
            text: Transcribed text
            avg_logprob: Average log probability from Whisper
            compression_ratio: Compression ratio from Whisper
            no_speech_prob: No-speech probability from Whisper
            word_timestamps: Word-level timestamps if available
            segment_duration: Duration of the audio segment

        Returns:
            ConfidenceMetrics with detailed scoring
        """
        component_scores = {}
        quality_indicators = {}

        # 1. Whisper Internal Metrics Score
        whisper_score = self._score_whisper_metrics(
            avg_logprob, compression_ratio, no_speech_prob
        )
        component_scores['whisper_metrics'] = whisper_score

        # 2. Text Quality Score
        text_score = self._score_text_quality(text)
        component_scores['text_quality'] = text_score

        # 3. Temporal Consistency Score
        temporal_score = self._score_temporal_consistency(
            word_timestamps, segment_duration, len(text.split()) if text else 0
        )
        component_scores['temporal_consistency'] = temporal_score

        # 4. Linguistic Coherence Score
        linguistic_score = self._score_linguistic_coherence(text)
        component_scores['linguistic_coherence'] = linguistic_score

        # Calculate overall score with weighted combination
        overall_score = (
            whisper_score * 0.4 +        # Whisper's own confidence is most important
            text_score * 0.25 +          # Text quality indicators
            temporal_score * 0.2 +       # Timing consistency
            linguistic_score * 0.15      # Language model coherence
        )

        # Quality indicators for analysis
        quality_indicators.update({
            'text_length': len(text) if text else 0,
            'word_count': len(text.split()) if text else 0,
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text else 0,
            'speech_rate': self._calculate_speech_rate(text, segment_duration),
            'repetition_rate': self._calculate_repetition_rate(text)
        })

        # Determine recommended threshold based on use case
        recommended_threshold = self._determine_threshold(overall_score, component_scores)

        return ConfidenceMetrics(
            overall_score=overall_score,
            component_scores=component_scores,
            quality_indicators=quality_indicators,
            recommended_threshold=recommended_threshold
        )

    def _score_whisper_metrics(
        self,
        avg_logprob: Optional[float],
        compression_ratio: Optional[float],
        no_speech_prob: Optional[float]
    ) -> float:
        """Score based on Whisper's internal metrics."""
        score = 0.7  # Default neutral score

        # Log probability scoring (higher is better, but values are negative)
        if avg_logprob is not None:
            if avg_logprob > -0.3:
                score += 0.3  # Very confident
            elif avg_logprob > -0.6:
                score += 0.2  # Confident
            elif avg_logprob > -1.0:
                score += 0.1  # Somewhat confident
            elif avg_logprob < -1.5:
                score -= 0.2  # Low confidence
            elif avg_logprob < -2.0:
                score -= 0.4  # Very low confidence

        # Compression ratio scoring (lower is better)
        if compression_ratio is not None:
            if compression_ratio < 1.5:
                score += 0.2  # Good compression
            elif compression_ratio < 2.0:
                score += 0.1  # Acceptable compression
            elif compression_ratio > 2.4:
                score -= 0.3  # High compression (likely repetitive)
            elif compression_ratio > 3.0:
                score -= 0.5  # Very high compression

        # No speech probability (lower is better for transcription)
        if no_speech_prob is not None:
            if no_speech_prob < 0.2:
                score += 0.1  # Confident there is speech
            elif no_speech_prob > 0.6:
                score -= 0.2  # Likely no speech
            elif no_speech_prob > 0.8:
                score -= 0.4  # Very likely no speech

        return max(0.0, min(1.0, score))

    def _score_text_quality(self, text: str) -> float:
        """Score based on text characteristics."""
        if not text or not text.strip():
            return 0.0

        text = text.strip()
        score = 0.5

        # Length-based scoring
        length = len(text)
        if 20 <= length <= 200:
            score += 0.2  # Good length
        elif 10 <= length < 20 or 200 < length <= 500:
            score += 0.1  # Acceptable length
        elif length < 10:
            score -= 0.2  # Too short
        elif length > 500:
            score -= 0.1  # Quite long

        # Character diversity
        unique_chars = len(set(text.lower()))
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0:
            diversity = unique_chars / total_chars
            if diversity > 0.3:
                score += 0.1
            elif diversity < 0.15:
                score -= 0.1

        # Punctuation presence (indicates structure)
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        word_count = len(text.split())
        if word_count > 0:
            punct_ratio = punctuation_count / word_count
            if 0.05 <= punct_ratio <= 0.3:  # Reasonable punctuation
                score += 0.1

        # Check for excessive capitalization
        if text.isupper() and len(text) > 10:
            score -= 0.15  # All caps is often transcription error

        return max(0.0, min(1.0, score))

    def _score_temporal_consistency(
        self,
        word_timestamps: Optional[List[dict]],
        segment_duration: Optional[float],
        word_count: int
    ) -> float:
        """Score based on temporal patterns in word timestamps."""
        if not word_timestamps or segment_duration is None or word_count == 0:
            return 0.5  # Neutral score when timing data unavailable

        score = 0.5

        try:
            # Calculate word durations
            word_durations = []
            for word_info in word_timestamps:
                if 'start' in word_info and 'end' in word_info:
                    duration = word_info['end'] - word_info['start']
                    word_durations.append(duration)

            if not word_durations:
                return 0.5

            # Analyze timing consistency
            avg_word_duration = np.mean(word_durations)
            word_duration_std = np.std(word_durations)

            # Reasonable word duration (0.1 to 1.0 seconds typically)
            if 0.1 <= avg_word_duration <= 1.0:
                score += 0.2
            elif avg_word_duration < 0.05 or avg_word_duration > 2.0:
                score -= 0.2

            # Consistency in word durations (lower std is better)
            if word_duration_std < avg_word_duration * 0.5:
                score += 0.15  # Consistent timing
            elif word_duration_std > avg_word_duration * 2.0:
                score -= 0.15  # Very inconsistent timing

            # Speech rate analysis
            estimated_speech_rate = word_count / segment_duration * 60  # words per minute
            if 120 <= estimated_speech_rate <= 200:
                score += 0.15  # Normal speech rate
            elif 80 <= estimated_speech_rate < 120 or 200 < estimated_speech_rate <= 300:
                score += 0.05  # Slightly unusual but acceptable
            elif estimated_speech_rate < 60 or estimated_speech_rate > 400:
                score -= 0.2   # Very unusual speech rate

        except (KeyError, TypeError, ValueError):
            # If there are issues with timestamp data, return neutral score
            return 0.5

        return max(0.0, min(1.0, score))

    def _score_linguistic_coherence(self, text: str) -> float:
        """Score based on linguistic patterns and coherence."""
        if not text or not text.strip():
            return 0.0

        text = text.strip()
        words = text.split()
        score = 0.5

        if len(words) == 0:
            return 0.0

        # Check for reasonable sentence structure
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            if 5 <= avg_sentence_length <= 20:
                score += 0.2
            elif 3 <= avg_sentence_length < 5 or 20 < avg_sentence_length <= 30:
                score += 0.1
            elif avg_sentence_length < 3 or avg_sentence_length > 30:
                score -= 0.1

        # Check for excessive repetition
        repetition_penalty = self._calculate_repetition_rate(text)
        score -= repetition_penalty * 0.3

        # Vocabulary diversity
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words) if words else 0
        if vocabulary_diversity > 0.7:
            score += 0.15
        elif vocabulary_diversity < 0.3:
            score -= 0.15

        # Check for common filler words (moderate amount is normal)
        filler_words = {'um', 'uh', 'er', 'ah', 'like', 'you know', 'so', 'well'}
        filler_count = sum(1 for word in words if word.lower() in filler_words)
        filler_ratio = filler_count / len(words) if words else 0
        if filler_ratio > 0.15:  # Too many fillers
            score -= 0.1
        elif filler_ratio > 0.3:   # Excessive fillers
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _calculate_speech_rate(self, text: str, duration: Optional[float]) -> float:
        """Calculate estimated speech rate in words per minute."""
        if not text or not duration or duration <= 0:
            return 0.0

        word_count = len(text.split())
        return word_count / duration * 60

    def _calculate_repetition_rate(self, text: str) -> float:
        """Calculate the rate of word repetitions in text."""
        if not text:
            return 0.0

        words = text.lower().split()
        if len(words) < 2:
            return 0.0

        repetitions = 0
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repetitions += 1

        return repetitions / len(words)

    def _determine_threshold(self, overall_score: float, component_scores: Dict[str, float]) -> float:
        """Determine recommended confidence threshold based on score characteristics."""
        # Base threshold
        if overall_score > 0.8:
            return 0.7    # High quality, can be more permissive
        elif overall_score > 0.6:
            return 0.5    # Medium quality, standard threshold
        else:
            return 0.3    # Lower quality, need lower threshold to get any results


def calculate_confidence_score(
    text: str,
    language: str = "en",
    **whisper_metrics
) -> ConfidenceMetrics:
    """
    Convenience function for quick confidence scoring.

    Args:
        text: Transcribed text to score
        language: Language code
        **whisper_metrics: Metrics from Whisper (avg_logprob, etc.)

    Returns:
        ConfidenceMetrics with scoring details
    """
    scorer = ConfidenceScorer(language)
    return scorer.score_segment_confidence(text, **whisper_metrics)


def filter_by_confidence(
    segments: List[dict],
    min_confidence: float = 0.5,
    language: str = "en"
) -> List[dict]:
    """
    Filter segments based on confidence scores.

    Args:
        segments: List of segment dictionaries from Whisper
        min_confidence: Minimum confidence threshold
        language: Language code

    Returns:
        Filtered list of high-confidence segments
    """
    scorer = ConfidenceScorer(language)
    filtered_segments = []

    for segment in segments:
        text = segment.get('text', '')

        # Extract metrics if available
        kwargs = {}
        for key in ['avg_logprob', 'compression_ratio', 'no_speech_prob']:
            if key in segment:
                kwargs[key] = segment[key]

        if 'words' in segment:
            kwargs['word_timestamps'] = segment['words']
        if 'start' in segment and 'end' in segment:
            kwargs['segment_duration'] = segment['end'] - segment['start']

        confidence_result = scorer.score_segment_confidence(text, **kwargs)

        if confidence_result.overall_score >= min_confidence:
            # Add confidence metadata
            segment['confidence_analysis'] = {
                'overall_score': confidence_result.overall_score,
                'component_scores': confidence_result.component_scores,
                'quality_indicators': confidence_result.quality_indicators
            }
            filtered_segments.append(segment)

    return filtered_segments
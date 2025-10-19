"""
Enhanced language detection with confidence scoring for OpenAI Whisper.
Addresses issues from GitHub Discussions #25, #16 regarding Chinese and Serbo-Croatian recognition.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class LanguageDetector:
    """Enhanced language detection with confidence scoring and fallback mechanisms."""

    # Language similarity groups for disambiguation
    SIMILAR_LANGUAGES = {
        'zh': ['zh-cn', 'zh-tw', 'yue'],  # Chinese variants
        'sr': ['hr', 'bs', 'me'],  # South Slavic languages
        'hr': ['sr', 'bs', 'me'],
        'bs': ['sr', 'hr', 'me'],
        'es': ['ca', 'gl'],  # Iberian languages
        'pt': ['gl', 'es'],
    }

    # Language-specific confidence thresholds
    CONFIDENCE_THRESHOLDS = {
        'zh': 0.8,  # Higher threshold for Chinese due to complexity
        'sr': 0.7,  # Higher threshold for Serbo-Croatian
        'hr': 0.7,
        'bs': 0.7,
        'ar': 0.8,  # Arabic variants
        'hi': 0.75,  # Hindi/Urdu confusion
        'ur': 0.75,
        'default': 0.6
    }

    def __init__(self, model):
        self.model = model
        self.detection_cache = {}

    def detect_language_enhanced(
        self,
        mel: torch.Tensor,
        segment_length: int = 30,
        use_multiple_segments: bool = True,
        return_probabilities: bool = True
    ) -> Dict:
        """
        Enhanced language detection with multiple sampling and confidence analysis.

        Args:
            mel: Log-mel spectrogram tensor
            segment_length: Length of segments for analysis (seconds)
            use_multiple_segments: Whether to analyze multiple segments
            return_probabilities: Whether to return detailed probabilities

        Returns:
            Dictionary with detected language, confidence, and analysis details
        """
        try:
            # Cache key for repeated detections
            cache_key = hash(mel.cpu().numpy().tobytes())
            if cache_key in self.detection_cache:
                return self.detection_cache[cache_key]

            # Single segment detection (standard Whisper)
            _, single_probs = self.model.detect_language(mel)

            result = {
                'language': max(single_probs, key=single_probs.get),
                'confidence': max(single_probs.values()),
                'probabilities': single_probs.copy(),
                'method': 'single_segment'
            }

            if use_multiple_segments and mel.shape[-1] > segment_length * 100:
                # Multi-segment analysis for longer audio
                multi_result = self._analyze_multiple_segments(mel, segment_length)

                # Compare results and use more confident one
                if multi_result['confidence'] > result['confidence']:
                    result = multi_result
                    result['method'] = 'multi_segment'
                else:
                    result['multi_segment_backup'] = multi_result

            # Apply confidence thresholds and similarity analysis
            result = self._apply_confidence_analysis(result)

            # Cache result
            self.detection_cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Language detection failed: {e}")
            return {
                'language': 'en',  # Fallback to English
                'confidence': 0.5,
                'probabilities': {'en': 1.0},
                'method': 'fallback',
                'error': str(e)
            }

    def _analyze_multiple_segments(self, mel: torch.Tensor, segment_length: int) -> Dict:
        """Analyze multiple segments of audio for more robust language detection."""
        segment_size = segment_length * 100  # Convert to mel frames
        segments = []

        # Extract segments
        for i in range(0, mel.shape[-1], segment_size):
            segment = mel[..., i:i + segment_size]
            if segment.shape[-1] >= segment_size // 2:  # Minimum half segment
                segments.append(segment)

        if len(segments) < 2:
            # Not enough segments for multi-segment analysis
            _, probs = self.model.detect_language(mel)
            return {
                'language': max(probs, key=probs.get),
                'confidence': max(probs.values()),
                'probabilities': probs
            }

        # Detect language for each segment
        segment_results = []
        language_votes = defaultdict(list)

        for i, segment in enumerate(segments):
            try:
                _, probs = self.model.detect_language(segment)
                detected_lang = max(probs, key=probs.get)
                confidence = probs[detected_lang]

                segment_results.append({
                    'segment': i,
                    'language': detected_lang,
                    'confidence': confidence,
                    'probabilities': probs
                })

                language_votes[detected_lang].append(confidence)

            except Exception as e:
                logger.warning(f"Segment {i} detection failed: {e}")

        # Aggregate results
        return self._aggregate_segment_results(language_votes, segment_results)

    def _aggregate_segment_results(self, language_votes: Dict, segment_results: List) -> Dict:
        """Aggregate results from multiple segments."""
        if not language_votes:
            return {
                'language': 'en',
                'confidence': 0.5,
                'probabilities': {'en': 1.0}
            }

        # Calculate weighted scores
        language_scores = {}
        for lang, confidences in language_votes.items():
            # Weighted average with vote count bonus
            avg_confidence = np.mean(confidences)
            vote_weight = len(confidences) / len(segment_results)
            consistency_bonus = 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0

            language_scores[lang] = avg_confidence * vote_weight * consistency_bonus

        # Select best language
        best_language = max(language_scores, key=language_scores.get)
        best_confidence = language_scores[best_language]

        # Create probability distribution
        total_score = sum(language_scores.values())
        probabilities = {
            lang: score / total_score
            for lang, score in language_scores.items()
        }

        return {
            'language': best_language,
            'confidence': best_confidence,
            'probabilities': probabilities,
            'segment_analysis': {
                'total_segments': len(segment_results),
                'language_votes': dict(language_votes),
                'consistency_score': self._calculate_consistency(segment_results)
            }
        }

    def _calculate_consistency(self, segment_results: List) -> float:
        """Calculate how consistent language detection is across segments."""
        if len(segment_results) < 2:
            return 1.0

        languages = [r['language'] for r in segment_results]
        most_common_lang = max(set(languages), key=languages.count)
        consistency = languages.count(most_common_lang) / len(languages)

        return consistency

    def _apply_confidence_analysis(self, result: Dict) -> Dict:
        """Apply language-specific confidence thresholds and similarity analysis."""
        detected_lang = result['language']
        confidence = result['confidence']
        probabilities = result['probabilities']

        # Get language-specific threshold
        threshold = self.CONFIDENCE_THRESHOLDS.get(
            detected_lang,
            self.CONFIDENCE_THRESHOLDS['default']
        )

        # Check if confidence meets threshold
        if confidence < threshold:
            result['low_confidence'] = True
            result['threshold'] = threshold

            # Check for similar languages
            if detected_lang in self.SIMILAR_LANGUAGES:
                similar_langs = self.SIMILAR_LANGUAGES[detected_lang]
                similar_probs = {
                    lang: probabilities.get(lang, 0.0)
                    for lang in similar_langs
                }

                # If similar language has higher probability, suggest it
                best_similar = max(similar_probs, key=similar_probs.get)
                if similar_probs[best_similar] > confidence:
                    result['alternative_language'] = best_similar
                    result['alternative_confidence'] = similar_probs[best_similar]
                    result['similarity_analysis'] = similar_probs

        # Additional metadata
        result['confidence_level'] = self._get_confidence_level(confidence)
        result['recommended_action'] = self._get_recommended_action(result)

        return result

    def _get_confidence_level(self, confidence: float) -> str:
        """Get human-readable confidence level."""
        if confidence >= 0.9:
            return 'very_high'
        elif confidence >= 0.8:
            return 'high'
        elif confidence >= 0.7:
            return 'medium'
        elif confidence >= 0.6:
            return 'low'
        else:
            return 'very_low'

    def _get_recommended_action(self, result: Dict) -> str:
        """Get recommended action based on detection results."""
        confidence = result['confidence']

        if confidence >= 0.8:
            return 'proceed'
        elif confidence >= 0.6:
            if 'alternative_language' in result:
                return 'consider_alternative'
            else:
                return 'proceed_with_caution'
        else:
            return 'manual_review_recommended'

    def detect_code_switching(self, mel: torch.Tensor, segment_length: int = 10) -> Dict:
        """
        Detect potential code-switching (multiple languages in same audio).

        Args:
            mel: Log-mel spectrogram tensor
            segment_length: Length of segments for analysis (seconds)

        Returns:
            Dictionary with code-switching analysis
        """
        segment_size = segment_length * 100
        segments = []

        # Extract shorter segments for code-switching detection
        for i in range(0, mel.shape[-1], segment_size):
            segment = mel[..., i:i + segment_size]
            if segment.shape[-1] >= segment_size // 2:
                segments.append((i // 100, segment))  # (timestamp, segment)

        if len(segments) < 2:
            return {
                'code_switching_detected': False,
                'languages': [],
                'confidence': 0.0
            }

        # Detect language for each segment
        segment_languages = []
        for timestamp, segment in segments:
            try:
                _, probs = self.model.detect_language(segment)
                detected_lang = max(probs, key=probs.get)
                confidence = probs[detected_lang]

                segment_languages.append({
                    'timestamp': timestamp,
                    'language': detected_lang,
                    'confidence': confidence
                })
            except Exception as e:
                logger.warning(f"Code-switching detection failed at {timestamp}s: {e}")

        # Analyze for code-switching
        unique_languages = list(set(sl['language'] for sl in segment_languages))

        if len(unique_languages) > 1:
            # Potential code-switching detected
            language_transitions = []
            for i in range(1, len(segment_languages)):
                prev_lang = segment_languages[i-1]['language']
                curr_lang = segment_languages[i]['language']

                if prev_lang != curr_lang:
                    language_transitions.append({
                        'timestamp': segment_languages[i]['timestamp'],
                        'from_language': prev_lang,
                        'to_language': curr_lang,
                        'confidence': segment_languages[i]['confidence']
                    })

            return {
                'code_switching_detected': True,
                'languages': unique_languages,
                'transitions': language_transitions,
                'segment_analysis': segment_languages,
                'confidence': np.mean([sl['confidence'] for sl in segment_languages])
            }
        else:
            return {
                'code_switching_detected': False,
                'languages': unique_languages,
                'confidence': np.mean([sl['confidence'] for sl in segment_languages])
            }

    def clear_cache(self):
        """Clear the detection cache."""
        self.detection_cache.clear()
"""
Language-aware processing and post-processing for OpenAI Whisper.
Addresses specific language issues mentioned in GitHub Discussions #25, #16.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class LanguageProcessor:
    """Language-specific processing and correction utilities."""

    # Language-specific corrections
    LANGUAGE_CORRECTIONS = {
        'zh': {
            # Chinese punctuation and spacing
            '，': ', ',
            '。': '. ',
            '？': '? ',
            '！': '! ',
            '：': ': ',
            '；': '; ',
            '（': ' (',
            '）': ') ',
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            # Common OCR-like errors in Chinese transcription
            ' 的 ': '的',
            ' 了 ': '了',
            ' 在 ': '在',
            ' 是 ': '是',
            ' 有 ': '有',
        },
        'sr': {
            # Serbian Cyrillic corrections
            ' ј ': 'ј',
            ' њ ': 'њ',
            ' љ ': 'љ',
            ' џ ': 'џ',
            ' ћ ': 'ћ',
            ' ђ ': 'ђ',
            # Common transcription issues
            'dj': 'ђ',
            'tj': 'ћ',
            'nj': 'њ',
            'lj': 'љ',
            'dz': 'џ',
        },
        'hr': {
            # Croatian corrections
            ' č ': 'č',
            ' ć ': 'ć',
            ' đ ': 'đ',
            ' š ': 'š',
            ' ž ': 'ž',
            # Digraph corrections
            'dj': 'đ',
            'ch': 'č',
            'sh': 'š',
            'zh': 'ž',
        },
        'de': {
            # German umlauts and ß
            ' ä ': 'ä',
            ' ö ': 'ö',
            ' ü ': 'ü',
            ' ß ': 'ß',
            ' Ä ': 'Ä',
            ' Ö ': 'Ö',
            ' Ü ': 'Ü',
            # Common transcription errors
            'ae': 'ä',
            'oe': 'ö',
            'ue': 'ü',
            'ss': 'ß',
        },
        'fr': {
            # French accents
            ' à ': 'à',
            ' â ': 'â',
            ' ç ': 'ç',
            ' è ': 'è',
            ' é ': 'é',
            ' ê ': 'ê',
            ' ë ': 'ë',
            ' î ': 'î',
            ' ï ': 'ï',
            ' ô ': 'ô',
            ' ù ': 'ù',
            ' û ': 'û',
            ' ü ': 'ü',
            ' ÿ ': 'ÿ',
        },
        'es': {
            # Spanish accents and ñ
            ' á ': 'á',
            ' é ': 'é',
            ' í ': 'í',
            ' ó ': 'ó',
            ' ú ': 'ú',
            ' ñ ': 'ñ',
            ' ü ': 'ü',
            # Inverted punctuation
            '¿': '¿',
            '¡': '¡',
        },
        'pt': {
            # Portuguese accents and cedilla
            ' á ': 'á',
            ' â ': 'â',
            ' ã ': 'ã',
            ' ç ': 'ç',
            ' é ': 'é',
            ' ê ': 'ê',
            ' í ': 'í',
            ' ó ': 'ó',
            ' ô ': 'ô',
            ' õ ': 'õ',
            ' ú ': 'ú',
            ' ü ': 'ü',
        },
        'ar': {
            # Arabic punctuation
            '،': ', ',
            '؟': '? ',
            '؛': '; ',
            '：': ': ',
        },
        'hi': {
            # Hindi Devanagari corrections
            '।': '. ',
            '॥': '. ',
            # Common transliteration issues
            ' ke ': ' के ',
            ' ki ': ' की ',
            ' ko ': ' को ',
            ' ka ': ' का ',
        }
    }

    # Language-specific regular expressions for cleaning
    LANGUAGE_REGEX_PATTERNS = {
        'zh': [
            # Remove extra spaces around Chinese characters
            (r'([^\u4e00-\u9fff])\s+([^\u4e00-\u9fff])', r'\1\2'),
            # Fix spacing around punctuation
            (r'\s+([，。？！：；])', r'\1'),
            # Remove spaces within Chinese text
            (r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2'),
        ],
        'ar': [
            # Arabic text direction and spacing
            (r'\s+([،؟؛])', r'\1'),
            # Remove extra spaces in Arabic text
            (r'([\u0600-\u06ff])\s+([\u0600-\u06ff])', r'\1\2'),
        ],
        'hi': [
            # Devanagari spacing
            (r'\s+([।॥])', r'\1'),
            # Remove extra spaces in Hindi text
            (r'([\u0900-\u097f])\s+([\u0900-\u097f])', r'\1\2'),
        ],
        'default': [
            # General cleanup patterns
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'^\s+|\s+$', ''),  # Trim whitespace
            (r'\s+([.!?;:,])', r'\1'),  # Remove space before punctuation
        ]
    }

    # Language-specific transcription options
    LANGUAGE_TRANSCRIPTION_OPTIONS = {
        'zh': {
            'temperature': 0.0,  # More deterministic for Chinese
            'compression_ratio_threshold': 2.8,
            'logprob_threshold': -1.2,
            'condition_on_previous_text': False,  # Reduce context confusion
            'fp16': False,  # Better accuracy
        },
        'sr': {
            'temperature': 0.1,
            'initial_prompt': "Говори јасно и споро.",  # "Speak clearly and slowly"
            'condition_on_previous_text': False,
        },
        'hr': {
            'temperature': 0.1,
            'initial_prompt': "Govorite jasno i polako.",  # "Speak clearly and slowly"
            'condition_on_previous_text': False,
        },
        'de': {
            'temperature': 0.0,
            'condition_on_previous_text': False,  # Reduce German hallucinations
            'compression_ratio_threshold': 2.6,
        },
        'ar': {
            'temperature': 0.0,
            'compression_ratio_threshold': 3.0,  # Higher threshold for Arabic
            'condition_on_previous_text': False,
        },
        'hi': {
            'temperature': 0.1,
            'compression_ratio_threshold': 2.5,
            'condition_on_previous_text': True,  # Context helps with Hindi
        },
        'ja': {
            'temperature': 0.0,
            'compression_ratio_threshold': 2.2,  # Lower for Japanese
            'condition_on_previous_text': False,
        },
        'ko': {
            'temperature': 0.0,
            'compression_ratio_threshold': 2.3,
            'condition_on_previous_text': False,
        }
    }

    def __init__(self):
        pass

    def get_language_options(self, language: str, **kwargs) -> Dict:
        """
        Get language-specific transcription options.

        Args:
            language: Language code
            **kwargs: Additional options to override defaults

        Returns:
            Dictionary of transcription options
        """
        options = self.LANGUAGE_TRANSCRIPTION_OPTIONS.get(language, {}).copy()
        options.update(kwargs)
        return options

    def postprocess_text(self, text: str, language: str) -> str:
        """
        Apply language-specific post-processing to transcribed text.

        Args:
            text: Transcribed text
            language: Language code

        Returns:
            Post-processed text
        """
        if not text:
            return text

        try:
            # Apply language-specific corrections
            processed_text = self._apply_corrections(text, language)

            # Apply language-specific regex patterns
            processed_text = self._apply_regex_patterns(processed_text, language)

            # Normalize Unicode characters
            processed_text = self._normalize_unicode(processed_text, language)

            # Final cleanup
            processed_text = self._final_cleanup(processed_text)

            return processed_text

        except Exception as e:
            logger.error(f"Post-processing failed for language {language}: {e}")
            return text  # Return original text if processing fails

    def _apply_corrections(self, text: str, language: str) -> str:
        """Apply language-specific character corrections."""
        corrections = self.LANGUAGE_CORRECTIONS.get(language, {})

        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)

        return text

    def _apply_regex_patterns(self, text: str, language: str) -> str:
        """Apply language-specific regex patterns."""
        patterns = self.LANGUAGE_REGEX_PATTERNS.get(
            language,
            self.LANGUAGE_REGEX_PATTERNS['default']
        )

        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)

        return text

    def _normalize_unicode(self, text: str, language: str) -> str:
        """Normalize Unicode characters for consistent representation."""
        # Different normalization strategies for different languages
        if language in ['ar', 'hi', 'zh', 'ja', 'ko']:
            # Use NFC for languages with complex character composition
            return unicodedata.normalize('NFC', text)
        else:
            # Use NFKC for Latin-based languages
            return unicodedata.normalize('NFKC', text)

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup operations."""
        # Remove multiple consecutive spaces
        text = re.sub(r'\s{2,}', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Fix common punctuation spacing issues
        text = re.sub(r'\s+([.!?;:,])', r'\1', text)
        text = re.sub(r'([.!?])\s*$', r'\1', text)

        return text

    def detect_text_quality_issues(self, text: str, language: str) -> Dict:
        """
        Detect potential quality issues in transcribed text.

        Args:
            text: Transcribed text
            language: Language code

        Returns:
            Dictionary with quality analysis
        """
        issues = {
            'repetitive_patterns': self._detect_repetitive_patterns(text),
            'unusual_character_frequency': self._detect_unusual_chars(text, language),
            'inconsistent_spacing': self._detect_spacing_issues(text, language),
            'mixed_scripts': self._detect_mixed_scripts(text, language),
            'quality_score': 1.0  # Default quality score
        }

        # Calculate overall quality score
        issue_count = sum(1 for issue_list in issues.values() if isinstance(issue_list, list) and issue_list)
        issues['quality_score'] = max(0.0, 1.0 - (issue_count * 0.2))

        return issues

    def _detect_repetitive_patterns(self, text: str) -> List[str]:
        """Detect repetitive patterns that might indicate hallucination."""
        issues = []
        words = text.split()

        if len(words) < 3:
            return issues

        # Check for repeated phrases
        for length in range(2, min(6, len(words) // 2)):
            for i in range(len(words) - length * 2 + 1):
                phrase = ' '.join(words[i:i+length])
                next_phrase = ' '.join(words[i+length:i+length*2])

                if phrase == next_phrase and len(phrase) > 5:
                    issues.append(f"Repeated phrase: '{phrase}'")

        # Check for word repetition
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1

        for word, count in word_counts.items():
            if count > len(words) * 0.1 and count > 3:  # More than 10% and at least 4 times
                issues.append(f"Overused word: '{word}' appears {count} times")

        return issues

    def _detect_unusual_chars(self, text: str, language: str) -> List[str]:
        """Detect unusual character frequency for the given language."""
        issues = []

        # Count character types
        char_counts = {
            'latin': 0,
            'chinese': 0,
            'arabic': 0,
            'cyrillic': 0,
            'devanagari': 0,
            'other': 0
        }

        for char in text:
            if '\u0020' <= char <= '\u007f':  # Basic Latin
                char_counts['latin'] += 1
            elif '\u4e00' <= char <= '\u9fff':  # Chinese
                char_counts['chinese'] += 1
            elif '\u0600' <= char <= '\u06ff':  # Arabic
                char_counts['arabic'] += 1
            elif '\u0400' <= char <= '\u04ff':  # Cyrillic
                char_counts['cyrillic'] += 1
            elif '\u0900' <= char <= '\u097f':  # Devanagari
                char_counts['devanagari'] += 1
            else:
                char_counts['other'] += 1

        total_chars = sum(char_counts.values())
        if total_chars == 0:
            return issues

        # Check for script consistency
        expected_scripts = {
            'zh': ['chinese'],
            'ar': ['arabic'],
            'sr': ['cyrillic'],
            'hi': ['devanagari'],
            'default': ['latin']
        }

        expected = expected_scripts.get(language, expected_scripts['default'])

        for script in char_counts:
            if script not in expected and char_counts[script] / total_chars > 0.1:
                issues.append(f"Unexpected {script} characters: {char_counts[script]}/{total_chars}")

        return issues

    def _detect_spacing_issues(self, text: str, language: str) -> List[str]:
        """Detect spacing inconsistencies."""
        issues = []

        # Check for multiple consecutive spaces
        if re.search(r'\s{3,}', text):
            issues.append("Multiple consecutive spaces found")

        # Language-specific spacing checks
        if language == 'zh':
            # Chinese shouldn't have spaces between characters
            if re.search(r'[\u4e00-\u9fff]\s+[\u4e00-\u9fff]', text):
                issues.append("Unnecessary spaces in Chinese text")

        elif language in ['ar', 'hi']:
            # Similar check for Arabic and Hindi
            if language == 'ar' and re.search(r'[\u0600-\u06ff]\s+[\u0600-\u06ff]', text):
                issues.append("Unnecessary spaces in Arabic text")
            elif language == 'hi' and re.search(r'[\u0900-\u097f]\s+[\u0900-\u097f]', text):
                issues.append("Unnecessary spaces in Hindi text")

        # Check spacing around punctuation
        if re.search(r'\s+[.!?;:,]', text):
            issues.append("Incorrect spacing before punctuation")

        return issues

    def _detect_mixed_scripts(self, text: str, language: str) -> List[str]:
        """Detect unexpected script mixing."""
        issues = []

        # This is a simplified check - in reality, some mixing is normal
        scripts_found = []

        if re.search(r'[\u4e00-\u9fff]', text):
            scripts_found.append('Chinese')
        if re.search(r'[\u0600-\u06ff]', text):
            scripts_found.append('Arabic')
        if re.search(r'[\u0400-\u04ff]', text):
            scripts_found.append('Cyrillic')
        if re.search(r'[\u0900-\u097f]', text):
            scripts_found.append('Devanagari')

        if len(scripts_found) > 1:
            issues.append(f"Multiple scripts detected: {', '.join(scripts_found)}")

        return issues

    def suggest_language_alternatives(self, text: str, original_language: str) -> List[Tuple[str, float]]:
        """
        Suggest alternative languages based on text analysis.

        Args:
            text: Transcribed text
            original_language: Originally detected language

        Returns:
            List of (language_code, confidence) tuples
        """
        suggestions = []

        # Analyze character composition
        char_analysis = self._analyze_character_composition(text)

        # Language-specific suggestions
        if original_language == 'sr':
            # Check if it might be Croatian or Bosnian
            if 'latin' in char_analysis and char_analysis['latin'] > 0.8:
                suggestions.append(('hr', 0.7))
                suggestions.append(('bs', 0.6))

        elif original_language == 'zh':
            # Check for Traditional vs Simplified Chinese
            if self._detect_traditional_chinese_chars(text):
                suggestions.append(('zh-tw', 0.8))
            else:
                suggestions.append(('zh-cn', 0.8))

        elif original_language in ['hi', 'ur']:
            # Hindi/Urdu can be confusing
            if 'arabic' in char_analysis and char_analysis['arabic'] > 0.3:
                suggestions.append(('ur', 0.7))
            elif 'devanagari' in char_analysis and char_analysis['devanagari'] > 0.3:
                suggestions.append(('hi', 0.7))

        return suggestions

    def _analyze_character_composition(self, text: str) -> Dict[str, float]:
        """Analyze the composition of characters in text."""
        char_counts = {
            'latin': 0,
            'chinese': 0,
            'arabic': 0,
            'cyrillic': 0,
            'devanagari': 0,
        }

        for char in text:
            if '\u0020' <= char <= '\u007f':
                char_counts['latin'] += 1
            elif '\u4e00' <= char <= '\u9fff':
                char_counts['chinese'] += 1
            elif '\u0600' <= char <= '\u06ff':
                char_counts['arabic'] += 1
            elif '\u0400' <= char <= '\u04ff':
                char_counts['cyrillic'] += 1
            elif '\u0900' <= char <= '\u097f':
                char_counts['devanagari'] += 1

        total = sum(char_counts.values())
        if total == 0:
            return {}

        return {script: count / total for script, count in char_counts.items()}

    def _detect_traditional_chinese_chars(self, text: str) -> bool:
        """Detect if text contains Traditional Chinese characters."""
        # Some characters that are different between Traditional and Simplified
        traditional_chars = ['繁', '體', '語', '學', '國', '華', '電', '話', '時', '間']
        simplified_chars = ['繁', '体', '语', '学', '国', '华', '电', '话', '时', '间']

        traditional_count = sum(1 for char in traditional_chars if char in text)
        simplified_count = sum(1 for char in simplified_chars if char in text)

        return traditional_count > simplified_count
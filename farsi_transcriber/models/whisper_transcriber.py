"""
Whisper Transcriber Module

Handles Farsi audio/video transcription using OpenAI's Whisper model.
"""

import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import whisper


class FarsiTranscriber:
    """
    Wrapper around Whisper model for Farsi transcription.

    Supports both audio and video files, with word-level timestamp extraction.
    """

    # Supported audio formats
    AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}

    # Supported video formats
    VIDEO_FORMATS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".wmv"}

    # Language code for Farsi/Persian
    FARSI_LANGUAGE = "fa"

    def __init__(self, model_name: str = "medium", device: Optional[str] = None):
        """
        Initialize Farsi Transcriber.

        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda', 'cpu'). Auto-detect if None.
        """
        self.model_name = model_name

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load model
        print(f"Loading Whisper model: {model_name}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = whisper.load_model(model_name, device=self.device)

        print("Model loaded successfully")

    def transcribe(
        self,
        file_path: str,
        language: str = FARSI_LANGUAGE,
        verbose: bool = False,
    ) -> Dict:
        """
        Transcribe an audio or video file in Farsi.

        Args:
            file_path: Path to audio or video file
            language: Language code (default: 'fa' for Farsi)
            verbose: Whether to print progress

        Returns:
            Dictionary with transcription results including word-level segments
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check format is supported
        if not self._is_supported_format(file_path):
            raise ValueError(
                f"Unsupported format: {file_path.suffix}. "
                f"Supported: {self.AUDIO_FORMATS | self.VIDEO_FORMATS}"
            )

        # Perform transcription
        print(f"Transcribing: {file_path.name}")

        result = self.model.transcribe(
            str(file_path),
            language=language,
            verbose=verbose,
        )

        # Enhance result with word-level segments
        enhanced_result = self._enhance_with_word_segments(result)

        return enhanced_result

    def _is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        suffix = file_path.suffix.lower()
        return suffix in (self.AUDIO_FORMATS | self.VIDEO_FORMATS)

    def _enhance_with_word_segments(self, result: Dict) -> Dict:
        """
        Enhance transcription result with word-level timing information.

        Args:
            result: Whisper transcription result

        Returns:
            Enhanced result with word-level segments
        """
        enhanced_segments = []

        for segment in result.get("segments", []):
            # Extract word-level timing if available
            word_segments = self._extract_word_segments(segment)

            enhanced_segment = {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "text": segment.get("text", ""),
                "words": word_segments,
            }
            enhanced_segments.append(enhanced_segment)

        result["segments"] = enhanced_segments
        return result

    def _extract_word_segments(self, segment: Dict) -> List[Dict]:
        """
        Extract word-level timing from a segment.

        Args:
            segment: Whisper segment with text

        Returns:
            List of word dictionaries with timing information
        """
        text = segment.get("text", "").strip()
        if not text:
            return []

        # For now, return simple word list
        # Whisper v3 includes word-level details in some configurations
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        duration = end_time - start_time

        words = text.split()
        if not words:
            return []

        # Distribute time evenly across words (simple approach)
        # More sophisticated timing can be extracted from Whisper's internal data
        word_duration = duration / len(words) if words else 0

        word_segments = []
        for i, word in enumerate(words):
            word_start = start_time + (i * word_duration)
            word_end = word_start + word_duration

            word_segments.append(
                {
                    "word": word,
                    "start": word_start,
                    "end": word_end,
                }
            )

        return word_segments

    def format_result_for_display(
        self, result: Dict, include_timestamps: bool = True
    ) -> str:
        """
        Format transcription result for display in UI.

        Args:
            result: Transcription result
            include_timestamps: Whether to include timestamps

        Returns:
            Formatted text string
        """
        lines = []

        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if not text:
                continue

            if include_timestamps:
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                timestamp = f"[{self._format_time(start)} - {self._format_time(end)}]"
                lines.append(f"{timestamp}\n{text}\n")
            else:
                lines.append(text)

        return "\n".join(lines)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)

        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

    def get_device_info(self) -> str:
        """Get information about current device and model."""
        return (
            f"Model: {self.model_name} | "
            f"Device: {self.device.upper()} | "
            f"VRAM: {torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f}GB "
            if self.device == "cuda"
            else f"Model: {self.model_name} | Device: {self.device.upper()}"
        )

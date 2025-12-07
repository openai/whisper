#!/usr/bin/env python3
"""
Real-time Speech Transcription with OpenAI Whisper

Enhanced features:
- Voice Activity Detection (VAD) - only transcribes when speech is detected
- Multiple language support with auto-detection
- Audio device selection
- Live audio level visualization
- Transcript saving with timestamps
- Word-level timestamps for detailed timing
- Speaker change detection hints
- Configurable via CLI arguments

Usage:
    python real_time_transcription.py --model base --language auto
    python real_time_transcription.py --list-devices
    python real_time_transcription.py --output transcript.txt --timestamps
    python real_time_transcription.py --word-timestamps
"""

import argparse
import datetime
import signal
import sys
import threading
import time
from collections import deque
from difflib import SequenceMatcher

import numpy as np
import torch

import whisper

try:
    import pyaudio
except ImportError:
    print("╭─────────────────────────────────────────────────────╮")
    print("│  pyaudio is required for microphone input          │")
    print("│  Install with: pip install pyaudio                 │")
    print("╰─────────────────────────────────────────────────────╯")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper's expected sample rate
CHUNK_SIZE = 1024

# VAD Configuration
VAD_ENERGY_THRESHOLD = 0.01  # Minimum energy to consider as speech
VAD_SILENCE_DURATION = 1.5  # Seconds of silence before stopping segment

# Transcription Configuration
MIN_SEGMENT_DURATION = 1.0  # Minimum audio duration to transcribe
MAX_SEGMENT_DURATION = 10.0  # Maximum audio duration per transcription
SIMILARITY_THRESHOLD = 0.85  # Threshold for duplicate detection


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def list_audio_devices():
    """List all available audio input devices."""
    p = pyaudio.PyAudio()
    print("\n╭─────────────────────────────────────────────────────╮")
    print("│              Available Audio Devices                │")
    print("├─────────────────────────────────────────────────────┤")

    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append((i, info["name"]))
            marker = "●" if info.get("isDefault", False) else "○"
            print(f"│  {marker} [{i:2d}] {info['name'][:42]:<42} │")

    print("╰─────────────────────────────────────────────────────╯")
    p.terminate()
    return devices


def get_audio_energy(audio_chunk: bytes) -> float:
    """Calculate the RMS energy of an audio chunk."""
    audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
    if len(audio_np) == 0:
        return 0.0
    return np.sqrt(np.mean(audio_np**2)) / 32768.0


def audio_level_bar(energy: float, width: int = 30) -> str:
    """Create a visual audio level bar."""
    level = min(1.0, energy * 10)  # Scale for visibility
    filled = int(level * width)
    bar = "█" * filled + "░" * (width - filled)

    # Color coding based on level
    if level < 0.3:
        color = "\033[90m"  # Gray (quiet)
    elif level < 0.6:
        color = "\033[92m"  # Green (normal)
    elif level < 0.85:
        color = "\033[93m"  # Yellow (loud)
    else:
        color = "\033[91m"  # Red (very loud)

    return f"{color}{bar}\033[0m"


def similar_text(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-TIME TRANSCRIBER
# ═══════════════════════════════════════════════════════════════════════════════


class RealTimeTranscriber:
    """
    Real-time speech transcription using OpenAI Whisper with VAD.

    Features:
    - Voice Activity Detection to segment speech
    - Duplicate transcript filtering
    - Live audio visualization
    - Multi-language support
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = None,
        language: str = "auto",
        input_device: int = None,
        output_file: str = None,
        show_timestamps: bool = False,
        word_timestamps: bool = False,
        detect_speakers: bool = False,
        energy_threshold: float = VAD_ENERGY_THRESHOLD,
    ):
        self.model_name = model_name
        self.language = language if language != "auto" else None
        self.input_device = input_device
        self.output_file = output_file
        self.show_timestamps = show_timestamps
        self.word_timestamps = word_timestamps
        self.detect_speakers = detect_speakers
        self.energy_threshold = energy_threshold

        # State
        self.running = False
        self.audio_buffer = deque()
        self.is_speaking = False
        self.silence_start = None
        self.speech_start = None
        self.current_energy = 0.0
        self.transcript_history = []
        self.start_time = None

        # Threading
        self.lock = threading.Lock()

        # Load model
        self._load_model(device)

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Output file handle
        self.output_handle = None

    def _load_model(self, device: str):
        """Load the Whisper model with progress indication."""
        print(f"\n\033[94m⟳ Loading Whisper model '{self.model_name}'...\033[0m")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = whisper.load_model(self.model_name, device=device)

        device_emoji = "🎮" if device == "cuda" else "💻"
        print(f"\033[92m✓ Model loaded on {device.upper()} {device_emoji}\033[0m\n")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process incoming audio in real-time."""
        energy = get_audio_energy(in_data)

        with self.lock:
            self.current_energy = energy

            # Voice Activity Detection
            if energy > self.energy_threshold:
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_start = time.time()
                    self.audio_buffer.clear()

                self.silence_start = None
                self.audio_buffer.append(in_data)
            else:
                if self.is_speaking:
                    self.audio_buffer.append(in_data)  # Include trailing silence

                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > VAD_SILENCE_DURATION:
                        # End of speech segment
                        self.is_speaking = False

            # Limit buffer size (max segment duration)
            max_chunks = int(MAX_SEGMENT_DURATION * RATE / CHUNK_SIZE)
            while len(self.audio_buffer) > max_chunks:
                self.audio_buffer.popleft()

        return (in_data, pyaudio.paContinue)

    def _get_audio_segment(self) -> np.ndarray:
        """Extract and convert buffered audio to numpy array."""
        with self.lock:
            if not self.audio_buffer:
                return None

            segment_bytes = b"".join(self.audio_buffer)
            self.audio_buffer.clear()

        audio_np = (
            np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )
        duration = len(audio_np) / RATE

        if duration < MIN_SEGMENT_DURATION:
            return None

        return audio_np

    def _is_duplicate(self, text: str) -> bool:
        """Check if transcript is too similar to recent ones."""
        text = text.strip()
        if not text:
            return True

        for prev in self.transcript_history[-5:]:  # Check last 5 transcripts
            if similar_text(text, prev) > SIMILARITY_THRESHOLD:
                return True
        return False

    def _format_timestamp(self) -> str:
        """Format elapsed time as timestamp."""
        if self.start_time is None:
            return "00:00:00"
        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _save_transcript(self, text: str, timestamp: str, words: list = None):
        """Save transcript to output file."""
        if self.output_handle:
            if self.show_timestamps:
                self.output_handle.write(f"[{timestamp}] {text}\n")
            else:
                self.output_handle.write(f"{text}\n")

            # Save word timestamps if enabled
            if self.word_timestamps and words:
                word_line = " | ".join(
                    f"{w.get('word', '').strip()}[{w.get('start', 0):.2f}-{w.get('end', 0):.2f}s]"
                    for w in words
                )
                self.output_handle.write(f"  Words: {word_line}\n")

            self.output_handle.flush()

    def _print_transcript(
        self,
        text: str,
        language: str = None,
        words: list = None,
        speaker_changed: bool = False,
    ):
        """Print transcript with formatting."""
        timestamp = self._format_timestamp()

        # Clear the audio level line and print transcript
        print("\r" + " " * 60, end="\r")

        # Speaker change indicator
        if speaker_changed and self.detect_speakers:
            print("\033[95m◆ Speaker change detected\033[0m")

        if self.show_timestamps:
            print(f"\033[96m[{timestamp}]\033[0m ", end="")

        if language and self.language is None:
            print(f"\033[90m({language})\033[0m ", end="")

        print(f"\033[97m{text}\033[0m")

        # Print word-level timestamps if enabled
        if self.word_timestamps and words:
            self._print_word_timestamps(words)

        self._save_transcript(text, timestamp, words)

    def _print_word_timestamps(self, words: list):
        """Print word-level timestamps in a formatted way."""
        print("  \033[90m┌─ Word Timestamps ─────────────────────────────┐\033[0m")

        line_words = []
        line_length = 0

        for word_info in words:
            word = word_info.get("word", "").strip()
            start = word_info.get("start", 0)

            formatted = f"{word}\033[90m[{start:.1f}s]\033[0m"
            display_len = len(word) + len("[{:.1f}s]".format(start)) + 1

            if line_length + display_len > 50 and line_words:
                print(f"  \033[90m│\033[0m {' '.join(line_words)}")
                line_words = []
                line_length = 0

            line_words.append(formatted)
            line_length += display_len

        if line_words:
            print(f"  \033[90m│\033[0m {' '.join(line_words)}")

        print("  \033[90m└───────────────────────────────────────────────┘\033[0m")

    def _detect_speaker_change(self, current_words: list) -> bool:
        """
        Heuristic speaker change detection based on pause patterns.

        This is a simple heuristic - for proper diarization,
        consider using pyannote.audio or similar libraries.
        """
        if not current_words or len(current_words) < 2:
            return False

        # Look for significant pauses (>0.5s) which might indicate speaker changes
        for i in range(1, len(current_words)):
            prev_end = current_words[i - 1].get("end", 0)
            curr_start = current_words[i].get("start", 0)
            pause = curr_start - prev_end

            if pause > 0.5:  # 500ms pause threshold
                return True

        return False

    def start(self):
        """Start the real-time transcription."""
        self.running = True
        self.start_time = time.time()

        # Open output file if specified
        if self.output_file:
            self.output_handle = open(self.output_file, "a", encoding="utf-8")
            header = f"\n{'='*50}\nSession: {datetime.datetime.now().isoformat()}\n{'='*50}\n"
            self.output_handle.write(header)

        # Configure input device
        stream_kwargs = {
            "format": FORMAT,
            "channels": CHANNELS,
            "rate": RATE,
            "input": True,
            "frames_per_buffer": CHUNK_SIZE,
            "stream_callback": self._audio_callback,
        }

        if self.input_device is not None:
            stream_kwargs["input_device_index"] = self.input_device

        # Start audio stream
        try:
            self.stream = self.p.open(**stream_kwargs)
            self.stream.start_stream()
        except Exception as e:
            print(f"\033[91m✗ Failed to open audio stream: {e}\033[0m")
            self._cleanup()
            return

        # Print header
        print("╭─────────────────────────────────────────────────────╮")
        print("│         🎙️  Real-Time Speech Transcription          │")
        print("├─────────────────────────────────────────────────────┤")
        print(f"│  Model: {self.model_name:<10}  Device: {self.device:<8}           │")
        lang_display = self.language or "auto-detect"
        print(f"│  Language: {lang_display:<40} │")
        print("├─────────────────────────────────────────────────────┤")
        print("│  Press Ctrl+C to stop  •  Say 'quit' to exit       │")
        print("╰─────────────────────────────────────────────────────╯\n")

        # Main transcription loop
        try:
            last_level_update = 0

            while self.running and self.stream.is_active():
                # Update audio level display
                now = time.time()
                if now - last_level_update > 0.05:  # 20 FPS update
                    with self.lock:
                        energy = self.current_energy

                    status = "🔴 Recording" if self.is_speaking else "⚪ Listening"
                    bar = audio_level_bar(energy)
                    print(f"\r  {status} {bar}", end="", flush=True)
                    last_level_update = now

                # Check if we have a complete speech segment
                if not self.is_speaking and self.speech_start is not None:
                    audio_segment = self._get_audio_segment()
                    self.speech_start = None

                    if audio_segment is not None:
                        self._transcribe_segment(audio_segment)

                time.sleep(0.01)

        except KeyboardInterrupt:
            print("\n\n\033[93m⚠ Stopping transcription...\033[0m")
        finally:
            self._cleanup()

    def _transcribe_segment(self, audio: np.ndarray):
        """Transcribe an audio segment."""
        # Pad or trim for Whisper
        audio_input = whisper.pad_or_trim(audio)

        # Transcription options
        options = {
            "fp16": self.device == "cuda",
        }

        if self.language:
            options["language"] = self.language

        # Request word timestamps if enabled
        if self.word_timestamps or self.detect_speakers:
            options["word_timestamps"] = True

        # Transcribe
        result = self.model.transcribe(audio_input, **options)
        text = result["text"].strip()
        detected_language = result.get("language", None)

        # Extract word-level data
        words = []
        if self.word_timestamps or self.detect_speakers:
            for segment in result.get("segments", []):
                words.extend(segment.get("words", []))

        # Skip duplicates and empty results
        if self._is_duplicate(text):
            return

        # Detect speaker changes (heuristic)
        speaker_changed = False
        if self.detect_speakers and words:
            speaker_changed = self._detect_speaker_change(words)

        self.transcript_history.append(text)
        self._print_transcript(
            text,
            detected_language,
            words if self.word_timestamps else None,
            speaker_changed,
        )

        # Check for quit command
        if "quit" in text.lower():
            print("\n\033[93m⚠ 'Quit' detected. Stopping...\033[0m")
            self.running = False

    def _cleanup(self):
        """Clean up resources."""
        self.running = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        if self.p is not None:
            self.p.terminate()

        if self.output_handle:
            self.output_handle.close()

        print("\n\033[92m✓ Transcription ended.\033[0m")

        if self.output_file:
            print(f"\033[90m  Transcript saved to: {self.output_file}\033[0m")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time speech transcription with OpenAI Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use defaults (base model, auto language)
  %(prog)s --model small            # Use small model for better accuracy
  %(prog)s --language es            # Transcribe Spanish
  %(prog)s --output transcript.txt  # Save to file
  %(prog)s --list-devices           # Show available microphones
  %(prog)s --device-id 2            # Use specific microphone
        """,
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large", "turbo"],
        help="Whisper model size (default: base)",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="auto",
        help="Language code (e.g., 'en', 'es', 'fr') or 'auto' for detection",
    )

    parser.add_argument(
        "--device",
        "-d",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device (default: auto-detect)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path for saving transcripts",
    )

    parser.add_argument(
        "--timestamps", "-t", action="store_true", help="Include timestamps in output"
    )

    parser.add_argument(
        "--word-timestamps",
        "-w",
        action="store_true",
        help="Show word-level timestamps for each transcription",
    )

    parser.add_argument(
        "--detect-speakers",
        action="store_true",
        help="Enable speaker change detection hints (experimental)",
    )

    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Audio input device ID (use --list-devices to see options)",
    )

    parser.add_argument(
        "--energy-threshold",
        "-e",
        type=float,
        default=VAD_ENERGY_THRESHOLD,
        help=f"VAD energy threshold (default: {VAD_ENERGY_THRESHOLD})",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle device listing
    if args.list_devices:
        list_audio_devices()
        return

    # Create and start transcriber
    transcriber = RealTimeTranscriber(
        model_name=args.model,
        device=args.device,
        language=args.language,
        input_device=args.device_id,
        output_file=args.output,
        show_timestamps=args.timestamps,
        word_timestamps=args.word_timestamps,
        detect_speakers=args.detect_speakers,
        energy_threshold=args.energy_threshold,
    )

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        transcriber.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start transcription
    transcriber.start()


if __name__ == "__main__":
    main()

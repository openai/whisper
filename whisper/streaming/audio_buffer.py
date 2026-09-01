"""
Audio buffer management for real-time streaming transcription.

This module handles audio buffering, chunking, and Voice Activity Detection (VAD)
for efficient real-time processing.
"""

import time
import collections
from typing import Optional, List, Iterator, Tuple, Union
from dataclasses import dataclass
import numpy as np
import threading
import queue


@dataclass
class AudioChunk:
    """Represents a chunk of audio data with metadata."""
    data: np.ndarray
    sample_rate: int
    timestamp: float
    duration: float
    chunk_id: int
    is_silence: bool = False
    vad_confidence: float = 0.0


class AudioBuffer:
    """
    Thread-safe audio buffer for real-time streaming.

    This buffer maintains a sliding window of audio data and provides
    chunks for processing while handling voice activity detection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 1000,
        buffer_duration_ms: int = 5000,
        overlap_duration_ms: int = 200,
        vad_threshold: float = 0.3,
        silence_timeout_ms: int = 1000
    ):
        """
        Initialize the audio buffer.

        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each processing chunk in milliseconds
            buffer_duration_ms: Total buffer duration in milliseconds
            overlap_duration_ms: Overlap between consecutive chunks in milliseconds
            vad_threshold: Voice Activity Detection threshold (0.0-1.0)
            silence_timeout_ms: Timeout for silence detection in milliseconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.buffer_duration_ms = buffer_duration_ms
        self.overlap_duration_ms = overlap_duration_ms
        self.vad_threshold = vad_threshold
        self.silence_timeout_ms = silence_timeout_ms

        # Calculate sizes in samples
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.buffer_size = int(sample_rate * buffer_duration_ms / 1000)
        self.overlap_size = int(sample_rate * overlap_duration_ms / 1000)
        self.silence_timeout_samples = int(sample_rate * silence_timeout_ms / 1000)

        # Initialize buffer
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_position = 0
        self.total_samples_received = 0
        self.chunk_counter = 0

        # Thread safety
        self.lock = threading.RLock()
        self.chunk_queue = queue.Queue()

        # State tracking
        self.last_vad_activity = 0
        self.is_speaking = False
        self.speech_start_sample = None

    def add_audio(self, audio_data: np.ndarray) -> None:
        """
        Add audio data to the buffer.

        Args:
            audio_data: Audio samples as numpy array
        """
        with self.lock:
            audio_data = audio_data.astype(np.float32)
            samples_to_add = len(audio_data)

            # Handle buffer wraparound
            if self.buffer_position + samples_to_add <= self.buffer_size:
                # Fits in buffer without wraparound
                self.buffer[self.buffer_position:self.buffer_position + samples_to_add] = audio_data
            else:
                # Need to wrap around
                first_part_size = self.buffer_size - self.buffer_position
                second_part_size = samples_to_add - first_part_size

                self.buffer[self.buffer_position:] = audio_data[:first_part_size]
                self.buffer[:second_part_size] = audio_data[first_part_size:]

            self.buffer_position = (self.buffer_position + samples_to_add) % self.buffer_size
            self.total_samples_received += samples_to_add

            # Check if we have enough data for a new chunk
            if self.total_samples_received >= self.chunk_size:
                self._create_chunks()

    def _create_chunks(self) -> None:
        """Create audio chunks from the current buffer state."""
        # Calculate how many chunks we can create
        available_samples = min(self.total_samples_received, self.buffer_size)

        # Create chunks with overlap
        chunk_start = 0
        while chunk_start + self.chunk_size <= available_samples:
            chunk_end = chunk_start + self.chunk_size

            # Extract chunk data (handle wraparound)
            start_pos = (self.buffer_position - available_samples + chunk_start) % self.buffer_size
            chunk_data = self._extract_circular_data(start_pos, self.chunk_size)

            # Perform voice activity detection
            vad_confidence = self._calculate_vad(chunk_data)
            is_silence = vad_confidence < self.vad_threshold

            # Update speech state
            timestamp = (self.total_samples_received - available_samples + chunk_start) / self.sample_rate
            self._update_speech_state(vad_confidence, timestamp)

            # Create chunk
            chunk = AudioChunk(
                data=chunk_data,
                sample_rate=self.sample_rate,
                timestamp=timestamp,
                duration=self.chunk_duration_ms / 1000.0,
                chunk_id=self.chunk_counter,
                is_silence=is_silence,
                vad_confidence=vad_confidence
            )

            self.chunk_queue.put(chunk)
            self.chunk_counter += 1

            # Move to next chunk position (with overlap)
            step_size = self.chunk_size - self.overlap_size
            chunk_start += step_size

    def _extract_circular_data(self, start_pos: int, length: int) -> np.ndarray:
        """Extract data from circular buffer."""
        if start_pos + length <= self.buffer_size:
            return self.buffer[start_pos:start_pos + length].copy()
        else:
            # Handle wraparound
            first_part_size = self.buffer_size - start_pos
            second_part_size = length - first_part_size

            result = np.zeros(length, dtype=np.float32)
            result[:first_part_size] = self.buffer[start_pos:]
            result[first_part_size:] = self.buffer[:second_part_size]
            return result

    def _calculate_vad(self, audio_data: np.ndarray) -> float:
        """
        Simple Voice Activity Detection using energy and zero-crossing rate.

        Args:
            audio_data: Audio chunk to analyze

        Returns:
            VAD confidence score (0.0-1.0)
        """
        if len(audio_data) == 0:
            return 0.0

        # Energy-based detection
        energy = np.mean(audio_data ** 2)
        energy_threshold = 0.001  # Adjust based on your use case

        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        zcr_threshold = 0.05

        # Combined score
        energy_score = min(1.0, energy / energy_threshold)
        zcr_score = min(1.0, zero_crossings / zcr_threshold)

        # Weight energy more heavily than ZCR
        vad_score = 0.8 * energy_score + 0.2 * zcr_score
        return min(1.0, vad_score)

    def _update_speech_state(self, vad_confidence: float, timestamp: float) -> None:
        """Update the speech activity state based on VAD results."""
        if vad_confidence >= self.vad_threshold:
            self.last_vad_activity = self.total_samples_received
            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_sample = self.total_samples_received
        else:
            # Check for end of speech
            silence_duration = self.total_samples_received - self.last_vad_activity
            if self.is_speaking and silence_duration > self.silence_timeout_samples:
                self.is_speaking = False
                self.speech_start_sample = None

    def get_chunk(self, timeout: Optional[float] = None) -> Optional[AudioChunk]:
        """
        Get the next available audio chunk.

        Args:
            timeout: Maximum time to wait for a chunk (None for no timeout)

        Returns:
            AudioChunk if available, None if timeout or no chunks
        """
        try:
            return self.chunk_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_chunks_batch(self, max_chunks: int = 10, timeout: float = 0.1) -> List[AudioChunk]:
        """
        Get multiple chunks in a batch.

        Args:
            max_chunks: Maximum number of chunks to return
            timeout: Maximum time to wait for first chunk

        Returns:
            List of AudioChunks (may be empty)
        """
        chunks = []
        try:
            # Get first chunk with timeout
            first_chunk = self.chunk_queue.get(timeout=timeout)
            chunks.append(first_chunk)

            # Get remaining chunks without blocking
            for _ in range(max_chunks - 1):
                try:
                    chunk = self.chunk_queue.get_nowait()
                    chunks.append(chunk)
                except queue.Empty:
                    break

        except queue.Empty:
            pass

        return chunks

    def is_speech_active(self) -> bool:
        """Check if speech is currently being detected."""
        return self.is_speaking

    def get_buffer_info(self) -> dict:
        """Get information about the current buffer state."""
        with self.lock:
            return {
                "buffer_size_samples": self.buffer_size,
                "buffer_size_ms": self.buffer_duration_ms,
                "chunk_size_samples": self.chunk_size,
                "chunk_size_ms": self.chunk_duration_ms,
                "total_samples_received": self.total_samples_received,
                "total_duration_ms": self.total_samples_received / self.sample_rate * 1000,
                "chunks_created": self.chunk_counter,
                "chunks_pending": self.chunk_queue.qsize(),
                "is_speaking": self.is_speaking,
                "buffer_position": self.buffer_position
            }

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        with self.lock:
            self.buffer.fill(0)
            self.buffer_position = 0
            self.total_samples_received = 0
            self.chunk_counter = 0
            self.last_vad_activity = 0
            self.is_speaking = False
            self.speech_start_sample = None

            # Clear the queue
            while not self.chunk_queue.empty():
                try:
                    self.chunk_queue.get_nowait()
                except queue.Empty:
                    break


class StreamingVAD:
    """
    Enhanced Voice Activity Detection for streaming audio.

    Uses a more sophisticated approach than the basic VAD in AudioBuffer.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 20,
        energy_threshold: float = 0.001,
        zcr_threshold: float = 0.05,
        smoothing_window: int = 5
    ):
        """
        Initialize the streaming VAD.

        Args:
            sample_rate: Audio sample rate
            frame_duration_ms: Duration of each analysis frame
            energy_threshold: Energy threshold for speech detection
            zcr_threshold: Zero-crossing rate threshold
            smoothing_window: Number of frames to smooth over
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.energy_threshold = energy_threshold
        self.zcr_threshold = zcr_threshold
        self.smoothing_window = smoothing_window

        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.energy_history = collections.deque(maxlen=smoothing_window)
        self.zcr_history = collections.deque(maxlen=smoothing_window)

    def analyze_frame(self, audio_frame: np.ndarray) -> Tuple[float, dict]:
        """
        Analyze an audio frame for voice activity.

        Args:
            audio_frame: Audio frame data

        Returns:
            Tuple of (vad_probability, analysis_details)
        """
        if len(audio_frame) == 0:
            return 0.0, {}

        # Energy calculation
        energy = np.mean(audio_frame ** 2)
        self.energy_history.append(energy)

        # Zero-crossing rate calculation
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_frame))))
        zcr = zero_crossings / (2 * len(audio_frame))
        self.zcr_history.append(zcr)

        # Smoothed values
        avg_energy = np.mean(self.energy_history)
        avg_zcr = np.mean(self.zcr_history)

        # Speech probability calculation
        energy_score = min(1.0, avg_energy / self.energy_threshold)
        zcr_score = min(1.0, avg_zcr / self.zcr_threshold)

        # Adaptive thresholding based on recent history
        if len(self.energy_history) == self.smoothing_window:
            energy_std = np.std(self.energy_history)
            if energy_std > self.energy_threshold * 0.5:
                # High variability suggests speech
                energy_score *= 1.2

        vad_probability = 0.7 * energy_score + 0.3 * zcr_score
        vad_probability = min(1.0, vad_probability)

        analysis_details = {
            "energy": energy,
            "zcr": zcr,
            "avg_energy": avg_energy,
            "avg_zcr": avg_zcr,
            "energy_score": energy_score,
            "zcr_score": zcr_score,
            "energy_std": np.std(self.energy_history) if len(self.energy_history) > 1 else 0.0
        }

        return vad_probability, analysis_details

    def is_speech(self, vad_probability: float, threshold: float = 0.5) -> bool:
        """Determine if the current frame contains speech."""
        return vad_probability >= threshold
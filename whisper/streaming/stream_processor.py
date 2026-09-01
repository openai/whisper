"""
Real-time streaming processor for Whisper transcription.

This module handles the core streaming logic, integrating audio buffering,
real-time transcription, and result management.
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json

from .audio_buffer import AudioBuffer, AudioChunk, StreamingVAD


class StreamState(Enum):
    """Possible states of the stream processor."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for the stream processor."""
    # Audio settings
    sample_rate: int = 16000
    chunk_duration_ms: int = 1000
    buffer_duration_ms: int = 5000
    overlap_duration_ms: int = 200

    # Transcription settings
    model_name: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"  # "transcribe" or "translate"
    temperature: float = 0.0
    condition_on_previous_text: bool = False

    # Streaming settings
    realtime_factor: float = 1.0  # Target processing speed vs real-time
    max_processing_delay_ms: int = 500
    min_silence_duration_ms: int = 1000
    max_segment_duration_ms: int = 30000

    # VAD settings
    vad_threshold: float = 0.5
    vad_frame_duration_ms: int = 20

    # Performance settings
    use_ctranslate2: bool = False
    device: str = "auto"  # "auto", "cpu", "cuda"
    compute_type: str = "float16"

    # Output settings
    return_timestamps: bool = True
    return_word_timestamps: bool = False
    return_confidence_scores: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StreamConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    text: str
    start_time: float
    end_time: float
    confidence: float
    language: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None
    processing_time_ms: float = 0.0
    is_final: bool = True
    segment_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class StreamProcessor:
    """
    Real-time streaming processor for Whisper transcription.

    This class manages the entire streaming pipeline:
    1. Audio buffering and chunking
    2. Voice Activity Detection
    3. Real-time transcription
    4. Result aggregation and delivery
    """

    def __init__(
        self,
        config: StreamConfig,
        model: Optional[Any] = None,
        result_callback: Optional[Callable[[TranscriptionResult], None]] = None,
        error_callback: Optional[Callable[[Exception], None]] = None
    ):
        """
        Initialize the stream processor.

        Args:
            config: Stream configuration
            model: Pre-loaded Whisper model (optional)
            result_callback: Callback for transcription results
            error_callback: Callback for errors
        """
        self.config = config
        self.model = model
        self.result_callback = result_callback
        self.error_callback = error_callback

        # State management
        self.state = StreamState.STOPPED
        self.start_time = None
        self.total_audio_duration = 0.0
        self.total_processing_time = 0.0

        # Audio processing
        self.audio_buffer = AudioBuffer(
            sample_rate=config.sample_rate,
            chunk_duration_ms=config.chunk_duration_ms,
            buffer_duration_ms=config.buffer_duration_ms,
            overlap_duration_ms=config.overlap_duration_ms,
            vad_threshold=config.vad_threshold
        )

        self.vad = StreamingVAD(
            sample_rate=config.sample_rate,
            frame_duration_ms=config.vad_frame_duration_ms
        )

        # Threading
        self.processing_thread = None
        self.stop_event = threading.Event()

        # Results management
        self.pending_segments = []
        self.completed_segments = []
        self.segment_counter = 0

        # Performance tracking
        self.processing_stats = {
            "chunks_processed": 0,
            "average_processing_time_ms": 0.0,
            "max_processing_time_ms": 0.0,
            "realtime_factor": 0.0,
            "dropped_chunks": 0
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def start(self) -> bool:
        """
        Start the streaming processor.

        Returns:
            True if started successfully, False otherwise
        """
        if self.state != StreamState.STOPPED:
            self.logger.warning(f"Cannot start processor in state {self.state}")
            return False

        try:
            self.state = StreamState.STARTING
            self.start_time = time.time()
            self.stop_event.clear()

            # Initialize model if not provided
            if self.model is None:
                self._load_model()

            # Start processing thread
            self.processing_thread = threading.Thread(
                target=self._processing_loop,
                name="WhisperStreamProcessor",
                daemon=True
            )
            self.processing_thread.start()

            self.state = StreamState.RUNNING
            self.logger.info("Stream processor started successfully")
            return True

        except Exception as e:
            self.state = StreamState.ERROR
            self.logger.error(f"Failed to start stream processor: {e}")
            if self.error_callback:
                self.error_callback(e)
            return False

    def stop(self) -> bool:
        """
        Stop the streaming processor.

        Returns:
            True if stopped successfully, False otherwise
        """
        if self.state not in [StreamState.RUNNING, StreamState.ERROR]:
            return True

        try:
            self.state = StreamState.STOPPING
            self.stop_event.set()

            # Wait for processing thread to finish
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=5.0)

            self.state = StreamState.STOPPED
            self.logger.info("Stream processor stopped successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping stream processor: {e}")
            return False

    def add_audio(self, audio_data: Union[bytes, list, tuple]) -> None:
        """
        Add audio data to the processing pipeline.

        Args:
            audio_data: Audio data as bytes, list, or tuple
        """
        if self.state != StreamState.RUNNING:
            return

        try:
            # Convert to numpy array if needed
            import numpy as np
            if isinstance(audio_data, bytes):
                # Assume 16-bit PCM
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            elif isinstance(audio_data, (list, tuple)):
                audio_array = np.array(audio_data, dtype=np.float32)
            else:
                audio_array = audio_data

            self.audio_buffer.add_audio(audio_array)
            self.total_audio_duration += len(audio_array) / self.config.sample_rate

        except Exception as e:
            self.logger.error(f"Error adding audio data: {e}")
            if self.error_callback:
                self.error_callback(e)

    def _load_model(self) -> None:
        """Load the Whisper model based on configuration."""
        try:
            if self.config.use_ctranslate2:
                from .ctranslate2_backend import CTranslate2Backend
                self.model = CTranslate2Backend(
                    model_name=self.config.model_name,
                    device=self.config.device,
                    compute_type=self.config.compute_type
                )
            else:
                import whisper
                self.model = whisper.load_model(
                    self.config.model_name,
                    device=self.config.device
                )

            self.logger.info(f"Loaded model: {self.config.model_name}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _processing_loop(self) -> None:
        """Main processing loop running in a separate thread."""
        self.logger.info("Processing loop started")

        while not self.stop_event.is_set():
            try:
                # Get audio chunks from buffer
                chunks = self.audio_buffer.get_chunks_batch(
                    max_chunks=5,
                    timeout=0.1
                )

                if not chunks:
                    continue

                # Process chunks
                for chunk in chunks:
                    if self.stop_event.is_set():
                        break

                    self._process_chunk(chunk)

            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                if self.error_callback:
                    self.error_callback(e)

        self.logger.info("Processing loop ended")

    def _process_chunk(self, chunk: AudioChunk) -> None:
        """
        Process a single audio chunk.

        Args:
            chunk: AudioChunk to process
        """
        processing_start = time.time()

        try:
            # Skip processing if chunk is silence (unless we need to finalize a segment)
            if chunk.is_silence and not self._should_process_silence(chunk):
                return

            # Prepare audio for transcription
            audio_data = chunk.data

            # Transcribe using the model
            if self.config.use_ctranslate2:
                result = self._transcribe_with_ctranslate2(audio_data, chunk)
            else:
                result = self._transcribe_with_whisper(audio_data, chunk)

            # Process the transcription result
            if result and result.text.strip():
                self._handle_transcription_result(result, chunk)

            # Update performance stats
            processing_time = (time.time() - processing_start) * 1000
            self._update_processing_stats(processing_time, chunk.duration)

        except Exception as e:
            self.logger.error(f"Error processing chunk {chunk.chunk_id}: {e}")
            self.processing_stats["dropped_chunks"] += 1

    def _should_process_silence(self, chunk: AudioChunk) -> bool:
        """Determine if we should process a silence chunk."""
        # Process silence if we have pending segments that need to be finalized
        return len(self.pending_segments) > 0

    def _transcribe_with_whisper(self, audio_data, chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """Transcribe using standard Whisper model."""
        try:
            # Prepare transcription options
            options = {
                "language": self.config.language,
                "task": self.config.task,
                "temperature": self.config.temperature,
                "condition_on_previous_text": self.config.condition_on_previous_text,
                "word_timestamps": self.config.return_word_timestamps,
            }

            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            # Transcribe
            result = self.model.transcribe(audio_data, **options)

            # Extract text and metadata
            text = result.get("text", "").strip()
            language = result.get("language")
            segments = result.get("segments", [])

            if not text:
                return None

            # Calculate confidence (simplified)
            confidence = 0.8  # Default confidence for standard Whisper
            if segments:
                # Use average log probability if available
                avg_logprobs = [s.get("avg_logprob", -1.0) for s in segments if s.get("avg_logprob")]
                if avg_logprobs:
                    # Convert log probability to confidence (rough approximation)
                    avg_logprob = sum(avg_logprobs) / len(avg_logprobs)
                    confidence = max(0.1, min(0.99, 1.0 + avg_logprob / 2.0))

            return TranscriptionResult(
                text=text,
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
                confidence=confidence,
                language=language,
                chunks=segments if self.config.return_timestamps else None,
                segment_id=f"seg_{self.segment_counter}_{chunk.chunk_id}"
            )

        except Exception as e:
            self.logger.error(f"Error in Whisper transcription: {e}")
            return None

    def _transcribe_with_ctranslate2(self, audio_data, chunk: AudioChunk) -> Optional[TranscriptionResult]:
        """Transcribe using CTranslate2 backend."""
        try:
            result = self.model.transcribe(
                audio_data,
                language=self.config.language,
                task=self.config.task,
                temperature=self.config.temperature,
                return_timestamps=self.config.return_timestamps,
                return_word_timestamps=self.config.return_word_timestamps
            )

            if not result or not result.get("text", "").strip():
                return None

            return TranscriptionResult(
                text=result["text"].strip(),
                start_time=chunk.timestamp,
                end_time=chunk.timestamp + chunk.duration,
                confidence=result.get("confidence", 0.8),
                language=result.get("language"),
                chunks=result.get("segments"),
                segment_id=f"seg_{self.segment_counter}_{chunk.chunk_id}"
            )

        except Exception as e:
            self.logger.error(f"Error in CTranslate2 transcription: {e}")
            return None

    def _handle_transcription_result(self, result: TranscriptionResult, chunk: AudioChunk) -> None:
        """
        Handle a transcription result.

        Args:
            result: Transcription result
            chunk: Source audio chunk
        """
        # Add processing time
        result.processing_time_ms = (time.time() - chunk.timestamp) * 1000

        # Determine if this is a final result
        result.is_final = not self.audio_buffer.is_speech_active()

        # Add to appropriate list
        if result.is_final:
            self.completed_segments.append(result)
            self.segment_counter += 1
        else:
            self.pending_segments.append(result)

        # Send result to callback
        if self.result_callback:
            try:
                self.result_callback(result)
            except Exception as e:
                self.logger.error(f"Error in result callback: {e}")

    def _update_processing_stats(self, processing_time_ms: float, chunk_duration_s: float) -> None:
        """Update processing performance statistics."""
        self.processing_stats["chunks_processed"] += 1
        self.total_processing_time += processing_time_ms / 1000

        # Update average processing time
        count = self.processing_stats["chunks_processed"]
        current_avg = self.processing_stats["average_processing_time_ms"]
        self.processing_stats["average_processing_time_ms"] = (
            (current_avg * (count - 1) + processing_time_ms) / count
        )

        # Update max processing time
        self.processing_stats["max_processing_time_ms"] = max(
            self.processing_stats["max_processing_time_ms"],
            processing_time_ms
        )

        # Calculate realtime factor
        if chunk_duration_s > 0:
            realtime_factor = chunk_duration_s / (processing_time_ms / 1000)
            self.processing_stats["realtime_factor"] = realtime_factor

    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        return {
            "state": self.state.value,
            "uptime_seconds": time.time() - self.start_time if self.start_time else 0,
            "total_audio_duration": self.total_audio_duration,
            "total_processing_time": self.total_processing_time,
            "buffer_info": self.audio_buffer.get_buffer_info(),
            "processing_stats": self.processing_stats.copy(),
            "segments_completed": len(self.completed_segments),
            "segments_pending": len(self.pending_segments)
        }

    def get_results(self, since_segment: int = 0) -> List[TranscriptionResult]:
        """
        Get transcription results.

        Args:
            since_segment: Return results after this segment number

        Returns:
            List of transcription results
        """
        all_results = self.completed_segments + self.pending_segments
        return [result for result in all_results if int(result.segment_id.split('_')[1]) >= since_segment]

    def clear_completed_segments(self) -> None:
        """Clear completed segments to free memory."""
        self.completed_segments.clear()

    def get_config(self) -> StreamConfig:
        """Get current configuration."""
        return self.config
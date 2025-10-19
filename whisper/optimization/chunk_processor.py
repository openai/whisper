"""
Chunk processor for memory-efficient transcription of large audio files.

This module handles intelligent chunking and processing of large audio files
to prevent memory issues and optimize performance.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .memory_manager import MemoryManager, ChunkingStrategy


class ProcessingMode(Enum):
    """Processing modes for chunk processing."""
    SEQUENTIAL = "sequential"      # Process chunks one by one
    PARALLEL = "parallel"          # Process chunks in parallel
    ADAPTIVE = "adaptive"          # Adapt based on system resources


@dataclass
class ChunkResult:
    """Result of processing a single chunk."""
    chunk_id: int
    start_time: float
    end_time: float
    text: str
    segments: List[Dict[str, Any]]
    processing_time: float
    memory_used_gb: float
    confidence_score: float = 0.0
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingStats:
    """Statistics from chunk processing."""
    total_chunks: int
    processed_chunks: int
    failed_chunks: int
    total_processing_time: float
    total_audio_duration: float
    average_processing_time: float
    realtime_factor: float
    peak_memory_usage_gb: float
    memory_cleanups: int


class ChunkProcessor:
    """
    Intelligent chunk processor for large audio files.

    Handles chunking, processing, and result aggregation with memory management
    and performance optimization.
    """

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        max_workers: Optional[int] = None,
        processing_mode: ProcessingMode = ProcessingMode.ADAPTIVE,
        enable_progress_callback: bool = True
    ):
        """
        Initialize chunk processor.

        Args:
            memory_manager: Memory manager instance (created if None)
            max_workers: Maximum number of parallel workers (auto-detect if None)
            processing_mode: Processing mode (sequential, parallel, adaptive)
            enable_progress_callback: Enable progress callbacks
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.max_workers = max_workers or self._determine_optimal_workers()
        self.processing_mode = processing_mode
        self.enable_progress_callback = enable_progress_callback

        # Processing state
        self.current_stats = ProcessingStats(
            total_chunks=0,
            processed_chunks=0,
            failed_chunks=0,
            total_processing_time=0.0,
            total_audio_duration=0.0,
            average_processing_time=0.0,
            realtime_factor=0.0,
            peak_memory_usage_gb=0.0,
            memory_cleanups=0
        )

        # Callbacks
        self.progress_callback: Optional[Callable[[int, int, float], None]] = None
        self.chunk_callback: Optional[Callable[[ChunkResult], None]] = None
        self.error_callback: Optional[Callable[[Exception], None]] = None

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers based on system resources."""
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=False) or 1
            memory_gb = psutil.virtual_memory().total / (1024**3)

            # Conservative worker count based on memory
            # Assume each worker needs ~2GB for Whisper processing
            memory_workers = max(1, int(memory_gb // 2))

            # Use fewer workers than CPU cores to avoid overload
            cpu_workers = max(1, cpu_count - 1)

            # Take the minimum to avoid resource exhaustion
            optimal_workers = min(memory_workers, cpu_workers, 4)  # Cap at 4 workers

            self.logger.info(f"Determined optimal workers: {optimal_workers}")
            return optimal_workers

        except Exception as e:
            self.logger.warning(f"Error determining optimal workers: {e}")
            return 1

    def process_audio_file(
        self,
        audio_path: str,
        model,
        model_size: str = "base",
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_duration: Optional[float] = None,
        overlap_duration: float = 1.0,
        **transcribe_options
    ) -> Dict[str, Any]:
        """
        Process a large audio file using intelligent chunking.

        Args:
            audio_path: Path to audio file
            model: Loaded Whisper model
            model_size: Model size for memory optimization
            language: Source language (None for auto-detect)
            task: Task type (transcribe/translate)
            chunk_duration: Chunk duration (auto-calculate if None)
            overlap_duration: Overlap between chunks
            **transcribe_options: Additional options for transcription

        Returns:
            Dictionary with aggregated results
        """
        start_time = time.time()

        try:
            # Load audio file
            audio_data, audio_duration = self._load_audio_file(audio_path)

            # Calculate chunks
            chunks = self._calculate_chunks(
                audio_duration, model_size, chunk_duration, overlap_duration
            )

            # Initialize stats
            self.current_stats = ProcessingStats(
                total_chunks=len(chunks),
                processed_chunks=0,
                failed_chunks=0,
                total_processing_time=0.0,
                total_audio_duration=audio_duration,
                average_processing_time=0.0,
                realtime_factor=0.0,
                peak_memory_usage_gb=0.0,
                memory_cleanups=0
            )

            # Process chunks
            chunk_results = self._process_chunks(
                audio_data, chunks, model, language, task, **transcribe_options
            )

            # Aggregate results
            final_result = self._aggregate_results(
                chunk_results, audio_duration, overlap_duration
            )

            # Finalize stats
            total_time = time.time() - start_time
            self.current_stats.total_processing_time = total_time
            self.current_stats.average_processing_time = (
                total_time / max(1, self.current_stats.processed_chunks)
            )
            self.current_stats.realtime_factor = audio_duration / total_time

            # Add processing metadata
            final_result.update({
                "processing_stats": self.current_stats.__dict__,
                "chunk_info": {
                    "total_chunks": len(chunks),
                    "chunk_duration": chunk_duration or "auto",
                    "overlap_duration": overlap_duration,
                    "processing_mode": self.processing_mode.value
                }
            })

            self.logger.info(f"Processed {audio_duration:.1f}s audio in {total_time:.1f}s "
                           f"(RTF: {self.current_stats.realtime_factor:.2f})")

            return final_result

        except Exception as e:
            self.logger.error(f"Error processing audio file: {e}")
            if self.error_callback:
                self.error_callback(e)
            raise

    def _load_audio_file(self, audio_path: str) -> Tuple[np.ndarray, float]:
        """Load audio file and return data with duration."""
        try:
            import whisper
            audio_data = whisper.load_audio(audio_path)
            duration = len(audio_data) / 16000.0  # Whisper uses 16kHz

            self.logger.info(f"Loaded audio file: {audio_path} ({duration:.1f}s)")
            return audio_data, duration

        except Exception as e:
            self.logger.error(f"Error loading audio file {audio_path}: {e}")
            raise

    def _calculate_chunks(
        self,
        audio_duration: float,
        model_size: str,
        chunk_duration: Optional[float],
        overlap_duration: float
    ) -> List[Tuple[float, float]]:
        """Calculate optimal chunks for the audio."""
        if chunk_duration is None:
            # Use memory manager to determine optimal chunk size
            chunking_strategy = ChunkingStrategy(self.memory_manager)
            chunks = chunking_strategy.calculate_optimal_chunks(
                audio_duration,
                model_size,
                overlap_seconds=overlap_duration
            )
        else:
            # Use specified chunk duration
            chunks = []
            current_start = 0.0
            while current_start < audio_duration:
                current_end = min(current_start + chunk_duration, audio_duration)
                chunks.append((current_start, current_end))
                current_start = current_end - overlap_duration
                if current_start >= current_end:
                    break

        self.logger.info(f"Split {audio_duration:.1f}s audio into {len(chunks)} chunks")
        return chunks

    def _process_chunks(
        self,
        audio_data: np.ndarray,
        chunks: List[Tuple[float, float]],
        model,
        language: Optional[str],
        task: str,
        **transcribe_options
    ) -> List[ChunkResult]:
        """Process audio chunks using the configured processing mode."""
        if self.processing_mode == ProcessingMode.SEQUENTIAL:
            return self._process_chunks_sequential(
                audio_data, chunks, model, language, task, **transcribe_options
            )
        elif self.processing_mode == ProcessingMode.PARALLEL:
            return self._process_chunks_parallel(
                audio_data, chunks, model, language, task, **transcribe_options
            )
        else:  # ADAPTIVE
            return self._process_chunks_adaptive(
                audio_data, chunks, model, language, task, **transcribe_options
            )

    def _process_chunks_sequential(
        self,
        audio_data: np.ndarray,
        chunks: List[Tuple[float, float]],
        model,
        language: Optional[str],
        task: str,
        **transcribe_options
    ) -> List[ChunkResult]:
        """Process chunks sequentially."""
        results = []

        for i, (start_time, end_time) in enumerate(chunks):
            try:
                with self.memory_manager.memory_context():
                    result = self._process_single_chunk(
                        audio_data, i, start_time, end_time, model,
                        language, task, **transcribe_options
                    )
                    results.append(result)

                    # Update progress
                    self.current_stats.processed_chunks += 1
                    if self.progress_callback:
                        self.progress_callback(
                            self.current_stats.processed_chunks,
                            self.current_stats.total_chunks,
                            (self.current_stats.processed_chunks / self.current_stats.total_chunks) * 100
                        )

                    # Call chunk callback
                    if self.chunk_callback:
                        self.chunk_callback(result)

            except Exception as e:
                self.logger.error(f"Error processing chunk {i}: {e}")
                self.current_stats.failed_chunks += 1

                # Create error result
                error_result = ChunkResult(
                    chunk_id=i,
                    start_time=start_time,
                    end_time=end_time,
                    text="",
                    segments=[],
                    processing_time=0.0,
                    memory_used_gb=0.0,
                    error=str(e)
                )
                results.append(error_result)

        return results

    def _process_chunks_parallel(
        self,
        audio_data: np.ndarray,
        chunks: List[Tuple[float, float]],
        model,
        language: Optional[str],
        task: str,
        **transcribe_options
    ) -> List[ChunkResult]:
        """Process chunks in parallel."""
        results = [None] * len(chunks)  # Pre-allocate results array

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunks
            future_to_index = {}
            for i, (start_time, end_time) in enumerate(chunks):
                future = executor.submit(
                    self._process_single_chunk_with_context,
                    audio_data, i, start_time, end_time, model,
                    language, task, **transcribe_options
                )
                future_to_index[future] = i

            # Collect results as they complete
            for future in as_completed(future_to_index):
                chunk_index = future_to_index[future]
                try:
                    result = future.result()
                    results[chunk_index] = result
                    self.current_stats.processed_chunks += 1

                    # Update progress
                    if self.progress_callback:
                        self.progress_callback(
                            self.current_stats.processed_chunks,
                            self.current_stats.total_chunks,
                            (self.current_stats.processed_chunks / self.current_stats.total_chunks) * 100
                        )

                    # Call chunk callback
                    if self.chunk_callback:
                        self.chunk_callback(result)

                except Exception as e:
                    self.logger.error(f"Error processing chunk {chunk_index}: {e}")
                    self.current_stats.failed_chunks += 1

                    start_time, end_time = chunks[chunk_index]
                    error_result = ChunkResult(
                        chunk_id=chunk_index,
                        start_time=start_time,
                        end_time=end_time,
                        text="",
                        segments=[],
                        processing_time=0.0,
                        memory_used_gb=0.0,
                        error=str(e)
                    )
                    results[chunk_index] = error_result

        # Filter out None results (shouldn't happen, but safety check)
        return [r for r in results if r is not None]

    def _process_chunks_adaptive(
        self,
        audio_data: np.ndarray,
        chunks: List[Tuple[float, float]],
        model,
        language: Optional[str],
        task: str,
        **transcribe_options
    ) -> List[ChunkResult]:
        """Process chunks adaptively based on system resources."""
        # Check memory status to decide processing mode
        memory_status = self.memory_manager.check_memory_status()
        cpu_info = memory_status["cpu"]

        # Use parallel processing if we have sufficient resources
        if cpu_info.usage_percent < 70 and len(chunks) > 2:
            self.logger.info("Using parallel processing mode (sufficient resources)")
            return self._process_chunks_parallel(
                audio_data, chunks, model, language, task, **transcribe_options
            )
        else:
            self.logger.info("Using sequential processing mode (limited resources)")
            return self._process_chunks_sequential(
                audio_data, chunks, model, language, task, **transcribe_options
            )

    def _process_single_chunk_with_context(self, *args, **kwargs) -> ChunkResult:
        """Process a single chunk with memory context (for parallel execution)."""
        with self.memory_manager.memory_context():
            return self._process_single_chunk(*args, **kwargs)

    def _process_single_chunk(
        self,
        audio_data: np.ndarray,
        chunk_id: int,
        start_time: float,
        end_time: float,
        model,
        language: Optional[str],
        task: str,
        **transcribe_options
    ) -> ChunkResult:
        """Process a single audio chunk."""
        chunk_start_time = time.time()

        try:
            # Extract audio chunk
            sample_rate = 16000  # Whisper standard sample rate
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            chunk_audio = audio_data[start_sample:end_sample]

            # Monitor memory before processing
            memory_before = self.memory_manager.check_memory_status()

            # Transcribe chunk
            result = model.transcribe(
                chunk_audio,
                language=language,
                task=task,
                **transcribe_options
            )

            # Monitor memory after processing
            memory_after = self.memory_manager.check_memory_status()
            memory_used = memory_after["cpu"].used_gb - memory_before["cpu"].used_gb

            # Update peak memory usage
            current_usage = memory_after["cpu"].used_gb
            if current_usage > self.current_stats.peak_memory_usage_gb:
                self.current_stats.peak_memory_usage_gb = current_usage

            # Calculate confidence score
            confidence = self._calculate_chunk_confidence(result)

            # Adjust segment timestamps to global time
            adjusted_segments = []
            for segment in result.get("segments", []):
                adjusted_segment = segment.copy()
                adjusted_segment["start"] += start_time
                adjusted_segment["end"] += start_time
                adjusted_segments.append(adjusted_segment)

            processing_time = time.time() - chunk_start_time

            return ChunkResult(
                chunk_id=chunk_id,
                start_time=start_time,
                end_time=end_time,
                text=result.get("text", ""),
                segments=adjusted_segments,
                processing_time=processing_time,
                memory_used_gb=max(0.0, memory_used),
                confidence_score=confidence,
                metadata={
                    "language": result.get("language"),
                    "chunk_duration": end_time - start_time
                }
            )

        except Exception as e:
            processing_time = time.time() - chunk_start_time
            self.logger.error(f"Error processing chunk {chunk_id}: {e}")

            return ChunkResult(
                chunk_id=chunk_id,
                start_time=start_time,
                end_time=end_time,
                text="",
                segments=[],
                processing_time=processing_time,
                memory_used_gb=0.0,
                error=str(e)
            )

    def _calculate_chunk_confidence(self, result: Dict[str, Any]) -> float:
        """Calculate confidence score for a chunk result."""
        try:
            segments = result.get("segments", [])
            if not segments:
                return 0.0

            # Use average log probability as confidence indicator
            log_probs = []
            for segment in segments:
                if "avg_logprob" in segment:
                    log_probs.append(segment["avg_logprob"])

            if log_probs:
                avg_log_prob = sum(log_probs) / len(log_probs)
                # Convert log probability to confidence (rough approximation)
                confidence = max(0.1, min(0.99, 1.0 + avg_log_prob / 2.0))
                return confidence

            return 0.5  # Default confidence when no log probs available

        except Exception:
            return 0.5

    def _aggregate_results(
        self,
        chunk_results: List[ChunkResult],
        total_duration: float,
        overlap_duration: float
    ) -> Dict[str, Any]:
        """Aggregate chunk results into final transcription."""
        # Filter out failed chunks
        successful_results = [r for r in chunk_results if r.error is None]

        if not successful_results:
            return {
                "text": "",
                "segments": [],
                "language": "en",
                "error": "All chunks failed to process"
            }

        # Combine text with overlap handling
        combined_text = self._combine_text_with_overlap(
            successful_results, overlap_duration
        )

        # Combine segments
        all_segments = []
        segment_id = 0
        for result in successful_results:
            for segment in result.segments:
                segment_copy = segment.copy()
                segment_copy["id"] = segment_id
                all_segments.append(segment_copy)
                segment_id += 1

        # Determine primary language
        languages = [r.metadata.get("language") for r in successful_results if r.metadata]
        languages = [lang for lang in languages if lang]
        primary_language = max(set(languages), key=languages.count) if languages else "en"

        # Calculate overall confidence
        confidences = [r.confidence_score for r in successful_results]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": combined_text,
            "segments": all_segments,
            "language": primary_language,
            "confidence": overall_confidence,
            "chunk_results": [r.__dict__ for r in chunk_results]  # Include all chunk info
        }

    def _combine_text_with_overlap(
        self,
        results: List[ChunkResult],
        overlap_duration: float
    ) -> str:
        """Combine text from chunks, handling overlaps intelligently."""
        if not results:
            return ""

        if len(results) == 1:
            return results[0].text.strip()

        combined_parts = []

        for i, result in enumerate(results):
            text = result.text.strip()

            if i == 0:
                # First chunk: use full text
                combined_parts.append(text)
            else:
                # Subsequent chunks: try to detect and remove overlap
                if overlap_duration > 0 and combined_parts:
                    # Simple overlap detection: split text and look for common endings/beginnings
                    words = text.split()
                    if len(words) > 5:
                        # Try to find overlap by comparing last words of previous chunk
                        # with first words of current chunk
                        prev_words = combined_parts[-1].split()

                        # Look for overlap (up to 1/3 of the words in the chunk)
                        max_overlap = min(len(prev_words), len(words)) // 3

                        best_overlap = 0
                        for overlap_size in range(1, max_overlap + 1):
                            if (prev_words[-overlap_size:] == words[:overlap_size] and
                                overlap_size > best_overlap):
                                best_overlap = overlap_size

                        if best_overlap > 0:
                            # Remove overlapped words from current chunk
                            text = " ".join(words[best_overlap:])

                if text:  # Only add if there's remaining text
                    combined_parts.append(text)

        return " ".join(combined_parts)

    def set_progress_callback(self, callback: Callable[[int, int, float], None]) -> None:
        """Set progress callback function."""
        self.progress_callback = callback

    def set_chunk_callback(self, callback: Callable[[ChunkResult], None]) -> None:
        """Set chunk completion callback function."""
        self.chunk_callback = callback

    def set_error_callback(self, callback: Callable[[Exception], None]) -> None:
        """Set error callback function."""
        self.error_callback = callback

    def get_current_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.current_stats


class AdaptiveChunker:
    """Adaptive chunker that adjusts chunk size based on system performance."""

    def __init__(self, memory_manager: Optional[MemoryManager] = None):
        """Initialize adaptive chunker."""
        self.memory_manager = memory_manager or MemoryManager()
        self.performance_history: List[Tuple[float, float, float]] = []  # (chunk_size, processing_time, memory_used)
        self.logger = logging.getLogger(__name__)

    def get_adaptive_chunk_size(
        self,
        base_chunk_size: float,
        model_size: str = "base",
        target_realtime_factor: float = 1.0
    ) -> float:
        """
        Get adaptively adjusted chunk size based on performance history.

        Args:
            base_chunk_size: Base chunk size in seconds
            model_size: Whisper model size
            target_realtime_factor: Target real-time factor (1.0 = real-time)

        Returns:
            Adjusted chunk size in seconds
        """
        if not self.performance_history:
            return base_chunk_size

        # Analyze recent performance
        recent_history = self.performance_history[-10:]  # Last 10 chunks

        avg_processing_time = sum(h[1] for h in recent_history) / len(recent_history)
        avg_chunk_size = sum(h[0] for h in recent_history) / len(recent_history)
        avg_memory_used = sum(h[2] for h in recent_history) / len(recent_history)

        current_rtf = avg_chunk_size / avg_processing_time if avg_processing_time > 0 else 1.0

        # Adjust chunk size based on performance
        if current_rtf < target_realtime_factor * 0.8:
            # Too slow, reduce chunk size
            adjustment_factor = 0.8
            self.logger.info(f"Reducing chunk size due to slow processing (RTF: {current_rtf:.2f})")
        elif current_rtf > target_realtime_factor * 1.5:
            # Too fast, can increase chunk size
            adjustment_factor = 1.2
            self.logger.info(f"Increasing chunk size due to fast processing (RTF: {current_rtf:.2f})")
        else:
            # Performance is acceptable
            adjustment_factor = 1.0

        # Also consider memory usage
        memory_status = self.memory_manager.check_memory_status()
        if memory_status["cpu"].usage_percent > 80:
            adjustment_factor *= 0.9  # Reduce size if memory is high

        adjusted_size = base_chunk_size * adjustment_factor

        # Apply reasonable bounds
        min_size = 5.0   # Minimum 5 seconds
        max_size = 300.0  # Maximum 5 minutes
        adjusted_size = max(min_size, min(adjusted_size, max_size))

        return adjusted_size

    def record_performance(
        self,
        chunk_size: float,
        processing_time: float,
        memory_used: float
    ) -> None:
        """Record performance for adaptive adjustment."""
        self.performance_history.append((chunk_size, processing_time, memory_used))

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

    def reset_history(self) -> None:
        """Reset performance history."""
        self.performance_history.clear()
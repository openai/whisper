"""
Memory management utilities for Whisper transcription.

This module provides intelligent memory management for GPU and CPU resources,
preventing out-of-memory errors and optimizing performance for large audio files.
"""

import gc
import psutil
import time
import logging
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class MemoryLevel(Enum):
    """Memory usage levels."""
    LOW = "low"          # < 30% usage
    MODERATE = "moderate"  # 30-60% usage
    HIGH = "high"        # 60-85% usage
    CRITICAL = "critical"  # > 85% usage


@dataclass
class MemoryInfo:
    """Memory information container."""
    total_gb: float
    used_gb: float
    available_gb: float
    usage_percent: float
    level: MemoryLevel


class MemoryManager:
    """
    System memory manager for CPU and GPU resources.

    Monitors memory usage and provides automatic cleanup and optimization
    to prevent out-of-memory errors during long transcriptions.
    """

    def __init__(
        self,
        cpu_threshold_percent: float = 80.0,
        gpu_threshold_percent: float = 85.0,
        cleanup_interval_seconds: float = 30.0,
        enable_automatic_cleanup: bool = True
    ):
        """
        Initialize the memory manager.

        Args:
            cpu_threshold_percent: CPU memory threshold for triggering cleanup
            gpu_threshold_percent: GPU memory threshold for triggering cleanup
            cleanup_interval_seconds: Interval between automatic cleanup checks
            enable_automatic_cleanup: Enable automatic memory cleanup
        """
        self.cpu_threshold = cpu_threshold_percent
        self.gpu_threshold = gpu_threshold_percent
        self.cleanup_interval = cleanup_interval_seconds
        self.enable_automatic_cleanup = enable_automatic_cleanup

        # State tracking
        self.last_cleanup_time = 0.0
        self.cleanup_callbacks: List[Callable[[], None]] = []
        self.memory_history: List[Tuple[float, MemoryInfo]] = []

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Initialize GPU manager if available
        self.gpu_manager = GPUMemoryManager() if TORCH_AVAILABLE else None

    def get_cpu_memory_info(self) -> MemoryInfo:
        """Get current CPU memory information."""
        memory = psutil.virtual_memory()

        total_gb = memory.total / (1024**3)
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        usage_percent = memory.percent

        # Determine memory level
        if usage_percent < 30:
            level = MemoryLevel.LOW
        elif usage_percent < 60:
            level = MemoryLevel.MODERATE
        elif usage_percent < 85:
            level = MemoryLevel.HIGH
        else:
            level = MemoryLevel.CRITICAL

        return MemoryInfo(
            total_gb=total_gb,
            used_gb=used_gb,
            available_gb=available_gb,
            usage_percent=usage_percent,
            level=level
        )

    def get_gpu_memory_info(self) -> Optional[MemoryInfo]:
        """Get current GPU memory information."""
        if self.gpu_manager:
            return self.gpu_manager.get_memory_info()
        return None

    def check_memory_status(self) -> Dict[str, MemoryInfo]:
        """Check current memory status for all devices."""
        status = {"cpu": self.get_cpu_memory_info()}

        gpu_info = self.get_gpu_memory_info()
        if gpu_info:
            status["gpu"] = gpu_info

        # Store in history
        current_time = time.time()
        self.memory_history.append((current_time, status["cpu"]))

        # Keep only recent history (last hour)
        hour_ago = current_time - 3600
        self.memory_history = [(t, info) for t, info in self.memory_history if t > hour_ago]

        return status

    def is_memory_critical(self) -> bool:
        """Check if any memory is at critical level."""
        status = self.check_memory_status()

        for device, info in status.items():
            if device == "cpu" and info.usage_percent > self.cpu_threshold:
                return True
            elif device == "gpu" and info.usage_percent > self.gpu_threshold:
                return True

        return False

    def cleanup_memory(self, force: bool = False) -> Dict[str, float]:
        """
        Perform memory cleanup.

        Args:
            force: Force cleanup even if thresholds aren't met

        Returns:
            Dictionary with memory freed for each device
        """
        if not force and not self.is_memory_critical():
            return {}

        memory_before = self.check_memory_status()
        cleanup_results = {}

        # Run cleanup callbacks
        for callback in self.cleanup_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.warning(f"Cleanup callback failed: {e}")

        # Python garbage collection
        gc.collect()

        # GPU memory cleanup
        if self.gpu_manager:
            gpu_freed = self.gpu_manager.cleanup_memory()
            if gpu_freed > 0:
                cleanup_results["gpu"] = gpu_freed

        # Check memory after cleanup
        memory_after = self.check_memory_status()

        # Calculate freed memory
        for device in memory_before:
            if device in memory_after:
                freed = memory_before[device].used_gb - memory_after[device].used_gb
                if freed > 0:
                    cleanup_results[device] = freed

        self.last_cleanup_time = time.time()

        if cleanup_results:
            self.logger.info(f"Memory cleanup freed: {cleanup_results}")

        return cleanup_results

    def add_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Add a cleanup callback function."""
        self.cleanup_callbacks.append(callback)

    def remove_cleanup_callback(self, callback: Callable[[], None]) -> None:
        """Remove a cleanup callback function."""
        if callback in self.cleanup_callbacks:
            self.cleanup_callbacks.remove(callback)

    @contextmanager
    def memory_context(self, cleanup_after: bool = True):
        """
        Context manager for automatic memory management.

        Args:
            cleanup_after: Perform cleanup when exiting context
        """
        initial_memory = self.check_memory_status()

        try:
            yield self
        finally:
            if cleanup_after:
                self.cleanup_memory()

            # Log memory usage
            final_memory = self.check_memory_status()
            for device in initial_memory:
                if device in final_memory:
                    initial_used = initial_memory[device].used_gb
                    final_used = final_memory[device].used_gb
                    memory_delta = final_used - initial_used

                    if abs(memory_delta) > 0.1:  # Only log significant changes
                        direction = "increased" if memory_delta > 0 else "decreased"
                        self.logger.info(
                            f"{device.upper()} memory {direction} by {abs(memory_delta):.2f}GB "
                            f"({final_used:.2f}GB used, {final_memory[device].usage_percent:.1f}%)"
                        )

    def auto_cleanup_if_needed(self) -> bool:
        """Automatically cleanup if thresholds are exceeded."""
        if not self.enable_automatic_cleanup:
            return False

        current_time = time.time()

        # Check if enough time has passed since last cleanup
        if current_time - self.last_cleanup_time < self.cleanup_interval:
            return False

        # Check if cleanup is needed
        if self.is_memory_critical():
            self.cleanup_memory()
            return True

        return False

    def get_memory_recommendations(self) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        status = self.check_memory_status()

        cpu_info = status.get("cpu")
        if cpu_info and cpu_info.level == MemoryLevel.CRITICAL:
            recommendations.append("Reduce batch size or chunk duration")
            recommendations.append("Enable memory cleanup callbacks")
            recommendations.append("Consider processing audio in smaller segments")

        gpu_info = status.get("gpu")
        if gpu_info and gpu_info.level == MemoryLevel.CRITICAL:
            recommendations.append("Use smaller Whisper model (e.g., base instead of large)")
            recommendations.append("Enable mixed precision (fp16)")
            recommendations.append("Reduce GPU batch size")
            recommendations.append("Clear GPU cache between segments")

        if cpu_info and cpu_info.level in [MemoryLevel.HIGH, MemoryLevel.CRITICAL]:
            recommendations.append("Close other applications to free memory")
            recommendations.append("Consider using swap memory if available")

        return recommendations

    def get_optimal_chunk_size(self, audio_length_seconds: float, target_memory_gb: float = 2.0) -> float:
        """
        Calculate optimal chunk size based on available memory.

        Args:
            audio_length_seconds: Total audio length
            target_memory_gb: Target memory usage per chunk

        Returns:
            Recommended chunk size in seconds
        """
        cpu_info = self.get_cpu_memory_info()
        available_memory = min(cpu_info.available_gb, target_memory_gb)

        # Rough estimate: 1 second of audio ~ 0.1GB memory for processing
        # This is very approximate and depends on model size and batch size
        memory_per_second = 0.1

        if self.gpu_manager and self.gpu_manager.is_cuda_available():
            # GPU processing is more memory efficient for longer segments
            memory_per_second = 0.05

        optimal_chunk_seconds = available_memory / memory_per_second

        # Clamp to reasonable bounds
        optimal_chunk_seconds = max(10.0, min(optimal_chunk_seconds, 300.0))  # 10s to 5min

        # Ensure we don't exceed total audio length
        optimal_chunk_seconds = min(optimal_chunk_seconds, audio_length_seconds)

        return optimal_chunk_seconds


class GPUMemoryManager:
    """GPU-specific memory management utilities."""

    def __init__(self):
        """Initialize GPU memory manager."""
        self.logger = logging.getLogger(__name__)
        self.device_count = 0

        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device_count = torch.cuda.device_count()

    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return TORCH_AVAILABLE and torch.cuda.is_available()

    def get_memory_info(self, device_id: int = 0) -> Optional[MemoryInfo]:
        """Get GPU memory information for specified device."""
        if not self.is_cuda_available() or device_id >= self.device_count:
            return None

        try:
            # Get memory info from PyTorch
            memory_allocated = torch.cuda.memory_allocated(device_id) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(device_id) / (1024**3)   # GB
            memory_total = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)  # GB

            memory_free = memory_total - memory_reserved
            usage_percent = (memory_reserved / memory_total) * 100

            # Determine memory level
            if usage_percent < 30:
                level = MemoryLevel.LOW
            elif usage_percent < 60:
                level = MemoryLevel.MODERATE
            elif usage_percent < 85:
                level = MemoryLevel.HIGH
            else:
                level = MemoryLevel.CRITICAL

            return MemoryInfo(
                total_gb=memory_total,
                used_gb=memory_reserved,
                available_gb=memory_free,
                usage_percent=usage_percent,
                level=level
            )

        except Exception as e:
            self.logger.error(f"Error getting GPU memory info: {e}")
            return None

    def cleanup_memory(self, device_id: Optional[int] = None) -> float:
        """
        Cleanup GPU memory.

        Args:
            device_id: Specific device to cleanup (None for all devices)

        Returns:
            Amount of memory freed in GB
        """
        if not self.is_cuda_available():
            return 0.0

        memory_before = 0.0
        memory_after = 0.0

        devices_to_cleanup = [device_id] if device_id is not None else range(self.device_count)

        for device in devices_to_cleanup:
            if device < self.device_count:
                try:
                    # Get memory before cleanup
                    if device == 0:  # Only measure for first device to avoid overhead
                        memory_before += torch.cuda.memory_allocated(device) / (1024**3)

                    # Clear cache
                    torch.cuda.empty_cache()

                    # Force garbage collection
                    gc.collect()

                    # Get memory after cleanup
                    if device == 0:
                        memory_after += torch.cuda.memory_allocated(device) / (1024**3)

                except Exception as e:
                    self.logger.warning(f"Error cleaning GPU {device}: {e}")

        memory_freed = max(0.0, memory_before - memory_after)

        if memory_freed > 0.1:  # Log only significant memory freeing
            self.logger.info(f"GPU memory cleanup freed {memory_freed:.2f}GB")

        return memory_freed

    @contextmanager
    def gpu_memory_context(self, device_id: int = 0):
        """Context manager for GPU memory management."""
        if not self.is_cuda_available():
            yield
            return

        # Set device
        original_device = torch.cuda.current_device()
        torch.cuda.set_device(device_id)

        initial_memory = torch.cuda.memory_allocated(device_id) / (1024**3)

        try:
            yield
        finally:
            # Cleanup and restore device
            torch.cuda.empty_cache()
            torch.cuda.set_device(original_device)

            final_memory = torch.cuda.memory_allocated(device_id) / (1024**3)
            memory_delta = final_memory - initial_memory

            if abs(memory_delta) > 0.1:
                direction = "increased" if memory_delta > 0 else "decreased"
                self.logger.debug(
                    f"GPU {device_id} memory {direction} by {abs(memory_delta):.2f}GB"
                )

    def optimize_model_for_memory(self, model, use_half_precision: bool = True) -> Any:
        """
        Optimize model for memory usage.

        Args:
            model: PyTorch model to optimize
            use_half_precision: Use FP16 to reduce memory usage

        Returns:
            Optimized model
        """
        if not self.is_cuda_available():
            return model

        try:
            # Move to GPU if not already
            if hasattr(model, 'device') and model.device.type == 'cpu':
                model = model.cuda()

            # Enable half precision if supported and requested
            if use_half_precision and hasattr(model, 'half'):
                if hasattr(torch.cuda, 'get_device_capability'):
                    # Check if GPU supports half precision
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 6:  # Pascal architecture or newer
                        model = model.half()
                        self.logger.info("Enabled half precision (FP16) for memory optimization")
                else:
                    model = model.half()
                    self.logger.info("Enabled half precision (FP16)")

            # Enable optimizations
            if hasattr(torch, 'compile') and hasattr(model, 'forward'):
                # PyTorch 2.0 compilation (if available)
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    self.logger.info("Applied PyTorch 2.0 compilation optimizations")
                except Exception:
                    pass  # Compilation may not be available

        except Exception as e:
            self.logger.warning(f"Error optimizing model for memory: {e}")

        return model

    def get_device_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPU devices."""
        if not self.is_cuda_available():
            return []

        devices = []
        for i in range(self.device_count):
            try:
                props = torch.cuda.get_device_properties(i)
                memory_info = self.get_memory_info(i)

                device_info = {
                    "device_id": i,
                    "name": props.name,
                    "total_memory_gb": props.total_memory / (1024**3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                    "current_memory_info": memory_info.__dict__ if memory_info else None
                }

                devices.append(device_info)

            except Exception as e:
                self.logger.warning(f"Error getting info for GPU {i}: {e}")

        return devices


class ChunkingStrategy:
    """Strategy for chunking large audio files based on memory constraints."""

    def __init__(self, memory_manager: MemoryManager):
        """Initialize chunking strategy with memory manager."""
        self.memory_manager = memory_manager
        self.logger = logging.getLogger(__name__)

    def calculate_optimal_chunks(
        self,
        audio_length_seconds: float,
        model_size: str = "base",
        target_memory_gb: float = 2.0,
        min_chunk_seconds: float = 10.0,
        max_chunk_seconds: float = 300.0,
        overlap_seconds: float = 1.0
    ) -> List[Tuple[float, float]]:
        """
        Calculate optimal chunk boundaries for processing large audio.

        Args:
            audio_length_seconds: Total audio length
            model_size: Whisper model size for memory estimation
            target_memory_gb: Target memory usage per chunk
            min_chunk_seconds: Minimum chunk size
            max_chunk_seconds: Maximum chunk size
            overlap_seconds: Overlap between chunks

        Returns:
            List of (start_time, end_time) tuples
        """
        # Get optimal chunk size from memory manager
        optimal_chunk_size = self.memory_manager.get_optimal_chunk_size(
            audio_length_seconds, target_memory_gb
        )

        # Adjust based on model size
        model_multipliers = {
            "tiny": 0.5,
            "base": 1.0,
            "small": 1.5,
            "medium": 2.0,
            "large": 3.0,
            "large-v2": 3.5,
            "large-v3": 3.5
        }

        multiplier = model_multipliers.get(model_size, 1.0)
        adjusted_chunk_size = optimal_chunk_size / multiplier

        # Apply bounds
        chunk_size = max(min_chunk_seconds, min(adjusted_chunk_size, max_chunk_seconds))

        # Generate chunks
        chunks = []
        current_start = 0.0

        while current_start < audio_length_seconds:
            current_end = min(current_start + chunk_size, audio_length_seconds)
            chunks.append((current_start, current_end))

            # Next chunk starts with overlap
            current_start = current_end - overlap_seconds
            if current_start >= current_end:
                break

        self.logger.info(f"Generated {len(chunks)} chunks for {audio_length_seconds:.1f}s audio")
        self.logger.info(f"Chunk size: {chunk_size:.1f}s, Overlap: {overlap_seconds:.1f}s")

        return chunks

    def get_memory_efficient_batch_size(self, model_size: str = "base") -> int:
        """Get memory-efficient batch size for the current system."""
        cpu_info = self.memory_manager.get_cpu_memory_info()
        gpu_info = self.memory_manager.get_gpu_memory_info()

        # Base batch size recommendations
        base_batch_sizes = {
            "tiny": 8,
            "base": 4,
            "small": 2,
            "medium": 1,
            "large": 1,
            "large-v2": 1,
            "large-v3": 1
        }

        base_batch_size = base_batch_sizes.get(model_size, 1)

        # Adjust based on available memory
        if gpu_info:
            # GPU processing
            if gpu_info.level == MemoryLevel.CRITICAL:
                return 1  # Minimal batch size
            elif gpu_info.level == MemoryLevel.HIGH:
                return max(1, base_batch_size // 2)
            elif gpu_info.level == MemoryLevel.LOW:
                return base_batch_size * 2
        else:
            # CPU processing
            if cpu_info.level == MemoryLevel.CRITICAL:
                return 1
            elif cpu_info.level == MemoryLevel.HIGH:
                return max(1, base_batch_size // 2)
            elif cpu_info.level == MemoryLevel.LOW:
                return min(base_batch_size * 2, 8)  # Cap CPU batch size

        return base_batch_size
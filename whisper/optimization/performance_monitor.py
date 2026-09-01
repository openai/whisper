"""
Performance monitoring and benchmarking utilities for Whisper optimization.

This module provides comprehensive performance monitoring, benchmarking,
and optimization recommendations for Whisper transcription.
"""

import time
import psutil
import logging
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict, deque
import threading

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


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    processing_time_seconds: float
    audio_duration_seconds: float
    realtime_factor: float
    cpu_usage_percent: float
    memory_usage_gb: float
    gpu_memory_usage_gb: Optional[float]
    model_size: str
    device: str
    batch_size: int
    timestamp: float


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    model_name: str
    device: str
    audio_duration: float
    processing_time: float
    realtime_factor: float
    memory_peak_gb: float
    gpu_memory_peak_gb: Optional[float]
    accuracy_score: Optional[float]
    configuration: Dict[str, Any]
    system_info: Dict[str, Any]


class PerformanceMonitor:
    """
    Real-time performance monitoring for Whisper transcription.

    Tracks processing performance, resource usage, and provides
    optimization recommendations.
    """

    def __init__(
        self,
        max_history_size: int = 1000,
        enable_gpu_monitoring: bool = True,
        sampling_interval: float = 0.1
    ):
        """
        Initialize performance monitor.

        Args:
            max_history_size: Maximum number of metrics to keep in history
            enable_gpu_monitoring: Enable GPU memory monitoring
            sampling_interval: Interval between resource usage samples
        """
        self.max_history_size = max_history_size
        self.enable_gpu_monitoring = enable_gpu_monitoring and TORCH_AVAILABLE
        self.sampling_interval = sampling_interval

        # Performance history
        self.metrics_history: deque = deque(maxlen=max_history_size)

        # Real-time monitoring
        self.current_session = {
            "start_time": None,
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "segments_processed": 0,
            "peak_memory_usage": 0.0,
            "peak_gpu_memory_usage": 0.0 if self.enable_gpu_monitoring else None
        }

        # Resource monitoring
        self.resource_history: List[Dict[str, Any]] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def start_session(self) -> None:
        """Start a monitoring session."""
        self.current_session = {
            "start_time": time.time(),
            "total_audio_processed": 0.0,
            "total_processing_time": 0.0,
            "segments_processed": 0,
            "peak_memory_usage": 0.0,
            "peak_gpu_memory_usage": 0.0 if self.enable_gpu_monitoring else None
        }

        # Start background resource monitoring
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_resources,
            daemon=True
        )
        self.monitoring_thread.start()

        self.logger.info("Performance monitoring session started")

    def stop_session(self) -> Dict[str, Any]:
        """Stop monitoring session and return summary."""
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=1.0)

        session_duration = time.time() - (self.current_session["start_time"] or time.time())

        summary = {
            "session_duration": session_duration,
            "total_audio_processed": self.current_session["total_audio_processed"],
            "total_processing_time": self.current_session["total_processing_time"],
            "segments_processed": self.current_session["segments_processed"],
            "peak_memory_usage_gb": self.current_session["peak_memory_usage"],
            "average_rtf": (
                self.current_session["total_audio_processed"] /
                max(0.001, self.current_session["total_processing_time"])
            ),
            "throughput_minutes_per_hour": (
                (self.current_session["total_audio_processed"] / 60) /
                max(0.001, session_duration / 3600)
            )
        }

        if self.enable_gpu_monitoring:
            summary["peak_gpu_memory_usage_gb"] = self.current_session["peak_gpu_memory_usage"]

        self.logger.info(f"Performance monitoring session ended: RTF={summary['average_rtf']:.2f}")
        return summary

    @contextmanager
    def monitor_transcription(
        self,
        model_size: str,
        device: str,
        batch_size: int = 1
    ):
        """
        Context manager for monitoring a transcription operation.

        Args:
            model_size: Whisper model size
            device: Processing device (cpu/cuda)
            batch_size: Batch size used
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage() if self.enable_gpu_monitoring else None

        try:
            yield self
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage() if self.enable_gpu_monitoring else None

            processing_time = end_time - start_time

            # Create metrics (will be completed by record_transcription)
            self._processing_start_time = start_time
            self._processing_time = processing_time
            self._memory_usage = max(start_memory, end_memory)
            self._gpu_memory_usage = max(start_gpu_memory or 0, end_gpu_memory or 0)
            self._model_size = model_size
            self._device = device
            self._batch_size = batch_size

    def record_transcription(
        self,
        audio_duration: float,
        processing_time: Optional[float] = None,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> PerformanceMetrics:
        """
        Record metrics for a completed transcription.

        Args:
            audio_duration: Duration of processed audio
            processing_time: Time taken for processing
            model_size: Model size used
            device: Processing device
            batch_size: Batch size used

        Returns:
            PerformanceMetrics object
        """
        # Use values from context manager if available
        processing_time = processing_time or getattr(self, '_processing_time', 0.0)
        model_size = model_size or getattr(self, '_model_size', 'unknown')
        device = device or getattr(self, '_device', 'unknown')
        batch_size = batch_size or getattr(self, '_batch_size', 1)

        # Calculate metrics
        realtime_factor = audio_duration / max(0.001, processing_time)
        cpu_usage = psutil.cpu_percent()
        memory_usage = getattr(self, '_memory_usage', self._get_memory_usage())
        gpu_memory_usage = getattr(self, '_gpu_memory_usage', None)

        metrics = PerformanceMetrics(
            processing_time_seconds=processing_time,
            audio_duration_seconds=audio_duration,
            realtime_factor=realtime_factor,
            cpu_usage_percent=cpu_usage,
            memory_usage_gb=memory_usage,
            gpu_memory_usage_gb=gpu_memory_usage,
            model_size=model_size,
            device=device,
            batch_size=batch_size,
            timestamp=time.time()
        )

        # Add to history
        self.metrics_history.append(metrics)

        # Update session stats
        self.current_session["total_audio_processed"] += audio_duration
        self.current_session["total_processing_time"] += processing_time
        self.current_session["segments_processed"] += 1
        self.current_session["peak_memory_usage"] = max(
            self.current_session["peak_memory_usage"], memory_usage
        )

        if gpu_memory_usage is not None:
            self.current_session["peak_gpu_memory_usage"] = max(
                self.current_session["peak_gpu_memory_usage"] or 0, gpu_memory_usage
            )

        # Clean up context manager attributes
        for attr in ['_processing_start_time', '_processing_time', '_memory_usage',
                     '_gpu_memory_usage', '_model_size', '_device', '_batch_size']:
            if hasattr(self, attr):
                delattr(self, attr)

        return metrics

    def get_performance_summary(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """
        Get performance summary statistics.

        Args:
            last_n: Number of recent metrics to analyze (None for all)

        Returns:
            Dictionary with performance statistics
        """
        if not self.metrics_history:
            return {"error": "No performance data available"}

        # Select metrics to analyze
        metrics_to_analyze = list(self.metrics_history)
        if last_n is not None:
            metrics_to_analyze = metrics_to_analyze[-last_n:]

        # Calculate statistics
        rtf_values = [m.realtime_factor for m in metrics_to_analyze]
        processing_times = [m.processing_time_seconds for m in metrics_to_analyze]
        audio_durations = [m.audio_duration_seconds for m in metrics_to_analyze]
        memory_usage = [m.memory_usage_gb for m in metrics_to_analyze]

        # GPU memory (if available)
        gpu_memory = [m.gpu_memory_usage_gb for m in metrics_to_analyze if m.gpu_memory_usage_gb is not None]

        summary = {
            "total_samples": len(metrics_to_analyze),
            "performance": {
                "average_rtf": sum(rtf_values) / len(rtf_values),
                "median_rtf": sorted(rtf_values)[len(rtf_values) // 2],
                "min_rtf": min(rtf_values),
                "max_rtf": max(rtf_values),
                "average_processing_time": sum(processing_times) / len(processing_times),
                "total_audio_processed": sum(audio_durations),
                "total_processing_time": sum(processing_times)
            },
            "resources": {
                "average_memory_usage_gb": sum(memory_usage) / len(memory_usage),
                "peak_memory_usage_gb": max(memory_usage),
                "min_memory_usage_gb": min(memory_usage)
            }
        }

        if gpu_memory:
            summary["resources"]["average_gpu_memory_gb"] = sum(gpu_memory) / len(gpu_memory)
            summary["resources"]["peak_gpu_memory_gb"] = max(gpu_memory)

        # Add model/device breakdown
        model_stats = defaultdict(list)
        device_stats = defaultdict(list)

        for metric in metrics_to_analyze:
            model_stats[metric.model_size].append(metric.realtime_factor)
            device_stats[metric.device].append(metric.realtime_factor)

        summary["breakdown"] = {
            "by_model": {
                model: {
                    "count": len(rtfs),
                    "average_rtf": sum(rtfs) / len(rtfs),
                    "median_rtf": sorted(rtfs)[len(rtfs) // 2]
                }
                for model, rtfs in model_stats.items()
            },
            "by_device": {
                device: {
                    "count": len(rtfs),
                    "average_rtf": sum(rtfs) / len(rtfs),
                    "median_rtf": sorted(rtfs)[len(rtfs) // 2]
                }
                for device, rtfs in device_stats.items()
            }
        }

        return summary

    def get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on performance history."""
        if not self.metrics_history:
            return ["No performance data available for recommendations"]

        recommendations = []
        recent_metrics = list(self.metrics_history)[-20:]  # Last 20 measurements

        # Analyze RTF performance
        avg_rtf = sum(m.realtime_factor for m in recent_metrics) / len(recent_metrics)

        if avg_rtf < 0.5:
            recommendations.append("Consider using a smaller Whisper model (e.g., base instead of large)")
            recommendations.append("Enable GPU acceleration if available")
            recommendations.append("Reduce audio chunk size to lower memory usage")
            recommendations.append("Close other applications to free system resources")

        elif avg_rtf < 1.0:
            recommendations.append("Performance is below real-time. Consider GPU acceleration")
            recommendations.append("Monitor system resources for bottlenecks")

        elif avg_rtf > 5.0:
            recommendations.append("Performance is excellent! Consider using a larger model for better accuracy")
            recommendations.append("You can increase chunk size for more efficient processing")

        # Memory analysis
        avg_memory = sum(m.memory_usage_gb for m in recent_metrics) / len(recent_metrics)

        if avg_memory > 8.0:
            recommendations.append("High memory usage detected. Consider smaller chunk sizes")
            recommendations.append("Enable memory cleanup between segments")

        # GPU memory analysis (if available)
        gpu_metrics = [m for m in recent_metrics if m.gpu_memory_usage_gb is not None]
        if gpu_metrics:
            avg_gpu_memory = sum(m.gpu_memory_usage_gb for m in gpu_metrics) / len(gpu_metrics)
            if avg_gpu_memory > 4.0:
                recommendations.append("High GPU memory usage. Consider using fp16 precision")
                recommendations.append("Reduce batch size or use smaller model")

        # Device-specific recommendations
        device_usage = defaultdict(int)
        for metric in recent_metrics:
            device_usage[metric.device] += 1

        if device_usage.get('cpu', 0) > device_usage.get('cuda', 0):
            if TORCH_AVAILABLE and torch.cuda.is_available():
                recommendations.append("GPU is available but not being used. Enable GPU acceleration")

        # Consistency analysis
        rtf_variance = np.var([m.realtime_factor for m in recent_metrics]) if NUMPY_AVAILABLE else 0
        if rtf_variance > 1.0:
            recommendations.append("Performance is inconsistent. Check for background processes")
            recommendations.append("Consider using fixed chunk sizes for more predictable performance")

        return recommendations or ["Performance looks good! No specific recommendations."]

    def export_metrics(self, filepath: str, format: str = "json") -> None:
        """
        Export performance metrics to file.

        Args:
            filepath: Path to output file
            format: Export format ("json", "csv")
        """
        if format.lower() == "json":
            self._export_json(filepath)
        elif format.lower() == "csv":
            self._export_csv(filepath)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, filepath: str) -> None:
        """Export metrics as JSON."""
        data = {
            "session_info": self.current_session,
            "metrics": [asdict(metric) for metric in self.metrics_history],
            "summary": self.get_performance_summary(),
            "recommendations": self.get_optimization_recommendations(),
            "export_timestamp": time.time()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")

    def _export_csv(self, filepath: str) -> None:
        """Export metrics as CSV."""
        import csv

        with open(filepath, 'w', newline='') as f:
            if not self.metrics_history:
                return

            writer = csv.DictWriter(f, fieldnames=asdict(self.metrics_history[0]).keys())
            writer.writeheader()

            for metric in self.metrics_history:
                writer.writerow(asdict(metric))

        self.logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")

    def _monitor_resources(self) -> None:
        """Background thread for monitoring system resources."""
        while not self.stop_monitoring.wait(self.sampling_interval):
            try:
                resource_sample = {
                    "timestamp": time.time(),
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_gb": psutil.virtual_memory().used / (1024**3)
                }

                if self.enable_gpu_monitoring:
                    gpu_memory = self._get_gpu_memory_usage()
                    if gpu_memory is not None:
                        resource_sample["gpu_memory_gb"] = gpu_memory

                self.resource_history.append(resource_sample)

                # Keep only recent history (last hour)
                cutoff_time = time.time() - 3600
                self.resource_history = [
                    sample for sample in self.resource_history
                    if sample["timestamp"] > cutoff_time
                ]

            except Exception as e:
                self.logger.warning(f"Error in resource monitoring: {e}")

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        return psutil.virtual_memory().used / (1024**3)

    def _get_gpu_memory_usage(self) -> Optional[float]:
        """Get current GPU memory usage in GB."""
        if not self.enable_gpu_monitoring:
            return None

        try:
            return torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            return None


class BenchmarkRunner:
    """
    Comprehensive benchmarking suite for Whisper models.

    Provides standardized benchmarks for comparing performance across
    different models, devices, and configurations.
    """

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        """Initialize benchmark runner."""
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.logger = logging.getLogger(__name__)

    def benchmark_model(
        self,
        model_name: str,
        device: str = "auto",
        test_audio_duration: float = 60.0,
        num_runs: int = 3,
        warmup_runs: int = 1,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """
        Benchmark a specific Whisper model configuration.

        Args:
            model_name: Whisper model to benchmark
            device: Device for testing ("cpu", "cuda", "auto")
            test_audio_duration: Duration of test audio
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            custom_config: Custom configuration parameters

        Returns:
            BenchmarkResult with performance metrics
        """
        self.logger.info(f"Starting benchmark: {model_name} on {device}")

        try:
            # Load model
            import whisper
            model = whisper.load_model(model_name, device=device)

            # Generate test audio
            test_audio = self._generate_test_audio(test_audio_duration)

            # Configuration
            config = {
                "temperature": 0.0,
                "language": None,
                "task": "transcribe"
            }
            if custom_config:
                config.update(custom_config)

            # Warmup runs
            self.logger.info(f"Running {warmup_runs} warmup iterations...")
            for _ in range(warmup_runs):
                model.transcribe(test_audio, **config)
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Benchmark runs
            self.logger.info(f"Running {num_runs} benchmark iterations...")
            results = []

            for run in range(num_runs):
                self.performance_monitor.start_session()

                with self.performance_monitor.monitor_transcription(
                    model_name, device, batch_size=1
                ):
                    start_time = time.time()
                    result = model.transcribe(test_audio, **config)
                    end_time = time.time()

                processing_time = end_time - start_time

                # Record metrics
                metrics = self.performance_monitor.record_transcription(
                    audio_duration=test_audio_duration,
                    processing_time=processing_time,
                    model_size=model_name,
                    device=device
                )

                session_summary = self.performance_monitor.stop_session()
                results.append((metrics, session_summary, result))

                self.logger.info(f"Run {run + 1}/{num_runs}: RTF={metrics.realtime_factor:.2f}")

                # Clean up between runs
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Calculate aggregate results
            processing_times = [r[0].processing_time_seconds for r in results]
            rtf_values = [r[0].realtime_factor for r in results]
            memory_peaks = [r[1]["peak_memory_usage_gb"] for r in results]

            # System info
            system_info = self._get_system_info()

            benchmark_result = BenchmarkResult(
                model_name=model_name,
                device=device,
                audio_duration=test_audio_duration,
                processing_time=sum(processing_times) / len(processing_times),
                realtime_factor=sum(rtf_values) / len(rtf_values),
                memory_peak_gb=max(memory_peaks),
                gpu_memory_peak_gb=max([
                    r[1].get("peak_gpu_memory_usage_gb", 0) or 0 for r in results
                ]) if TORCH_AVAILABLE else None,
                accuracy_score=None,  # Would need reference transcription
                configuration=config,
                system_info=system_info
            )

            self.logger.info(f"Benchmark completed: RTF={benchmark_result.realtime_factor:.2f}")
            return benchmark_result

        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise

    def compare_models(
        self,
        model_names: List[str],
        device: str = "auto",
        test_audio_duration: float = 60.0
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple models on the same test conditions.

        Args:
            model_names: List of model names to compare
            device: Device for testing
            test_audio_duration: Duration of test audio

        Returns:
            Dictionary mapping model names to benchmark results
        """
        results = {}

        for model_name in model_names:
            try:
                self.logger.info(f"Benchmarking {model_name}...")
                results[model_name] = self.benchmark_model(
                    model_name, device, test_audio_duration
                )
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model_name}: {e}")
                results[model_name] = None

        return results

    def _generate_test_audio(self, duration: float) -> np.ndarray:
        """Generate synthetic test audio."""
        sample_rate = 16000
        samples = int(duration * sample_rate)

        # Generate speech-like audio with varying frequencies
        t = np.linspace(0, duration, samples)
        frequencies = 440 + 200 * np.sin(2 * np.pi * 0.5 * t)  # Varying pitch
        audio = 0.3 * np.sin(2 * np.pi * frequencies * t)

        # Add some noise to make it more realistic
        noise = 0.05 * np.random.randn(samples)
        audio = audio + noise

        return audio.astype(np.float32)

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        }

        if TORCH_AVAILABLE and torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
            info["cuda_version"] = torch.version.cuda

        return info
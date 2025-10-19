# Whisper Optimization Module
"""
This module provides memory and performance optimization utilities for OpenAI Whisper.
Includes GPU memory management, efficient chunking, and performance monitoring.
"""

from .memory_manager import MemoryManager, GPUMemoryManager, ChunkingStrategy
from .chunk_processor import ChunkProcessor, AdaptiveChunker
from .performance_monitor import PerformanceMonitor, BenchmarkRunner

__all__ = [
    'MemoryManager',
    'GPUMemoryManager',
    'ChunkingStrategy',
    'ChunkProcessor',
    'AdaptiveChunker',
    'PerformanceMonitor',
    'BenchmarkRunner'
]

# Version info
__version__ = "1.0.0"
# Whisper Real-time Streaming Module
"""
This module provides real-time streaming capabilities for OpenAI Whisper.
Supports WebSocket-based streaming, chunked processing, and low-latency transcription.
"""

from .stream_processor import StreamProcessor, StreamConfig
from .websocket_server import WhisperWebSocketServer
from .audio_buffer import AudioBuffer, AudioChunk
from .ctranslate2_backend import CTranslate2Backend

__all__ = [
    'StreamProcessor',
    'StreamConfig',
    'WhisperWebSocketServer',
    'AudioBuffer',
    'AudioChunk',
    'CTranslate2Backend'
]

# Version info
__version__ = "1.0.0"
"""
WebSocket server for real-time Whisper transcription.

This module provides a WebSocket-based API for streaming audio and receiving
real-time transcription results.
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, Set, Callable
import websockets
import websockets.server
from websockets.exceptions import ConnectionClosed, WebSocketException
import threading
import base64
import struct

from .stream_processor import StreamProcessor, StreamConfig, TranscriptionResult, StreamState


class WhisperWebSocketServer:
    """
    WebSocket server for real-time Whisper transcription.

    Supports multiple concurrent connections with independent processing streams.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        default_config: Optional[StreamConfig] = None,
        model_cache: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the WebSocket server.

        Args:
            host: Server host address
            port: Server port
            default_config: Default stream configuration
            model_cache: Optional model cache for performance
        """
        self.host = host
        self.port = port
        self.default_config = default_config or StreamConfig()
        self.model_cache = model_cache or {}

        # Connection management
        self.active_connections: Set[websockets.WebSocketServerProtocol] = set()
        self.connection_processors: Dict[str, StreamProcessor] = {}
        self.connection_configs: Dict[str, StreamConfig] = {}

        # Server state
        self.server = None
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "connections_total": 0,
            "connections_active": 0,
            "messages_received": 0,
            "messages_sent": 0,
            "audio_bytes_processed": 0,
            "start_time": None
        }

    async def start_server(self) -> None:
        """Start the WebSocket server."""
        try:
            self.stats["start_time"] = time.time()
            self.server = await websockets.serve(
                self._handle_connection,
                self.host,
                self.port,
                ping_interval=20,
                ping_timeout=10,
                max_size=10 * 1024 * 1024,  # 10MB max message size
                compression=None  # Disable compression for audio data
            )

            self.is_running = True
            self.logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

            # Wait until shutdown
            await self.shutdown_event.wait()

        except Exception as e:
            self.logger.error(f"Error starting WebSocket server: {e}")
            raise

    async def stop_server(self) -> None:
        """Stop the WebSocket server."""
        if not self.is_running:
            return

        try:
            self.is_running = False
            self.shutdown_event.set()

            # Stop all active processors
            for processor in self.connection_processors.values():
                processor.stop()

            # Close all connections
            if self.active_connections:
                await asyncio.gather(
                    *[conn.close() for conn in self.active_connections.copy()],
                    return_exceptions=True
                )

            # Close the server
            if self.server:
                self.server.close()
                await self.server.wait_closed()

            self.logger.info("WebSocket server stopped")

        except Exception as e:
            self.logger.error(f"Error stopping WebSocket server: {e}")

    async def _handle_connection(self, websocket: websockets.WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection."""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}:{id(websocket)}"

        self.logger.info(f"New connection: {connection_id}")
        self.active_connections.add(websocket)
        self.stats["connections_total"] += 1
        self.stats["connections_active"] += 1

        try:
            # Initialize connection
            await self._initialize_connection(websocket, connection_id)

            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, connection_id, message)

        except ConnectionClosed:
            self.logger.info(f"Connection closed: {connection_id}")
        except WebSocketException as e:
            self.logger.warning(f"WebSocket error for {connection_id}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error for {connection_id}: {e}")
            await self._send_error(websocket, "Internal server error", str(e))

        finally:
            await self._cleanup_connection(websocket, connection_id)

    async def _initialize_connection(self, websocket: websockets.WebSocketServerProtocol, connection_id: str) -> None:
        """Initialize a new connection."""
        # Send welcome message
        welcome_msg = {
            "type": "connection_established",
            "connection_id": connection_id,
            "server_info": {
                "version": "1.0.0",
                "supported_formats": ["pcm16", "pcm32", "float32"],
                "supported_sample_rates": [8000, 16000, 22050, 44100, 48000],
                "max_audio_chunk_size": 1024 * 1024  # 1MB
            },
            "default_config": self.default_config.to_dict()
        }

        await websocket.send(json.dumps(welcome_msg))

    async def _cleanup_connection(self, websocket: websockets.WebSocketServerProtocol, connection_id: str) -> None:
        """Clean up connection resources."""
        # Remove from active connections
        self.active_connections.discard(websocket)
        self.stats["connections_active"] -= 1

        # Stop processor if exists
        if connection_id in self.connection_processors:
            processor = self.connection_processors[connection_id]
            processor.stop()
            del self.connection_processors[connection_id]

        # Remove config
        self.connection_configs.pop(connection_id, None)

    async def _handle_message(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, message: str) -> None:
        """Handle incoming WebSocket message."""
        self.stats["messages_received"] += 1

        try:
            # Parse JSON message
            if isinstance(message, bytes):
                # Handle binary audio data
                await self._handle_binary_audio(websocket, connection_id, message)
                return

            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "configure":
                await self._handle_configure(websocket, connection_id, data)
            elif message_type == "start_stream":
                await self._handle_start_stream(websocket, connection_id, data)
            elif message_type == "stop_stream":
                await self._handle_stop_stream(websocket, connection_id, data)
            elif message_type == "audio_data":
                await self._handle_audio_data(websocket, connection_id, data)
            elif message_type == "get_status":
                await self._handle_get_status(websocket, connection_id, data)
            elif message_type == "get_results":
                await self._handle_get_results(websocket, connection_id, data)
            else:
                await self._send_error(websocket, "Unknown message type", f"Unsupported message type: {message_type}")

        except json.JSONDecodeError as e:
            await self._send_error(websocket, "Invalid JSON", str(e))
        except Exception as e:
            self.logger.error(f"Error handling message from {connection_id}: {e}")
            await self._send_error(websocket, "Message processing error", str(e))

    async def _handle_configure(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle stream configuration."""
        try:
            config_data = data.get("config", {})
            config = StreamConfig.from_dict({**self.default_config.to_dict(), **config_data})

            self.connection_configs[connection_id] = config

            response = {
                "type": "configuration_updated",
                "config": config.to_dict(),
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(response))

        except Exception as e:
            await self._send_error(websocket, "Configuration error", str(e))

    async def _handle_start_stream(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle stream start request."""
        try:
            # Get or create config
            config = self.connection_configs.get(connection_id, self.default_config)

            # Create result callback
            def result_callback(result: TranscriptionResult):
                asyncio.create_task(self._send_transcription_result(websocket, result))

            def error_callback(error: Exception):
                asyncio.create_task(self._send_error(websocket, "Processing error", str(error)))

            # Create and start processor
            processor = StreamProcessor(
                config=config,
                model=self.model_cache.get(config.model_name),
                result_callback=result_callback,
                error_callback=error_callback
            )

            if processor.start():
                self.connection_processors[connection_id] = processor

                response = {
                    "type": "stream_started",
                    "connection_id": connection_id,
                    "config": config.to_dict(),
                    "timestamp": time.time()
                }
            else:
                response = {
                    "type": "error",
                    "error": "Failed to start stream processor",
                    "timestamp": time.time()
                }

            await websocket.send(json.dumps(response))

        except Exception as e:
            await self._send_error(websocket, "Stream start error", str(e))

    async def _handle_stop_stream(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle stream stop request."""
        try:
            if connection_id in self.connection_processors:
                processor = self.connection_processors[connection_id]
                processor.stop()
                del self.connection_processors[connection_id]

            response = {
                "type": "stream_stopped",
                "connection_id": connection_id,
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(response))

        except Exception as e:
            await self._send_error(websocket, "Stream stop error", str(e))

    async def _handle_audio_data(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle audio data from JSON message."""
        try:
            processor = self.connection_processors.get(connection_id)
            if not processor:
                await self._send_error(websocket, "Stream not started", "Start stream before sending audio")
                return

            # Decode audio data
            audio_format = data.get("format", "pcm16")
            audio_b64 = data.get("audio")

            if not audio_b64:
                await self._send_error(websocket, "Missing audio data", "Audio data field is required")
                return

            audio_bytes = base64.b64decode(audio_b64)
            audio_data = self._decode_audio(audio_bytes, audio_format)

            processor.add_audio(audio_data)
            self.stats["audio_bytes_processed"] += len(audio_bytes)

        except Exception as e:
            await self._send_error(websocket, "Audio processing error", str(e))

    async def _handle_binary_audio(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, audio_bytes: bytes) -> None:
        """Handle binary audio data."""
        try:
            processor = self.connection_processors.get(connection_id)
            if not processor:
                return  # Silently ignore if no processor

            # Assume PCM16 format for binary data
            audio_data = self._decode_audio(audio_bytes, "pcm16")
            processor.add_audio(audio_data)
            self.stats["audio_bytes_processed"] += len(audio_bytes)

        except Exception as e:
            self.logger.error(f"Error processing binary audio from {connection_id}: {e}")

    async def _handle_get_status(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle status request."""
        try:
            processor_status = {}
            if connection_id in self.connection_processors:
                processor = self.connection_processors[connection_id]
                processor_status = processor.get_status()

            response = {
                "type": "status",
                "connection_id": connection_id,
                "processor": processor_status,
                "server_stats": self.stats.copy(),
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(response))

        except Exception as e:
            await self._send_error(websocket, "Status error", str(e))

    async def _handle_get_results(self, websocket: websockets.WebSocketServerProtocol, connection_id: str, data: Dict[str, Any]) -> None:
        """Handle results request."""
        try:
            processor = self.connection_processors.get(connection_id)
            if not processor:
                await self._send_error(websocket, "Stream not started", "No active stream")
                return

            since_segment = data.get("since_segment", 0)
            results = processor.get_results(since_segment)

            response = {
                "type": "results",
                "connection_id": connection_id,
                "results": [result.to_dict() for result in results],
                "total_results": len(results),
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(response))

        except Exception as e:
            await self._send_error(websocket, "Results error", str(e))

    def _decode_audio(self, audio_bytes: bytes, audio_format: str) -> list:
        """Decode audio bytes to list of samples."""
        if audio_format == "pcm16":
            # 16-bit PCM
            samples = struct.unpack(f"<{len(audio_bytes)//2}h", audio_bytes)
            return [s / 32768.0 for s in samples]  # Normalize to [-1, 1]
        elif audio_format == "pcm32":
            # 32-bit PCM
            samples = struct.unpack(f"<{len(audio_bytes)//4}i", audio_bytes)
            return [s / 2147483648.0 for s in samples]  # Normalize to [-1, 1]
        elif audio_format == "float32":
            # 32-bit float
            samples = struct.unpack(f"<{len(audio_bytes)//4}f", audio_bytes)
            return list(samples)
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

    async def _send_transcription_result(self, websocket: websockets.WebSocketServerProtocol, result: TranscriptionResult) -> None:
        """Send transcription result to client."""
        try:
            message = {
                "type": "transcription_result",
                "result": result.to_dict(),
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(message))
            self.stats["messages_sent"] += 1

        except Exception as e:
            self.logger.error(f"Error sending transcription result: {e}")

    async def _send_error(self, websocket: websockets.WebSocketServerProtocol, error_type: str, message: str) -> None:
        """Send error message to client."""
        try:
            error_msg = {
                "type": "error",
                "error_type": error_type,
                "message": message,
                "timestamp": time.time()
            }

            await websocket.send(json.dumps(error_msg))
            self.stats["messages_sent"] += 1

        except Exception as e:
            self.logger.error(f"Error sending error message: {e}")

    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        stats = self.stats.copy()
        if stats["start_time"]:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
        return stats


def run_websocket_server(
    host: str = "localhost",
    port: int = 8765,
    config: Optional[StreamConfig] = None,
    model_cache: Optional[Dict[str, Any]] = None
) -> None:
    """
    Convenience function to run the WebSocket server.

    Args:
        host: Server host
        port: Server port
        config: Default stream configuration
        model_cache: Model cache for performance
    """
    server = WhisperWebSocketServer(host, port, config, model_cache)

    async def run():
        try:
            await server.start_server()
        except KeyboardInterrupt:
            print("\\nShutting down server...")
            await server.stop_server()

    asyncio.run(run())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Whisper WebSocket Server")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--model", default="base", help="Whisper model name")
    parser.add_argument("--device", default="auto", help="Device for inference")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create default config
    default_config = StreamConfig(
        model_name=args.model,
        device=args.device
    )

    print(f"Starting Whisper WebSocket server on ws://{args.host}:{args.port}")
    print(f"Model: {args.model}, Device: {args.device}")

    run_websocket_server(args.host, args.port, default_config)
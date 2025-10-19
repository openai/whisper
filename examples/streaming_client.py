#!/usr/bin/env python3
"""
Example client for Whisper WebSocket streaming server.

This script demonstrates how to connect to the WebSocket server and stream audio
for real-time transcription.
"""

import asyncio
import json
import base64
import numpy as np
import time
import logging
from typing import Optional
import argparse

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class WhisperStreamingClient:
    """Client for connecting to Whisper WebSocket streaming server."""

    def __init__(self, server_url: str = "ws://localhost:8765"):
        """Initialize the streaming client."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library is required: pip install websockets")

        self.server_url = server_url
        self.websocket = None
        self.is_connected = False
        self.is_streaming = False

        # Audio settings
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.audio_format = pyaudio.paInt16 if PYAUDIO_AVAILABLE else None

        # Setup logging
        self.logger = logging.getLogger(__name__)

    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        try:
            self.websocket = await websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=10
            )
            self.is_connected = True
            self.logger.info(f"Connected to {self.server_url}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self) -> None:
        """Disconnect from the WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.logger.info("Disconnected from server")

    async def configure_stream(self, config: dict) -> bool:
        """Configure the streaming parameters."""
        try:
            message = {
                "type": "configure",
                "config": config
            }
            await self.websocket.send(json.dumps(message))

            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("type") == "configuration_updated":
                self.logger.info("Stream configured successfully")
                return True
            else:
                self.logger.error(f"Configuration failed: {response_data}")
                return False

        except Exception as e:
            self.logger.error(f"Error configuring stream: {e}")
            return False

    async def start_stream(self) -> bool:
        """Start the transcription stream."""
        try:
            message = {"type": "start_stream"}
            await self.websocket.send(json.dumps(message))

            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("type") == "stream_started":
                self.is_streaming = True
                self.logger.info("Stream started successfully")
                return True
            else:
                self.logger.error(f"Failed to start stream: {response_data}")
                return False

        except Exception as e:
            self.logger.error(f"Error starting stream: {e}")
            return False

    async def stop_stream(self) -> bool:
        """Stop the transcription stream."""
        try:
            message = {"type": "stop_stream"}
            await self.websocket.send(json.dumps(message))

            # Wait for response
            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("type") == "stream_stopped":
                self.is_streaming = False
                self.logger.info("Stream stopped successfully")
                return True
            else:
                self.logger.error(f"Failed to stop stream: {response_data}")
                return False

        except Exception as e:
            self.logger.error(f"Error stopping stream: {e}")
            return False

    async def send_audio_data(self, audio_data: np.ndarray) -> None:
        """Send audio data to the server."""
        try:
            if not self.is_streaming:
                return

            # Convert audio to bytes
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)

            audio_bytes = audio_data.tobytes()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            message = {
                "type": "audio_data",
                "format": "pcm16",
                "audio": audio_b64
            }

            await self.websocket.send(json.dumps(message))

        except Exception as e:
            self.logger.error(f"Error sending audio data: {e}")

    async def listen_for_results(self) -> None:
        """Listen for transcription results from the server."""
        try:
            while self.is_connected:
                response = await self.websocket.recv()
                response_data = json.loads(response)

                message_type = response_data.get("type")

                if message_type == "transcription_result":
                    self._handle_transcription_result(response_data)
                elif message_type == "error":
                    self._handle_error(response_data)
                elif message_type == "connection_established":
                    self._handle_connection_established(response_data)
                else:
                    self.logger.info(f"Received: {response_data}")

        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Connection closed by server")
        except Exception as e:
            self.logger.error(f"Error listening for results: {e}")

    def _handle_transcription_result(self, data: dict) -> None:
        """Handle transcription result from server."""
        result = data.get("result", {})
        text = result.get("text", "")
        confidence = result.get("confidence", 0.0)
        is_final = result.get("is_final", True)

        status = "FINAL" if is_final else "PARTIAL"
        print(f"[{status}] ({confidence:.2f}): {text}")

    def _handle_error(self, data: dict) -> None:
        """Handle error message from server."""
        error_type = data.get("error_type", "Unknown")
        message = data.get("message", "")
        print(f"ERROR [{error_type}]: {message}")

    def _handle_connection_established(self, data: dict) -> None:
        """Handle connection established message."""
        server_info = data.get("server_info", {})
        print(f"Connected to server version {server_info.get('version', 'unknown')}")
        print(f"Supported formats: {server_info.get('supported_formats', [])}")

    async def get_status(self) -> dict:
        """Get status from the server."""
        try:
            message = {"type": "get_status"}
            await self.websocket.send(json.dumps(message))

            response = await self.websocket.recv()
            response_data = json.loads(response)

            if response_data.get("type") == "status":
                return response_data
            else:
                self.logger.error(f"Unexpected status response: {response_data}")
                return {}

        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}


class MicrophoneStreamer:
    """Stream audio from microphone to Whisper server."""

    def __init__(self, client: WhisperStreamingClient):
        """Initialize microphone streamer."""
        if not PYAUDIO_AVAILABLE:
            raise ImportError("pyaudio library is required: pip install pyaudio")

        self.client = client
        self.audio = None
        self.stream = None
        self.is_recording = False

    def start_recording(self) -> bool:
        """Start recording from microphone."""
        try:
            self.audio = pyaudio.PyAudio()

            self.stream = self.audio.open(
                format=self.client.audio_format,
                channels=self.client.channels,
                rate=self.client.sample_rate,
                input=True,
                frames_per_buffer=self.client.chunk_size
            )

            self.is_recording = True
            print(f"Started recording from microphone (SR: {self.client.sample_rate}Hz)")
            return True

        except Exception as e:
            print(f"Error starting microphone: {e}")
            return False

    def stop_recording(self) -> None:
        """Stop recording from microphone."""
        self.is_recording = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.audio:
            self.audio.terminate()

        print("Stopped recording")

    async def stream_audio(self) -> None:
        """Stream audio from microphone to server."""
        print("Streaming audio... (Press Ctrl+C to stop)")

        try:
            while self.is_recording:
                # Read audio data
                data = self.stream.read(self.client.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Send to server
                await self.client.send_audio_data(audio_data)

                # Small delay to avoid overwhelming the server
                await asyncio.sleep(0.01)

        except KeyboardInterrupt:
            print("\\nStopping audio stream...")
        except Exception as e:
            print(f"Error streaming audio: {e}")


async def run_demo_client(server_url: str, model: str = "base", use_microphone: bool = False):
    """Run a demo of the streaming client."""
    client = WhisperStreamingClient(server_url)

    try:
        # Connect to server
        if not await client.connect():
            return

        # Start listening for results in background
        listen_task = asyncio.create_task(client.listen_for_results())

        # Wait a bit for connection to be established
        await asyncio.sleep(1)

        # Configure stream
        config = {
            "model_name": model,
            "sample_rate": 16000,
            "language": None,  # Auto-detect
            "temperature": 0.0,
            "return_timestamps": True
        }

        if not await client.configure_stream(config):
            return

        # Start stream
        if not await client.start_stream():
            return

        # Stream audio
        if use_microphone and PYAUDIO_AVAILABLE:
            # Use microphone
            mic_streamer = MicrophoneStreamer(client)
            if mic_streamer.start_recording():
                try:
                    await mic_streamer.stream_audio()
                finally:
                    mic_streamer.stop_recording()
        else:
            # Use synthetic audio for demo
            print("Streaming synthetic audio... (Press Ctrl+C to stop)")
            try:
                duration = 0
                while duration < 30:  # Stream for 30 seconds
                    # Generate 1 second of synthetic speech-like audio
                    t = np.linspace(0, 1, 16000)
                    frequency = 440 + 50 * np.sin(2 * np.pi * 0.5 * duration)  # Varying frequency
                    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
                    audio_data = (audio * 32767).astype(np.int16)

                    await client.send_audio_data(audio_data)
                    await asyncio.sleep(1)
                    duration += 1

            except KeyboardInterrupt:
                print("\\nStopping synthetic audio stream...")

        # Stop stream
        await client.stop_stream()

        # Get final status
        status = await client.get_status()
        if status:
            print(f"\\nFinal Status:")
            print(f"  Processor state: {status.get('processor', {}).get('state', 'unknown')}")
            print(f"  Segments processed: {status.get('processor', {}).get('segments_completed', 0)}")

    except Exception as e:
        print(f"Demo error: {e}")

    finally:
        await client.disconnect()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Whisper Streaming Client Demo")
    parser.add_argument("--server", default="ws://localhost:8765", help="WebSocket server URL")
    parser.add_argument("--model", default="base", help="Whisper model to use")
    parser.add_argument("--microphone", action="store_true", help="Use microphone input")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check dependencies
    if not WEBSOCKETS_AVAILABLE:
        print("Error: websockets library is required")
        print("Install with: pip install websockets")
        return

    if args.microphone and not PYAUDIO_AVAILABLE:
        print("Error: pyaudio library is required for microphone input")
        print("Install with: pip install pyaudio")
        return

    print(f"Whisper Streaming Client Demo")
    print(f"Server: {args.server}")
    print(f"Model: {args.model}")
    print(f"Input: {'Microphone' if args.microphone else 'Synthetic audio'}")
    print()

    # Run the demo
    try:
        asyncio.run(run_demo_client(args.server, args.model, args.microphone))
    except KeyboardInterrupt:
        print("\\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed: {e}")


if __name__ == "__main__":
    main()
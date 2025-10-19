# Whisper Real-time Streaming Module

This module provides real-time streaming capabilities for OpenAI Whisper, enabling low-latency transcription for live audio streams.

## Features

- **Real-time Processing**: Stream audio and receive transcription results in real-time
- **WebSocket Server**: WebSocket-based API for easy integration
- **Voice Activity Detection**: Intelligent audio segmentation using VAD
- **CTranslate2 Acceleration**: Optional CTranslate2 backend for faster inference
- **Configurable Buffering**: Adaptive audio buffering with overlap handling
- **Multi-client Support**: Handle multiple concurrent streaming connections

## Quick Start

### Starting the WebSocket Server

```python
from whisper.streaming import WhisperWebSocketServer, StreamConfig

# Create default configuration
config = StreamConfig(
    model_name="base",
    sample_rate=16000,
    chunk_duration_ms=1000,
    language=None  # Auto-detect
)

# Start server
server = WhisperWebSocketServer(
    host="localhost",
    port=8765,
    default_config=config
)

# Run server
import asyncio
asyncio.run(server.start_server())
```

### Using the Stream Processor Directly

```python
from whisper.streaming import StreamProcessor, StreamConfig
import numpy as np

# Configure streaming
config = StreamConfig(
    model_name="base",
    chunk_duration_ms=1000,
    return_timestamps=True
)

# Result callback
def on_result(result):
    print(f"[{result.confidence:.2f}]: {result.text}")

# Create processor
processor = StreamProcessor(config, result_callback=on_result)
processor.start()

# Send audio data
audio_data = np.random.randn(16000).astype(np.float32)  # 1 second of audio
processor.add_audio(audio_data)

# Stop when done
processor.stop()
```

## WebSocket API

### Connection Messages

**Connect**: Connect to `ws://localhost:8765`

**Configure Stream**:
```json
{
    "type": "configure",
    "config": {
        "model_name": "base",
        "sample_rate": 16000,
        "language": "en",
        "temperature": 0.0,
        "return_timestamps": true
    }
}
```

**Start Stream**:
```json
{
    "type": "start_stream"
}
```

**Send Audio**:
```json
{
    "type": "audio_data",
    "format": "pcm16",
    "audio": "<base64-encoded-audio-data>"
}
```

**Stop Stream**:
```json
{
    "type": "stop_stream"
}
```

### Response Messages

**Transcription Result**:
```json
{
    "type": "transcription_result",
    "result": {
        "text": "Hello world",
        "start_time": 0.0,
        "end_time": 2.0,
        "confidence": 0.95,
        "is_final": true,
        "language": "en"
    }
}
```

## Configuration Options

### StreamConfig Parameters

- `sample_rate`: Audio sample rate (default: 16000)
- `chunk_duration_ms`: Duration of each processing chunk (default: 1000)
- `buffer_duration_ms`: Total buffer duration (default: 5000)
- `overlap_duration_ms`: Overlap between chunks (default: 200)
- `model_name`: Whisper model to use (default: "base")
- `language`: Source language (None for auto-detect)
- `temperature`: Sampling temperature (default: 0.0)
- `vad_threshold`: Voice activity detection threshold (default: 0.5)
- `use_ctranslate2`: Enable CTranslate2 acceleration (default: False)
- `device`: Device for inference ("auto", "cpu", "cuda")

## Performance Optimization

### CTranslate2 Backend

For better performance, especially on GPU:

```python
config = StreamConfig(
    model_name="base",
    use_ctranslate2=True,
    device="cuda",
    compute_type="float16"
)
```

### Chunking Strategy

Optimize chunk and buffer sizes based on your use case:

```python
# Low latency (faster response, higher CPU usage)
config = StreamConfig(
    chunk_duration_ms=500,
    buffer_duration_ms=2000,
    overlap_duration_ms=100
)

# Balanced (default settings)
config = StreamConfig(
    chunk_duration_ms=1000,
    buffer_duration_ms=5000,
    overlap_duration_ms=200
)

# High accuracy (slower response, better accuracy)
config = StreamConfig(
    chunk_duration_ms=2000,
    buffer_duration_ms=10000,
    overlap_duration_ms=500
)
```

## Client Examples

### Python WebSocket Client

See `examples/streaming_client.py` for a complete example of connecting to the WebSocket server and streaming audio.

### JavaScript Client

```javascript
const ws = new WebSocket('ws://localhost:8765');

ws.onopen = function() {
    // Configure stream
    ws.send(JSON.stringify({
        type: 'configure',
        config: {
            model_name: 'base',
            language: 'en'
        }
    }));

    // Start stream
    ws.send(JSON.stringify({type: 'start_stream'}));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === 'transcription_result') {
        console.log('Transcription:', data.result.text);
    }
};

// Send audio data
function sendAudio(audioBuffer) {
    const audioBase64 = btoa(String.fromCharCode(...audioBuffer));
    ws.send(JSON.stringify({
        type: 'audio_data',
        format: 'pcm16',
        audio: audioBase64
    }));
}
```

## Supported Audio Formats

- PCM 16-bit (`pcm16`)
- PCM 32-bit (`pcm32`)
- IEEE Float 32-bit (`float32`)
- Sample rates: 8000, 16000, 22050, 44100, 48000 Hz

## Error Handling

The streaming system provides comprehensive error handling:

```python
def error_callback(error):
    print(f"Processing error: {error}")
    # Handle error (retry, fallback, etc.)

processor = StreamProcessor(
    config=config,
    error_callback=error_callback
)
```

## Dependencies

### Required
- `numpy`
- `torch` (for standard Whisper backend)

### Optional
- `ctranslate2` (for accelerated inference)
- `transformers` (for CTranslate2 integration)
- `websockets` (for WebSocket server)
- `pyaudio` (for microphone input in examples)

## Installation

```bash
# Install core streaming dependencies
pip install websockets

# For CTranslate2 acceleration
pip install ctranslate2 transformers

# For microphone input examples
pip install pyaudio
```

## Performance Benchmarks

Typical performance on different hardware:

| Model | Device | RTF* | Latency |
|-------|--------|------|---------|
| tiny | CPU | 0.1x | ~100ms |
| base | CPU | 0.3x | ~300ms |
| small | CPU | 0.6x | ~600ms |
| base | GPU | 0.05x | ~50ms |
| small | GPU | 0.1x | ~100ms |

*RTF = Real-time Factor (lower is better)

## Contributing

When contributing to the streaming module:

1. Maintain backward compatibility
2. Add comprehensive error handling
3. Include performance benchmarks
4. Update documentation
5. Add tests for new features

## License

Same as the main Whisper project (MIT License).
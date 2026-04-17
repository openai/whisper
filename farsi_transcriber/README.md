# Farsi Transcriber

A professional desktop application for transcribing Farsi audio and video files using OpenAI's Whisper model.

## Features

✨ **Core Features**
- 🎙️ Transcribe audio files (MP3, WAV, M4A, FLAC, OGG, AAC, WMA)
- 🎬 Extract audio from video files (MP4, MKV, MOV, WebM, AVI, FLV, WMV)
- 🇮🇷 High-accuracy Farsi/Persian language transcription
- ⏱️ Word-level timestamps for precise timing
- 📤 Export to multiple formats (TXT, SRT, VTT, JSON, TSV)
- 💻 Clean, intuitive PyQt6-based GUI
- 🚀 GPU acceleration support (CUDA) with automatic fallback to CPU
- 🔄 Progress indicators and real-time status updates

## System Requirements

**Minimum:**
- Python 3.8 or higher
- 4GB RAM
- ffmpeg installed

**Recommended:**
- Python 3.10+
- 8GB+ RAM
- NVIDIA GPU with CUDA support (optional but faster)
- SSD for better performance

## Installation

### Step 1: Install ffmpeg

Choose your operating system:

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Fedora/CentOS:**
```bash
sudo dnf install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows (Chocolatey):**
```bash
choco install ffmpeg
```

**Windows (Scoop):**
```bash
scoop install ffmpeg
```

### Step 2: Set up Python environment

```bash
# Navigate to the repository
cd whisper/farsi_transcriber

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyQt6 (GUI framework)
- openai-whisper (transcription engine)
- PyTorch (deep learning framework)
- NumPy, tiktoken, tqdm (supporting libraries)

## Usage

### Running the Application

```bash
python main.py
```

### Step-by-Step Guide

1. **Launch the app** - Run `python main.py`
2. **Select a file** - Click "Select File" button to choose audio/video
3. **Transcribe** - Click "Transcribe" and wait for completion
4. **View results** - See transcription with timestamps
5. **Export** - Click "Export Results" to save in your preferred format

### Supported Export Formats

- **TXT** - Plain text (content only)
- **SRT** - SubRip subtitle format (with timestamps)
- **VTT** - WebVTT subtitle format (with timestamps)
- **JSON** - Structured format with segments and metadata
- **TSV** - Tab-separated values (spreadsheet compatible)

## Configuration

Edit `config.py` to customize:

```python
# Model size (tiny, base, small, medium, large)
DEFAULT_MODEL = "medium"

# Language code
LANGUAGE_CODE = "fa"  # Farsi

# Supported formats
SUPPORTED_AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ...}
SUPPORTED_VIDEO_FORMATS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ...}
```

## Model Information

### Available Models

| Model | Size | Speed | Accuracy | VRAM |
|-------|------|-------|----------|------|
| tiny | 39M | ~10x | Good | ~1GB |
| base | 74M | ~7x | Very Good | ~1GB |
| small | 244M | ~4x | Excellent | ~2GB |
| medium | 769M | ~2x | Excellent | ~5GB |
| large | 1550M | 1x | Best | ~10GB |

**Default**: `medium` (recommended for Farsi)

### Performance Notes

- Larger models provide better accuracy but require more VRAM
- GPU (CUDA) dramatically speeds up transcription (8-10x faster)
- First run downloads the model (~500MB-3GB depending on model size)
- Subsequent runs use cached model files

## Project Structure

```
farsi_transcriber/
├── ui/                          # User interface components
│   ├── __init__.py
│   ├── main_window.py          # Main application window
│   └── styles.py               # Styling and theming
├── models/                      # Model management
│   ├── __init__.py
│   └── whisper_transcriber.py  # Whisper wrapper
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── export.py               # Export functionality
├── config.py                    # Configuration settings
├── main.py                      # Application entry point
├── __init__.py                  # Package init
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Troubleshooting

### Issue: "ffmpeg not found"
**Solution**: Install ffmpeg using your package manager (see Installation section)

### Issue: "CUDA out of memory"
**Solution**: Use a smaller model or reduce audio processing in chunks

### Issue: "Model download fails"
**Solution**: Check internet connection, try again. Models are cached in `~/.cache/whisper/`

### Issue: Slow transcription
**Solution**: Ensure CUDA is detected (`nvidia-smi`), or upgrade to a smaller/faster model

## Advanced Usage

### Custom Model Selection

Update `config.py`:
```python
DEFAULT_MODEL = "large"  # For maximum accuracy
# or
DEFAULT_MODEL = "tiny"   # For fastest processing
```

### Batch Processing (Future)

Script to process multiple files:
```python
from farsi_transcriber.models.whisper_transcriber import FarsiTranscriber

transcriber = FarsiTranscriber(model_name="medium")
for audio_file in audio_files:
    result = transcriber.transcribe(audio_file)
    # Process results
```

## Performance Tips

1. **Use GPU** - Ensure NVIDIA CUDA is properly installed
2. **Choose appropriate model** - Balance speed vs accuracy
3. **Close other applications** - Free up RAM/VRAM
4. **Use SSD** - Faster model loading and temporary file I/O
5. **Local processing** - All processing happens locally, no cloud uploads

## Development

### Code Style

```bash
# Format code
black farsi_transcriber/

# Check style
flake8 farsi_transcriber/

# Sort imports
isort farsi_transcriber/
```

### Future Features

- [ ] Batch processing
- [ ] Real-time transcription preview
- [ ] Speaker diarization
- [ ] Multi-language support UI
- [ ] Settings dialog
- [ ] Keyboard shortcuts
- [ ] Drag-and-drop support
- [ ] Recent files history

## License

MIT License - Personal use and modifications allowed

## Acknowledgments

Built with:
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI framework
- [PyTorch](https://pytorch.org/) - Deep learning

## Support

For issues or suggestions:
1. Check the troubleshooting section
2. Verify ffmpeg is installed
3. Ensure Python 3.8+ is used
4. Check available disk space
5. Verify CUDA setup (for GPU users)

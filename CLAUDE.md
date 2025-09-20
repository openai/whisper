# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is OpenAI's Whisper repository - a general-purpose speech recognition model that performs multilingual speech recognition, speech translation, and language identification. The codebase is built as a Python package with PyTorch.

## Architecture

### Core Components

- **whisper/__init__.py**: Main entry point with model loading (`load_model()`) and available models registry
- **whisper/model.py**: Core Whisper transformer model implementation
- **whisper/transcribe.py**: High-level transcription interface with CLI entry point
- **whisper/decoding.py**: Lower-level decoding logic and options
- **whisper/audio.py**: Audio processing utilities (loading, mel spectrograms, padding)
- **whisper/tokenizer.py**: Text tokenization and language handling
- **whisper/normalizers/**: Text normalization for different languages

### Model Architecture
- Transformer sequence-to-sequence model
- Multiple model sizes: tiny, base, small, medium, large, turbo
- Both English-only (.en) and multilingual variants
- Models downloaded from Azure CDN and cached locally

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test file
pytest -m requires_cuda   # Run CUDA-specific tests
```

### Code Quality
```bash
black .                   # Format code
isort .                   # Sort imports
flake8                    # Lint code
pre-commit run --all-files # Run all pre-commit hooks
```

### Installation for Development
```bash
pip install -e .[dev]     # Install in development mode with dev dependencies
```

## Package Structure

- Built using setuptools with pyproject.toml configuration
- Entry point: `whisper` command maps to `whisper.transcribe:cli`
- Dependencies: torch, numpy, tiktoken, tqdm, numba, more-itertools
- Optional triton dependency for Linux x86_64 optimization

## Key APIs

### High-level Usage
```python
import whisper
model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
```

### Lower-level Usage
```python
audio = whisper.load_audio("audio.mp3")
mel = whisper.log_mel_spectrogram(audio)
result = whisper.decode(model, mel, options)
```

## Testing Notes

- Tests use pytest with custom markers for CUDA requirements
- Random seeds fixed for reproducibility (seed=42)
- Test coverage includes audio processing, normalization, timing, tokenization, and transcription
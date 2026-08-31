# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenAI Whisper is a robust automatic speech recognition (ASR) system built on a Transformer sequence-to-sequence model. It performs multilingual speech recognition, speech translation, spoken language identification, and voice activity detection as a unified multitask model.

## Development Commands

### Installation
```bash
# Install package in development mode with dependencies
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
```

### Code Quality & Linting
```bash
# Format code with black
black .

# Sort imports with isort
isort .

# Lint with flake8
flake8

# Run all pre-commit hooks
pre-commit run --all-files
```

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_transcribe.py

# Run tests requiring CUDA
pytest -m requires_cuda
```

### Package Building
```bash
# Build package
python -m build

# Install built package
pip install dist/openai_whisper-*.whl
```

## Architecture Overview

### Core Components

**whisper/__init__.py**: Main entry point with model loading (`load_model()`) and model registry (`_MODELS` dict mapping model names to download URLs)

**whisper/model.py**:
- `ModelDimensions`: Configuration dataclass for model architecture
- `Whisper`: Main model class implementing the Transformer architecture
- Audio encoder and text decoder components with multi-head attention
- Optimized layers (`LayerNorm`, `Linear`) for mixed-precision training

**whisper/transcribe.py**:
- `transcribe()`: High-level transcription function with sliding window processing
- `cli()`: Command-line interface implementation
- Handles batch processing, temperature sampling, and output formatting

**whisper/decoding.py**:
- `DecodingOptions`/`DecodingResult`: Configuration and result classes
- `decode()`: Core decoding logic with beam search and sampling strategies
- `detect_language()`: Language identification functionality

**whisper/audio.py**: Audio preprocessing utilities including mel-spectrogram computation, padding/trimming to 30-second windows

**whisper/tokenizer.py**: BPE tokenization with special tokens for task specification (transcription vs translation) and language identification

**whisper/timing.py**: Word-level timestamp alignment using cross-attention weights from specific attention heads

**whisper/normalizers/**: Text normalization for different languages to improve transcription accuracy

### Model Pipeline Flow

1. Audio → Mel-spectrogram (whisper/audio.py)
2. Spectrogram → Audio encoder features (whisper/model.py)
3. Language detection via decoder (whisper/decoding.py)
4. Text generation with task-specific tokens (whisper/transcribe.py)
5. Optional word-level timestamp alignment (whisper/timing.py)

### Available Models

Six model sizes with different accuracy/speed tradeoffs:
- `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- English-only variants: `*.en` (better for English)
- Models auto-download to `~/.cache/whisper/`

## Testing Structure

- **tests/conftest.py**: pytest configuration with CUDA markers and random seeds
- **tests/jfk.flac**: Reference audio file for integration tests
- Tests cover audio processing, tokenization, normalization, timing, and transcription functionality

## Code Style

- **Black** formatter (88 char line length)
- **isort** for import sorting (black profile)
- **flake8** linting with specific ignores for E203, E501, W503, W504
- **pre-commit hooks** enforce consistency

## Key Dependencies

- **PyTorch**: Core ML framework
- **tiktoken**: Fast BPE tokenization
- **numba**: JIT compilation for audio processing
- **tqdm**: Progress bars for model downloads and processing
- **triton**: GPU kernel optimization (Linux x86_64)
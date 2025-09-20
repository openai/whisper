# Voice-to-Text Technical Documentation

## Overview

The `voice_to_text.py` script is an enhanced voice recording and transcription system that converts speech to text using OpenAI's Whisper model, with intelligent processing to create optimized prompts for Claude Code.

## Architecture

### Core Components

#### 1. PromptProcessor Class
**Purpose**: Transforms raw transcriptions into Claude Code-optimized prompts

**Key Features**:
- Pattern-based text replacement using regex
- Context-aware transformations for development workflows
- Automatic capitalization and punctuation correction

**Pattern Categories**:
- **Agent References**: Converts natural speech about agents to `@agent` format
- **Tool References**: Transforms tool mentions to `@tool` format
- **File/Directory References**: Standardizes file and directory mentions
- **Code Elements**: Formats function, class, and variable references with backticks
- **Task Management**: Optimizes todo and task-related language
- **Commands**: Standardizes common development commands

#### 2. VoiceRecorder Class
**Purpose**: Handles audio recording, transcription, and output processing

**Key Features**:
- Real-time audio recording with configurable duration
- Whisper model integration for speech-to-text
- Automatic transcript organization in `/transcripts` folder
- Clipboard integration for immediate prompt usage

### Data Flow

1. **Audio Capture**: Records microphone input using sounddevice
2. **Audio Processing**: Converts to WAV format for Whisper compatibility
3. **Transcription**: Uses Whisper model to convert speech to text
4. **Text Processing**: Applies intelligent pattern matching via PromptProcessor
5. **Output**: Displays both raw and processed text, copies to clipboard
6. **Storage**: Saves complete transcript to timestamped file in `/transcripts`

## Configuration

### Audio Settings
- **Sample Rate**: 16kHz (configurable)
- **Channels**: Mono (configurable)
- **Format**: 16-bit WAV for Whisper compatibility

### Whisper Model
- **Default Model**: `base` (good balance of speed/accuracy)
- **Alternatives**: `tiny`, `small`, `medium`, `large`, `turbo`
- **Language**: Auto-detected

### File Organization
- **Transcript Location**: `./transcripts/transcription_YYYYMMDD_HHMMSS.txt`
- **Naming Convention**: ISO timestamp format for chronological sorting
- **Content Structure**: Raw transcription + processed prompt in single file

## Pattern Matching System

### Regex Patterns
The system uses compiled regex patterns for efficient text transformation:

```python
# Example patterns
(r'\\buse agent ([\\w-]+)\\b', r'@agent \\1')           # Agent calls
(r'\\brun tool (\\w+)\\b', r'@tool \\1')               # Tool references
(r'\\bfile ([\\w/\\\\.-]+\\.[\\w]+)\\b', r'@file \\1') # File references
```

### Processing Order
1. Agent and tool references (highest priority)
2. File and directory references
3. Code element formatting
4. Task management language
5. Command standardization
6. Final capitalization and punctuation

## Dependencies

### Required Packages
- `whisper`: OpenAI's speech recognition model
- `sounddevice`: Cross-platform audio recording
- `pyperclip`: Cross-platform clipboard access
- `numpy`: Audio data processing
- `torch`: PyTorch backend for Whisper

### System Requirements
- **Python**: 3.8+
- **Audio**: Microphone access
- **Platform**: Windows/Mac/Linux clipboard support
- **Memory**: Sufficient RAM for Whisper model (varies by model size)

## Error Handling

### Audio Issues
- Microphone permission checks
- Device availability validation
- Recording timeout handling

### Transcription Issues
- Model loading error recovery
- Empty audio detection
- Whisper processing timeouts

### Clipboard Issues
- Platform-specific clipboard access
- Fallback to manual copy instructions
- Graceful degradation when pyperclip fails

## Performance Considerations

### Model Selection
- **tiny**: Fastest, lower accuracy, ~39M parameters
- **base**: Balanced, recommended default, ~74M parameters
- **large**: Highest accuracy, slower, ~1550M parameters

### Memory Usage
- Model caching reduces load times
- Audio buffer management for long recordings
- Transcript file size monitoring

### Processing Speed
- Pattern matching is O(n) with text length
- Clipboard operations are near-instantaneous
- File I/O is optimized for small transcript files

## Extensibility

### Adding New Patterns
```python
# Add to PromptProcessor.__init__()
self.patterns.append((
    r'\\bnew pattern\\b',     # Regex pattern
    r'replacement text'       # Replacement string
))
```

### Custom Processing
The `PromptProcessor.process()` method can be extended with:
- Additional text transformations
- Context-aware replacements
- Language-specific processing
- User-defined pattern files

### Integration Points
- Pre/post-processing hooks
- Custom clipboard formatters
- Alternative output destinations
- Real-time processing callbacks
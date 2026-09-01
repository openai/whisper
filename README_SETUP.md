# Voice to Text Converter - Setup Guide

## Quick Start

### Prerequisites
- Windows 10/11
- Internet connection (for initial setup)
- Microphone access

### Installation Steps

1. **Install Python** (if not already installed)
   - Download from [python.org](https://python.org)
   - **IMPORTANT**: Check "Add Python to PATH" during installation
   - Minimum version: Python 3.8

2. **Run Setup**
   - Double-click `setup.bat`
   - Wait for dependencies to install (may take 5-10 minutes)
   - Setup is complete when you see "Setup Complete!"

3. **Start Using**
   - **GUI Mode**: Double-click `voice_to_text_gui.bat`
   - **Terminal Mode**: Double-click `voice_to_text_terminal.bat`

## Usage Modes

### GUI Mode (Recommended)
- **Launch**: `voice_to_text_gui.bat` (batch window closes, GUI stays open)
- **Features**:
  - Visual interface with buttons
  - F1 global hotkey for recording
  - Always on top option
  - System tray integration
  - Settings dialog
- **Best for**: Daily use and continuous workflow

### Terminal Mode
- **Launch**: `voice_to_text_terminal.bat`
- **Features**:
  - Simple text interface
  - Press Enter to stop recording
  - Lightweight and fast
- **Best for**: Quick one-off recordings

## Creating Desktop Shortcuts

1. **Right-click** on desktop → **New** → **Shortcut**
2. **Browse** to the batch file you want (e.g., `voice_to_text_gui.bat`)
3. **Name** the shortcut (e.g., "Voice to Text")
4. **Optional**: Right-click shortcut → **Properties** → **Change Icon**

## First Run

- **Model Download**: First run will download Whisper model (~150MB)
- **Microphone Permission**: Windows may ask for microphone access
- **Settings**: GUI mode creates `voice_to_text_settings.json` for preferences

## File Structure

```
voice-to-text/
├── voice_to_text.py           # Main application
├── setup.bat                  # One-time setup
├── voice_to_text_gui.bat      # GUI launcher
├── voice_to_text_terminal.bat # Terminal launcher
├── requirements.txt           # Python dependencies
├── transcripts/               # Saved transcriptions
├── voice_to_text_settings.json # Settings (created after first GUI run)
└── README_SETUP.md           # This file
```

## Voice Commands

The system automatically converts natural speech into Claude Code prompts:

| Say This | Gets Converted To |
|----------|-------------------|
| "use agent python pro" | `@agent python-pro` |
| "run tool bash" | `@tool bash` |
| "file package.json" | `@file package.json` |
| "directory source" | `@dir source/` |
| "function get user" | `` `getUser()` function`` |

## Troubleshooting

### "Python is not installed"
- Install Python from [python.org](https://python.org)
- **Must check "Add Python to PATH"** during installation
- Restart command prompt/computer after installation

### "Failed to install dependencies"
- Check internet connection
- Try running `setup.bat` as administrator
- Manually run: `pip install -r requirements.txt`

### "No audio recorded"
- Check microphone permissions in Windows Settings
- Ensure microphone is not muted
- Try a different microphone

### "Poor transcription accuracy"
- Speak clearly and at normal pace
- Reduce background noise
- Move closer to microphone
- In GUI mode: Settings → Change to larger Whisper model

### GUI Hotkey Not Working
- Check if another application is using F1
- Try running as administrator
- Change hotkey in Settings dialog

### System Tray Issues
- If tray doesn't work, app falls back to normal minimize
- Some Windows configurations don't support system tray
- This doesn't affect core functionality

## Advanced Settings (GUI Mode)

Access via Settings button:
- **Global Hotkey**: Change from F1 to F2-F12
- **Whisper Model**: tiny (fast) to large (accurate)
- **Always on Top**: Keep window visible
- **Auto Copy**: Automatically copy to clipboard

## Support

- Check transcripts in `/transcripts` folder
- Settings saved in `voice_to_text_settings.json`
- For issues, check the console output in terminal mode

## Updates

To update the application:
1. Replace `voice_to_text.py` with new version
2. Update `requirements.txt` if needed
3. Run `setup.bat` again if dependencies changed
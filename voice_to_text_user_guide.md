# Voice-to-Text User Guide

## Quick Start

### Terminal Mode (Default)
1. **Run the script**: `python voice_to_text.py`
2. **Choose option 1**: Record (Enter to stop)
3. **Speak your prompt**: Use natural language with smart commands
4. **Get your prompt**: Processed text is automatically copied to clipboard
5. **Paste in Claude Code**: Ctrl+V to paste the optimized prompt

### GUI Mode
1. **Run with UI flag**: `python voice_to_text.py ui`
2. **Click Record button** or **press F1** anywhere to start recording
3. **Speak your prompt**: Use natural language with smart commands
4. **Click Stop** or **press F1 again** to finish recording
5. **Get your prompt**: Processed text is automatically copied to clipboard
6. **Paste in Claude Code**: Ctrl+V to paste the optimized prompt

## Installation

### Prerequisites
- Python 3.8 or higher
- Microphone access
- Internet connection (for initial Whisper model download)

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the script
python voice_to_text.py
```

## How to Use

### Terminal Mode
- **Option 1**: Record (Press Enter to stop)
  - Choose option `1`
  - Speak your prompt after \"Recording...\" appears
  - Press Enter to stop recording
- **Option 2**: Quit
  - Choose option `2` to exit the program

### GUI Mode
- **Global Hotkey**: Press F1 (or custom key) anywhere on your system to start/stop recording
- **Record Button**: Click the microphone button to start/stop recording
- **Visual Feedback**: Button changes color and text during recording
- **Real-time Status**: Status bar shows current recording state
- **Results Display**: Both raw and processed transcriptions shown in text areas
- **Always on Top**: Optional setting to keep window visible above other apps
- **System Tray**: Minimize to tray, access from system tray icon
- **Settings**: Configurable hotkeys, Whisper models, and preferences

### Smart Voice Commands

The system automatically converts natural speech into Claude Code-optimized prompts:

#### Agent Commands
| Say This | Gets Converted To |
|----------|-------------------|
| \"use agent python-pro\" | `@agent python-pro` |
| \"launch agent debug specialist\" | `@agent debug-specialist` |
| \"call agent javascript pro\" | `@agent javascript-pro` |

#### Tool References
| Say This | Gets Converted To |
|----------|-------------------|
| \"run tool bash\" | `@tool bash` |
| \"use the grep tool\" | `@tool grep` |
| \"call the read tool\" | `@tool read` |

#### File & Directory References
| Say This | Gets Converted To |
|----------|-------------------|
| \"directory downloads\" | `@dir downloads/` |
| \"file package.json\" | `@file package.json` |
| \"the readme file\" | `@file README.md` |
| \"folder source\" | `@dir source/` |

#### Code Elements
| Say This | Gets Converted To |
|----------|-------------------|
| \"function get user\" | `` `getUser()` function`` |
| \"class user manager\" | `` `UserManager` class`` |
| \"variable user name\" | `` `userName` variable`` |
| \"method save data\" | `` `saveData()` method`` |

#### Task Management
| Say This | Gets Converted To |
|----------|-------------------|
| \"add to todo\" | `add to todo:` |
| \"new task\" | `new todo:` |
| \"mark complete\" | `mark todo complete` |
| \"mark done\" | `mark todo complete` |

#### Common Commands
| Say This | Gets Converted To |
|----------|-------------------|
| \"run tests\" | `run tests` |
| \"commit changes\" | `commit changes` |
| \"create pull request\" | `create PR` |
| \"install dependencies\" | `install dependencies` |

## Example Workflow

### Terminal Mode Example
1. Run `python voice_to_text.py`
2. Choose option `1` (Record)
3. Speak: *\"Use agent python pro to review file auth.py and run tests\"*
4. Press Enter to stop
5. See processed result: *\"@agent python-pro to review @file auth.py and run tests.\"*
6. Text automatically copied to clipboard
7. Choose option `1` to record again or `2` to quit

### GUI Mode Example
1. Run `python voice_to_text.py ui`
2. Press F1 (or click Record button)
3. Speak: *\"Add to todo fix the authentication bug in function login user\"*
4. Press F1 again (or click Stop)
5. See both raw and processed results in the GUI
6. Processed text: *\"Add to todo: fix the authentication bug in `loginUser()` function.\"*
7. Text automatically copied to clipboard
8. Press F1 again for next recording

### Voice Command Examples

**Testing a Feature**:
- **Say**: *\"I just finished implementing the user authentication feature. Can you use agent python pro to review the code in file auth.py and then run tests to make sure everything works?\"*
- **Gets processed to**: *\"I just finished implementing the user authentication feature. Can you @agent python-pro to review the code in @file auth.py and then run tests to make sure everything works?\"*

**File Operations**:
- **Say**: *\"Please read file package.json and check the dependencies in folder node modules then use tool bash to run npm install\"*
- **Gets processed to**: *\"Please read @file package.json and check the dependencies in @dir node_modules/ then @tool bash to run npm install.\"*

**Task Management**:
- **Say**: *\"Add to todo fix the authentication bug in function login user and mark the previous task as complete\"*
- **Gets processed to**: *\"Add to todo: fix the authentication bug in `loginUser()` function and mark todo complete.\"*

## Tips for Better Results

### Speaking Clearly
- Speak at normal pace (not too fast or slow)
- Use clear pronunciation
- Pause briefly between different concepts
- Speak in a quiet environment

### Effective Commands
- Use specific file names: \"file config.json\" not \"the config file\"
- Mention directories explicitly: \"directory source\" not \"the source\"
- Use consistent naming: \"function getUserData\" not \"the get user data function\"

### Natural Language
- Speak naturally - the system handles capitalization and punctuation
- Use complete sentences when possible
- Don't worry about perfect grammar - focus on clarity

## Output

### What You See
1. **Raw Transcription**: Exactly what Whisper heard
2. **Processed Prompt**: Optimized version for Claude Code
3. **Clipboard Confirmation**: \"✓ Processed prompt copied to clipboard!\"
4. **File Location**: Path to saved transcript in `/transcripts` folder

### File Storage
All transcripts are saved in the `transcripts/` folder with timestamps:
- **Format**: `transcription_YYYYMMDD_HHMMSS.txt`
- **Content**: Both raw and processed versions
- **Sorting**: Files are chronologically ordered

## Mode Comparison

| Feature | Terminal Mode | GUI Mode |
|---------|---------------|----------|
| **Launch** | `python voice_to_text.py` | `python voice_to_text.py ui` |
| **Recording** | Enter to stop | Button or custom hotkey |
| **Global Hotkey** | ❌ No | ✅ Customizable (F1-F12) |
| **Visual Feedback** | Text only | Button colors, status bar |
| **Results Display** | Console output | Scrollable text areas |
| **Multiple Sessions** | Menu driven | Always available |
| **Background Use** | ❌ Terminal focused | ✅ Hotkey works anywhere |
| **Always on Top** | ❌ No | ✅ Optional setting |
| **System Tray** | ❌ No | ✅ Minimize to tray |
| **Settings** | ❌ No | ✅ Full settings dialog |
| **Best For** | Quick one-off recordings | Continuous workflow |

## Advanced Usage

### GUI Settings Dialog

Access settings through:
1. **Settings Button**: Click the ⚙️ Settings button in the GUI
2. **System Tray**: Right-click tray icon → Settings (when minimized)

**Available Settings**:
- **Global Hotkey**: Choose F1-F12 for recording control
- **Whisper Model**: Select from tiny, base, small, medium, large, turbo
- **Always on Top**: Keep window above other applications
- **Minimize to Tray**: Hide to system tray instead of closing
- **Auto Copy Clipboard**: Automatically copy processed text

**Model Trade-offs**:
- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Balanced, recommended (~74M parameters)
- **large**: Most accurate, slower (~1550M parameters)
- **turbo**: Fast and accurate (~809M parameters)

### System Tray Features

When minimized to tray, right-click the tray icon for:
- **Show**: Restore the main window
- **Record**: Start/stop recording directly from tray
- **Settings**: Open settings dialog
- **Quit**: Exit the application completely

### Settings File

Settings are automatically saved to `voice_to_text_settings.json` with:
```json
{
  \"hotkey\": \"f1\",
  \"always_on_top\": false,
  \"minimize_to_tray\": true,
  \"whisper_model\": \"base\",
  \"auto_copy_clipboard\": true
}
```

### Manual Customization

For advanced users, you can:
1. **Add Custom Patterns**: Edit the `PromptProcessor` class patterns list
2. **Modify Default Settings**: Edit `default_settings` in `SettingsManager`
3. **Custom Hotkeys**: Use any key combination supported by pynput

## Troubleshooting

### Audio Issues
**Problem**: \"No microphone detected\"
- **Solution**: Check microphone permissions and connections
- **Windows**: Settings > Privacy > Microphone
- **Mac**: System Preferences > Security & Privacy > Microphone

**Problem**: \"Recording sounds muffled\"
- **Solution**: Check microphone positioning and background noise
- Move closer to microphone
- Reduce background noise

### GUI Mode Issues
**Problem**: \"F1 hotkey not working\"
- **Solution**:
  - Check if another application is using F1
  - Try running as administrator (Windows)
  - Restart the application

**Problem**: \"GUI window not responding\"
- **Solution**:
  - Wait for Whisper model to load (first time is slow)
  - Check task manager for hung processes
  - Restart the application

### Transcription Issues
**Problem**: \"Poor transcription accuracy\"
- **Solution**:
  - Speak more clearly and slowly
  - Reduce background noise
  - Check microphone quality
  - Consider upgrading to larger Whisper model

**Problem**: \"Model loading takes too long\"
- **Solution**: First run downloads the model (~150MB for base model)
- Subsequent runs are much faster
- Consider using smaller `tiny` model for speed

### Clipboard Issues
**Problem**: \"Could not copy to clipboard\"
- **Solution**:
  - Copy the processed text manually
  - Check clipboard permissions
  - Restart the application

### Processing Issues
**Problem**: \"Smart commands not working\"
- **Solution**:
  - Check pronunciation of keywords
  - Use exact phrases from the reference table
  - Speak clearly and pause between concepts

## Advanced Usage

### Changing Whisper Model
Edit line 29 in `voice_to_text.py`:
```python
model = whisper.load_model(\"base\")  # Change to: tiny, small, medium, large, turbo
```

**Model Trade-offs**:
- **tiny**: Fastest, least accurate
- **base**: Balanced (recommended)
- **large**: Most accurate, slower

### Adding Custom Patterns
To add your own smart commands, edit the `PromptProcessor` class patterns list in `voice_to_text.py`.

### Batch Processing
For processing multiple audio files, consider modifying the script to accept file arguments rather than recording live audio.

## Support

### Common Questions
**Q**: Can I use this offline?
**A**: Yes, after the initial model download, everything runs locally.

**Q**: What audio formats are supported?
**A**: The script records in WAV format. For existing files, Whisper supports many formats.

**Q**: Can I change the recording quality?
**A**: Yes, modify the `sample_rate` parameter in the `VoiceRecorder` constructor.

### Getting Help
- Check the technical documentation for implementation details
- Review the troubleshooting section above
- Ensure all dependencies are properly installed
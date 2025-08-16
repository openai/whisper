# Real-Time Whisper Transcription Example

This example demonstrates live microphone transcription using OpenAI's Whisper.

## Features
- Real-time audio capture from microphone
- Automatic sample rate detection
- Continuous transcription output

## Installation
```bash
# System requirements (Linux)
sudo apt install portaudio19-dev alsa-utils

# Python packages
pip install -e .  # Install whisper
pip install sounddevice numpy
```

## Usage
```bash
python examples/realtime_streaming.py
```

## Verification Steps
To confirm accurate transcription:

1. **Test Setup** (run in terminal):
   ```bash
   # Check audio devices
   python3 -c "import sounddevice as sd; print(sd.query_devices())"
   
   # Verify microphone input
   python3 -c "import sounddevice as sd; import numpy as np; \
   def print_vol(indata, frames, time, status): \
       print(f'Volume: {np.sqrt(np.mean(indata**2)):.4f}'); \
   with sd.InputStream(callback=print_vol): sd.sleep(5000)"
   ```
   - Speak normally - you should see volume values between 0.1-0.5
   - If values are <0.01, check mic permissions/volume

2. **Accuracy Test**:
   - Say clearly: "The quick brown fox jumps over the lazy dog"
   - Expected output should match closely
   - If inaccurate, try:
     ```python
     model = whisper.load_model("base")  # In script - more accurate than "tiny"
     ```

## Troubleshooting
| Symptom | Solution |
|---------|----------|
| No transcription | 1. Run `alsamixer` to increase mic volume<br>2. Try different device IDs (0,1,4,11) |
| Wrong words | 1. Speak closer to mic<br>2. Use `model="base"` or `"small"` |
| Delayed output | Reduce `blocksize=1024` in code |

## Expected Output
```
Starting transcription... (Press Ctrl+C to stop)
The quick brown fox jumps over the lazy dog
```
```

### **Key Improvements**:
1. Added **verification steps** to confirm mic is working
2. Included **accuracy testing** with standard test sentence
3. Added **troubleshooting table** for common issues
4. Shows **expected output** example

### **How to Update**:
1. Open `examples/README.md`
2. Replace contents with the above markdown
3. Commit changes:
   ```bash
   git add examples/README.md
   git commit -m "docs: Add detailed verification steps"
   git push
   ```

This will help users (including yourself) verify if the transcription is working properly. The test sentence "The quick brown fox..." is particularly useful because:
- Contains all English letters
- Easy to recognize when correct
- Helps identify specific sound recognition issues


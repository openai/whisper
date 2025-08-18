import whisper
import numpy as np
import sounddevice as sd
import subprocess
import sys

def get_audio_config():
    """Find any working audio configuration"""
    # Try different approaches in order
    approaches = [
        try_standard_rates,
        try_pulseaudio,
        try_direct_hw
    ]
    
    for approach in approaches:
        config = approach()
        if config:
            return config
            
    print("\nFAILED TO FIND WORKING AUDIO CONFIGURATION")
    print("Possible solutions:")
    print("1. Run: sudo apt install alsa-utils pulseaudio")
    print("2. Check mic permissions: ls -l /dev/snd/")
    print("3. Try USB microphone")
    sys.exit(1)

def try_standard_rates():
    """Try common sample rates"""
    for device_id in [None, 0, 1, 4, 11]:
        for rate in [16000, 44100, 48000]:
            try:
                sd.check_input_settings(device=device_id, samplerate=rate)
                return {'device': device_id, 'rate': rate}
            except:
                continue
    return None


def try_pulseaudio():
    """Force PulseAudio configuration"""
    try:
        subprocess.run(['pacmd', 'list-sources'], check=True)
        return {'device': 'pulse', 'rate': 44100}
    except:
        return None

def try_direct_hw():
    """Last-resort direct hardware access"""
    try:
        return {'device': 'hw:0,0', 'rate': 48000}
    except:
        return None

# Load model first to fail fast if issues
MODEL = whisper.load_model("base")

# Get audio config
config = get_audio_config()
print(f"\nUsing audio config: {config}")

def callback(indata, frames, time, status):
    audio = indata[:, 0].astype(np.float32)
    result = MODEL.transcribe(audio)
    if result["text"].strip():
        print(result["text"], end=" ", flush=True)

print("\nStarting transcription... Speak now!")
with sd.InputStream(
    device=config['device'],
    samplerate=config['rate'],
    channels=1,
    blocksize=2048,
    dtype='float32',
    callback=callback
):
    while True:
        sd.sleep(100)
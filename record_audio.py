import os
import pyaudio
import wave
import subprocess
from datetime import datetime

# Constants
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (samples per second)
RECORD_SECONDS = 5     # Duration of recording
# Specify the complete path where you want to save the recordings
OUTPUT_FOLDER = "recordings"

# Create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Initialize PyAudio
audio = pyaudio.PyAudio()

# Create a stream for recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=1024)

print("Recording...")

frames = []

# Record audio data
for _ in range(0, int(RATE / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("Finished recording.")

# Stop and close the audio stream
stream.stop_stream()
stream.close()

# Terminate PyAudio
audio.terminate()

# Generate a dynamic filename based on the current date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"recording_{timestamp}.mp3"

# Save recorded audio to the specified folder with the dynamic filename
output_path = os.path.join(OUTPUT_FOLDER, output_filename)
with wave.open(output_path, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved to {output_path}")



# Define the command to run the other Python script
command = ["python", "test.py",output_path]

# Use subprocess to execute the command
try:
    subprocess.run(command, check=True)
    print("Other script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error executing the other script: {e}")


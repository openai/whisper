import whisper
import time
import numpy as np
from collections import deque
import torch

try:
    import pyaudio
except ImportError:
    print("pyaudio not installed, please install with 'pip install pyaudio'")
    exit()


# Configuration for audio capture
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000 # Whisper's expected sample rate
CHUNK_SIZE = 1024 # Audio buffer size
RECORD_SECONDS_PER_SEGMENT = 5 # Process audio in 5-second chunks
MAX_AUDIO_BUFFER_SECONDS = 15 # Keep up to 15 seconds of audio for context

class RealTimeTranscriber:
    def __init__(self, model_name="base", device=None):
        """
        Initializes the real-time transcriber.

        Args:
            model_name (str): The Whisper model to use (e.g., "tiny", "base", "small").
            device (str, optional): "cuda" for GPU, "cpu" for CPU. Defaults to None (auto).
        """
        print(f"Loading Whisper model '{model_name}'...")
        self.model = whisper.load_model(model_name, device=device)
        print("Model loaded.")

        self.p = pyaudio.PyAudio()
        self.stream = None
        self.audio_buffer = deque() # Stores raw audio chunks
        self.buffer_duration = 0.0 # Current duration of audio in buffer
        self.total_processed_duration = 0.0 # Total duration of audio already transcribed

        # Calculate max buffer size in chunks
        self.max_buffer_chunks = int(MAX_AUDIO_BUFFER_SECONDS * (RATE / CHUNK_SIZE))

        # Calculate chunks per segment
        self.chunks_per_segment = int(RECORD_SECONDS_PER_SEGMENT * (RATE / CHUNK_SIZE))

    def start_stream(self):
        """Starts the audio input stream."""
        print("Starting audio stream...")
        self.stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK_SIZE,
                                stream_callback=self._audio_callback)
        self.stream.start_stream()
        print("Audio stream started. Start speaking...")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function to append audio chunks to the buffer."""
        self.audio_buffer.append(in_data)
        self.buffer_duration += frame_count / RATE

        # Keep buffer under max_buffer_chunks
        while len(self.audio_buffer) > self.max_buffer_chunks:
            removed_data = self.audio_buffer.popleft()
            self.buffer_duration -= len(removed_data) / (2 * CHANNELS * RATE) # 2 bytes per sample for paInt16
        return (in_data, pyaudio.paContinue)

    def stop_stream(self):
        """Stops and cleans up the audio input stream."""
        if self.stream is not None:
            print("Stopping audio stream...")
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        print("Audio stream stopped and resources released.")

    def get_audio_segment(self):
        """
        Extracts a segment of audio from the buffer for transcription.
        Returns a numpy array of the audio, or None if not enough audio.
        """
        if self.buffer_duration < RECORD_SECONDS_PER_SEGMENT:
            return None

        # Convert deque of bytes to a single numpy array
        segment_bytes = b"".join(list(self.audio_buffer)[:self.chunks_per_segment])
        audio_np = np.frombuffer(segment_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return audio_np

    def transcribe_loop(self):
        """Continuously transcribes audio from the microphone."""
        try:
            self.start_stream()
            print("\n[INFO] Transcribing in real-time. Say 'quit' to exit.")

            last_transcript = ""
            while self.stream.is_active():
                time.sleep(0.1) # Small delay to prevent busy-waiting

                if self.buffer_duration >= RECORD_SECONDS_PER_SEGMENT:
                    audio_segment_np = self.get_audio_segment()
                    if audio_segment_np is None:
                        continue

                    # Remove transcribed segment from buffer (assuming successful transcription)
                    # This is a bit simplistic; for full robustness, you might only pop after successful result.
                    for _ in range(self.chunks_per_segment):
                        if self.audio_buffer:
                            removed_data = self.audio_buffer.popleft()
                            self.buffer_duration -= len(removed_data) / (2 * CHANNELS * RATE)


                    # Pad or trim to 30 seconds as Whisper expects
                    # This is simplified for real-time segments, Whisper will handle padding internally
                    # if the input is shorter than 30s during actual transcription call.
                    # For better context, a sliding window on a larger buffer might be used.
                    
                    # Whisper expects 16kHz, single-channel float32
                    audio_input = whisper.pad_or_trim(audio_segment_np)

                    # Transcribe the audio segment
                    result = self.model.transcribe(audio_input, language="en", fp16=torch.cuda.is_available())
                    current_transcript = result["text"].strip()

                    if current_transcript and current_transcript != last_transcript:
                        print(f"You said: {current_transcript}")
                        last_transcript = current_transcript
                        if "quit" in current_transcript.lower():
                            print("Detected 'quit'. Exiting.")
                            break
                
        except KeyboardInterrupt:
            print("\n[INFO] Transcription stopped by user (Ctrl+C).")
        except Exception as e:
            print(f"\n[ERROR] An error occurred: {e}")
        finally:
            self.stop_stream()

if __name__ == "__main__":
    # You might want to let the user specify the model or device
    # For demonstration, 'base' model is a good balance.
    transcriber = RealTimeTranscriber(model_name="base", device="cuda" if torch.cuda.is_available() else "cpu")
    transcriber.transcribe_loop()

    
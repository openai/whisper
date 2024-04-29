# Import necessary libraries
import os
import glob
import sqlite3
from whisper import Whisper

# Initialize Whisper model
model = Whisper()

# Function to transcribe audio file
def transcribe_audio(audio_file):
    transcription = model.transcribe(audio_file)
    return transcription

# Function to extract audio from video file
def extract_audio(video_file, output_path):
    # Use ffmpeg to extract audio from video
    os.system(f'ffmpeg -i {video_file} -vn -ab 256 {output_path}')

# Function to log file paths and transcriptions
def log_file(file_path, transcription):
    # Connect to SQLite database
    conn = sqlite3.connect('files.db')
    c = conn.cursor()

    # Create table if it doesn't exist
    c.execute('''CREATE TABLE IF NOT EXISTS files
                 (file_path TEXT PRIMARY KEY, transcription TEXT)''')

    # Insert file path and transcription into table
    c.execute("INSERT OR REPLACE INTO files (file_path, transcription) VALUES (?,?)", (file_path, transcription))

    # Commit changes and close connection
    conn.commit()
    conn.close()

# Get list of video and audio files in directories
video_files = []
audio_files = []
for dirpath, dirnames, filenames in os.walk('C:\\Users\\lundg\\Videos'):
    for filename in filenames:
        if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mkv'):
            video_files.append(os.path.join(dirpath, filename))
        elif filename.endswith('.mp3') or filename.endswith('.wav'):
            audio_files.append(os.path.join(dirpath, filename))

# Extract audio from video files and transcribe
for video_file in video_files:
    audio_path = os.path.join(os.path.dirname(video_file), 'audio_' + os.path.basename(video_file))
    extract_audio(video_file, audio_path)
    transcription = transcribe_audio(audio_path)
    log_file(video_file, transcription)

# Transcribe existing audio files
for audio_file in audio_files:
    transcription = transcribe_audio(audio_file)
    log_file(audio_file, transcription)
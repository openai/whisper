from flask import Flask, request, render_template
import os
from datetime import datetime
import whisper

app = Flask(__name__)

# Define the folder where you want to save the audio files
upload_folder = "uploads"


@app.route("/")
def home():
    return render_template("form.html")


@app.route("/upload", methods=["POST"])
def upload():
    audio_file = request.files["audio"]

    if audio_file:
        # Generate a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Clear all existing files in the upload folder
        clear_upload_folder()

        # Save the uploaded audio with the timestamp in the filename
        os.makedirs(upload_folder, exist_ok=True)
        audio_filename = f"recorded_audio_{timestamp}.mp3"
        audio_path = os.path.join(upload_folder, audio_filename)
        audio_file.save(audio_path)
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        # Process the audio file as needed
        # For this example, we'll just return a success message
        return f'{result["text"]}'

    return "No audio file provided."


def clear_upload_folder():
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")


if __name__ == "__main__":
    app.run(debug=True)

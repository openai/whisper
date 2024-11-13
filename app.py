import streamlit as st
import os
from datetime import datetime
from whisper import load_model, transcribe

# Set up the app title and description
st.title("Whisper Audio Transcription")
st.write("Upload an audio file and choose a model to transcribe it using OpenAI's Whisper.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "m4a", "mp4"])

# Model selection widget
model_size = st.selectbox("Choose model size:", ["tiny", "base", "small", "medium", "large"])

# Define folders for temporary uploads and results
temp_upload_folder = "TempUploads"
results_folder = "Results"
os.makedirs(temp_upload_folder, exist_ok=True)  # Create TempUploads if it doesn't exist
os.makedirs(results_folder, exist_ok=True)  # Create Results if it doesn't exist

# Function to create a unique output folder for each transcription run
def create_output_folder(audio_file):
    # Use the audio file name (without extension) and a timestamp to create a unique folder name
    folder_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(results_folder, f"{folder_name}_{timestamp}")

    # Create the output folder if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Button to start transcription
if st.button("Transcribe"):
    if uploaded_file is not None:
        # Save the uploaded file temporarily with its original name in TempUploads
        temp_file_path = os.path.join(temp_upload_folder, uploaded_file.name)

        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the chosen Whisper model
        model = load_model(model_size)

        # Create a unique folder for the transcription output in Results
        output_folder = create_output_folder(uploaded_file.name)

        # Run transcription
        try:
            result = transcribe(model, temp_file_path)

            # Save transcription to a text file in the output folder
            output_file = os.path.join(output_folder, "transcription.txt")
            with open(output_file, "w") as f:
                f.write(result["text"])

            # Display the transcription result in the app
            st.write("### Transcription Result")
            st.write(result["text"])
            st.write(f"Transcription saved to {output_file}")

        except Exception as e:
            st.write("An error occurred:", e)
    else:
        st.write("Please upload an audio file.")
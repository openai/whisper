import streamlit as st
import os
from datetime import datetime
from whisper import load_model, transcribe
from googletrans import Translator

# Set up the app title and description
st.title("Whisper Audio Transcription and Translation")
st.write("Upload an audio file, choose a model, and optionally translate the transcription.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "m4a", "mp4"])

# Model selection widget
model_size = st.selectbox("Choose model size:", ["tiny", "base", "small", "medium", "large"])

# Translation selection
target_language = st.selectbox("Translate Transcription To", ["None", "Spanish", "French", "German", "Chinese", "Japanese", "Turkish", "English"])

# Define folders for temporary uploads and results
temp_upload_folder = "TempUploads"
results_folder = "Results"
os.makedirs(temp_upload_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

# Initialize Google Translator
translator = Translator()

# Function to create a unique output folder for each transcription run
def create_output_folder(audio_file):
    folder_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(results_folder, f"{folder_name}_{timestamp}")
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

# Button to start transcription
if st.button("Transcribe"):
    if uploaded_file is not None:
        temp_file_path = os.path.join(temp_upload_folder, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load the chosen Whisper model
        model = load_model(model_size)

        # Create a unique folder for the transcription output in Results
        output_folder = create_output_folder(uploaded_file.name)

        # Run transcription
        try:
            result = transcribe(model, temp_file_path, task="transcribe")

            # Save the English transcription to a text file
            output_file = os.path.join(output_folder, "transcription.txt")
            with open(output_file, "w") as f:
                f.write(result["text"])

            # Display the transcription result in the app
            st.write("### Transcription Result (English)")
            st.write(result["text"])
            #st.write(f"Transcription saved to {output_file}")

            # Translate if a target language is selected
            if target_language != "None":
                translation = translator.translate(result["text"], dest=target_language.lower()).text
                # Save the translation to a text file in a "Translations" subfolder
                translations_folder = os.path.join(output_folder, "Translations")
                os.makedirs(translations_folder, exist_ok=True)
                translation_file = os.path.join(translations_folder, f"{target_language}_translation.txt")
                with open(translation_file, "w") as f:
                    f.write(translation)

                # Display the translated result
                st.write(f"### Translated Transcription ({target_language})")
                st.write(translation)
                #st.write(f"Translation saved to {translation_file}")

        except Exception as e:
            st.write("An error occurred:", e)
    else:
        st.write("Please upload an audio file.")
import argparse
import os
from whisper import load_model  # For loading the Whisper model
from whisper.transcribe import transcribe  # Import the transcribe function
from colorama import Fore, Style

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using OpenAI's Whisper model.")
    parser.add_argument("file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="tiny",
                        help="Choose the model size for transcription (default is 'tiny').")

    args = parser.parse_args()

    # Check if the audio file exists
    if not os.path.isfile(args.file):
        print(Fore.RED + f"Error: The file '{args.file}' does not exist." + Style.RESET_ALL)
        return

    print(Fore.CYAN + f"Transcribing '{args.file}' using the '{args.model}' model..." + Style.RESET_ALL)

    try:
        # Load the model
        model = load_model(args.model)

        # Transcribe the audio file
        result = transcribe(model, args.file)

        # Print the transcription result
        print(Fore.GREEN + "Transcription completed successfully!" + Style.RESET_ALL)
        print(result)

    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)

if __name__ == "__main__":
    main()
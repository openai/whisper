import argparse
import os
from datetime import datetime
from whisper import load_model, transcribe  # Import necessary functions from Whisper
from colorama import Fore, Style


def create_output_folder(audio_file):
    # Base folder where all test folders will be created
    base_folder = "Results"

    # Use the audio file name (without extension) and a timestamp to create a unique folder name
    folder_name = os.path.splitext(os.path.basename(audio_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(base_folder, f"{folder_name}_{timestamp}")

    # Create the Tests folder and the output folder if they don’t exist
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files and save results to a unique folder.")
    parser.add_argument("file", type=str, help="Path to the audio file.")
    parser.add_argument("--model", choices=["tiny", "base", "small", "medium", "large"], default="tiny",
                        help="Choose the model size for transcription (default is 'tiny').")

    args = parser.parse_args()

    # Check if the audio file exists
    if not os.path.isfile(args.file):
        print(Fore.RED + f"Error: The file '{args.file}' does not exist." + Style.RESET_ALL)
        return

    # Load the Whisper model
    print(Fore.CYAN + f"Loading model '{args.model}'..." + Style.RESET_ALL)
    model = load_model(args.model)

    # Create a unique folder under "Tests" for this run
    output_folder = create_output_folder(args.file)
    print(Fore.CYAN + f"Created folder for results: {output_folder}" + Style.RESET_ALL)

    # Run the transcription
    try:
        print(Fore.CYAN + f"Transcribing '{args.file}' using the '{args.model}' model..." + Style.RESET_ALL)
        result = transcribe(model, args.file)

        # Save transcription to a text file in the output folder
        output_file = os.path.join(output_folder, "transcription.txt")
        with open(output_file, "w") as f:
            f.write(result["text"])
        print(Fore.GREEN + f"Transcription completed successfully! Saved to {output_file}" + Style.RESET_ALL)

    except Exception as e:
        print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()
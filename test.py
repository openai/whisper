import whisper
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test.py <output_path>")
    else:
        output_path = sys.argv[1]
        print(f"Received output_path: {output_path}")
        model = whisper.load_model("base")
        result = model.transcribe(output_path)
        print(result["text"])







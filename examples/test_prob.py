import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import whisper
import argparse
import colorsys
from whisper.utils import exact_div
from typing import List
from whisper.tokenizer import get_tokenizer
from colorama import init, Style



# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio_from_source(audio_source):
    audio = whisper.load_audio(audio_source)
    audio = whisper.pad_or_trim(audio)
    return audio


def decode_audio(model, audio, language="en", f16=True):
    dtype = torch.float16 if f16 else torch.float32
    # mel = whisper.log_mel_spectrogram(audio).to(model.device)
    mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES).to(model.device)
    mel_segment =whisper.pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)

    print('Decoding audio') # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel_segment, options)

    tokenizer = get_tokenizer(multilingual=model.is_multilingual, language=language, task=options.task)

    text_tokens = [tokenizer.decode([t]) for t in result.tokens]

    return text_tokens, result.token_probs

def get_colored_text(text_tokens: List[int], token_probs: List[float]):
    init(autoreset=False)  # Initialize colorama with autoreset=True to reset colors after each print
    output_text = ""
    for i, (token, prob) in enumerate(zip(text_tokens, token_probs)):
        # Interpolate between red and green in the HSV color space
        r, g, b = colorsys.hsv_to_rgb(prob * (1/3), 1, 1)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color_code = f"\033[38;2;{r};{g};{b}m"
        colored_token = f"{color_code}{Style.BRIGHT}{str(token)}{Style.RESET_ALL}"
        output_text += colored_token
    return output_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, help='the path of the audio file')
    parser.add_argument('--model', type=str, default="large", help='The version of the model to be used')


    args = parser.parse_args()

    model = args.model
    audio = args.audio

    # Load model
    model = whisper.load_model(model)
    audio = load_audio_from_source(audio_source=audio)
    text, proba = decode_audio(model=model, audio=audio)
    print(get_colored_text(text, proba))
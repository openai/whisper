import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import whisper
import argparse
import colorsys
from typing import List
from whisper.tokenizer import get_tokenizer
from colorama import init, Style



def load_audio_from_source(audio_source):
    audio = whisper.load_audio(audio_source)
    audio = whisper.pad_or_trim(audio)
    return audio


def decode_audio(model, audio, language="en"):
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    print('Decoding audio') # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    tokenizer = get_tokenizer(multilingual=model.is_multilingual, language=language, task=options.task)

    text_tokens = [tokenizer.decode([t]) for t in result.tokens]

    return text_tokens, result.token_probs

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
    print(text)
    print(proba)

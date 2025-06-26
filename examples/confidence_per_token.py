# IMPORTANT: This is just for using the local whisper dir as the package directly. Delete until next comment when just installing whisper normally.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# end of dev import
import whisper

import colorsys
from typing import List
from whisper.tokenizer import get_tokenizer
from colorama import init, Style


print('Loading model')
model = whisper.load_model("large")


print('Loading audio') # load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("samples/your_audio.wav")
audio = whisper.pad_or_trim(audio)


mel = whisper.log_mel_spectrogram(audio).to(model.device) # make log-Mel spectrogram and move to the same device as the model


detect_lang = False
language = "en"
if detect_lang: # detect the spoken language
    print('Detecting language')
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    language=max(probs, key=probs.get)


print('Decoding audio') # decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)


def get_colored_text(tokens: List[int], token_probs: List[float], tokenizer, prompt: str=""):
    init(autoreset=False)  # Initialize colorama
    text_tokens = [tokenizer.decode([t]) for t in tokens]

    output_text = ""
    for i, (token, prob) in enumerate(zip(text_tokens, token_probs)):
        # Interpolate between red and green in the HSV color space
        r, g, b = colorsys.hsv_to_rgb(prob * (1/3), 1, 1)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        color_code = f"\033[38;2;{r};{g};{b}m"

        colored_token = f"{color_code}{Style.BRIGHT}{token}{Style.RESET_ALL}"
        output_text += colored_token

    return output_text


tokenizer = get_tokenizer(multilingual=model.is_multilingual, language=language, task=options.task)
print(get_colored_text(result.tokens, result.token_probs, tokenizer))  # print text with fancy confidence colors
# HINT: when using a prompt, you must provide it in the get_colored_text as well

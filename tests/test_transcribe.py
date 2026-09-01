import os

import pytest
import torch

import whisper
from whisper.tokenizer import get_tokenizer


class TestingProgressReceiver(whisper.TranscribeProgressReceiver):
    def start(self, total: int):
        self.result = ""
        self.total = total
        self.progress = 0
        return self
    def update_line(self, start: float, end: float, text: str):
        self.result += text
    def update(self, n):
        self.progress += n
    def get_result(self):
        return self.result
    def verify_total(self):
        return self.total == self.progress

@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    receiver = TestingProgressReceiver()

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True,
        progress_receiver=receiver
    )
    assert receiver.verify_total()
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])
    assert result["text"] == receiver.get_result()

    transcription = result["text"].lower()
    assert "my fellow americans" in transcription
    assert "your country" in transcription
    assert "do for you" in transcription

    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages)
    all_tokens = [t for s in result["segments"] for t in s["tokens"]]
    assert tokenizer.decode(all_tokens) == result["text"]
    assert tokenizer.decode_with_timestamps(all_tokens).startswith("<|0.00|>")

    timing_checked = False
    for segment in result["segments"]:
        for timing in segment["words"]:
            assert timing["start"] < timing["end"]
            if timing["word"].strip(" ,") == "Americans":
                assert timing["start"] <= 1.8
                assert timing["end"] >= 1.8
                timing_checked = True

    assert timing_checked

import os
from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import whisper
from whisper.tokenizer import get_tokenizer

transcribe_module = import_module("whisper.transcribe")


def model_returning_batched_decode():
    return SimpleNamespace(
        device=torch.device("cpu"),
        dims=SimpleNamespace(n_mels=80, n_audio_ctx=1500, n_text_ctx=448),
        is_multilingual=False,
        num_languages=99,
        decode=lambda *args, **kwargs: [object()],
    )


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((2, 160), dtype=np.float32),
        np.zeros((1, 160), dtype=np.float32),
        torch.zeros(2, 160),
        torch.zeros(1, 160),
    ],
)
def test_transcribe_rejects_non_mono_waveform(audio):
    with pytest.raises(
        ValueError, match=r"1-D mono waveform.*shape \([12], 160\).*[Dd]ownmix"
    ):
        whisper.transcribe(model_returning_batched_decode(), audio, fp16=False)


@pytest.mark.parametrize(
    "audio",
    ["audio.wav", np.zeros(160, dtype=np.float32), torch.zeros(160)],
)
def test_transcribe_accepts_supported_audio_input(monkeypatch, audio):
    class ReachedLogMelSpectrogram(Exception):
        pass

    def stop_after_shape_validation(*args, **kwargs):
        raise ReachedLogMelSpectrogram

    monkeypatch.setattr(
        transcribe_module, "log_mel_spectrogram", stop_after_shape_validation
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = SimpleNamespace(device=torch.device("cpu"), dims=SimpleNamespace(n_mels=80))

    with pytest.raises(ReachedLogMelSpectrogram):
        whisper.transcribe(model, audio, fp16=False)


@pytest.mark.parametrize("model_name", whisper.available_models())
def test_transcribe(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None
    result = model.transcribe(
        audio_path, language=language, temperature=0.0, word_timestamps=True
    )
    assert result["language"] == "en"
    assert result["text"] == "".join([s["text"] for s in result["segments"]])

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

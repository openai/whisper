import json
import os

import numpy as np
import pytest
import torch

import whisper
from whisper.model import ModelDimensions, Whisper
from whisper.tokenizer import get_tokenizer


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


@pytest.mark.parametrize(
    "temperature,logprob_threshold,expected_temperature",
    [
        (np.float32(0.0), None, 0.0),
        (np.int64(0), None, 0.0),
        ((np.float32(0.0), np.float32(0.2)), None, 0.0),
        (
            (np.float32(0.0), np.float32(0.2)),
            0.0,
            float(np.float32(0.2)),
        ),
    ],
)
def test_transcribe_numpy_scalar_temperature(
    temperature, logprob_threshold, expected_temperature
):
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=1,
    )
    audio = np.zeros(1600, dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = Whisper(dims).eval()
        torch.nn.init.normal_(model.decoder.positional_embedding, std=0.01)
        result = model.transcribe(
            audio,
            temperature=temperature,
            language="en",
            without_timestamps=True,
            sample_len=1,
            fp16=False,
            no_speech_threshold=None,
            logprob_threshold=logprob_threshold,
            compression_ratio_threshold=None,
        )

    serialized_result = json.loads(json.dumps(result))
    assert [s["temperature"] for s in serialized_result["segments"]] == [
        expected_temperature
    ]

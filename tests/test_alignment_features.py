import importlib
from types import SimpleNamespace

import pytest
import torch

from whisper.decoding import DecodingResult
from whisper.model import ModelDimensions, Whisper, disable_sdpa
from whisper.timing import find_alignment
from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize("cached_sdpa", [False, True])
def test_find_alignment_reuses_audio_features(monkeypatch, cached_sdpa):
    torch.manual_seed(0)
    dims = ModelDimensions(80, 12, 8, 2, 2, 51865, 64, 8, 2, 2)
    model = Whisper(dims)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    tokenizer = get_tokenizer(True, language="en")
    tokens = tokenizer.encode(" hello world")
    mel = torch.randn(80, 24)

    encoder_calls = []
    model.encoder.register_forward_hook(lambda *args: encoder_calls.append(True))
    expected = find_alignment(model, tokenizer, tokens, mel, 24)
    assert len(encoder_calls) == 1
    with torch.no_grad():
        if cached_sdpa:
            features = model.embed_audio(mel.unsqueeze(0))[0]
        else:
            with disable_sdpa():
                features = model.embed_audio(mel.unsqueeze(0))[0]

    def unexpected_encoder(*args, **kwargs):
        pytest.fail("cached word alignment must not encode the audio again")

    monkeypatch.setattr(model.encoder, "forward", unexpected_encoder)
    actual = find_alignment(model, tokenizer, tokens, mel, 24, audio_features=features)

    assert [(w.word, w.tokens) for w in actual] == [
        (w.word, w.tokens) for w in expected
    ]
    if not cached_sdpa:
        assert [(w.start, w.end) for w in actual] == [
            (w.start, w.end) for w in expected
        ]
    assert all(0 <= w.start <= w.end <= 0.24 for w in actual)
    assert [w.probability for w in actual] == pytest.approx(
        [w.probability for w in expected], rel=1e-5, abs=1e-7
    )


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("clip_timestamps", ["0", "5,20,35,50"])
@pytest.mark.parametrize("word_timestamps", [False, True])
def test_word_alignment_uses_current_decoding_features(
    monkeypatch, fallback, clip_timestamps, word_timestamps
):
    transcribe_module = importlib.import_module("whisper.transcribe")
    tokenizer = get_tokenizer(True, language="en")
    decoded_features = []
    aligned_features = []

    def decode(mel, options):
        features = torch.full((1500, 8), len(decoded_features), dtype=torch.float32)
        decoded_features.append(features)
        return DecodingResult(
            audio_features=features,
            language="en",
            tokens=[
                tokenizer.timestamp_begin,
                *tokenizer.encode(" hello"),
                tokenizer.timestamp_begin + 500,
            ],
            text=" hello",
            avg_logprob=-2.0 if fallback and options.temperature == 0 else -0.1,
            no_speech_prob=0.0,
            temperature=options.temperature,
            compression_ratio=1.0,
        )

    def add_word_timestamps(*, segments, audio_features, **kwargs):
        aligned_features.append(audio_features)
        for segment in segments:
            segment["words"] = [
                {
                    "word": " hello",
                    "start": segment["start"],
                    "end": segment["end"],
                    "probability": 1.0,
                }
            ]

    model = SimpleNamespace(
        device=torch.device("cpu"),
        dims=SimpleNamespace(n_mels=80, n_audio_ctx=1500, n_text_ctx=448),
        is_multilingual=True,
        num_languages=99,
        decode=decode,
    )
    monkeypatch.setattr(
        transcribe_module,
        "log_mel_spectrogram",
        lambda *args, **kwargs: torch.zeros(80, 9000),
    )
    monkeypatch.setattr(transcribe_module, "add_word_timestamps", add_word_timestamps)
    transcribe_module.transcribe(
        model,
        torch.zeros(1),
        language="en",
        fp16=False,
        temperature=(0.0, 0.2) if fallback else 0.0,
        word_timestamps=word_timestamps,
        clip_timestamps=clip_timestamps,
    )

    if not word_timestamps:
        assert aligned_features == []
        return

    assert len(aligned_features) == 2
    expected_features = decoded_features[1::2] if fallback else decoded_features
    assert len(expected_features) == 2
    assert all(
        actual is expected
        for actual, expected in zip(aligned_features, expected_features)
    )


def test_find_alignment_empty_text_with_cached_features():
    assert find_alignment(None, None, [], None, 0, audio_features=torch.zeros(1)) == []

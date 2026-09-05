import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from whisper.audio import N_FRAMES, pad_or_trim
from whisper.decoding import DecodingResult

transcribe_module = importlib.import_module("whisper.transcribe")


@pytest.fixture
def make_transcription(monkeypatch):
    def make(content_frames=6000):
        # Frame indices expose which part of the input reaches language detection.
        mel = torch.arange(content_frames + N_FRAMES, dtype=torch.float32)
        mel = mel.unsqueeze(0).expand(80, -1)
        monkeypatch.setattr(
            transcribe_module, "log_mel_spectrogram", Mock(return_value=mel)
        )
        model = SimpleNamespace(
            device=torch.device("cpu"),
            dims=SimpleNamespace(n_mels=80, n_audio_ctx=1500, n_text_ctx=448),
            is_multilingual=True,
            num_languages=99,
            detect_language=Mock(return_value=(None, {"en": 0.9, "fr": 0.1})),
            decode=Mock(
                return_value=DecodingResult(
                    audio_features=torch.empty(0),
                    language="en",
                    tokens=[],
                    text="",
                    avg_logprob=-2.0,
                    no_speech_prob=1.0,
                    temperature=0.0,
                    compression_ratio=0.0,
                )
            ),
        )
        return model, mel

    return make


@pytest.mark.parametrize(
    "clips,start,end",
    [
        ("35,40", 3500, 4000),
        ([35.0, 40.0], 3500, 4000),
        ("0,10", 0, 1000),
        ("5,40", 500, 3500),
        ("35", 3500, 6500),
        ([35.0], 3500, 6500),
        ("35,60", 3500, 6500),
        ("35,40,45,50", 3500, 4000),
        ("0,0,35,40", 3500, 4000),
        ("40,35,45,50", 4500, 5000),
    ],
)
def test_language_detection_uses_first_clip(make_transcription, clips, start, end):
    model, mel = make_transcription()

    result = transcribe_module.transcribe(
        model, torch.empty(0), clip_timestamps=clips, fp16=False, verbose=None
    )

    model.detect_language.assert_called_once()
    detected_mel = model.detect_language.call_args.args[0]
    assert torch.equal(detected_mel, pad_or_trim(mel[:, start:end], N_FRAMES))
    assert result["language"] == "en"
    assert model.decode.called
    assert all(call.args[1].language == "en" for call in model.decode.call_args_list)


@pytest.mark.parametrize("content_frames", [1200, 6000])
@pytest.mark.parametrize("clips", [None, "", "0", [0.0]])
def test_whole_file_language_detection_is_unchanged(
    make_transcription, content_frames, clips
):
    model, mel = make_transcription(content_frames)
    options = {} if clips is None else {"clip_timestamps": clips}

    transcribe_module.transcribe(
        model, torch.empty(0), fp16=False, verbose=None, **options
    )

    # Preserve the existing audio-domain silence padding for short whole-file inputs.
    detected_mel = model.detect_language.call_args.args[0]
    assert torch.equal(detected_mel, mel[:, :N_FRAMES])


@pytest.mark.parametrize("multilingual,language", [(True, "fr"), (False, None)])
def test_language_detection_can_be_skipped(make_transcription, multilingual, language):
    model, _ = make_transcription()
    model.is_multilingual = multilingual

    result = transcribe_module.transcribe(
        model,
        torch.empty(0),
        clip_timestamps="35,40",
        language=language,
        fp16=False,
        verbose=None,
    )

    model.detect_language.assert_not_called()
    assert result["language"] == (language or "en")


@pytest.mark.parametrize("clips", ["5,5", "0,0,35,35", [40.0, 35.0, 50.0, 50.0]])
def test_empty_clips_preserve_language_detection_fallback(make_transcription, clips):
    model, mel = make_transcription()

    transcribe_module.transcribe(
        model, torch.empty(0), clip_timestamps=clips, fp16=False, verbose=None
    )

    assert torch.equal(model.detect_language.call_args.args[0], mel[:, :N_FRAMES])
    model.decode.assert_not_called()

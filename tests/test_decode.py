import os

import pytest
import torch

import whisper


@pytest.mark.parametrize("model_name", whisper.available_models())
def test_decode(model_name: str):
    # Regression test: batch_size and beam_size should work together
    beam_size = 2
    batch_size = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_name).to(device)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")

    language = "en" if model_name.endswith(".en") else None

    options = whisper.DecodingOptions(language=language, beam_size=beam_size)

    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(device)

    # Create a small batch
    batch_mel = mel.unsqueeze(0).repeat(batch_size, 1, 1)

    results = model.decode(batch_mel, options)

    # Since both examples are the same, results should be identical
    assert len(results) == batch_size
    assert results[0].text == results[1].text

    decoded_text = results[0].text.lower()
    assert "my fellow americans" in decoded_text
    assert "your country" in decoded_text
    assert "do for you" in decoded_text

    timing_checked = False
    if hasattr(results[0], "segments"):
        for segment in results[0].segments:
            for timing in segment["words"]:
                assert timing["start"] < timing["end"]
                if timing["word"].strip(" ,") == "Americans":
                    assert timing["start"] <= 1.8
                    assert timing["end"] >= 1.8
                    timing_checked = True

    if hasattr(results[0], "segments"):
        assert timing_checked

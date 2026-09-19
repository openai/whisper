import os.path

import numpy as np
import pytest
import torch

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram


def test_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    audio = load_audio(audio_path)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(audio_path)

    assert np.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0


@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("padding", [0, 1600])
def test_batched_spectrogram_matches_individual_audio(n_mels, padding):
    time = torch.arange(SAMPLE_RATE, dtype=torch.float32) / SAMPLE_RATE
    tone = torch.sin(2 * torch.pi * 440 * time)
    audio = torch.stack([tone, tone * 1e-4, torch.zeros_like(tone)])

    batched = log_mel_spectrogram(audio, n_mels=n_mels, padding=padding)
    individual = torch.stack(
        [log_mel_spectrogram(item, n_mels=n_mels, padding=padding) for item in audio]
    )

    torch.testing.assert_close(batched, individual)

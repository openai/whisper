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


@pytest.mark.parametrize("step", [-2, -1, 1, 2])
@pytest.mark.parametrize("n_mels", [80, 128])
def test_strided_audio(step, n_mels):
    time = np.arange(3200, dtype=np.float32) / SAMPLE_RATE
    source = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    original = source.copy()
    audio = source[::step]

    actual = log_mel_spectrogram(audio, n_mels=n_mels)
    expected = log_mel_spectrogram(audio.copy(), n_mels=n_mels)

    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    assert np.array_equal(source, original)
    assert audio.dtype == np.float32

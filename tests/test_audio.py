import os.path

import numpy as np

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram


def test_audio():
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    audio = load_audio(audio_path)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(audio_path)
    mel_from_float64 = log_mel_spectrogram(audio.astype(np.float64))

    assert np.allclose(mel_from_audio, mel_from_file)
    assert np.allclose(mel_from_audio, mel_from_float64, rtol=1e-4, atol=1e-4)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0

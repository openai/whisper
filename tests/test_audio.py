import os.path

import numpy as np
import pytest

from whisper.audio import SAMPLE_RATE, load_audio, log_mel_spectrogram

@pytest.mark.parametrize("read_bytes", [True, False])
def test_audio(read_bytes):
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    audio_input = audio_path
    if (read_bytes): 
        with open(audio_path, 'rb') as f:
            audio_input = f.read()
    audio = load_audio(audio_input)
    assert audio.ndim == 1
    assert SAMPLE_RATE * 10 < audio.shape[0] < SAMPLE_RATE * 12
    assert 0 < audio.std() < 1

    mel_from_audio = log_mel_spectrogram(audio)
    mel_from_file = log_mel_spectrogram(audio_input)

    assert np.allclose(mel_from_audio, mel_from_file)
    assert mel_from_audio.max() - mel_from_audio.min() <= 2.0

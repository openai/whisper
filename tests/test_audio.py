import os.path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import whisper.audio as audio_module
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


def reference_log_mel(audio, n_mels=80, padding=0):
    """The unchunked implementation, including its recording-wide floor."""
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(audio_module.N_FFT).to(audio.device)
    stft = torch.stft(
        audio,
        audio_module.N_FFT,
        audio_module.HOP_LENGTH,
        window=window,
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs() ** 2
    mel = audio_module.mel_filters(audio.device, n_mels) @ magnitudes
    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


@pytest.mark.parametrize("n_mels", [80, 128])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("padding", [0, 321])
def test_chunked_mel_boundaries(monkeypatch, n_mels, batched, padding):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 7, raising=False)
    generator = torch.Generator().manual_seed(42)
    # Cover every possible final-frame position relative to the waveform end.
    for remainder in range(audio_module.HOP_LENGTH):
        length = 15 * audio_module.HOP_LENGTH + remainder
        shape = (2, length * 2) if batched else (length * 2,)
        audio = torch.randn(shape, generator=generator)[..., ::2]
        original = audio.clone()
        expected = reference_log_mel(audio, n_mels, padding)
        actual = log_mel_spectrogram(audio, n_mels, padding)
        assert actual.shape == expected.shape
        assert torch.allclose(actual, expected, atol=5e-7, rtol=0)
        assert torch.equal(audio, original)


@pytest.mark.parametrize("n_mels", [80, 128])
def test_chunked_mel_global_normalization(monkeypatch, n_mels):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 7, raising=False)
    generator = torch.Generator().manual_seed(42)
    audio = torch.zeros(2, 24 * audio_module.HOP_LENGTH)
    # A distant loud region must determine the floor of earlier silence and
    # quiet batch entries, as in the original normalization.
    audio[0, 15 * audio_module.HOP_LENGTH :] = 100 * torch.randn(
        9 * audio_module.HOP_LENGTH, generator=generator
    )
    audio[1] = 1e-5 * torch.randn(audio.shape[-1], generator=generator)
    expected = reference_log_mel(audio, n_mels)
    actual = log_mel_spectrogram(audio, n_mels)
    assert torch.allclose(actual, expected, atol=5e-7, rtol=0)


@pytest.mark.parametrize("frames", [6000, 6001, 12000, 12001])
def test_chunked_mel_stft_allocation_bound(frames):
    chunk_frames = 6000
    # Exercise the production threshold and final chunks as small as one frame.
    audio = torch.randn(frames * audio_module.HOP_LENGTH)
    expected = reference_log_mel(audio)
    with patch.object(torch, "stft", wraps=torch.stft) as stft:
        actual = log_mel_spectrogram(audio)
    assert torch.allclose(actual, expected, atol=5e-7, rtol=0)
    if frames <= chunk_frames:
        assert stft.call_count == 1
        assert stft.call_args[1].get("center", True)
    else:
        assert stft.call_count == (frames + chunk_frames - 1) // chunk_frames
        for args, kwargs in stft.call_args_list:
            assert kwargs["center"] is False
            assert args[0].shape[-1] <= (
                (chunk_frames - 1) * audio_module.HOP_LENGTH + audio_module.N_FFT
            )


def test_chunked_mel_numpy_and_file(monkeypatch):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 100, raising=False)
    audio_path = os.path.join(os.path.dirname(__file__), "jfk.flac")
    audio = load_audio(audio_path)
    expected = reference_log_mel(torch.from_numpy(audio), padding=480000)
    for source in (audio, audio_path):
        actual = log_mel_spectrogram(source, padding=480000, device="cpu")
        assert torch.allclose(actual, expected, atol=5e-7, rtol=0)


def test_chunked_mel_autograd_fallback(monkeypatch):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 7, raising=False)
    audio = torch.randn(19 * audio_module.HOP_LENGTH, requires_grad=True)
    expected = reference_log_mel(audio, padding=321)
    with patch.object(torch, "stft", wraps=torch.stft) as stft:
        actual = log_mel_spectrogram(audio, padding=321)
    assert stft.call_count == 1
    assert stft.call_args[1].get("center", True)
    assert torch.equal(actual, expected)
    actual_grad = torch.autograd.grad(actual.sum(), audio)[0]
    expected_grad = torch.autograd.grad(expected.sum(), audio)[0]
    assert torch.equal(actual_grad, expected_grad)

    # A tensor's requires_grad flag must not disable chunking during inference.
    with torch.no_grad(), patch.object(torch, "stft", wraps=torch.stft) as stft:
        actual = log_mel_spectrogram(audio, padding=321)
    assert stft.call_count > 1
    assert torch.allclose(actual, expected, atol=5e-7, rtol=0)


@pytest.mark.parametrize("n_mels", [80, 128])
def test_chunked_mel_cpu_autocast(monkeypatch, n_mels):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 7, raising=False)
    audio = torch.randn(19 * audio_module.HOP_LENGTH)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        expected = reference_log_mel(audio, n_mels)
        actual = log_mel_spectrogram(audio, n_mels)
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.requires_cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_chunked_mel_keeps_cuda_path(monkeypatch):
    monkeypatch.setattr(audio_module, "_MEL_CHUNK_FRAMES", 7, raising=False)
    audio = torch.randn(19 * audio_module.HOP_LENGTH, device="cuda")
    expected = reference_log_mel(audio)
    with patch.object(torch, "stft", wraps=torch.stft) as stft:
        actual = log_mel_spectrogram(audio)
    assert stft.call_count == 1
    assert stft.call_args[1].get("center", True)
    assert torch.equal(actual, expected)

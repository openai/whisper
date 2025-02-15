import os
import subprocess
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = N_SAMPLES // HOP_LENGTH

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
    except FileNotFoundError:
        raise RuntimeError("ffmpeg is required but not found.  Please install it.")
    except Exception as e:
        raise RuntimeError(f"Error processing audio with ffmpeg: {e}")

    if process.returncode != 0:
        error_message = err.decode("utf-8").strip()
        raise RuntimeError(f"ffmpeg error: {error_message}")

    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio


def pad_or_trim(array: Union[np.ndarray, torch.Tensor], length: int = N_SAMPLES, axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
    array_len = array.shape[axis]
    if array_len > length:
        if torch.is_tensor(array):
            index = torch.arange(length, device=array.device)
            array = torch.index_select(array, dim=axis, index=index)
        else:
            array = np.take(array, indices=range(length), axis=axis)
    elif array_len < length:
        pad_width = [(0, 0)] * array.ndim
        pad_width[axis] = (0, length - array_len)
        if torch.is_tensor(array):
            array = F.pad(array, [pad for sizes in pad_width[::-1] for pad in sizes])
        else:
            array = np.pad(array, pad_width)
    return array


@lru_cache(maxsize=2)
def mel_filters(device: Union[str, torch.device], n_mels: int) -> torch.Tensor:
    if n_mels not in {80, 128}:
        raise ValueError(f"Unsupported n_mels: {n_mels}.  Must be 80 or 128.")

    filters_file = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    try:
        with np.load(filters_file, allow_pickle=False) as data:
            filters = torch.from_numpy(data[f"mel_{n_mels}"]).to(device)
            return filters
    except FileNotFoundError:
        raise FileNotFoundError(f"Mel filters file not found: {filters_file}")
    except KeyError:
        raise KeyError(f"Mel filter with {n_mels} not found in {filters_file}")
    except Exception as e:
        raise RuntimeError(f"Error loading mel filters: {e}")


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:

    if isinstance(audio, str):
        audio = load_audio(audio)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)

    if device:
        audio = audio.to(device)

    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT, device=audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft.abs() ** 2
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec

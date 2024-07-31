import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token
def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Loads an audio file as a mono waveform, resampling to the specified sample rate.

    Parameters
    ----------
    file : str
        Path to the audio file.
    sr : int, optional
        Target sample rate for resampling, defaults to SAMPLE_RATE.

    Returns
    -------
    np.ndarray
        1D NumPy array of the audio waveform, normalized between -1 and 1.

    Raises
    ------
    RuntimeError
        If the audio cannot be loaded.

    Notes
    -----
    Requires ffmpeg installed and accessible in the system's PATH.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    
    cmd = [
    "ffmpeg",           # Command to run the ffmpeg tool.
    "-nostdin",         # Prevents ffmpeg from reading from stdin.
    "-threads", "0",    # Uses all available CPU cores for processing.
    "-i", file,         # Specifies the input file path.
    "-f", "s16le",      # Sets the output format to 16-bit PCM.
    "-ac", "1",         # Converts audio to mono (1 channel).
    "-acodec", "pcm_s16le",  # Specifies the audio codec as PCM signed 16-bit little-endian.
    "-ar", str(sr),     # Resamples the audio to the specified sample rate.
    "-"                 # Outputs the processed audio to stdout.
]

    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    
    """
    Pads or trims the input array to a specified length along the given axis.

    Parameters:
    - array: Input array (torch.Tensor or np.ndarray).
    - length: Target length along the specified axis (default is N_SAMPLES).
    - axis: Axis to pad or trim (default is -1 for the last axis).

    Returns:
    - The modified array, either padded with zeros or trimmed to the target length.
    
     Note:
    - The function handles both PyTorch tensors and NumPy arrays, applying appropriate methods 
      for padding and trimming depending on the array type.
    
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    Loads a precomputed Mel filterbank matrix for converting STFT to a Mel spectrogram.

    Parameters
    ----------
    device : torch.device
        The device (CPU or GPU) to load the tensor onto.
    n_mels : int
        The number of Mel bands, must be either 80 or 128.

    Returns
    -------
    torch.Tensor
        A tensor containing the Mel filterbank matrix.

    Raises
    ------
    AssertionError
        If `n_mels` is not supported.

    Notes
    -----
    The Mel filterbank matrices are saved in a compressed npz file, which decouples 
    the dependency on librosa for generating these filters.
    
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Computes the log-Mel spectrogram of an audio waveform.

    Parameters
    ----------
    audio : Union[str, np.ndarray, torch.Tensor]
        The audio input, either as a file path, NumPy array, or Torch tensor. 
        The waveform should be in 16 kHz.
    n_mels : int, optional
        The number of Mel-frequency filters, only 80 is supported. Defaults to 80.
    padding : int, optional
        Number of zero samples to pad at the end of the audio. Defaults to 0.
    device : Optional[Union[str, torch.device]], optional
        The device to perform computations on. If provided, the audio tensor is moved 
        to this device. Defaults to None.

    Returns
    -------
    torch.Tensor
        A tensor containing the Mel spectrogram with shape (80, n_frames).

    Notes
    -----
    The function expects a 16 kHz sampling rate for the input audio and uses a Hann 
    window for the Short-Time Fourier Transform (STFT).
    """
    
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

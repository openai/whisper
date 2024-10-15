import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

# Importação corrigida
from utils import exact_div

# Constants for audio processing
class AudioConstants:
    SAMPLE_RATE = 16000
    N_FFT = 400
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input
    N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # Initial convolutions have stride 2
    FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
    TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

def load_audio(file: str, sr: int = AudioConstants.SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    Parameters
    ----------
    file: str
        The audio file to open.

    sr: int
        The sample rate to resample the audio if necessary.

    Returns
    -------
    np.ndarray
        A NumPy array containing the audio waveform, in float32 dtype.
    """
    
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ]
    
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio from {file}: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

def pad_or_trim(array: Union[np.ndarray, torch.Tensor], length: int = AudioConstants.N_SAMPLES, *, axis: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """
    Pad or trim the audio array to a specified length.

    Parameters
    ----------
    array: Union[np.ndarray, torch.Tensor]
        The input array (NumPy or PyTorch tensor).

    length: int
        The target length to pad or trim to.

    axis: int
        The axis along which to pad or trim.

    Returns
    -------
    Union[np.ndarray, torch.Tensor]
        The padded or trimmed array.
    """
    
    if isinstance(array, torch.Tensor):
        if array.shape[axis] > length:
            array = array.index_select(dim=axis, index=torch.arange(length, device=array.device))
        
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    
    else:  # Assume it's a NumPy array
        if array.shape[axis] > length:
            array = np.take(array, indices=range(length), axis=axis)
        
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array

@lru_cache(maxsize=None)
def mel_filters(device: torch.device, n_mels: int) -> torch.Tensor:
    """
    Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    Parameters
    ----------
    device: torch.device
        The device to which the tensor will be moved.

    n_mels: int
        The number of Mel-frequency filters.

    Returns
    -------
    torch.Tensor
        A tensor containing the mel filterbank.
    
    Raises
    ------
    AssertionError
        If n_mels is not supported.
    """
    
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filters_path = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    
    with np.load(filters_path, allow_pickle=False) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

def get_hann_window(size: int, device: torch.device) -> torch.Tensor:
    """
    Get a Hann window of specified size on the given device.

    Parameters
    ----------
    size: int
        The size of the window.

    device: torch.device
        The device to which the window will be moved.

    Returns
    -------
    torch.Tensor
        A Hann window tensor.
    
    Cache the windows for efficiency.
    """
    
    # Cache for Hann windows based on size and device.
    if not hasattr(get_hann_window, 'cache'):
        get_hann_window.cache = {}
    
    key = (size, str(device))
    if key not in get_hann_window.cache:
        get_hann_window.cache[key] = torch.hann_window(size).to(device)
    
    return get_hann_window.cache[key]

def log_mel_spectrogram(
   audio: Union[str, np.ndarray, torch.Tensor],
   n_mels: int = 80,
   padding: int = 0,
   device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
   """
   Compute the log-Mel spectrogram of an audio waveform.

   Parameters
   ----------
   audio: Union[str, np.ndarray, torch.Tensor]
       The path to audio or a NumPy array or Tensor containing the audio waveform.

   n_mels: int
       The number of Mel-frequency filters (only supports 80 and 128).

   padding: int
       Number of zero samples to pad to the right.

   device: Optional[Union[str, torch.device]]
       If given, moves the audio tensor to this device before STFT.

   Returns
   -------
   torch.Tensor
       A Tensor containing the log-Mel spectrogram.
   """
   
   # Load audio if necessary and convert to tensor if needed.
   if isinstance(audio, str):
       audio_tensor = load_audio(audio)
       audio_tensor = torch.from_numpy(audio_tensor)
   else:
       audio_tensor = audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio)

   # Move to specified device if provided.
   if device is not None:
       audio_tensor = audio_tensor.to(device)

   # Padding the audio tensor.
   if padding > 0:
       audio_tensor = F.pad(audio_tensor, (0, padding))

   # Compute STFT and magnitudes.
   window = get_hann_window(AudioConstants.N_FFT, audio_tensor.device)
   stft = torch.stft(audio_tensor, AudioConstants.N_FFT,
                      AudioConstants.HOP_LENGTH,
                      window=window,
                      return_complex=True)
   
   magnitudes = stft.abs() ** 2

   # Calculate Mel spectrogram and apply logarithmic scaling.
   filters = mel_filters(audio_tensor.device, n_mels)
   mel_spec = filters @ magnitudes

   log_spec = torch.clamp(mel_spec, min=1e-10).log10()
   
   log_spec_normalized = (log_spec + 4.0) / 4.0
   
   return log_spec_normalized

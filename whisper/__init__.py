import hashlib
import io
import os
import urllib
import warnings
from typing import List, Optional, Union

import torch
from tqdm import tqdm

from .audio import load_audio, log_mel_spectrogram, pad_or_trim
from .decoding import DecodingOptions, DecodingResult, decode, detect_language
from .model import Whisper, ModelDimensions
from .transcribe import transcribe


_MODELS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large": "https://openaipublic.azureedge.net/main/whisper/models/e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a/large.pt",
}


def _download(url: str, root: str, in_memory: bool) -> Union[bytes, str]:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return model_bytes if in_memory else download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")

    return model_bytes if in_memory else download_target


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_model(name: str) -> Whisper:
    """
    Load a Whisper ASR model

    Parameters
    ----------
    name : str
        one of the official model names listed by `whisper.available_models()`, or
        path to a model checkpoint containing the model dimensions and the model state_dict.

    Returns
    -------
    model : Whisper
        The Whisper ASR model instance
    """

    if name == "tiny":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "tiny.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 384, 'n_audio_head': 6, 'n_audio_layer': 4, 'n_text_ctx': 448, 'n_text_state': 384, 'n_text_head': 6, 'n_text_layer': 4}
    elif name == "base":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "base.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 512, 'n_audio_head': 8, 'n_audio_layer': 6, 'n_text_ctx': 448, 'n_text_state': 512, 'n_text_head': 8, 'n_text_layer': 6}
    elif name == "small":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "small.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 768, 'n_audio_head': 12, 'n_audio_layer': 12, 'n_text_ctx': 448, 'n_text_state': 768, 'n_text_head': 12, 'n_text_layer': 12}
    elif name == "medium":
        dims_config = {'n_mels': 80, 'n_vocab': 51865, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    elif name == "medium.en":
        dims_config = {'n_mels': 80, 'n_vocab': 51864, 'n_audio_ctx': 1500, 'n_audio_state': 1024, 'n_audio_head': 16, 'n_audio_layer': 24, 'n_text_ctx': 448, 'n_text_state': 1024, 'n_text_head': 16, 'n_text_layer': 24}
    else:
        raise ValueError(f"model type {name} not supported")

    dims = ModelDimensions(**dims_config)
    model = Whisper(dims, name)
    return model

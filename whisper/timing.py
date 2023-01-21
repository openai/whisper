from typing import List, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .model import Whisper


def median_filter(x: torch.Tensor, filter_width: int):
    """Apply a median filter of width `filter_width` along the last dimension of `x`"""
    assert 3 <= x.ndim <= 4, "`median_filter()` is implemented for only 3D or 4D tensors"
    assert filter_width > 0 and filter_width % 2 == 1, "`filter_width` should be an odd number"

    padded = F.pad(x, (0, 0, filter_width // 2, filter_width // 2), mode='replicate')
    slices = padded.unfold(-1, filter_width, 1)
    return slices.median(dim=-1).values


def add_word_timestamps(
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: torch.Tensor,
    num_frames: int,
    segments: List[dict],
    *,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
):
    if len(segments) == 0:
        return

    from dtw import dtw

    # install hooks on the cross attention layers to retrieve the attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.timestamp_begin,
            *[t for segment in segments for t in segment["tokens"]],
            tokenizer.timestamp_begin + mel.shape[-1] // 2,
            tokenizer.eot,
        ]
    ).to(model.device)

    with torch.no_grad():
        model(mel.unsqueeze(0), tokens.unsqueeze(0))

    for hook in hooks:
        hook.remove()

    weights = torch.cat(QKs)  # layers * heads * tokens * frames
    weights = weights[:, :, :, : num_frames // 2]
    weights = median_filter(weights, medfilt_width)
    weights = (weights * qk_scale).softmax(dim=-1)

    w = weights / weights.norm(dim=-2, keepdim=True)
    matrix = w.mean(axis=(0, 1)).neg().double().cpu().numpy()

    alignment = dtw(matrix)

    jumps = np.pad(np.diff(alignment.index1s), (1, 0), constant_values=1).astype(bool)
    jump_times = alignment.index2s[jumps] / TOKENS_PER_SECOND

    if tokenizer.language in {"zh", "ja", "th", "lo", "my"}:
        # These languages don't typically use spaces, so it is difficult to split words
        # without morpheme analysis. Here, we instead split words at any
        # position where the tokens are decoded as valid unicode points
        split_tokens = tokenizer.split_tokens_on_unicode
    else:
        split_tokens = tokenizer.split_tokens_on_spaces

    words, word_tokens = split_tokens(tokens[1:].tolist())

    token_sources = np.repeat(np.arange(len(segments)), [len(s["tokens"]) for s in segments])
    token_sources = [None] * len(tokenizer.sot_sequence) + list(token_sources)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens]), (1, 0))
    start_times = time_offset + jump_times[word_boundaries[:-1]]
    end_times = time_offset + jump_times[word_boundaries[1:]]

    for segment in segments:
        segment["words"] = []

    for i, (word, start, end) in enumerate(zip(words, start_times, end_times)):
        if word.startswith("<|") or word.strip() in ".,!?、。":
            continue

        segment = segments[token_sources[word_boundaries[i]]]
        segment["words"].append(dict(word=word, start=round(start, 2), end=round(end, 2)))

    # adjust the segment-level timestamps based on the word-level timestamps
    for segment in segments:
        if len(segment["words"]) > 0:
            segment["start"] = segment["words"][0]["start"]
            segment["end"] = segment["words"][-1]["end"]

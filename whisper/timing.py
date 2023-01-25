from typing import List, TYPE_CHECKING
from dataclasses import dataclass
import numba
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

    x = F.pad(x, (filter_width // 2, filter_width // 2, 0, 0), mode='replicate')
    if x.is_cuda:
        from .triton_ops import median_filter_cuda
        return median_filter_cuda(x, filter_width)

    # sort() is faster than torch.median (https://github.com/pytorch/pytorch/issues/51450)
    return x.unfold(-1, filter_width, 1).sort()[0][..., filter_width // 2]


@numba.jit
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")

    result = np.array(result)
    return result[::-1, :].T


@numba.jit(nopython=True, parallel=True)
def dtw_cpu(x: np.ndarray):
    N, M = x.shape
    cost = np.ones((N + 1, M + 1), dtype=np.float32) * np.inf
    trace = -np.ones((N + 1, M + 1), dtype=np.float32)

    cost[0, 0] = 0
    for j in range(1, M + 1):
        for i in range(1, N + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]

            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2

            cost[i, j] = x[i - 1, j - 1] + c
            trace[i, j] = t

    return backtrace(trace)


def dtw_cuda(x, BLOCK_SIZE=1024):
    from .triton_ops import dtw_kernel

    M, N = x.shape
    assert M < BLOCK_SIZE, f"M should be smaller than {BLOCK_SIZE=}"

    x_skew = F.pad(x, (0, M + 1), value=np.inf).flatten()[: M * (N + M)].reshape(M, N + M)
    x_skew = x_skew.T.contiguous()
    cost = torch.ones(N + M + 2, M + 2) * np.inf
    cost[0, 0] = 0
    cost = cost.cuda()
    trace = torch.zeros_like(cost, dtype=torch.int32)

    dtw_kernel[(1,)](
        cost,
        trace,
        x_skew,
        x_skew.stride(0),
        cost.stride(0),
        trace.stride(0),
        N,
        M,
        BLOCK_SIZE=BLOCK_SIZE
    )

    trace = trace.T.flatten()[:(M + 1) * (M + N + 3)].reshape(M + 1, M + N + 3)[:, :N + 1]
    return backtrace(trace.cpu().numpy())


def dtw(x: torch.Tensor) -> np.ndarray:
    if x.is_cuda:
        return dtw_cuda(x)

    return dtw_cpu(x.double().cpu().numpy())


@dataclass
class Alignment:
    words: List[str]
    word_tokens: List[List[int]]
    start_times: np.ndarray
    end_times: np.ndarray


def find_alignment(
    model: "Whisper",
    tokenizer: Tokenizer,
    text_tokens: List[int],
    mel: torch.Tensor,
    num_frames: int,
    *,
    max_qk_layers: int = 6,
    medfilt_width: int = 7,
    qk_scale: float = 1.0,
) -> Alignment:
    tokens = torch.tensor(
        [
            *tokenizer.sot_sequence,
            tokenizer.timestamp_begin,
            *text_tokens,
            tokenizer.timestamp_begin + num_frames // 2,
            tokenizer.eot,
        ]
    ).to(model.device)

    # install hooks on the cross attention layers to retrieve the attention weights
    QKs = [None] * model.dims.n_text_layer
    hooks = [
        block.cross_attn.register_forward_hook(
            lambda _, ins, outs, index=i: QKs.__setitem__(index, outs[-1])
        )
        for i, block in enumerate(model.decoder.blocks)
    ]

    with torch.no_grad():
        model(mel.unsqueeze(0), tokens.unsqueeze(0))

    for hook in hooks:
        hook.remove()

    weights = torch.cat(QKs[-max_qk_layers:])  # layers * heads * tokens * frames
    weights = weights[:, :, :, : num_frames // 2]
    weights = median_filter(weights, medfilt_width)
    weights = (weights * qk_scale).softmax(dim=-1)
    weights = weights / weights.norm(dim=-2, keepdim=True)
    matrix = weights.mean(axis=(0, 1)).neg()

    text_indices, time_indices = dtw(matrix)

    jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
    jump_times = time_indices[jumps] / TOKENS_PER_SECOND

    if tokenizer.language in {"zh", "ja", "th", "lo", "my"}:
        # These languages don't typically use spaces, so it is difficult to split words
        # without morpheme analysis. Here, we instead split words at any
        # position where the tokens are decoded as valid unicode points
        words, word_tokens = tokenizer.split_tokens_on_unicode(tokens[1:].tolist())
    else:
        words, word_tokens = tokenizer.split_tokens_on_spaces(tokens[1:].tolist())

    word_boundaries = np.pad(np.cumsum([len(t) for t in word_tokens]), (1, 0))
    start_times = jump_times[word_boundaries[:-1]]
    end_times = jump_times[word_boundaries[1:]]

    return Alignment(words, word_tokens, start_times, end_times)


def add_word_timestamps(
    segments: List[dict],
    model: "Whisper",
    tokenizer: Tokenizer,
    mel: torch.Tensor,
    num_frames: int,
    **hyperparams,
):
    if len(segments) == 0:
        return

    text_tokens = [t for segment in segments for t in segment["tokens"]]
    alignment = find_alignment(model, tokenizer, text_tokens, mel, num_frames, **hyperparams)

    time_offset = segments[0]["seek"] * HOP_LENGTH / SAMPLE_RATE
    alignment.start_times = time_offset + alignment.start_times
    alignment.end_times = time_offset + alignment.end_times

    token_sources = np.repeat(np.arange(len(segments)), [len(s["tokens"]) for s in segments])
    token_sources: List[int] = [None] * len(tokenizer.sot_sequence) + list(token_sources)

    for segment in segments:
        segment["words"] = []

    word_boundaries = np.pad(np.cumsum([len(t) for t in alignment.word_tokens]), (1, 0))
    for i, (word, start, end) in enumerate(zip(alignment.words, alignment.start_times, alignment.end_times)):
        if word.startswith("<|") or word.strip() in ".,!?、。":  # TODO: expand
            continue

        segment = segments[token_sources[word_boundaries[i]]]
        segment["words"].append(dict(word=word, start=round(start, 2), end=round(end, 2)))

    # adjust the segment-level timestamps based on the word-level timestamps
    for segment in segments:
        if len(segment["words"]) > 0:
            segment["start"] = segment["words"][0]["start"]
            segment["end"] = segment["words"][-1]["end"]

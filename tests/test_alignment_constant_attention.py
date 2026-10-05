"""Offline alignment-normalization regressions with synthetic attention heads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import whisper.timing as timing


class Attention(torch.nn.Module):
    def forward(self, weights):
        return None, weights


class Model:
    device = torch.device("cpu")
    dims = SimpleNamespace(n_text_layer=1)

    def __init__(self, pattern, frames, dtype):
        self.attention = Attention()
        self.decoder = SimpleNamespace(
            blocks=[SimpleNamespace(cross_attn=self.attention)]
        )
        self.alignment_heads = torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [0, 1]]), torch.ones(2), (1, 2)
        ).coalesce()
        self.pattern, self.frames, self.dtype = pattern, frames, dtype
        self.logits = None

    def __call__(self, mel, tokens):
        count = tokens.shape[1]
        positions = torch.arange(self.frames, dtype=self.dtype)
        repeated = positions.repeat(count, 1)
        varied = (
            positions[None, :]
            * (torch.arange(count, dtype=self.dtype)[:, None] + 1)
            / 5
        )
        if self.pattern == "constant":
            heads = [repeated, repeated]
        elif self.pattern == "mixed":
            heads = [repeated, varied]
        else:
            heads = [varied, varied * 2]
        self.logits = torch.stack(heads).unsqueeze(0)
        self.attention(self.logits)
        return torch.zeros((1, count, 8), dtype=self.dtype)


class Tokenizer:
    sot_sequence = (5,)
    no_timestamps = 6
    eot = 4

    def split_to_word_tokens(self, tokens):
        return [" first", " second", "<eot>"], [[0], [1], [4]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("frames", [1, 7])
@pytest.mark.parametrize("pattern", ["constant", "mixed", "varying"])
def test_constant_attention_produces_finite_alignment_costs(
    monkeypatch, dtype, frames, pattern
):
    model = Model(pattern, frames, dtype)
    captured = []

    def align(cost):
        captured.append(cost.detach().float().numpy())
        assert torch.isfinite(cost).all(), (
            "A zero-variance attention head poisoned the alignment matrix"
        )
        # Isolate normalization from the independent DTW tie-breaking behavior.
        return np.array([np.arange(3), np.minimum(np.arange(3), frames - 1)])

    monkeypatch.setattr(timing, "dtw", align)
    result = timing.find_alignment(
        model, Tokenizer(), [0, 1], torch.zeros(2, 2), frames * 2, medfilt_width=1
    )
    weights = model.logits[0].softmax(dim=-1)
    std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
    centered = (weights - mean).float().numpy()
    scale = std.float().numpy()
    expected = np.zeros_like(centered)
    np.divide(centered, scale, out=expected, where=scale != 0)
    np.testing.assert_allclose(
        captured[0], -expected.mean(axis=0)[1:-1], rtol=0.02, atol=2e-3
    )
    assert len(result) == 2
    assert all(
        np.isfinite(word.start)
        and np.isfinite(word.end)
        and np.isfinite(word.probability)
        for word in result
    )
    assert all(word.start <= word.end for word in result)
    assert not model.attention._forward_hooks


def test_empty_text_retains_empty_result_without_running_model():
    class UnusedModel:
        def __call__(self, *args):
            raise AssertionError("Empty text must not run inference")

    assert (
        timing.find_alignment(UnusedModel(), Tokenizer(), [], torch.zeros(2, 2), 2)
        == []
    )


@pytest.mark.parametrize("frames", [1, 7])
def test_constant_heads_produce_valid_timings_through_real_dtw(frames):
    result = timing.find_alignment(
        Model("constant", frames, torch.float32),
        Tokenizer(),
        [0, 1],
        torch.zeros(2, 2),
        frames * 2,
        medfilt_width=1,
    )
    assert len(result) == 2
    for word in result:
        assert 0 <= word.start <= word.end <= frames / timing.TOKENS_PER_SECOND
        assert np.isfinite(word.probability)

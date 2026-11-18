import pytest
import torch

from whisper.model import SDPA_AVAILABLE, ModelDimensions, Whisper, disable_sdpa


def _make_model() -> Whisper:
    torch.manual_seed(0)
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=16,
        n_audio_state=16,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=100,
        n_text_ctx=16,
        n_text_state=16,
        n_text_head=2,
        n_text_layer=2,
    )
    return Whisper(dims).eval()


def _decode_in_chunks(model: Whisper, tokens: torch.Tensor, xa: torch.Tensor):
    cache, hooks = model.install_kv_cache_hooks()
    try:
        out0 = model.decoder(tokens[:, :3], xa, kv_cache=cache)
        out1 = model.decoder(tokens[:, 3:6], xa, kv_cache=cache)
        out2 = model.decoder(tokens[:, 6:], xa, kv_cache=cache)
        return torch.cat([out0, out1, out2], dim=1)
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.skipif(not SDPA_AVAILABLE, reason="SDPA requires PyTorch 2.0+")
def test_decoder_kv_cache_multi_token_chunk_sdpa():
    model = _make_model()
    xa = torch.randn(2, 16, 16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

    with torch.no_grad():
        expected = model.decoder(tokens, xa)
        actual = _decode_in_chunks(model, tokens, xa)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_decoder_kv_cache_multi_token_chunk_without_sdpa():
    model = _make_model()
    xa = torch.randn(2, 16, 16)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]])

    with torch.no_grad(), disable_sdpa():
        expected = model.decoder(tokens, xa)
        actual = _decode_in_chunks(model, tokens, xa)

    assert torch.allclose(actual, expected, atol=1e-5)


@pytest.mark.skipif(not SDPA_AVAILABLE, reason="SDPA requires PyTorch 2.0+")
def test_qkv_attention_offset_mask_fp16():
    model = _make_model()
    attn = model.decoder.blocks[0].attn
    mask = model.decoder.mask

    q = torch.randn(1, 6, 16, dtype=torch.float16)
    k = torch.randn(1, 6, 16, dtype=torch.float16)
    v = torch.randn(1, 6, 16, dtype=torch.float16)

    with torch.no_grad():
        expected_full, _ = attn.qkv_attention(q, k, v, mask=mask)
        actual_chunk, _ = attn.qkv_attention(q[:, 3:], k, v, mask=mask)

    assert torch.allclose(actual_chunk, expected_full[:, 3:], atol=1e-3)

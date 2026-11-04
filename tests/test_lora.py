import os

import pytest
import torch

import whisper
from whisper.lora import (
    LoRALinear,
    apply_lora,
    has_lora,
    load_lora,
    lora_state_dict,
    merge_lora,
    save_lora,
)
from whisper.model import Linear


@pytest.fixture(scope="module")
def checkpoint():
    model = whisper.load_model("tiny", device="cpu")
    return model.dims, model.state_dict()


def make_model(checkpoint):
    dims, state_dict = checkpoint
    model = whisper.model.Whisper(dims)
    model.load_state_dict(state_dict)
    return model.eval()


def make_inputs(model):
    torch.manual_seed(0)
    mel = torch.randn(2, model.dims.n_mels, 3000)
    tokens = torch.randint(0, 50000, (2, 12))
    return mel, tokens


def randomize_lora(model):
    torch.manual_seed(1)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            torch.nn.init.normal_(module.lora_B, std=0.02)


def test_output_unchanged_at_init(checkpoint):
    model = make_model(checkpoint)
    mel, tokens = make_inputs(model)
    with torch.no_grad():
        expected = model(mel, tokens)
        apply_lora(model, r=8, alpha=16, dropout=0.0)
        actual = model(mel, tokens)
    assert torch.equal(expected, actual)


def test_only_lora_is_trainable(checkpoint):
    model = make_model(checkpoint)
    apply_lora(model, r=4, target_modules=("query", "value"))

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)

    dims = model.dims
    per_layer_enc = 4 * (dims.n_audio_state + dims.n_audio_state)
    per_layer_dec = 4 * (dims.n_text_state + dims.n_text_state)
    # query and value in the encoder self-attention, and in both the decoder
    # self-attention and cross-attention
    expected = (
        2 * dims.n_audio_layer * per_layer_enc + 4 * dims.n_text_layer * per_layer_dec
    )
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected


def test_target_options(checkpoint):
    model = make_model(checkpoint)
    apply_lora(model, target_modules="all", apply_to="decoder")
    block = model.decoder.blocks[0]
    for attn in (block.attn, block.cross_attn):
        for name in ("query", "key", "value", "out"):
            assert isinstance(getattr(attn, name), LoRALinear)
    assert isinstance(block.mlp[0], LoRALinear) and isinstance(block.mlp[2], LoRALinear)
    assert not has_lora(model.encoder)

    with pytest.raises(ValueError):
        apply_lora(make_model(checkpoint), target_modules=("qkv",))
    with pytest.raises(RuntimeError):
        apply_lora(model)


def test_save_load_round_trip(checkpoint, tmp_path):
    model = make_model(checkpoint)
    apply_lora(model, r=4, alpha=8, target_modules=("query", "key", "value"))
    randomize_lora(model)
    path = os.path.join(tmp_path, "adapter.pt")
    save_lora(model, path, base_model="tiny")

    adapter = torch.load(path, weights_only=True)
    assert adapter["config"]["base_model"] == "tiny"
    assert adapter["config"]["r"] == 4
    assert all(v.dtype == torch.float32 for v in adapter["state_dict"].values())
    assert set(adapter["state_dict"]) == set(lora_state_dict(model))

    loaded = load_lora(make_model(checkpoint), path).eval()
    mel, tokens = make_inputs(model)
    with torch.no_grad():
        assert torch.equal(model(mel, tokens), loaded(mel, tokens))


def test_load_model_with_lora(checkpoint, tmp_path):
    model = make_model(checkpoint)
    apply_lora(model)
    randomize_lora(model)
    path = os.path.join(tmp_path, "adapter.pt")
    save_lora(model, path, base_model="tiny")

    loaded = whisper.load_model("tiny", device="cpu", lora=path).eval()
    mel, tokens = make_inputs(model)
    with torch.no_grad():
        assert torch.equal(model(mel, tokens), loaded(mel, tokens))


def test_mismatched_base_model(checkpoint, tmp_path):
    model = make_model(checkpoint)
    apply_lora(model)
    path = os.path.join(tmp_path, "adapter.pt")
    save_lora(model, path, base_model="tiny")

    dims, state_dict = checkpoint
    other = whisper.model.Whisper(
        whisper.model.ModelDimensions(**{**dims.__dict__, "n_text_layer": 2})
    )
    with pytest.raises(ValueError, match="different base model"):
        load_lora(other, path)


def test_merge_matches_unmerged(checkpoint, tmp_path):
    model = make_model(checkpoint)
    apply_lora(model, target_modules="all")
    randomize_lora(model)
    mel, tokens = make_inputs(model)
    with torch.no_grad():
        expected = model(mel, tokens)
        merge_lora(model)
        actual = model(mel, tokens)

    assert not has_lora(model)
    assert all(
        type(m) is Linear for m in model.modules() if isinstance(m, torch.nn.Linear)
    )
    assert torch.allclose(expected, actual, atol=1e-3, rtol=1e-3)

    # the merged model is a plain Whisper checkpoint
    path = os.path.join(tmp_path, "merged.pt")
    torch.save(
        {"dims": model.dims.__dict__, "model_state_dict": model.state_dict()}, path
    )
    reloaded = whisper.load_model(path, device="cpu").eval()
    with torch.no_grad():
        assert torch.equal(actual, reloaded(mel, tokens))


def test_transcribe_with_lora(checkpoint):
    model = make_model(checkpoint)
    expected = model.transcribe(
        os.path.join(os.path.dirname(__file__), "jfk.flac"), fp16=False
    )["text"]
    apply_lora(model)
    actual = model.transcribe(
        os.path.join(os.path.dirname(__file__), "jfk.flac"), fp16=False
    )["text"]
    assert actual == expected

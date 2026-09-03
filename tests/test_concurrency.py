from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch

import pytest
import torch

from whisper.decoding import PyTorchInference
from whisper.model import ModelDimensions, Whisper


def make_test_model(model_class=Whisper) -> Whisper:
    torch.manual_seed(0)
    model = model_class(
        ModelDimensions(
            n_mels=4,
            n_audio_ctx=4,
            n_audio_state=4,
            n_audio_head=1,
            n_audio_layer=1,
            n_vocab=16,
            n_text_ctx=8,
            n_text_state=4,
            n_text_head=1,
            n_text_layer=2,
        )
    )
    for parameter in model.parameters():
        torch.nn.init.uniform_(parameter, -0.1, 0.1)
    return model.eval()


def make_mel(batch_size=1, reverse=False):
    values = torch.linspace(-1, 1, steps=batch_size * 4 * 8)
    if reverse:
        values = values.flip(0)
    return values.reshape(batch_size, 4, 8)


def forward_hook_count(model: Whisper) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def self_attention_modules(model):
    for block in model.decoder.blocks:
        yield block.attn.key
        yield block.attn.value


def cross_attention_modules(model):
    for block in model.decoder.blocks:
        yield block.cross_attn.key
        yield block.cross_attn.value


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


class HookTrackingWhisper(Whisper):
    def __init__(self, dims):
        super().__init__(dims)
        self.cache_hook_installations = 0

    def install_kv_cache_hooks(self, cache=None):
        self.cache_hook_installations += 1
        return super().install_kv_cache_hooks(cache)


class DerivedWhisper(Whisper):
    pass


def test_request_local_cache_decodes_and_reorders_beams():
    model = make_test_model()
    inference = PyTorchInference(model, initial_token_length=2)
    tokens = torch.tensor([[1, 2], [3, 4]])

    try:
        with torch.no_grad():
            audio_features = model.encoder(make_mel())
            expected = model.decoder(tokens, audio_features)
            actual = inference.logits(tokens, audio_features)
            torch.testing.assert_close(actual, expected)

            cross_cache = {
                module: inference.kv_cache[module]
                for module in cross_attention_modules(model)
            }
            tokens = torch.tensor([[1, 2, 5], [3, 4, 6]])
            actual = inference.logits(tokens, audio_features)[:, -1]
            expected = model.decoder(tokens, audio_features)[:, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

            self_cache = {
                module: inference.kv_cache[module].clone()
                for module in self_attention_modules(model)
            }
            inference.rearrange_kv_cache([1, 0])

            for module, cached_value in self_cache.items():
                torch.testing.assert_close(
                    inference.kv_cache[module], cached_value[[1, 0]]
                )
            for module, cached_value in cross_cache.items():
                assert inference.kv_cache[module] is cached_value

            tokens = torch.tensor([[3, 4, 6, 7], [1, 2, 5, 8]])
            actual = inference.logits(tokens, audio_features)[:, -1]
            expected = model.decoder(tokens, audio_features)[:, -1]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        assert not inference.hooks
        assert forward_hook_count(model) == 0
    finally:
        inference.cleanup_caching()


@pytest.mark.parametrize("model_class", [Whisper, DerivedWhisper])
def test_native_inference_is_isolated_during_overlap(model_class):
    model = make_test_model(model_class)
    tokens_a = torch.tensor([[1, 2]])
    tokens_b = torch.tensor([[4, 5]])

    with torch.no_grad():
        audio_a = model.encoder(make_mel())
        audio_b = model.encoder(make_mel(reverse=True))
        expected_a = model.decoder(tokens_a, audio_a)
        expected_b = model.decoder(tokens_b, audio_b)

    inference_a = PyTorchInference(model, initial_token_length=2)
    inference_b = PyTorchInference(model, initial_token_length=2)
    key = model.decoder.blocks[0].attn.key

    def run(inference, tokens, audio):
        with torch.no_grad():
            return inference.logits(tokens, audio)

    def run_overlapped(tokens_a, tokens_b):
        barrier = Barrier(2)
        original_forward = key.forward

        def synchronized_forward(x):
            barrier.wait(timeout=15)
            return original_forward(x)

        with patch.object(key, "forward", new=synchronized_forward):
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_a = executor.submit(run, inference_a, tokens_a, audio_a)
                future_b = executor.submit(run, inference_b, tokens_b, audio_b)
                return future_a.result(timeout=30), future_b.result(timeout=30)

    try:
        actual_a, actual_b = run_overlapped(tokens_a, tokens_b)

        torch.testing.assert_close(actual_a, expected_a)
        torch.testing.assert_close(actual_b, expected_b)
        assert inference_a.kv_cache is not inference_b.kv_cache
        assert inference_a.kv_cache.keys() == inference_b.kv_cache.keys()
        assert not inference_a.hooks and not inference_b.hooks

        tokens_a = torch.tensor([[1, 2, 3]])
        tokens_b = torch.tensor([[4, 5, 6]])
        with torch.no_grad():
            expected_a = model.decoder(tokens_a, audio_a)[:, -1]
            expected_b = model.decoder(tokens_b, audio_b)[:, -1]

        actual_a, actual_b = run_overlapped(tokens_a, tokens_b)
        torch.testing.assert_close(actual_a[:, -1], expected_a)
        torch.testing.assert_close(actual_b[:, -1], expected_b)
    finally:
        inference_a.cleanup_caching()
        inference_b.cleanup_caching()


def test_whisper_subclass_keeps_its_cache_hook_contract():
    model = make_test_model(HookTrackingWhisper)
    inference = PyTorchInference(model, initial_token_length=2)

    try:
        with torch.no_grad():
            audio_features = model.encoder(make_mel())
            inference.logits(torch.tensor([[1, 2]]), audio_features)

        assert inference._use_legacy_cache
        assert model.cache_hook_installations == 1
        assert inference.hooks
    finally:
        inference.cleanup_caching()


def test_class_cache_hook_override_keeps_legacy_path():
    original_installer = Whisper.install_kv_cache_hooks

    def install_kv_cache_hooks(self, cache=None):
        return original_installer(self, cache)

    with patch.object(Whisper, "install_kv_cache_hooks", new=install_kv_cache_hooks):
        model = make_test_model()
        inference = PyTorchInference(model, initial_token_length=2)
        assert inference._use_legacy_cache


def test_legacy_hooks_preserve_external_cache_contract():
    model = make_test_model()
    batch_size = 2
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])

    with torch.no_grad():
        audio_features = model.encoder(make_mel(batch_size=batch_size))
        expected = model.decoder(tokens, audio_features)

    captured, hooks = model.install_kv_cache_hooks()
    try:
        passed_cache = {}
        with torch.no_grad():
            actual = model.decoder(tokens, audio_features, kv_cache=passed_cache)

        torch.testing.assert_close(actual, expected)
        assert passed_cache == {}
        for module in self_attention_modules(model):
            assert captured[module].shape == (batch_size, 3, 4)
        for module in cross_attention_modules(model):
            assert captured[module].shape == (batch_size, 4, 4)

    finally:
        remove_hooks(hooks)

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import whisper.model as model_module
from whisper.model import MultiHeadAttention, disable_sdpa


def make_attention_inputs():
    return tuple(torch.randn(1, 2, 4) for _ in range(3))


def uses_sdpa(attention, inputs):
    _, weights = attention.qkv_attention(*inputs)
    return weights is None


@pytest.fixture
def sdpa_probe(monkeypatch):
    calls = []

    def scaled_dot_product_attention(query, key, value, *, is_causal):
        del key, value, is_causal
        calls.append(threading.get_ident())
        return query

    monkeypatch.setattr(model_module, "SDPA_AVAILABLE", True)
    monkeypatch.setattr(
        model_module,
        "scaled_dot_product_attention",
        scaled_dot_product_attention,
    )
    monkeypatch.setattr(MultiHeadAttention, "use_sdpa", True)
    return calls


def test_disable_sdpa_does_not_affect_another_thread(sdpa_probe):
    disabled_attention = MultiHeadAttention(4, 1)
    enabled_attention = MultiHeadAttention(4, 1)
    inputs = make_attention_inputs()
    disabled_context_ready = threading.Event()
    enabled_context_done = threading.Event()

    def run_disabled():
        with disable_sdpa():
            disabled_context_ready.set()
            if not enabled_context_done.wait(timeout=10):
                raise TimeoutError("enabled context did not finish")
            return threading.get_ident(), uses_sdpa(disabled_attention, inputs)

    def run_enabled():
        if not disabled_context_ready.wait(timeout=10):
            raise TimeoutError("disabled context did not start")
        try:
            return threading.get_ident(), uses_sdpa(enabled_attention, inputs)
        finally:
            enabled_context_done.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        disabled = executor.submit(run_disabled)
        enabled = executor.submit(run_enabled)
        disabled_thread, disabled_result = disabled.result(timeout=15)
        enabled_thread, enabled_result = enabled.result(timeout=15)

    assert disabled_thread != enabled_thread
    assert disabled_result is False
    assert enabled_result is True
    assert sdpa_probe == [enabled_thread]


def test_overlapping_disable_sdpa_contexts_restore_independently(sdpa_probe):
    first_attention = MultiHeadAttention(4, 1)
    second_attention = MultiHeadAttention(4, 1)
    unrelated_attention = MultiHeadAttention(4, 1)
    inputs = make_attention_inputs()
    contexts_ready = threading.Barrier(3)
    allow_first_exit = threading.Event()
    first_exited = threading.Event()
    second_checked = threading.Event()
    allow_second_exit = threading.Event()

    def run_first():
        try:
            with disable_sdpa():
                contexts_ready.wait(timeout=10)
                if not allow_first_exit.wait(timeout=10):
                    raise TimeoutError("first context was not released")
                return threading.get_ident(), uses_sdpa(first_attention, inputs)
        finally:
            first_exited.set()

    def run_second():
        with disable_sdpa():
            contexts_ready.wait(timeout=10)
            if not first_exited.wait(timeout=10):
                raise TimeoutError("first context did not exit")
            result = threading.get_ident(), uses_sdpa(second_attention, inputs)
            second_checked.set()
            if not allow_second_exit.wait(timeout=10):
                raise TimeoutError("second context was not released")
            return result

    main_thread = threading.get_ident()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_first)
        second = executor.submit(run_second)
        try:
            contexts_ready.wait(timeout=10)
            assert uses_sdpa(unrelated_attention, inputs) is True
            allow_first_exit.set()
            first_thread, first_result = first.result(timeout=15)
            if not second_checked.wait(timeout=10):
                raise TimeoutError("second context did not observe the first exit")
            assert uses_sdpa(unrelated_attention, inputs) is True
        finally:
            allow_first_exit.set()
            allow_second_exit.set()

        second_thread, second_result = second.result(timeout=15)

    assert first_thread != second_thread
    assert first_result is False
    assert second_result is False
    assert sdpa_probe == [main_thread, main_thread]


def test_disable_sdpa_supports_nested_contexts(sdpa_probe):
    attention = MultiHeadAttention(4, 1)
    inputs = make_attention_inputs()

    assert uses_sdpa(attention, inputs) is True
    with disable_sdpa():
        assert uses_sdpa(attention, inputs) is False
        with disable_sdpa():
            assert uses_sdpa(attention, inputs) is False
        assert uses_sdpa(attention, inputs) is False
    assert uses_sdpa(attention, inputs) is True

    assert len(sdpa_probe) == 2


def test_disable_sdpa_restores_context_after_an_exception(sdpa_probe):
    attention = MultiHeadAttention(4, 1)
    inputs = make_attention_inputs()

    with pytest.raises(RuntimeError, match="failed inside context"):
        with disable_sdpa():
            assert uses_sdpa(attention, inputs) is False
            raise RuntimeError("failed inside context")

    assert uses_sdpa(attention, inputs) is True
    assert len(sdpa_probe) == 1


def test_manual_global_sdpa_disable_is_preserved(sdpa_probe, monkeypatch):
    attention = MultiHeadAttention(4, 1)
    inputs = make_attention_inputs()
    monkeypatch.setattr(MultiHeadAttention, "use_sdpa", False)

    assert uses_sdpa(attention, inputs) is False
    with disable_sdpa():
        assert uses_sdpa(attention, inputs) is False
    assert uses_sdpa(attention, inputs) is False
    assert sdpa_probe == []

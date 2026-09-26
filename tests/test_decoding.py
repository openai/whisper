import pytest

from whisper.decoding import DecodingOptions, DecodingTask
from whisper.model import ModelDimensions, Whisper


@pytest.fixture
def model():
    dims = ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=8,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=8,
        n_text_head=2,
        n_text_layer=1,
    )
    return Whisper(dims)


def test_short_prefix(model):
    prefix = [1000, 2000, 3000]
    task = DecodingTask(model, DecodingOptions(prefix=prefix))

    assert task.initial_tokens == task.sot_sequence + tuple(prefix)


def test_long_prefix_is_truncated(model):
    prefix = list(range(1000))
    task = DecodingTask(model, DecodingOptions(prefix=prefix))

    assert len(task.initial_tokens) == model.dims.n_text_ctx // 2
    assert task.initial_tokens[-1] == prefix[-1]


def test_long_prompt_and_prefix_fit_the_context(model):
    task = DecodingTask(
        model, DecodingOptions(prompt=list(range(1000)), prefix=list(range(1000)))
    )

    assert len(task.initial_tokens) <= model.dims.n_text_ctx

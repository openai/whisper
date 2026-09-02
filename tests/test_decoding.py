from types import SimpleNamespace

import pytest

from whisper.decoding import DecodingOptions, DecodingTask
from whisper.tokenizer import get_tokenizer


def get_suppress_tokens(**kwargs):
    tokenizer = get_tokenizer(multilingual=False)
    task = SimpleNamespace(options=DecodingOptions(**kwargs), tokenizer=tokenizer)
    return DecodingTask._get_suppress_tokens(task), tokenizer


@pytest.mark.parametrize("value", [(5,), iter([5]), {5}])
def test_suppress_tokens_accepts_any_iterable(value):
    from_iterable, _ = get_suppress_tokens(suppress_tokens=value)
    from_list, _ = get_suppress_tokens(suppress_tokens=[5])
    assert from_iterable == from_list
    assert 5 in from_iterable


def test_suppress_tokens_does_not_mutate_the_options_list():
    tokens = [5]
    get_suppress_tokens(suppress_tokens=tokens)
    assert tokens == [5]


@pytest.mark.parametrize("value", [None, "", []])
def test_suppress_tokens_empty_means_only_special_tokens(value):
    suppress, tokenizer = get_suppress_tokens(suppress_tokens=value)
    specials = {
        tokenizer.transcribe,
        tokenizer.translate,
        tokenizer.sot,
        tokenizer.sot_prev,
        tokenizer.sot_lm,
    }
    if tokenizer.no_speech is not None:
        specials.add(tokenizer.no_speech)
    assert set(suppress) == specials


def test_suppress_tokens_default_includes_non_speech_tokens():
    suppress, tokenizer = get_suppress_tokens(suppress_tokens="-1")
    assert set(tokenizer.non_speech_tokens) <= set(suppress)
    assert -1 not in suppress


@pytest.mark.parametrize("value", ["1,,2", ",1", "1,", "1,x"])
def test_suppress_tokens_rejects_malformed_strings(value):
    # only the empty string means "no tokens"; anything else is parsed
    # strictly, so an empty or non-numeric component still raises
    with pytest.raises(ValueError):
        get_suppress_tokens(suppress_tokens=value)

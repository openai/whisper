import pytest

from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize("multilingual", [True, False])
def test_tokenizer(multilingual):
    tokenizer = get_tokenizer(multilingual=False)
    assert tokenizer.sot in tokenizer.sot_sequence
    assert len(tokenizer.all_language_codes) == len(tokenizer.all_language_tokens)
    assert all(c < tokenizer.timestamp_begin for c in tokenizer.all_language_tokens)


def test_multilingual_tokenizer():
    gpt2_tokenizer = get_tokenizer(multilingual=False)
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    text = "다람쥐 헌 쳇바퀴에 타고파"
    gpt2_tokens = gpt2_tokenizer.encode(text)
    multilingual_tokens = multilingual_tokenizer.encode(text)

    assert gpt2_tokenizer.decode(gpt2_tokens) == text
    assert multilingual_tokenizer.decode(multilingual_tokens) == text
    assert len(gpt2_tokens) > len(multilingual_tokens)


def test_split_on_unicode():
    multilingual_tokenizer = get_tokenizer(multilingual=True)

    tokens = [8404, 871, 287, 6, 246, 526, 3210, 20378]
    words, word_tokens = multilingual_tokenizer.split_tokens_on_unicode(tokens)

    assert words == [" elle", " est", " l", "'", "\ufffd", "é", "rit", "oire"]
    assert word_tokens == [[8404], [871], [287], [6], [246], [526], [3210], [20378]]


def test_language_unsupported_by_model():
    # "yue" is the 100th entry of LANGUAGES and only exists in large-v3 / turbo,
    # whose checkpoints report num_languages=100. Every other multilingual
    # checkpoint reports 99, so asking it for "yue" has to fail with a message
    # that names the language instead of an IndexError-style ValueError.
    assert get_tokenizer(True, num_languages=100, language="yue").sot_sequence

    with pytest.raises(ValueError, match="yue"):
        get_tokenizer(True, num_languages=99, language="yue")

    with pytest.raises(ValueError, match="yue"):
        get_tokenizer(True, num_languages=99, language="cantonese")

    # languages that every multilingual checkpoint supports are unaffected
    assert get_tokenizer(True, num_languages=99, language="zh").sot_sequence

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


def test_cjk_quotation_marks_are_not_suppressed():
    """「」『』 are ordinary CJK punctuation, not speaker tags.

    They are in the default `--suppress_tokens=-1` set, so every CLI run silently
    dropped the quotation marks from Chinese and Japanese output.
    """
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    non_speech = set(tokenizer.non_speech_tokens)

    for char in "「」『』":
        tokens = tokenizer.encoding.encode(char)
        assert not any(token in non_speech for token in tokens), char


def test_non_speech_symbols_are_still_suppressed():
    """Control: the suppression set keeps doing its actual job."""
    tokenizer = get_tokenizer(multilingual=True, language="zh", task="transcribe")
    non_speech = set(tokenizer.non_speech_tokens)

    for char in "♪#()*":
        tokens = tokenizer.encoding.encode(char)
        assert any(token in non_speech for token in tokens), char

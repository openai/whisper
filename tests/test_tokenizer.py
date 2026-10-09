import pytest

from whisper.tokenizer import get_tokenizer


@pytest.mark.parametrize(
    "multilingual, expected_prefix",
    [
        (True, "<|startoftranscript|><|en|><|transcribe|>"),
        (False, "<|startoftranscript|>"),
    ],
)
def test_tokenizer(multilingual, expected_prefix):
    tokenizer = get_tokenizer(multilingual=multilingual)
    assert tokenizer.decode(tokenizer.sot_sequence) == expected_prefix
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
